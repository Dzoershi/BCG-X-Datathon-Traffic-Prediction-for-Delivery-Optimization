"""Chronos-2 zero-shot forecasts (issue #19).

Amazon's pretrained Chronos-2 (``amazon/chronos-2``, revision pinned below)
forecasts every evaluated arc from the cutoff-limited history, without training
on our data. It follows the ``ForecastModel`` contract (``models/base.py``).

- Context: the last ``context_hours`` hours (8,192 at most, about 11 months).
  Hours whose ``qualite_mesure`` is not ``ok`` become NaN, which Chronos-2 masks.
- Horizon: from the hour after the last history label to the last target hour,
  i.e. 144 hours from Thursday 23:00 (the Friday gap, then the 120 targets kept).
- Covariates: calendar and weather values, known for the whole horizon.
- Defaults: the best backtest run (run 4) -- both targets forecast jointly, the six
  calendar covariates, the median. Weather was tested and brought no gain.
- Fallback: an arc with fewer than 168 ``ok`` hours (one week) in the last 28 days
  takes the ``HolidayAwarePreviousWeekBaseline`` forecast. With an all-NaN context,
  Chronos-2 would silently forecast about 0 (Champs-Elysées since 2025-11-19).
- Forecasts are clipped to flow >= 0 and occupancy in [0, 100].

torch and chronos are imported only when the weights are loaded, so this module
imports without them (the CI installs neither). To work offline, download the
weights once (478 MB), then forbid network access::

    .\\.venv\\Scripts\\hf.exe download amazon/chronos-2 --revision <MODEL_REVISION>
    $env:HF_HUB_OFFLINE = "1"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from traffic_prediction.data.build import DEFAULT_WEATHER_DIR
from traffic_prediction.evaluate import KEYS, TARGETS
from traffic_prediction.models.baselines import (
    DEFAULT_CALENDAR, SUNDAY, HolidayAwarePreviousWeekBaseline, _valid_history,
)

MODEL_ID = "amazon/chronos-2"
# Hugging Face commit of the weights (2026-06-05): pinned for reproducible forecasts.
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
MAX_CONTEXT_HOURS = 8192
SEED = 42
DEFAULT_WEATHER = DEFAULT_WEATHER_DIR / "weather.parquet"
# Native Chronos-2 levels 0.05 to 0.95, evenly spaced: their mean approximates the
# forecast mean. Rounding makes them equal to the model's levels (no interpolation).
QUANTILES = [round(0.05 * k, 2) for k in range(1, 20)]
RECENT_HOURS = 28 * 24
MIN_RECENT_OK_HOURS = 168
ID_SEPARATOR = "|"
CALENDAR_COVARIATES = (
    "is_holiday", "is_school_holiday", "is_day_before_holiday", "is_bridge", "is_sunday_or_holiday",
    "day_type",
)
WEATHER_COVARIATES = ("temperature_c", "wind_speed_kmh", "rain_3h")
DAY_KINDS = ("ouvre", "weekend", "ferie")


def forecast_horizon(last: pd.Timestamp, target_times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Hours from the one after ``last`` to the last target (144 from Thursday 23:00)."""
    if target_times.min() <= last:
        raise ValueError("Target hours must all follow the last history hour")
    return pd.date_range(last + pd.Timedelta(hours=1), target_times.max(), freq="h")


def context_values(
    history: pd.DataFrame, arcs: Sequence[str], last: pd.Timestamp, hours: int
) -> pd.DataFrame:
    """Targets of each arc on a regular hourly grid ending at ``last``; non-ok hours are NaN."""
    start = max(history["datetime"].min(), last - pd.Timedelta(hours=hours - 1))
    grid = pd.date_range(start, last, freq="h")
    return _valid_history(history).reindex(pd.MultiIndex.from_product([arcs, grid], names=KEYS))


def has_recent_history(values: pd.DataFrame, last: pd.Timestamp) -> pd.Series:
    """Per arc: at least ``MIN_RECENT_OK_HOURS`` measured hours in the last 28 days."""
    recent = values[values.index.get_level_values("datetime") > last - pd.Timedelta(hours=RECENT_HOURS)]
    counts = recent.notna().all(axis=1).groupby(level="arc", sort=False).sum()
    arcs = values.index.get_level_values("arc").unique()
    return counts.reindex(arcs, fill_value=0).ge(MIN_RECENT_OK_HOURS)


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar indexed by datetime, plus two ways of saying that a holiday looks like a Sunday (#17).

    ``is_sunday_or_holiday`` is a 0/1 flag; ``day_type`` is the day of week (0 = Monday)
    with holidays coded 6, as in the baselines. Holidays are rare in the context, but
    Sundays occur every week: ``day_type`` lets Chronos-2 copy their profile.
    """
    table = calendar.set_index("datetime")
    holiday = table["is_holiday"].astype(bool)
    return table.assign(
        is_sunday_or_holiday=table["day_of_week"].eq(SUNDAY) | holiday,
        day_type=table["day_of_week"].where(~holiday, SUNDAY),
    )


def day_kinds(times: pd.DatetimeIndex, calendar: pd.DataFrame) -> np.ndarray:
    """``DAY_KINDS`` label of each hour: public holiday first, then Saturday and Sunday."""
    days = calendar.set_index("datetime").reindex(times)
    if days["day_of_week"].isna().any():
        raise ValueError("Calendar does not cover every forecast hour; rebuild it")
    return np.where(days["is_holiday"].astype(bool), "ferie",
                    np.where(days["day_of_week"] >= 5, "weekend", "ouvre"))


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Weather indexed by datetime, plus rain_3h: hourly rain is noisy (#21), its 3-hour sum less so."""
    table = weather.set_index("datetime").sort_index()
    return table.assign(rain_3h=table["precipitation_mm"].rolling(3, min_periods=1).sum())


def chronos_inputs(
    values: pd.DataFrame, covariates: pd.DataFrame, horizon: pd.DatetimeIndex, multivariate: bool
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Long frames for ``predict_df`` and the target column names.

    One series per arc with both targets (multivariate), or per "arc|variable".
    Covariates are joined on the context (df) and on the horizon (future_df).
    """
    frame = values.reset_index().rename(columns={"arc": "id", "datetime": "timestamp"})
    targets = list(TARGETS)
    if not multivariate:
        frame = frame.melt(["id", "timestamp"], targets, var_name="variable", value_name="target")
        frame["id"] = frame["id"] + ID_SEPARATOR + frame.pop("variable")
        targets = ["target"]
    if covariates.columns.empty:
        return frame, None, targets
    future = pd.MultiIndex.from_product(
        [frame["id"].unique(), horizon], names=["id", "timestamp"]
    ).to_frame(index=False)
    return frame.join(covariates, on="timestamp"), future.join(covariates, on="timestamp"), targets


def point_forecast(
    raw: pd.DataFrame, point: str, multivariate: bool,
    kinds: np.ndarray | None = None, by_kind: dict[str, float] | None = None,
) -> pd.DataFrame:
    """``predict_df`` output to (arc, datetime) rows: median, or mean of ``QUANTILES``.

    ``by_kind`` replaces that value by one quantile on the rows of each listed day kind
    (``kinds`` gives the kind of each row of ``raw``).
    """
    levels = ["0.5"] if point == "median" else [str(level) for level in QUANTILES]
    values = raw[levels].astype("float64").mean(axis=1)  # Chronos-2 returns float32.
    for kind, level in (by_kind or {}).items():
        rows = kinds == kind
        values.loc[rows] = raw.loc[rows, str(level)].astype("float64")
    frame = pd.DataFrame({"datetime": raw["timestamp"], "value": values})
    if multivariate:
        frame["arc"], frame["variable"] = raw["id"], raw["target_name"]
    else:
        frame[["arc", "variable"]] = raw["id"].str.rsplit(ID_SEPARATOR, n=1, expand=True)
    table = frame.pivot(index=KEYS, columns="variable", values="value")
    return table[list(TARGETS)].rename_axis(columns=None)


class Chronos2Forecaster:
    """Chronos-2 forecasts of every evaluated arc, with the baseline as fallback.

    Options select the experiment: known covariates from the calendar and the
    weather (names in ``CALENDAR_COVARIATES`` and ``WEATHER_COVARIATES``, or any
    column of those tables), both targets forecast jointly (``multivariate``),
    information shared across arcs (``cross_learning``), and the point forecast
    (``"median"`` or ``"mean"`` of the quantiles). ``point_by_day_type`` takes one
    quantile per day kind instead, e.g. ``{"ouvre": 0.5, "weekend": 0.2, "ferie": 0.2}``:
    Chronos-2 overestimates traffic on non-working days (#19 backtest); kinds left out
    keep ``point``. The defaults are the retained run 4, with the median everywhere;
    ``Chronos2Forecaster(calendar)`` is what the submission script needs.
    The weather is read from ``data/processed/weather`` unless given. After each
    call, ``sources_`` gives "chronos" or the fallback rule per (arc, datetime),
    and ``raw_forecast_`` the quantiles returned by Chronos-2.
    """

    def __init__(
        self,
        calendar: pd.DataFrame | None = None,
        *,
        weather: pd.DataFrame | None = None,
        calendar_covariates: Sequence[str] = CALENDAR_COVARIATES,
        weather_covariates: Sequence[str] = (),
        multivariate: bool = True,
        cross_learning: bool = False,
        point: str = "median",
        point_by_day_type: Mapping[str, float] | None = None,
        context_hours: int = MAX_CONTEXT_HOURS,
        pipeline: Any = None,
    ) -> None:
        if point not in ("median", "mean"):
            raise ValueError("point must be 'median' or 'mean'")
        by_kind = dict(point_by_day_type or {})
        if set(by_kind) - set(DAY_KINDS) or any(level not in QUANTILES for level in by_kind.values()):
            raise ValueError(f"point_by_day_type maps {DAY_KINDS} to quantiles among {QUANTILES}")
        if not 0 < context_hours <= MAX_CONTEXT_HOURS:
            raise ValueError(f"context_hours must be between 1 and {MAX_CONTEXT_HOURS}")
        self.calendar = calendar
        self.weather = weather
        self.calendar_covariates = tuple(calendar_covariates)
        self.weather_covariates = tuple(weather_covariates)
        self.multivariate = multivariate
        self.cross_learning = cross_learning
        self.point = point
        self.point_by_day_type = by_kind
        self.context_hours = context_hours
        self.pipeline = pipeline
        self.sources_: pd.Series | None = None
        self.raw_forecast_: pd.DataFrame | None = None

    def describe(self) -> str:
        """Settings in one line, for docs/experiments.csv."""
        covariates = [*self.calendar_covariates, *self.weather_covariates]
        mode = "multivariate" if self.multivariate else "univariate"
        return (
            f"Chronos-2 zero-shot ({MODEL_ID}@{MODEL_REVISION[:8]}), context {self.context_hours} h, "
            f"covariates {covariates or 'none'}, {mode}{', cross-learning' if self.cross_learning else ''}, "
            f"{self.point} of quantiles"
            f"{f', quantile by day type {self.point_by_day_type}' if self.point_by_day_type else ''}; "
            f"fallback HolidayAwarePreviousWeekBaseline if fewer than "
            f"{MIN_RECENT_OK_HOURS} ok hours in the last 28 days"
        )

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        self.raw_forecast_ = None  # Stays None when every arc takes the fallback.
        last = history["datetime"].max()
        horizon = forecast_horizon(last, target_times)
        arcs = list(history.loc[history["role"].eq("evalue"), "arc"].unique())
        values = context_values(history, arcs, last, self.context_hours)
        recent = has_recent_history(values, last)
        parts, sources = [], []
        chronos_arcs = [arc for arc in arcs if recent[arc]]
        if chronos_arcs:
            index = pd.MultiIndex.from_product([chronos_arcs, target_times], names=KEYS)
            part = self._chronos(values.loc[chronos_arcs], horizon).reindex(index)
            parts.append(part)
            sources.append(pd.Series("chronos", index=index, dtype=object))
        if len(chronos_arcs) < len(arcs):
            baseline = HolidayAwarePreviousWeekBaseline(self._calendar())
            fallback = baseline.predict(history, target_times).set_index(KEYS)
            keep = ~fallback.index.get_level_values("arc").isin(chronos_arcs)
            parts.append(fallback.loc[keep, list(TARGETS)])
            sources.append("fallback_" + baseline.sources_[keep])
        index = pd.MultiIndex.from_product([arcs, target_times], names=KEYS)
        forecast = pd.concat(parts).reindex(index)
        if forecast.isna().any().any():
            raise ValueError("Chronos-2 returned missing forecasts")
        forecast["debit_horaire"] = forecast["debit_horaire"].clip(lower=0)
        forecast["taux_occupation"] = forecast["taux_occupation"].clip(0, 100)
        self.sources_ = pd.concat(sources).reindex(index)
        return forecast.reset_index()

    def _chronos(self, values: pd.DataFrame, horizon: pd.DatetimeIndex) -> pd.DataFrame:
        """Point forecasts of the given arcs over the whole horizon."""
        times = values.index.get_level_values("datetime").unique().append(horizon)
        df, future_df, targets = chronos_inputs(values, self._covariates(times), horizon, self.multivariate)
        raw = self._pipeline().predict_df(
            df, future_df=future_df, id_column="id", timestamp_column="timestamp",
            target=targets, prediction_length=len(horizon), quantile_levels=QUANTILES,
            context_length=self.context_hours, cross_learning=self.cross_learning, freq="h",
        )
        self.raw_forecast_ = raw
        kinds = None
        if self.point_by_day_type:
            kinds = day_kinds(pd.DatetimeIndex(raw["timestamp"]), self._calendar())
        return point_forecast(raw, self.point, self.multivariate, kinds, self.point_by_day_type)

    def _covariates(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """Selected covariates at ``times`` as 0/1 or real floats; fail if any value is missing."""
        tables = []
        if self.calendar_covariates:
            tables.append(calendar_features(self._calendar())[list(self.calendar_covariates)])
        if self.weather_covariates:
            tables.append(weather_features(self._weather())[list(self.weather_covariates)])
        if not tables:
            return pd.DataFrame(index=times)
        table = pd.concat(tables, axis=1).reindex(times)
        if table.isna().any().any():
            raise ValueError("Calendar or weather does not cover every context and horizon hour; rebuild them")
        # Floats, not booleans: Chronos-2 would encode booleans as categories.
        return table.astype("float32")

    def _calendar(self) -> pd.DataFrame:
        if self.calendar is None:
            self.calendar = pd.read_parquet(DEFAULT_CALENDAR)
        return self.calendar

    def _weather(self) -> pd.DataFrame:
        if self.weather is None:
            self.weather = pd.read_parquet(DEFAULT_WEATHER)
        return self.weather

    def _pipeline(self) -> Any:
        """Load the pinned weights once, on the GPU if available, otherwise on the CPU."""
        if self.pipeline is None:
            import torch
            from chronos import Chronos2Pipeline

            # Chronos-2 predicts quantiles without sampling; the seed guards any random step.
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = Chronos2Pipeline.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, device_map=device
            )
        return self.pipeline
