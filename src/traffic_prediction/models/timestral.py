"""Pinned Timestral zero-shot forecasts, using the shared backtest contract.

Preparation and outage eligibility match Chronos-2. Non-ok measurements remain
NaN; arcs without enough recent measurements use the explicit baseline fallback.
Neither torch nor the upstream timestral package is imported until inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from traffic_prediction.data.traffic import REPO_ROOT
from traffic_prediction.evaluate import KEYS, TARGETS, _check_frame
from traffic_prediction.models.baselines import DEFAULT_CALENDAR, HolidayAwarePreviousWeekBaseline
from traffic_prediction.models.chronos import (
    CALENDAR_COVARIATES, MIN_RECENT_OK_HOURS, SEED, calendar_features,
    context_values, forecast_horizon, has_recent_history,
)

SOURCE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
MODEL_REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
MAX_CONTEXT_HOURS = 32768
DEFAULT_CONTEXT_HOURS = 8192  # Match the first Chronos-2 comparisons.


class TimestralForecaster:
    """Forecast flow and occupancy, independently by default, with a pinned checkpoint.

    ``calendar_covariates=()`` selects the history-only experiment; otherwise the
    selected calendar columns cover both context and the full forecast horizon.
    ``multivariate=True`` forecasts both variables jointly within each arc.
    ``pipeline`` injects an upstream-compatible backend for offline tests.
    ``sources_`` identifies model/fallback hours after every call.
    """

    def __init__(
        self, calendar: pd.DataFrame | None = None, *, model_size: str = "5M",
        calendar_covariates: Sequence[str] = CALENDAR_COVARIATES,
        context_hours: int = DEFAULT_CONTEXT_HOURS, multivariate: bool = False,
        pipeline: Any = None,
    ) -> None:
        if model_size not in MODEL_REVISIONS:
            raise ValueError(f"model_size must be one of {tuple(MODEL_REVISIONS)}")
        if (isinstance(context_hours, bool) or not isinstance(context_hours, int)
                or not MIN_RECENT_OK_HOURS <= context_hours <= MAX_CONTEXT_HOURS):
            raise ValueError(f"context_hours must be an integer from {MIN_RECENT_OK_HOURS} to {MAX_CONTEXT_HOURS}")
        selected = tuple(calendar_covariates)
        if len(set(selected)) != len(selected) or set(selected) - set(CALENDAR_COVARIATES):
            raise ValueError(f"Choose unique calendar_covariates from {CALENDAR_COVARIATES}")
        self.calendar, self.model_size = calendar, model_size
        self.calendar_covariates, self.context_hours = selected, context_hours
        self.multivariate, self.pipeline = multivariate, pipeline
        self.sources_: pd.Series | None = None

    def describe(self) -> str:
        mode = "multivariate" if self.multivariate else "univariate"
        return (
            f"Timestral-{self.model_size}@{MODEL_REVISIONS[self.model_size][:8]}, "
            f"source {SOURCE_REVISION[:8]}, context {self.context_hours} h, {mode}, "
            f"calendar {list(self.calendar_covariates) or 'none'}, median, upstream clampsym defaults; "
            f"HolidayAwarePreviousWeekBaseline fallback if fewer than {MIN_RECENT_OK_HOURS} "
            "ok hours in the retained context's last 28 days"
        )

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        self.sources_ = None
        _check_frame(history, [*KEYS, *TARGETS, "role", "qualite_mesure"], "history")
        if (not isinstance(target_times, pd.DatetimeIndex) or target_times.empty
                or target_times.tz is not None or target_times.hasnans
                or target_times.has_duplicates or not target_times.is_monotonic_increasing
                or not target_times.equals(target_times.floor("h"))):
            raise ValueError("target_times must be unique, increasing, timezone-free whole hours")
        arcs = list(history.loc[history["role"].eq("evalue"), "arc"].unique())
        if history.empty or not arcs:
            raise ValueError("History must contain evaluated arcs")
        last = history["datetime"].max()
        horizon = forecast_horizon(last, target_times)
        values = context_values(history, arcs, last, self.context_hours)
        eligible = has_recent_history(values, last)
        active = [arc for arc in arcs if eligible[arc]]
        parts, sources = [], []
        if active:
            index = pd.MultiIndex.from_product([active, target_times], names=KEYS)
            parts.append(self._forecast(values.loc[active], active, horizon).reindex(index))
            sources.append(pd.Series("timestral", index=index, dtype=object))
        if len(active) < len(arcs):
            baseline = HolidayAwarePreviousWeekBaseline(self._calendar())
            fallback = baseline.predict(history, target_times).set_index(KEYS)
            keep = ~fallback.index.get_level_values("arc").isin(active)
            parts.append(fallback.loc[keep, list(TARGETS)])
            sources.append("fallback_" + baseline.sources_[keep])
        index = pd.MultiIndex.from_product([arcs, target_times], names=KEYS)
        result = pd.concat(parts).reindex(index).astype("float64")
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Timestral returned missing/nonfinite forecasts")
        result["debit_horaire"] = result["debit_horaire"].clip(lower=0)
        result["taux_occupation"] = result["taux_occupation"].clip(0, 100)
        self.sources_ = pd.concat(sources).reindex(index)
        return result.reset_index()

    def _forecast(self, values: pd.DataFrame, arcs: list[str], horizon: pd.DatetimeIndex) -> pd.DataFrame:
        times = values.index.get_level_values("datetime").unique()
        context = values[list(TARGETS)].to_numpy(dtype="float32").reshape(len(arcs), len(times), 2)
        context = np.ascontiguousarray(context.transpose(0, 2, 1))
        if not self.multivariate:
            context = context.reshape(len(arcs) * 2, 1, len(times))
        known = None
        if self.calendar_covariates:
            calendar = calendar_features(self._calendar())
            if calendar.index.has_duplicates:
                raise ValueError("Calendar contains duplicate timestamps")
            selected = calendar.reindex(times.append(horizon))[list(self.calendar_covariates)]
            known = selected.to_numpy(dtype="float32").T
            if not np.isfinite(known).all():
                raise ValueError("Calendar must cover every context and forecast hour with finite values")
            known = np.broadcast_to(known, (len(context), *known.shape)).copy()
        forecast = self._pipeline().forecast(
            context, prediction_length=len(horizon), known_covariates=known,
            symmetric_averaging=True, clip_nonnegative=True,
        )
        median = forecast.median.detach().cpu().numpy()
        expected = (len(context), 2 if self.multivariate else 1, len(horizon))
        if median.shape != expected or not np.isfinite(median).all():
            raise ValueError(f"Timestral returned invalid median forecasts; expected shape {expected}")
        median = median.reshape(len(arcs), 2, len(horizon)).transpose(0, 2, 1)
        index = pd.MultiIndex.from_product([arcs, horizon], names=KEYS)
        return pd.DataFrame(median.reshape(-1, 2), index=index, columns=TARGETS)

    def _calendar(self) -> pd.DataFrame:
        if self.calendar is None:
            self.calendar = pd.read_parquet(DEFAULT_CALENDAR)
        return self.calendar

    def _pipeline(self) -> Any:
        if self.pipeline is None:
            try:
                import torch
                from timestral import TimestralForecaster as NativeTimestral
            except ImportError as error:
                raise ImportError("Timestral inference needs torch and the pinned timestral dependency; see docs/methodology.md") from error
            torch.manual_seed(SEED)
            self.pipeline = NativeTimestral.from_pretrained(
                f"mistralai/Timestral-{self.model_size}", revision=MODEL_REVISIONS[self.model_size],
                device="cuda" if torch.cuda.is_available() else "cpu", max_context=self.context_hours,
                cache_dir=REPO_ROOT / "models/huggingface",
            )
        return self.pipeline
