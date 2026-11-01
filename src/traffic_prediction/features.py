"""Cutoff-aware features for tabular forecasts (issue #30).

Fit once per backtest window, then transform training rows and forecast rows.
Measurements are looked up by local hour, never shifted by row position.
Training profiles use a conservative seven-day gap, matching the long lags;
forecast profiles use all fitted history. No model is fitted or files read here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from traffic_prediction.data.calendar_events import CALENDAR_COLUMNS
from traffic_prediction.data.weather import VALUE_COLUMNS
from traffic_prediction.evaluate import KEYS, TARGETS, _check_frame

DEFAULT_LAGS = (168, 336, 504, 672)
CALENDAR_FLAGS = tuple(column for column in CALENDAR_COLUMNS if column.startswith("is_"))
CYCLICAL_FEATURES = tuple(
    f"{name}_{component}"
    for name in ("hour", "day_of_week", "month") for component in ("sin", "cos")
)
PROFILE_KEYS = ["arc", "day_type", "hour"]
PROFILE_FEATURES = [f"{target}_profile" for target in TARGETS]
WEATHER_LAG = 168


def _check_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate keys and normalize timestamp resolution for exact/as-of joins."""
    if missing := set(KEYS) - set(frame.columns):
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    times = frame["datetime"]
    if (not pd.api.types.is_datetime64_any_dtype(times.dtype)
            or times.dt.tz is not None or times.isna().any()
            or not times.eq(times.dt.floor("h")).all()):
        raise ValueError(f"{label}: datetime must contain timezone-free whole Paris hours")
    if frame["arc"].isna().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{label}: null arc or duplicate (arc, datetime)")
    result = frame.copy()
    result["datetime"] = times.astype("datetime64[ns]")
    result["arc"] = result["arc"].astype(object)
    return result


def _day_type(times: pd.Series, holiday: pd.Series) -> np.ndarray:
    """Monday=0 ... Sunday=6; public holidays share the Sunday profile (#17)."""
    return np.where(holiday.to_numpy(dtype=bool), 6, times.dt.dayofweek).astype("int8")


class TabularFeatures:
    """Reusable features fitted only on observations at or before ``cutoff``.

    ``history`` is the assembled #22 dataset. ``rows`` contains requested arc/hour
    keys; ``covariates`` is its separate covariates.parquet (or equivalent).
    Only evaluated arcs are supported; proxies never become labels or truth.
    Missing lags/profiles remain NaN for the downstream model to handle.

    Weather defaults to the same hour seven days earlier. ``weather="known"``
    instead selects target-hour forecasts and requires a real, timezone-free
    Paris ``weather_available_at`` timestamp per covariate row. Values issued
        after the cutoff (or a training row's seven-day origin) become NaN. The current
    #22 table has no issuance timestamps, so it supports the default mode only.
    """

    def __init__(self, lags: Sequence[int] = DEFAULT_LAGS, *, weather: str = "lagged") -> None:
        lags = tuple(lags)
        if (not lags or len(set(lags)) != len(lags)
                or any(isinstance(lag, bool) or not isinstance(lag, (int, np.integer))
                       or lag < 168 for lag in lags)):
            raise ValueError("lags must contain unique integer hour offsets of at least 168")
        if weather not in ("lagged", "known"):
            raise ValueError("weather must be 'lagged' or 'known'")
        self.lags = lags
        self.weather = weather

    @property
    def feature_columns(self) -> list[str]:
        """Model inputs in stable order; datetime is a join key, never a feature."""
        weather = [f"{column}_lag_{WEATHER_LAG}h" for column in VALUE_COLUMNS]
        if self.weather == "known":
            weather = list(VALUE_COLUMNS)
        return ["arc", *CYCLICAL_FEATURES, *CALENDAR_FLAGS,
                *(f"{target}_lag_{lag}h" for lag in self.lags for target in TARGETS),
                *weather, *PROFILE_FEATURES, "profile_count"]

    def fit(self, history: pd.DataFrame, *, cutoff: object) -> TabularFeatures:
        """Reset state from cutoff-limited, measured evaluated-arc history.

        Passing a complete dataset is safe: rows after cutoff are excluded before
        any fitted statistics are computed. Each subsequent fit replaces state.
        """
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff) or cutoff.tz is not None or cutoff != cutoff.floor("h"):
            raise ValueError("cutoff must be a timezone-free whole Paris hour")
        _check_frame(history, [*KEYS, *TARGETS, "role", "qualite_mesure", "is_holiday"], "history")
        retained = _check_rows(history.loc[history["datetime"].le(cutoff)], "history")
        if retained.groupby("arc")["role"].nunique(dropna=False).gt(1).any():
            raise ValueError("history: an arc must have a consistent role")
        retained = retained.loc[retained["role"].eq("evalue")].sort_values("datetime")
        if retained.empty:
            raise ValueError("history: no evaluated arcs at or before cutoff")
        if (not pd.api.types.is_bool_dtype(retained["is_holiday"])
                or retained["is_holiday"].isna().any()):
            raise ValueError("history: is_holiday must contain non-null booleans")
        values = retained[list(TARGETS)].to_numpy(dtype=float, na_value=np.nan)
        valid = (retained["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool)
                 & np.isfinite(values).all(axis=1) & (values[:, 0] >= 0)
                 & (values[:, 1] >= 0) & (values[:, 1] <= 100))
        measured = retained[[*KEYS, *TARGETS]].copy()
        measured.loc[~valid, list(TARGETS)] = np.nan
        if self.weather == "lagged":
            for column in VALUE_COLUMNS:
                if column not in retained or not pd.api.types.is_numeric_dtype(retained[column]):
                    raise ValueError(f"history: numeric weather column required: {column}")
                measured[column] = retained[column].where(np.isfinite(retained[column]))

        # Expanding sufficient statistics permit one as-of lookup per request.
        profiles = retained.loc[valid, [*KEYS, *TARGETS]].copy()
        profiles["hour"] = profiles["datetime"].dt.hour.astype("int8")
        profiles["day_type"] = _day_type(retained.loc[valid, "datetime"],
                                          retained.loc[valid, "is_holiday"])
        grouped = profiles.groupby(PROFILE_KEYS, sort=False)
        profiles["profile_count"] = grouped.cumcount() + 1
        for target, feature in zip(TARGETS, PROFILE_FEATURES):
            profiles[feature] = grouped[target].cumsum() / profiles["profile_count"]
        profiles = profiles.rename(columns={"datetime": "profile_time"})
        # Publish state only after every input check succeeded.
        self.cutoff_ = cutoff
        self.arcs_ = tuple(sorted(retained["arc"].unique()))
        self._values = measured.set_index(KEYS)
        self._profiles = profiles[[*PROFILE_KEYS, "profile_time", *PROFILE_FEATURES, "profile_count"]]
        return self

    def transform(self, rows: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
        """Return keys + feature_columns, preserving requested row order.

        Extra columns, including contemporary targets, quality, traffic states,
        disruption flags and weather source labels, are never model inputs.
        For historical training rows, profiles stop seven days before that row;
        for future rows, they stop at the fitted cutoff. A lag whose source is
        after cutoff is missing, even if the caller supplied later observations.
        """
        if not hasattr(self, "cutoff_"):
            raise ValueError("Call fit(history, cutoff=...) before transform")
        frame = _check_rows(rows, "rows")[KEYS].reset_index(drop=True)
        if not frame["arc"].isin(self.arcs_).all():
            raise ValueError("rows: only evaluated arcs present in fitted history are supported")
        covariates = _check_rows(covariates, "covariates").set_index(KEYS)
        if missing := set(CALENDAR_FLAGS) - set(covariates.columns):
            raise ValueError(f"covariates: missing calendar columns {sorted(missing)}")
        index = pd.MultiIndex.from_frame(frame[KEYS])
        known = covariates.reindex(index).reset_index(drop=True)
        for column in CALENDAR_FLAGS:
            if not pd.api.types.is_bool_dtype(known[column]) or known[column].isna().any():
                raise ValueError(f"covariates: {column} must cover every requested hour with booleans")
            frame[column] = known[column].astype("int8")
        times = frame["datetime"]
        for name, values, period in (("hour", times.dt.hour, 24),
                                      ("day_of_week", times.dt.dayofweek, 7),
                                      ("month", times.dt.month - 1, 12)):
            phase = 2 * np.pi * values / period
            frame[f"{name}_sin"], frame[f"{name}_cos"] = np.sin(phase), np.cos(phase)
        for lag in self.lags:
            lookup = pd.MultiIndex.from_arrays([frame["arc"], times - pd.Timedelta(hours=lag)], names=KEYS)
            values = self._values.reindex(lookup)
            for target in TARGETS:
                frame[f"{target}_lag_{lag}h"] = values[target].to_numpy(dtype=float, na_value=np.nan)
        # Historical examples must not learn from their own forecast interval.
        # A seven-day origin is conservative for the operational 25..144 h lead.
        available = (times - pd.Timedelta(hours=168)).where(times.le(self.cutoff_), self.cutoff_)
        if self.weather == "lagged":
            lookup = pd.MultiIndex.from_arrays(
                [frame["arc"], times - pd.Timedelta(hours=WEATHER_LAG)], names=KEYS)
            values = self._values.reindex(lookup)
            for column in VALUE_COLUMNS:
                frame[f"{column}_lag_{WEATHER_LAG}h"] = values[column].to_numpy(dtype=float, na_value=np.nan)
        else:
            if "weather_available_at" not in known:
                raise ValueError("covariates: known weather requires weather_available_at issuance timestamps")
            issued = known["weather_available_at"]
            if not pd.api.types.is_datetime64_any_dtype(issued.dtype) or issued.dt.tz is not None:
                raise ValueError("weather_available_at must contain timezone-free Paris timestamps")
            for column in VALUE_COLUMNS:
                if column not in known or not pd.api.types.is_numeric_dtype(known[column]):
                    raise ValueError(f"covariates: numeric weather column required: {column}")
                frame[column] = known[column].where(issued.le(available) & np.isfinite(known[column]))
        if frame.empty or self._profiles.empty:
            frame[PROFILE_FEATURES] = np.nan
            frame["profile_count"] = np.zeros(len(frame), dtype="int32")
        else:
            request = frame[KEYS].assign(
                hour=times.dt.hour.astype("int8"), day_type=_day_type(times, known["is_holiday"]),
                available_until=available, position=np.arange(len(frame)),
            )
            aligned = pd.merge_asof(
                request.sort_values("available_until"), self._profiles,
                left_on="available_until", right_on="profile_time", by=PROFILE_KEYS,
                direction="backward",
            ).sort_values("position")
            frame[PROFILE_FEATURES] = aligned[PROFILE_FEATURES].to_numpy(dtype=float, na_value=np.nan)
            frame["profile_count"] = aligned["profile_count"].fillna(0).to_numpy(dtype="int32")
        frame["arc"] = pd.Categorical(frame["arc"], categories=self.arcs_)
        return frame[[*KEYS, *self.feature_columns[1:]]]
