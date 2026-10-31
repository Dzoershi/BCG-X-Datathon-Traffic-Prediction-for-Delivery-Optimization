"""Derive shared chronological forecast windows from Thursday cutoffs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

from traffic_prediction.data.traffic import REPO_ROOT

from traffic_prediction.models.base import ForecastModel

TARGETS = ("debit_horaire", "taux_occupation")
KEYS = ["arc", "datetime"]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"


def target_window(cutoff: object) -> pd.DatetimeIndex:
    """Thursday date -> Saturday 00:00 through Wednesday 23:00 (120 labels)."""
    day = pd.Timestamp(cutoff)
    if pd.isna(day) or day.tz is not None or day != day.normalize() or day.weekday() != 3:
        raise ValueError("Cutoffs must be timezone-free Thursday dates at midnight")
    return pd.date_range(day + pd.Timedelta(days=2), periods=120, freq="h")


def load_cutoffs(path: str | Path = DEFAULT_CONFIG) -> list[pd.Timestamp]:
    """Read the shared Thursday dates, without accessing traffic data."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [pd.Timestamp(day) for day in config["backtest"]["cutoffs"]]


def _check_frame(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    times = frame["datetime"]
    if (not pd.api.types.is_datetime64_any_dtype(times.dtype)
            or times.dt.tz is not None or times.isna().any()
            or not times.eq(times.dt.floor("h")).all()):
        raise ValueError(f"{label}: datetime must contain timezone-free whole Paris hours")
    if frame["arc"].isna().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{label}: null arc or duplicate (arc, datetime)")
    for variable in TARGETS:
        if not pd.api.types.is_numeric_dtype(frame[variable]):
            raise ValueError(f"{label}: {variable} must be numeric")


def backtest(
    model: ForecastModel,
    data: pd.DataFrame,
    cutoffs: Iterable[object] | None = None,
) -> list[pd.DataFrame]:
    """Replay windows and return aligned predictions and truth for each cutoff.

    The model sees history only through Thursday 23:00, including proxies.
    Copies isolate mutations made by user code from the original observations.
    """
    _check_frame(data, [*KEYS, *TARGETS, "role", "qualite_mesure"], "data")
    arcs = sorted(data["arc"].unique())
    days = list(load_cutoffs() if cutoffs is None else cutoffs)
    windows = [(pd.Timestamp(day), target_window(day)) for day in days]
    if not windows or len({day for day, _ in windows}) != len(windows):
        raise ValueError("Supply at least one cutoff, without duplicates")
    indexed_data = data.set_index(KEYS)
    results = []
    for day, times in windows:
        history = data.loc[data["datetime"] < day + pd.Timedelta(days=1)].copy(deep=True)
        predictions = model.predict(history, times.copy(deep=True))
        _check_frame(predictions, [*KEYS, *TARGETS], "predictions")
        if not predictions["datetime"].isin(times).all():
            raise ValueError("predictions: datetime outside target window")
        index = pd.MultiIndex.from_product([arcs, times], names=KEYS)
        truth = indexed_data.reindex(index)
        predictions = predictions.set_index(KEYS).reindex(index)
        aligned = truth.join(predictions[list(TARGETS)], rsuffix="_prediction")
        results.append(aligned.reset_index().assign(cutoff=day))
    return results
