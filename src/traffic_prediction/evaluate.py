"""Derive shared chronological forecast windows from Thursday cutoffs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from traffic_prediction.data.traffic import REPO_ROOT

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
