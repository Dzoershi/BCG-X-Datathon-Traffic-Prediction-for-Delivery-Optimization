"""Chronological backtests reproducing the BCG X submission window."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

from traffic_prediction.data.traffic import REPO_ROOT
from traffic_prediction.models.base import ForecastModel

TARGETS = ("debit_horaire", "taux_occupation")
KEYS = ["arc", "datetime"]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
DEFAULT_EXPERIMENTS = REPO_ROOT / "docs" / "experiments.csv"
LOG_COLUMNS = ["date", "author", "model", "description", "scores"]


@dataclass
class BacktestResult:
    """Long-form scores; undefined RMSE/ratios remain NaN, with coverage counts."""

    scores: pd.DataFrame
    per_arc: pd.DataFrame
    summary: pd.DataFrame


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


class PreviousWeekModel:
    """Reference forecast: copy the same local hour seven days earlier.

    No imputation: absent or non-ok source measurements produce NaN.
    """

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        arcs = history.loc[history["role"].eq("evalue"), "arc"].unique()
        index = pd.MultiIndex.from_product([arcs, target_times], names=KEYS)
        source = history.loc[history["qualite_mesure"].eq("ok")].set_index(KEYS)
        lookup = pd.MultiIndex.from_arrays(
            [index.get_level_values("arc"), index.get_level_values("datetime") - pd.Timedelta(days=7)],
            names=KEYS,
        )
        values = source.reindex(lookup)[list(TARGETS)].to_numpy()
        return pd.DataFrame(values, index=index, columns=TARGETS).reset_index()


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


def _aggregate(scores: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = scores.groupby(keys, sort=False, dropna=False).agg(
        rmse=("rmse", "mean"),
        baseline_rmse=("baseline_rmse", "mean"),
        normalized_score=("normalized_score", "mean"),
        n_expected=("n_expected", "sum"),
        n_scored=("n_scored", "sum"),
        n_ignored=("n_ignored", "sum"),
        baseline_n_scored=("baseline_n_scored", "sum"),
        n_groups=("rmse", "size"),
        n_groups_scored=("rmse", "count"),
        n_groups_normalized=("normalized_score", "count"),
    )
    return result.reset_index()


def backtest(
    model: ForecastModel,
    data: pd.DataFrame,
    cutoffs: Iterable[object] | None = None,
    *,
    author: str | None = None,
    description: str = "",
    experiments_path: str | Path | None = DEFAULT_EXPERIMENTS,
) -> BacktestResult:
    """Score a model using only history through each Thursday 23:00.

    Only role=evalue arcs are scored; proxies remain available in history.
    Missing/invalid truth is ignored per variable and counted on a complete
    arc/hour grid. A missing/nonfinite forecast for valid truth is an error.
    Atypical means Saturday, Sunday or a metropolitan France public holiday.
    Aggregates are unweighted means of available arc/window RMSEs, not pooled
    errors; coverage and numbers of contributing groups accompany every mean.
    Ratios require full baseline coverage and a positive baseline RMSE.
    Successful calls append one JSON score record to the experiment CSV;
    pass experiments_path=None to disable logging (e.g. in tests).
    """
    _check_frame(data, [*KEYS, *TARGETS, "role", "qualite_mesure"], "data")
    if data.groupby("arc")["role"].nunique(dropna=False).gt(1).any():
        raise ValueError("data: an arc must have a consistent role")
    arcs = sorted(data.loc[data["role"].eq("evalue"), "arc"].unique())
    if not arcs:
        raise ValueError("data: no evaluated arcs")
    days = list(load_cutoffs() if cutoffs is None else cutoffs)
    windows = [(pd.Timestamp(day), target_window(day)) for day in days]
    if not windows or len({day for day, _ in windows}) != len(windows):
        raise ValueError("Supply at least one cutoff, without duplicates")
    calendar = holidays.France(years=sorted({time.year for _, times in windows for time in times}))
    records = []
    indexed_data = data.set_index(KEYS)
    for day, times in windows:
        history = data.loc[data["datetime"] < day + pd.Timedelta(days=1)].copy(deep=True)
        # Compute the reference before calling user code, which may mutate its input.
        reference = PreviousWeekModel().predict(history, times).set_index(KEYS)
        predictions = model.predict(history, times.copy(deep=True))
        _check_frame(predictions, [*KEYS, *TARGETS], "predictions")
        if not predictions["datetime"].isin(times).all():
            raise ValueError("predictions: datetime outside target window")
        predictions = predictions.set_index(KEYS)
        index = pd.MultiIndex.from_product([arcs, times], names=KEYS)
        truth = indexed_data.reindex(index)
        predictions = predictions.reindex(index)
        reference = reference.reindex(index)
        atypical = np.array([time.weekday() >= 5 or time.date() in calendar for time in times])
        for arc in arcs:
            actual, predicted, baseline = truth.loc[arc], predictions.loc[arc], reference.loc[arc]
            for variable in TARGETS:
                y = actual[variable].to_numpy(dtype=float, na_value=np.nan)
                p = predicted[variable].to_numpy(dtype=float, na_value=np.nan)
                b = baseline[variable].to_numpy(dtype=float, na_value=np.nan)
                valid = actual["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool) & np.isfinite(y)
                valid &= (y >= 0) if variable == "debit_horaire" else ((y >= 0) & (y <= 100))
                if not np.isfinite(p[valid]).all():
                    raise ValueError(f"Missing/nonfinite prediction for valid truth: {arc}, {day.date()}, {variable}")
                for subset, mask in (("all", np.ones(120, dtype=bool)), ("atypical", atypical)):
                    selected = valid & mask
                    covered = selected & np.isfinite(b)
                    n = int(selected.sum())
                    bn = int(covered.sum())
                    rmse = float(np.sqrt(np.mean((p[selected] - y[selected]) ** 2))) if n else np.nan
                    brmse = float(np.sqrt(np.mean((b[covered] - y[covered]) ** 2))) if bn else np.nan
                    ratio = rmse / brmse if n and bn == n and brmse > 0 else np.nan
                    records.append(dict(
                        cutoff=day, arc=arc, variable=variable, subset=subset,
                        rmse=rmse, baseline_rmse=brmse, normalized_score=ratio,
                        n_expected=int(mask.sum()), n_scored=n,
                        n_ignored=int(mask.sum()) - n, baseline_n_scored=bn,
                    ))
    scores = pd.DataFrame.from_records(records)
    result = BacktestResult(scores, _aggregate(scores, ["arc", "variable", "subset"]),
                            _aggregate(scores, ["variable", "subset"]))
    if experiments_path is not None:
        _append_experiment(result, Path(experiments_path), author or getpass.getuser(),
                           type(model).__name__, description)
    return result


def _append_experiment(result: BacktestResult, path: Path, author: str, model: str, description: str) -> None:
    # Pandas serializes undefined scores as JSON null, never non-standard NaN.
    payload = {name: json.loads(getattr(result, name).to_json(orient="records", date_format="iso"))
               for name in ("scores", "per_arc", "summary")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as stream:
        stream.seek(0)
        header = next(csv.reader(stream), None)
        if header is not None and header != LOG_COLUMNS:
            raise ValueError(f"Experiment CSV must have columns {LOG_COLUMNS}")
        stream.seek(0, 2)
        writer = csv.writer(stream)
        if header is None:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([datetime.now(timezone.utc).isoformat(), author, model, description,
                         json.dumps(payload, ensure_ascii=False, allow_nan=False)])
