"""Rerun the Timestral experiments of issue #44 on the local #22 dataset.

From the repository root, after ``pip install -r requirements-timestral.txt``::

    .venv/Scripts/python.exe scripts/run_timestral_44.py --list
    .venv/Scripts/python.exe scripts/run_timestral_44.py RUN --window 2026-10-22
    .venv/Scripts/python.exe scripts/run_timestral_44.py RUN --author "Your Name"
    .venv/Scripts/python.exe scripts/run_timestral_44.py RUN --select-quantiles

``--window`` times one cutoff and writes nothing. A full run backtests the
eight shared cutoffs, saves hourly forecasts and native quantiles under
``outputs/metrics/timestral_44/`` (outside Git) and appends one
``Issue #44; RUN; ...; fingerprint ...`` row to ``docs/experiments.csv``.
``--select-quantiles`` then chooses one quantile per day kind and variable,
leave-one-window-out, from those saved files only. Every run states all its settings, so changing the
class defaults never changes what a run name means. Scores are pooled over
Convention and Saint-Pères: ``sqrt(sum of squared errors / valid hours)``.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import json
import time

import holidays
import numpy as np
import pandas as pd

from traffic_prediction import evaluate
from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR, DEFAULT_DATASET_DIR
from traffic_prediction.data.dataset import content_sha256
from traffic_prediction.data.traffic import REPO_ROOT, file_sha256
from traffic_prediction.evaluate import KEYS, TARGETS
from traffic_prediction.models.timestral import TimestralForecaster

OUTPUT_DIR = REPO_ROOT / "outputs/metrics/timestral_44"
POOLED_ARCS = ("Convention", "Saint-Pères")
DAY_KINDS = ("ouvre", "weekend", "ferie")
# Native levels tried by --select-quantiles, on both sides of the median.
QUANTILE_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Explicit settings: the reference repeats the best #20 run (journal fingerprint 9e714afe).
RUNS = {
    "timestral_5m_calendar_multivariate": dict(model_size="5M", multivariate=True, context_hours=8192),
    "timestral_5m_calendar_multivariate_ctx32768": dict(model_size="5M", multivariate=True, context_hours=32768),
    "timestral_123m_calendar_multivariate_ctx32768": dict(model_size="123M", multivariate=True, context_hours=32768),
    "timestral_209m_calendar_multivariate_ctx32768": dict(model_size="209M", multivariate=True, context_hours=32768),
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Dataset, calendar and manifest, checked against the manifest's content hashes."""
    manifest = json.loads((DEFAULT_DATASET_DIR / "_manifest.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(DEFAULT_DATASET_DIR / "dataset.parquet")
    if content_sha256(data) != manifest["content_sha256"]["dataset.parquet"]:
        raise ValueError("dataset.parquet does not match its manifest; rebuild the #22 dataset")
    calendar_path = DEFAULT_CALENDAR_DIR / "calendar.parquet"
    if file_sha256(calendar_path) != manifest["inputs_sha256"]["calendar"]:
        raise ValueError("calendar.parquet differs from the one used to build the dataset")
    return data, pd.read_parquet(calendar_path), manifest


def pooled_rows(predictions: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Forecast rows of the pooled arcs joined to the truth, and the day kind of each row."""
    truth = data.set_index(KEYS)[[*TARGETS, "qualite_mesure"]]
    frame = predictions.loc[predictions["arc"].isin(POOLED_ARCS)].join(truth, on=KEYS, rsuffix="_true")
    frame = frame.reset_index(drop=True)
    times = frame["datetime"]
    calendar = holidays.France(years=sorted(times.dt.year.unique()))
    holiday = np.array([time.date() in calendar for time in times])
    return frame, np.where(holiday, "ferie", np.where(times.dt.dayofweek >= 5, "weekend", "ouvre"))


def valid_truth(frame: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Observed values and the backtest's validity mask (ok, finite, within bounds)."""
    y = frame[f"{variable}_true"].to_numpy(dtype=float)
    valid = frame["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool) & np.isfinite(y)
    valid &= (y >= 0) if variable == "debit_horaire" else ((y >= 0) & (y <= 100))
    return y, valid


def rmse_by_kind(squared: np.ndarray, valid: np.ndarray, kinds: np.ndarray) -> dict:
    """Pooled RMSE and valid hours, overall and per day kind."""
    scores = {}
    for kind in ("all", *DAY_KINDS):
        selected = valid & ((kinds == kind) if kind != "all" else True)
        scores[kind] = float(np.sqrt(np.mean(squared[selected])))
        scores[f"n_{kind}"] = int(selected.sum())
    return scores


def day_kind_scores(predictions: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Pooled RMSE and valid hours per variable and day kind, with the backtest's truth rules."""
    frame, kinds = pooled_rows(predictions, data)
    scores = {}
    for variable in TARGETS:
        y, valid = valid_truth(frame, variable)
        scores[variable] = rmse_by_kind((frame[variable].to_numpy(dtype=float) - y) ** 2, valid, kinds)
    return scores


def select_quantiles(run: str, data: pd.DataFrame) -> dict:
    """Leave-one-window-out choice of one quantile per day kind and variable, as in #19.

    For each held-out cutoff, the level with the lowest pooled error on the seven other
    windows is chosen per day kind and variable, then scored on the held-out window.
    Fallback hours have no quantiles and keep their forecast. Reads saved files only.
    """
    predictions = pd.read_parquet(OUTPUT_DIR / f"{run}_predictions.parquet")
    quantiles = pd.read_parquet(OUTPUT_DIR / f"{run}_quantiles.parquet")
    frame, kinds = pooled_rows(predictions, data)
    keys = pd.MultiIndex.from_frame(frame[["cutoff", *KEYS]])
    cutoffs = sorted(frame["cutoff"].unique())
    median = QUANTILE_GRID.index(0.5)
    report = {}
    for variable in TARGETS:
        y, valid = valid_truth(frame, variable)
        native = quantiles.loc[quantiles["variable"].eq(variable)].set_index(["cutoff", *KEYS]).reindex(keys)
        upper = None if variable == "debit_horaire" else 100
        candidates = np.column_stack([
            np.where(native[str(level)].notna(), native[str(level)].clip(0, upper), frame[variable])
            for level in QUANTILE_GRID
        ])
        squared = (candidates - y[:, None]) ** 2
        held_out = np.full(len(frame), np.nan)
        folds = {}
        for cutoff in cutoffs:
            train = frame["cutoff"].ne(cutoff).to_numpy()
            folds[str(pd.Timestamp(cutoff).date())] = choice = {}
            for kind in DAY_KINDS:
                rows = valid & (kinds == kind)
                fit, test = rows & train, rows & ~train
                best = int(np.argmin(squared[fit].mean(axis=0))) if fit.any() else median
                choice[kind] = QUANTILE_GRID[best]
                held_out[test] = squared[test, best]
        chosen = {kind: QUANTILE_GRID[int(np.argmin(squared[valid & (kinds == kind)].mean(axis=0)))]
                  for kind in DAY_KINDS}
        in_sample = np.choose(np.array([QUANTILE_GRID.index(chosen[kind]) for kind in kinds]), squared.T)
        report[variable] = {
            "median": rmse_by_kind(squared[:, median], valid, kinds),
            "cross_validated": rmse_by_kind(held_out, valid, kinds),
            "levels_all_windows": chosen,
            "in_sample_with_those_levels": rmse_by_kind(in_sample, valid, kinds),
            "fold_choices": folds,
        }
    return report


def print_selection(run: str, report: dict) -> None:
    print(f"\n{run} -- quantile per day kind, leave-one-window-out ({', '.join(map(str, QUANTILE_GRID))})")
    for variable, digits in (("debit_horaire", 1), ("taux_occupation", 2)):
        part = report[variable]
        print(f"\n{variable:34s}{'all':>8s}{'working':>9s}{'weekend':>9s}{'holiday':>9s}")
        for label, key in (("median", "median"), ("cross-validated (honest)", "cross_validated"),
                           ("levels of all 8 windows (optimistic)", "in_sample_with_those_levels")):
            row = part[key]
            print(f"{label:34s}" + "".join(f"{row[kind]:>{8 if kind == 'all' else 9}.{digits}f}"
                                            for kind in ("all", *DAY_KINDS)))
        print(f"levels chosen on all windows: {part['levels_all_windows']}")
        for kind in DAY_KINDS:
            print(f"  {kind:8s} per held-out window: {[part['fold_choices'][day][kind] for day in part['fold_choices']]}")


def print_scores(label: str, scores: dict) -> None:
    print(f"\n{label} -- pooled RMSE, {' + '.join(POOLED_ARCS)}")
    print(f"{'':16s}{'all':>8s}{'working':>9s}{'weekend':>9s}{'holiday':>9s}   valid hours")
    for variable, digits in (("debit_horaire", 1), ("taux_occupation", 2)):
        row = scores[variable]
        values = "".join(f"{row[kind]:>{8 if kind == 'all' else 9}.{digits}f}" for kind in ("all", *DAY_KINDS))
        counts = "/".join(str(row[f"n_{kind}"]) for kind in DAY_KINDS)
        print(f"{variable:16s}{values}   {counts}")


def timestral_commit() -> str:
    direct = importlib.metadata.distribution("timestral").read_text("direct_url.json")
    return json.loads(direct)["vcs_info"]["commit_id"] if direct else "unknown"


def time_one_window(model: TimestralForecaster, data: pd.DataFrame, day: pd.Timestamp) -> float:
    """Seconds for one forecast at cutoff ``day`` (weights already loaded); nothing is written."""
    history = data.loc[data["datetime"] < day + pd.Timedelta(days=1)]
    started = time.perf_counter()
    model.predict(history, evaluate.target_window(day))
    return time.perf_counter() - started


def run_backtest(model: TimestralForecaster, data: pd.DataFrame) -> tuple[
        evaluate.BacktestResult, pd.DataFrame, pd.DataFrame, float]:
    """Backtest without logging, keeping every window's forecast, its source and native quantiles."""
    frames, quantiles = [], []
    predict = model.predict

    def capture(history, target_times):
        table = predict(history, target_times)
        cutoff = history["datetime"].max().normalize()
        saved = table.copy()
        saved.insert(0, "cutoff", cutoff)
        saved["forecast_source"] = model.sources_.reindex(pd.MultiIndex.from_frame(table[KEYS])).to_numpy()
        frames.append(saved)
        if model.raw_forecast_ is not None:
            raw = model.raw_forecast_.loc[model.raw_forecast_["datetime"].isin(target_times)]
            quantiles.append(raw.assign(cutoff=cutoff))
        return table

    model.predict = capture
    started = time.perf_counter()
    try:
        result = evaluate.backtest(model, data, experiments_path=None)
    finally:
        del model.predict
    elapsed = time.perf_counter() - started
    return result, pd.concat(frames, ignore_index=True), pd.concat(quantiles, ignore_index=True), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", choices=RUNS, help="run name, see --list")
    parser.add_argument("--list", action="store_true", help="print the run names and their settings")
    parser.add_argument("--window", help="time one cutoff (Thursday date) and write nothing")
    parser.add_argument("--select-quantiles", action="store_true",
                        help="choose quantiles per day kind from RUN's saved files (no inference, no journal row)")
    parser.add_argument("--author", default=getpass.getuser(), help="author recorded in docs/experiments.csv")
    parser.add_argument("--no-log", action="store_true",
                        help="save files but add no journal row (e.g. to add quantiles to a logged run)")
    parser.add_argument("--threads", type=int, default=6, help="torch CPU threads (logged #20 runs used 6)")
    args = parser.parse_args()
    if args.list or args.run is None:
        for name, options in RUNS.items():
            print(f"{name}: {options}")
        return

    if args.select_quantiles:
        data, _, manifest = load_inputs()
        report = select_quantiles(args.run, data)
        print_selection(args.run, report)
        (OUTPUT_DIR / f"{args.run}_quantile_selection.json").write_text(json.dumps(
            {"run": args.run, "dataset_fingerprint": manifest["fingerprint"], "grid": QUANTILE_GRID,
             "pooled_arcs": POOLED_ARCS, **report}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    import torch

    torch.set_num_threads(args.threads)
    data, calendar, manifest = load_inputs()
    model = TimestralForecaster(calendar, **RUNS[args.run])
    started = time.perf_counter()
    model._pipeline()
    loading = time.perf_counter() - started
    cutoffs = evaluate.load_cutoffs()
    print(f"{args.run}: {model.describe()}")
    print(f"dataset fingerprint {manifest['fingerprint']}; weights loaded in {loading:.1f} s; "
          f"{torch.get_num_threads()} threads")

    if args.window:
        seconds = time_one_window(model, data, pd.Timestamp(args.window))
        print(f"one window ({args.window}): {seconds:.1f} s -> about {seconds * len(cutoffs):.0f} s "
              f"for {len(cutoffs)} windows at most (earlier cutoffs have less history)")
        return

    result, predictions, quantiles, elapsed = run_backtest(model, data)
    scores = day_kind_scores(predictions, data)
    print_scores(args.run, scores)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT_DIR / f"{args.run}_predictions.parquet", index=False)
    quantiles.to_parquet(OUTPUT_DIR / f"{args.run}_quantiles.parquet", index=False)
    settings = model.describe()
    (OUTPUT_DIR / f"{args.run}_run.json").write_text(json.dumps({
        "run": args.run, "options": RUNS[args.run], "settings": settings,
        "dataset_fingerprint": manifest["fingerprint"], "cutoffs": [str(day.date()) for day in cutoffs],
        "timestral_commit": timestral_commit(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "loading_seconds": loading, "backtest_seconds": elapsed,
        "pooled_arcs": POOLED_ARCS, "scores": scores,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    description = (
        f"Issue #44; {args.run}; {settings}; fingerprint {manifest['fingerprint']}; "
        f"{len(cutoffs)} shared cutoffs; CPU {torch.get_num_threads()} threads; "
        f"loading {loading:.3f}s; backtest {elapsed:.3f}s; "
        f"hourly predictions: {OUTPUT_DIR.name}/{args.run}_predictions.parquet"
    )
    if not args.no_log:
        evaluate._append_experiment(result, evaluate.DEFAULT_EXPERIMENTS, args.author, type(model).__name__, description)
    print(f"\nbacktest {elapsed:.1f} s; {'not logged' if args.no_log else 'logged to docs/experiments.csv'}; "
          f"forecasts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
