"""Backtest the candidate models on the eight shared cutoffs and keep their hourly forecasts.

From the repository root::

    .venv/bin/python scripts/run_local_backtests.py               # every model not yet saved
    .venv/bin/python scripts/run_local_backtests.py lightgbm_raw  # one model, rerun even if saved
    .venv/bin/python scripts/run_local_backtests.py --list        # models and their settings

Data: the #22 dataset when it exists, else the processed traffic, every stored day or through
``--end`` (enough for models without weather: baselines, Chronos-2, Timestral). LightGBM
requires the dataset.

Every run writes to ``outputs/metrics/local/`` (outside Git):

- ``<name>_predictions.parquet``: cutoff, arc, datetime, both forecasts, forecast source;
- ``<name>_quantiles.parquet``: native quantiles, foundation models only;
- ``<name>_run.json``: model settings, data source and content hash, cutoffs, package
  versions, machine, timings and scores;
- one line appended to ``RUN_LOG.md``.

Nothing is appended to ``docs/experiments.csv``. Scores are the pooled RMSE of issue #44
(Convention + Saint-Pères, per day kind), comparable with docs/methodology.md.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_timestral_44 import day_kind_scores  # noqa: E402

from traffic_prediction import evaluate  # noqa: E402
from traffic_prediction.data import dataset, traffic  # noqa: E402
from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR, DEFAULT_DATASET_DIR  # noqa: E402
from traffic_prediction.data.query import DEFAULT_TRAFFIC_DIR  # noqa: E402
from traffic_prediction.data.traffic import REPO_ROOT, file_sha256  # noqa: E402
from traffic_prediction.evaluate import KEYS  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs/metrics/local"
RUN_LOG = OUTPUT_DIR / "RUN_LOG.md"
PACKAGES = ("numpy", "pandas", "lightgbm", "torch", "chronos-forecasting", "timestral")


def load_data(end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Model input, LightGBM covariates (None without the dataset) and a description of the source.

    `end` (inclusive Paris day) only limits the processed traffic; the dataset has its own end.
    """
    if (DEFAULT_DATASET_DIR / "dataset.parquet").exists():
        data = pd.read_parquet(DEFAULT_DATASET_DIR / "dataset.parquet")
        covariates = pd.read_parquet(DEFAULT_DATASET_DIR / "covariates.parquet")
        manifest = json.loads((DEFAULT_DATASET_DIR / "_manifest.json").read_text(encoding="utf-8"))
        source = {"kind": "dataset #22", "path": str(DEFAULT_DATASET_DIR.relative_to(REPO_ROOT)),
                  "fingerprint": manifest.get("fingerprint")}
    else:
        data = dataset.read_traffic(DEFAULT_TRAFFIC_DIR, traffic.load_arcs(), end=end)
        covariates = None
        source = {"kind": "processed traffic (no dataset yet)",
                  "path": str(DEFAULT_TRAFFIC_DIR.relative_to(REPO_ROOT)), "end": end}
    source.update(content_sha256=dataset.content_sha256(data), rows=len(data),
                  first=str(data["datetime"].min()), last=str(data["datetime"].max()))
    return data, covariates, source


def build_models(calendar: pd.DataFrame, covariates: pd.DataFrame | None) -> dict:
    """Name -> factory. Settings are explicit so that a name always means the same model."""
    from traffic_prediction.models.baselines import DayTypeProfileBaseline, HolidayAwarePreviousWeekBaseline

    def lightgbm():
        from traffic_prediction.train import LightGBMForecaster
        if covariates is None:
            raise RuntimeError("LightGBM needs covariates.parquet: build the #22 dataset first")
        return LightGBMForecaster(covariates)  # #31 "lightgbm_raw": 200 trees, depth 6, lr 0.05.

    def chronos():
        from traffic_prediction.models.chronos import Chronos2Forecaster
        return Chronos2Forecaster(calendar, multivariate=True, point="median")  # #19 run 4 (retained).

    def timestral(size: str):
        def factory():
            from traffic_prediction.models.timestral import TimestralForecaster
            return TimestralForecaster(calendar, model_size=size, multivariate=True, context_hours=32768)
        return factory

    def ensemble():
        from traffic_prediction.predict import e5_ensemble
        if covariates is None:
            raise RuntimeError("the ensemble's LightGBM needs covariates.parquet: build the #22 dataset first")
        return e5_ensemble(calendar, covariates)  # #32 submitted model, frozen E5 weights.

    return {
        "ensemble_e5": ensemble,
        "baseline_holiday_aware": lambda: HolidayAwarePreviousWeekBaseline(calendar),
        "baseline_day_type": lambda: DayTypeProfileBaseline(calendar),
        "lightgbm_raw": lightgbm,
        "chronos2": chronos,
        "timestral_5m": timestral("5M"),
        "timestral_209m": timestral("209M"),
    }


def describe(model) -> str:
    return model.describe() if hasattr(model, "describe") else f"{type(model).__name__} (default settings)"


def run_backtest(model, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, list[float]]:
    """Backtest without logging; return forecasts, native quantiles and seconds per window."""
    frames, quantiles, seconds = [], [], []
    predict = model.predict

    def capture(history, target_times):
        started = time.perf_counter()
        table = predict(history, target_times)
        seconds.append(time.perf_counter() - started)
        cutoff = history["datetime"].max().normalize()
        saved = table.copy()
        saved.insert(0, "cutoff", cutoff)
        sources = getattr(model, "sources_", None)
        saved["forecast_source"] = (sources.reindex(pd.MultiIndex.from_frame(table[KEYS])).to_numpy()
                                    if sources is not None else type(model).__name__)
        frames.append(saved)
        raw = getattr(model, "raw_forecast_", None)
        if raw is not None:  # Timestral labels hours "datetime", Chronos-2 "timestamp".
            column = "datetime" if "datetime" in raw.columns else "timestamp"
            quantiles.append(raw.loc[raw[column].isin(target_times)].assign(cutoff=cutoff))
        return table

    model.predict = capture
    try:
        evaluate.backtest(model, data, experiments_path=None)
    finally:
        del model.predict
    return pd.concat(frames, ignore_index=True), (pd.concat(quantiles, ignore_index=True) if quantiles else None), seconds


def versions() -> dict:
    found = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def log_run(name: str, record: dict) -> None:
    (OUTPUT_DIR / f"{name}_run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    if not RUN_LOG.exists():
        RUN_LOG.write_text("# Local backtest runs\n\nOne line per run; full details in `<name>_run.json`.\n\n"
                           "| Started (UTC) | Run | Data (sha256) | Seconds | Flow pooled | Occupancy pooled | Settings |\n"
                           "|---|---|---|---:|---:|---:|---|\n", encoding="utf-8")
    flow, occupancy = record["scores"]["debit_horaire"]["all"], record["scores"]["taux_occupation"]["all"]
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(f"| {record['started_utc'][:19]} | {name} | {record['data']['kind']} "
                  f"({record['data']['content_sha256'][:8]}) | {record['seconds_total']:.0f} | {flow:.2f} | "
                  f"{occupancy:.3f} | {record['settings']} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model names (default: all not yet saved)")
    parser.add_argument("--list", action="store_true", help="print the model names and settings")
    parser.add_argument("--end", help="last Paris day of processed traffic, YYYY-MM-DD (default: every stored day; "
                                      "ignored when the #22 dataset exists)")
    args = parser.parse_args()

    data, covariates, source = load_data(args.end)
    calendar_path = DEFAULT_CALENDAR_DIR / "calendar.parquet"
    calendar = pd.read_parquet(calendar_path)
    factories = build_models(calendar, covariates)
    if args.list:
        for name, factory in factories.items():
            try:
                print(f"{name}: {describe(factory())}")
            except RuntimeError as error:
                print(f"{name}: unavailable ({error})")
        return
    if unknown := set(args.models) - set(factories):
        parser.error(f"unknown models {sorted(unknown)}; choose among {list(factories)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data: {source}")

    for name in args.models or factories:
        path = OUTPUT_DIR / f"{name}_predictions.parquet"
        if path.exists() and not args.models:
            print(f"{name}: already saved, skipped")
            continue
        try:
            model = factories[name]()
        except RuntimeError as error:
            print(f"{name}: skipped ({error})")
            continue
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        predictions, quantiles, seconds = run_backtest(model, data)
        total = time.perf_counter() - started
        predictions.to_parquet(path, index=False)
        if quantiles is not None:
            quantiles.to_parquet(OUTPUT_DIR / f"{name}_quantiles.parquet", index=False)
        scores = day_kind_scores(predictions, data)
        log_run(name, {
            "run": name, "model_class": type(model).__name__, "settings": describe(model),
            "started_utc": started_utc, "seconds_total": total,
            "seconds_per_window": dict(zip((str(day.date()) for day in evaluate.load_cutoffs()), seconds)),
            "cutoffs": [str(day.date()) for day in evaluate.load_cutoffs()],
            "data": source, "calendar_sha256": file_sha256(calendar_path),
            "scored_arcs": ["Convention", "Saint-Pères"], "metric": "pooled RMSE over valid hours",
            "forecast_sources": predictions["forecast_source"].value_counts().to_dict(),
            "scores": scores, "versions": versions(),
            "machine": f"{platform.platform()} {platform.machine()}",
        })
        print(f"{name} ({total:.0f} s) -- pooled RMSE Convention + Saint-Pères")
        for variable, by_kind in scores.items():
            print(f"  {variable:16s} all {by_kind['all']:7.2f} | ouvre {by_kind['ouvre']:7.2f} | "
                  f"weekend {by_kind['weekend']:7.2f} | ferie {by_kind['ferie']:7.2f}")


if __name__ == "__main__":
    main()
