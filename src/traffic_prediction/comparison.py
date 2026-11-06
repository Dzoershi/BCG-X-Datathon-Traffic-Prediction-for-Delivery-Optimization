"""Controlled model/feature experiments on the frozen #53 protocol (issue #55)."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from traffic_prediction import evaluate, validation
from traffic_prediction.data import dataset, traffic
from traffic_prediction.data.audit import _json
from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR
from traffic_prediction.data.weather import VALUE_COLUMNS
from traffic_prediction.models.baselines import DayTypeProfileBaseline, HolidayAwarePreviousWeekBaseline
from traffic_prediction.train import FEATURE_GROUPS, LightGBMForecaster

PACKAGES = ("numpy", "pandas", "pyarrow", "holidays", "lightgbm", "torch", "chronos-forecasting", "timestral")


def factories(calendar: pd.DataFrame, covariates: pd.DataFrame) -> dict:
    from traffic_prediction.models.chronos import Chronos2Forecaster
    from traffic_prediction.models.timestral import TimestralForecaster
    result = {"baseline_holiday_aware": lambda: HolidayAwarePreviousWeekBaseline(calendar),
              "baseline_day_type": lambda: DayTypeProfileBaseline(calendar),
              "lightgbm_all": lambda: LightGBMForecaster(covariates, n_jobs=1)}
    for omitted in FEATURE_GROUPS:
        groups = tuple(group for group in FEATURE_GROUPS if group != omitted)
        result[f"lightgbm_without_{omitted}"] = lambda groups=groups: LightGBMForecaster(
            covariates, n_jobs=1, feature_groups=groups)
    result["chronos_calendar"] = lambda: Chronos2Forecaster(calendar, multivariate=True)
    result["chronos_history"] = lambda: Chronos2Forecaster(calendar, multivariate=True, calendar_covariates=())
    for size in ("5M", "28M", "209M"):
        result[f"timestral_{size.lower()}_calendar"] = lambda size=size: TimestralForecaster(
            calendar, model_size=size, context_hours=32768, multivariate=True)
    result["timestral_5m_history"] = lambda: TimestralForecaster(
        calendar, model_size="5M", context_hours=32768, multivariate=True, calendar_covariates=())
    return result


class SensorOutage:
    """Hide target history only; held-out observations remain separate truth."""
    def __init__(self, model, target: str, days: int):
        self.model, self.target, self.days = model, target, days

    def predict(self, history, target_times):
        hidden = history.copy(deep=True)
        start = history.datetime.max().normalize() + pd.Timedelta(days=1 - self.days)
        mask = hidden.arc.eq(self.target) & hidden.datetime.ge(start)
        hidden.loc[mask, list(evaluate.TARGETS)] = np.nan
        hidden.loc[mask, "qualite_mesure"] = "absente"
        result = self.model.predict(hidden, target_times)
        self.sources_ = self.model.sources_
        return result


def measure_run(model, data, cutoffs, recent, durations=(28, 365)) -> tuple[pd.DataFrame, dict]:
    validation.validate_cutoffs(data, [*cutoffs, recent])
    loading = 0.0
    if hasattr(model, "_pipeline"):
        started = perf_counter()
        model._pipeline()
        loading = perf_counter() - started
    records, frames = [], []
    for phase, days in [("development", cutoffs), ("recent_check", [recent])]:
        for day in days:
            started = perf_counter()
            result = evaluate.backtest(model, data, [day], experiments_path=None)
            records.append({"phase": phase, "cutoff": str(day), "wall_seconds": perf_counter() - started,
                            **getattr(model, "timings_", {})})
            frames.append(result.hourly.assign(phase=phase, scenario="observed"))
    target = next(arc.name for arc in traffic.load_arcs() if arc.iu_ac == 4264)
    outage_data = data.copy()
    outage_data["role"] = np.where(outage_data.arc.eq(target), "evalue", "proxy")
    skipped = []
    for duration in durations:
        for day in cutoffs:
            start = day + pd.Timedelta(days=1 - duration)
            truth = outage_data.loc[outage_data.arc.eq(target) & outage_data.datetime.isin(evaluate.target_window(day))]
            if start - data.datetime.min() < pd.Timedelta(days=56) or not any(
                    evaluate.valid_truth(truth, variable).any() for variable in evaluate.TARGETS):
                skipped.append({"days": duration, "cutoff": str(day)})
                continue
            result = evaluate.backtest(SensorOutage(model, target, duration), outage_data, [day], experiments_path=None)
            frames.append(result.hourly.assign(phase="development", scenario=f"outage_{duration}d"))
    return pd.concat(frames, ignore_index=True), {"cold_load_seconds": loading, "windows": records,
                                               "skipped_outages": skipped}


def environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            item = distribution(name)
            packages[name] = {"version": item.version, "installed_bytes": sum(
                path.stat().st_size for file in item.files or [] if (path := Path(item.locate_file(file))).is_file())}
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "packages": packages}


def memory() -> dict:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {"rss_bytes": info.rss, "process_lifetime_peak_bytes": getattr(info, "peak_wset", None)}
    except ImportError:
        return {"rss_bytes": None, "process_lifetime_peak_bytes": None}


def cache_weights() -> dict:
    return {path.name: sum(file.stat().st_size for file in path.glob("snapshots/*/*.safetensors"))
            for path in (traffic.REPO_ROOT / "models/huggingface").glob("models--*")}


def ablation_effects(summary: pd.DataFrame, per_window: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "scenario", "variable"]
    base = summary.loc[summary.model.eq("lightgbm_all"), [*keys, "rmse", "n_scored"]]
    rows = summary.loc[summary.model.str.startswith("lightgbm_without_")].merge(
        base, on=keys, suffixes=("", "_full"), validate="many_to_one")
    rows["removed_group"] = rows.model.str.removeprefix("lightgbm_without_")
    rows["rmse_change_when_removed"] = rows.rmse - rows.rmse_full
    # Positive means the full model performed better; this is not a causal claim.
    return rows[[*keys, "removed_group", "n_scored", "rmse_full", "rmse", "rmse_change_when_removed"]]


def run(args) -> None:
    os.environ.setdefault("HF_HUB_CACHE", str(traffic.REPO_ROOT / "models/huggingface"))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if any(name.startswith(("chronos", "timestral")) for name in args.runs):
        import torch
        torch.set_num_threads(args.threads)
    started = perf_counter()
    calendar = pd.read_parquet(args.calendar)
    data = validation.load_audit(args.audit_dir, calendar)
    covariates = data[["arc", "role"]].drop_duplicates().merge(calendar, how="cross").assign(
        **dict.fromkeys(VALUE_COLUMNS, np.nan))
    constructors = factories(calendar, covariates)
    if not args.runs or len(set(args.runs)) != len(args.runs) or set(args.runs) - constructors.keys():
        raise ValueError(f"Choose unique runs from {list(constructors)}")
    cutoffs, recent = evaluate.load_cutoffs(), validation.RECENT_CUTOFF
    validation.validate_cutoffs(data, [*cutoffs, recent])
    preparation = perf_counter() - started
    env = environment()
    code = {str(path.relative_to(traffic.REPO_ROOT)): traffic.file_sha256(path) for path in [
        Path(__file__), Path(evaluate.__file__), Path(validation.__file__),
        traffic.REPO_ROOT / "src/traffic_prediction/train.py", traffic.REPO_ROOT / "src/traffic_prediction/features.py",
        traffic.REPO_ROOT / "src/traffic_prediction/models/chronos.py", traffic.REPO_ROOT / "src/traffic_prediction/models/timestral.py",
        traffic.REPO_ROOT / "src/traffic_prediction/models/baselines.py"]}
    frozen = {"data_sha256": dataset.content_sha256(data), "calendar_sha256": traffic.file_sha256(args.calendar),
              "audit_observations_sha256": traffic.file_sha256(args.audit_dir / "observations.parquet"),
              "cutoffs": [str(day) for day in cutoffs], "recent": str(recent), "code_sha256": code,
              "threads": args.threads, "versions": {key: value["version"] if value else None
                                                   for key, value in env["packages"].items()}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, runs = [], {}
    for name in args.runs:
        model = constructors[name]()
        settings = model.describe() if hasattr(model, "describe") else type(model).__name__
        key = hashlib.sha256(json.dumps({**frozen, "name": name, "settings": settings}, sort_keys=True).encode()).hexdigest()
        file, receipt = args.output_dir / f"{name}_hourly.parquet", args.output_dir / f"{name}_run.json"
        prior = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if file.exists() and prior.get("cache_key") == key and prior.get("hourly_sha256") == traffic.file_sha256(file):
            print(f"Reuse verified {name}", flush=True)
            frame, details = pd.read_parquet(file), prior
        else:
            print(f"Evaluate {name}", flush=True)
            before = memory()
            started = perf_counter()
            frame, timings = measure_run(model, data, cutoffs, recent)
            frame["model"] = name
            frame.to_parquet(file, index=False, compression="zstd")
            details = {"cache_key": key, "settings": settings, "wall_seconds": perf_counter() - started,
                       "memory_before": before, "memory_after": memory(), **timings,
                       "hourly_sha256": traffic.file_sha256(file)}
            _json(receipt, details)
        runs[name] = details
        frames.append(frame)
        del model
        import gc
        gc.collect()
    hourly = pd.concat(frames, ignore_index=True)
    hourly.to_parquet(args.output_dir / "hourly_predictions.parquet", index=False, compression="zstd")
    validation.save_scores(hourly, args.output_dir)
    summary = pd.read_csv(args.output_dir / "summary.csv")
    ablation_effects(summary, pd.read_csv(args.output_dir / "per_window.csv")).to_csv(
        args.output_dir / "ablations.csv", index=False, lineterminator="\n")
    timing_rows = [{"model": name, "cold_load_seconds": item["cold_load_seconds"],
                    "observed_windows_seconds": sum(window["wall_seconds"] for window in item["windows"]),
                    "all_scenarios_seconds": item["wall_seconds"],
                    "rss_before_bytes": item["memory_before"]["rss_bytes"],
                    "rss_after_bytes": item["memory_after"]["rss_bytes"]} for name, item in runs.items()]
    pd.DataFrame(timing_rows).to_csv(args.output_dir / "runtime.csv", index=False, lineterminator="\n")
    _json(args.output_dir / "run.json", {**frozen, "runs": runs, "environment": env,
          "shared_preparation_seconds": preparation, "cached_weights_bytes": cache_weights(),
          "reference": "HolidayAwarePreviousWeekBaseline, same frozen #53 inputs; #54 remains separately owned",
          "weather": "Excluded: all weather values missing. Archived target-hour forecast issuance unverified.",
          "recent_trends": "Not implemented in existing features; no marginal gain claimed",
          "proxy": "No production proxy feature; #53 copy/offset/linear study is separate; outages compare existing model fallbacks",
          "timing_notes": "Single CPU process; windows include preparation/fit/prediction; LightGBM sub-times exclude fallback. RSS is before/after, not isolated peak. Installed package sizes include files, not download size.",
          "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()})
    print(json.dumps(validation.verify_metrics(args.output_dir)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/quality_52")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_DIR / "calendar.parquet")
    parser.add_argument("--output-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/complexity_55")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--allow-download", action="store_true", help="Permit pinned foundation checkpoints to be fetched")
    parser.add_argument("--runs", nargs="+", default=["baseline_holiday_aware", "baseline_day_type", "lightgbm_all",
                        *[f"lightgbm_without_{group}" for group in FEATURE_GROUPS], "chronos_calendar", "chronos_history",
                        "timestral_5m_calendar", "timestral_5m_history", "timestral_28m_calendar", "timestral_209m_calendar"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
