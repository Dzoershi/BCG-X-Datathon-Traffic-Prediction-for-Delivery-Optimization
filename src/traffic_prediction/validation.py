"""Comparable, reproducible historical validation (#53), excluding the final test.

Run ``python -m traffic_prediction.validation --help`` for the offline audit-backed
experiment or independently verify/rescore its saved hourly predictions.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from traffic_prediction import evaluate
from traffic_prediction.constants import TARGET_START
from traffic_prediction.data import dataset, traffic
from traffic_prediction.data.audit import _json
from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR
from traffic_prediction.data.weather import VALUE_COLUMNS
from traffic_prediction.models.baselines import DayTypeProfileBaseline, HolidayAwarePreviousWeekBaseline
from traffic_prediction.models.proxy import ProxyOutageForecaster

KEYS = ["phase", "scenario", "cutoff", *evaluate.KEYS, "variable"]
GROUPS = {
    "summary": ["phase", "scenario", "model", "variable"],
    "per_arc": ["phase", "scenario", "model", "variable", "arc"],
    "per_window": ["phase", "scenario", "model", "variable", "cutoff"],
    "per_day_kind": ["phase", "scenario", "model", "variable", "day_kind"],
    "per_forecast_source": ["phase", "scenario", "model", "variable", "forecast_source"],
    "scores": ["phase", "scenario", "model", "variable", "cutoff", "arc"],
}
RECENT_CUTOFF = pd.Timestamp("2026-10-29")


def check_alignment(hourly: pd.DataFrame) -> None:
    """Require a complete common truth grid, never a convenient prediction intersection."""
    required = {*KEYS, "model", "truth", "truth_source", "truth_valid", "prediction",
                "day_kind", "forecast_source", "is_fallback"}
    if missing := required - set(hourly):
        raise ValueError(f"Hourly predictions missing columns: {sorted(missing)}")
    if hourly.empty or hourly.duplicated(["model", *KEYS]).any() or hourly[KEYS + ["model"]].isna().any().any():
        raise ValueError("Empty hourly predictions, missing keys, or duplicate prediction keys")
    if hourly.truth_valid.isna().any() or not pd.api.types.is_bool_dtype(hourly.truth_valid):
        raise ValueError("Truth eligibility must be explicit booleans")
    if not hourly.variable.isin(evaluate.TARGETS).all():
        raise ValueError("Unknown target variable")
    if not hourly.day_kind.isin(["working", "weekend", "holiday"]).all():
        raise ValueError("Unknown day kind")
    if (not pd.api.types.is_datetime64_any_dtype(hourly.datetime.dtype)
            or hourly.datetime.dt.tz is not None or hourly.datetime.max() >= TARGET_START):
        raise ValueError("Expected historical Paris hours; the final test is reserved")
    for cutoff, rows in hourly.groupby("cutoff", sort=False):
        if not rows.datetime.isin(evaluate.target_window(cutoff)).all():
            raise ValueError("Predictions contain hours outside their cutoff's window")
    counts = hourly.groupby(["model", "phase", "scenario", "cutoff", "arc", "variable"]).size()
    if not counts.eq(120).all():
        raise ValueError("Every comparison needs the complete 120-hour grid, including missing truth")
    if not hourly.groupby(["model", "phase", "scenario", "cutoff", "arc"]).variable.nunique().eq(2).all():
        raise ValueError("Both targets must appear in every comparison")
    for _, comparison in hourly.groupby(["phase", "scenario"], sort=False):
        reference = None
        for _, candidate in comparison.groupby("model", sort=False):
            truth = candidate.set_index(KEYS)[["truth", "truth_source", "truth_valid", "day_kind"]].sort_index()
            if reference is not None and not truth.equals(reference):
                raise ValueError("Candidate models must have identical truth rows, values and eligibility")
            reference = truth
    valid = hourly.truth_valid
    domain = np.isfinite(hourly.truth) & hourly.truth.ge(0)
    domain &= hourly.variable.ne("taux_occupation") | hourly.truth.le(100)
    if (valid & (~domain | hourly.truth_source.ne("observed"))).any():
        raise ValueError("Only valid observed measurements can be eligible truth")
    predicted_domain = np.isfinite(hourly.prediction) & hourly.prediction.ge(0)
    predicted_domain &= hourly.variable.ne("taux_occupation") | hourly.prediction.le(100)
    if (valid & ~predicted_domain).any():
        raise ValueError("Missing/nonfinite or out-of-domain prediction on common eligible truth")


def metrics(hourly: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pooled RMSE = sqrt(sum squared error / scored hours), never a mean of RMSEs."""
    selected = hourly.truth_valid
    work = hourly.assign(
        n_expected=1, n_scored=selected.astype(int), n_ignored=(~selected).astype(int),
        n_estimated_truth=hourly.truth_source.str.startswith("proxy", na=False).astype(int),
        n_fallback=(selected & hourly.is_fallback.eq(True).fillna(False)).astype(int),
        n_native=(selected & hourly.is_fallback.eq(False).fillna(False)).astype(int),
        n_source_unknown=(selected & hourly.is_fallback.isna()).astype(int),
        n_forecast_fallback=hourly.is_fallback.eq(True).fillna(False).astype(int),
        n_forecast_native=hourly.is_fallback.eq(False).fillna(False).astype(int),
        n_forecast_unknown=hourly.is_fallback.isna().astype(int),
        sum_squared_error=((hourly.prediction - hourly.truth).where(selected, 0) ** 2),
    )
    counts = ["n_expected", "n_scored", "n_ignored", "n_estimated_truth", "n_fallback", "n_native", "n_source_unknown",
              "n_forecast_fallback", "n_forecast_native", "n_forecast_unknown",
              "sum_squared_error"]
    result = work.groupby(keys, sort=False, dropna=False)[counts].sum().reset_index()
    result["rmse"] = np.sqrt(result.sum_squared_error / result.n_scored.replace(0, np.nan))
    return result


def save_scores(hourly: pd.DataFrame, output: Path) -> None:
    check_alignment(hourly)
    output.mkdir(parents=True, exist_ok=True)
    scores = metrics(hourly, GROUPS["scores"])
    macro = scores.groupby(GROUPS["summary"], sort=False).rmse.mean().rename("macro_rmse")
    for name, keys in GROUPS.items():
        table = scores if name == "scores" else metrics(hourly, keys)
        if name == "summary":
            table = table.merge(macro, on=GROUPS["summary"], validate="one_to_one")
            table["pooled_rmse"] = table.rmse
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")


def verify_metrics(output: Path) -> dict:
    """Independently recompute every table via Python loops and math.fsum."""
    hourly = pd.read_parquet(output / "hourly_predictions.parquet")
    check_alignment(hourly)
    records = hourly.to_dict("records")
    checked = 0

    def group_key(row: dict, keys: list[str]) -> tuple:
        return tuple(str(pd.Timestamp(row[key])) if key == "cutoff" else str(row[key]) for key in keys)

    macro_groups = {}
    for row in records:
        cells = macro_groups.setdefault(group_key(row, GROUPS["summary"]), {})
        errors = cells.setdefault((str(row["cutoff"]), row["arc"]), [])
        if row["truth_valid"]:
            errors.append((float(row["prediction"]) - float(row["truth"])) ** 2)
    for name, keys in GROUPS.items():
        stored = pd.read_csv(output / f"{name}.csv", keep_default_na=False)
        expected = {}
        for row in records:
            group = group_key(row, keys)
            value = expected.setdefault(group, {"n_expected": 0, "n_scored": 0, "errors": [],
                                                "n_fallback": 0, "n_native": 0, "n_source_unknown": 0,
                                                "n_estimated_truth": 0, "n_forecast_fallback": 0,
                                                "n_forecast_native": 0, "n_forecast_unknown": 0})
            value["n_expected"] += 1
            value["n_estimated_truth"] += str(row["truth_source"]).startswith("proxy")
            origin = ("n_forecast_unknown" if pd.isna(row["is_fallback"])
                      else "n_forecast_fallback" if row["is_fallback"] else "n_forecast_native")
            value[origin] += 1
            if row["truth_valid"]:
                value["n_scored"] += 1
                value["errors"].append((float(row["prediction"]) - float(row["truth"])) ** 2)
                kind = ("n_source_unknown" if pd.isna(row["is_fallback"])
                        else "n_fallback" if row["is_fallback"] else "n_native")
                value[kind] += 1
        if len(stored) != len(expected):
            raise ValueError(f"Verification failed: {name} group count")
        for row in stored.to_dict("records"):
            group = group_key(row, keys)
            value = expected.pop(group)
            for count in ["n_expected", "n_scored", "n_fallback", "n_native", "n_source_unknown", "n_estimated_truth",
                          "n_forecast_fallback", "n_forecast_native", "n_forecast_unknown"]:
                if int(row[count]) != value[count]:
                    raise ValueError(f"Verification failed: {name} {count}")
            if int(row["n_ignored"]) != value["n_expected"] - value["n_scored"]:
                raise ValueError(f"Verification failed: {name} ignored count")
            total = math.fsum(value["errors"])
            if not math.isclose(float(row["sum_squared_error"]), total, rel_tol=1e-11, abs_tol=1e-8):
                raise ValueError(f"Verification failed: {name} squared errors")
            rmse = math.sqrt(total / value["n_scored"]) if value["n_scored"] else None
            if (rmse is None and row["rmse"] != "") or (
                    rmse is not None and not math.isclose(float(row["rmse"]), rmse, rel_tol=1e-11, abs_tol=1e-8)):
                raise ValueError(f"Verification failed: {name} RMSE")
            if name == "summary":
                values = [math.sqrt(math.fsum(errors) / len(errors)) for errors in macro_groups[group].values() if errors]
                macro = math.fsum(values) / len(values) if values else None
                for field, number in [("macro_rmse", macro), ("pooled_rmse", rmse)]:
                    if (number is None and row[field] != "") or (number is not None and not math.isclose(
                            float(row[field]), number, rel_tol=1e-11, abs_tol=1e-8)):
                        raise ValueError(f"Verification failed: {name} {field}")
            checked += 1
    result = {"hourly_rows": len(hourly), "aggregate_rows_independently_checked": checked,
              "comparison_alignment_verified": True, "hourly_sha256": traffic.file_sha256(output / "hourly_predictions.parquet")}
    _json(output / "verification.json", result)
    return result


def validate_cutoffs(data: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> None:
    if not cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Supply unique development cutoffs")
    for day in cutoffs:
        times = evaluate.target_window(day)
        if times.max() >= TARGET_START:
            raise ValueError("The November 7-11 final test is reserved and cannot be evaluated or tuned here")
        if times.max() > data.datetime.max() or day < data.datetime.min():
            raise ValueError("Cutoff/forecast extends outside the frozen data availability")


def run_candidates(models: dict, data: pd.DataFrame, cutoffs: list[pd.Timestamp], phase: str) -> pd.DataFrame:
    validate_cutoffs(data, cutoffs)
    frames = []
    for name, model in models.items():
        print(f"{phase}: {name}", flush=True)
        result = evaluate.backtest(model, data, cutoffs, experiments_path=None)
        frames.append(result.hourly.assign(model=name, phase=phase, scenario="observed"))
    result = pd.concat(frames, ignore_index=True)
    check_alignment(result)
    return result


def run_outages(data: pd.DataFrame, calendar: pd.DataFrame, cutoffs: list[pd.Timestamp],
                durations: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    validate_cutoffs(data, cutoffs)
    arcs = traffic.load_arcs()
    target = next(arc.name for arc in arcs if arc.role == "evalue" and arc.iu_ac == 4264)
    proxies = [arc.name for arc in arcs if arc.role == "proxy"]
    frames, skipped = [], []
    for days in durations:
        for cutoff in cutoffs:
            start = cutoff + pd.Timedelta(days=1 - days)
            truth = data.loc[data.arc.eq(target) & data.datetime.isin(evaluate.target_window(cutoff))]
            if start - data.datetime.min() < pd.Timedelta(days=56) or not any(
                    evaluate.valid_truth(truth, variable).any() for variable in evaluate.TARGETS):
                skipped.append({"cutoff": str(cutoff), "outage_days": days,
                                "reason": "less than 56 pre-outage calendar days or no observed target truth"})
                continue
            options = [("profile", None, "profile")]
            options += [(f"{method}:{proxy}", proxy, method) for proxy in proxies for method in ("copy", "offset", "linear")]
            for name, proxy, method in options:
                model = ProxyOutageForecaster(calendar, target, proxy, days, method)
                result = evaluate.backtest(model, data, [cutoff], experiments_path=None)
                frames.append(result.hourly.loc[result.hourly.arc.eq(target)].assign(
                    model=name, phase="development", scenario=f"outage_{days}d"))
        print(f"Outage experiment: {days} days", flush=True)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        check_alignment(result)
    return result, skipped


def load_audit(path: Path, calendar: pd.DataFrame) -> pd.DataFrame:
    """Use the verified #52 sidecar; estimates never become measurement truth."""
    raw = pd.read_parquet(path / "observations.parquet")
    frame = raw[[*evaluate.KEYS, "role", "legacy_quality"]].rename(columns={"legacy_quality": "qualite_mesure"})
    for short, target in zip(("q", "k"), evaluate.TARGETS):
        frame[target] = raw[f"{short}_value"].where(raw[f"{short}_status"].eq("valid")).astype(float)
        frame[f"{target}_source"] = np.where(raw.source_file.notna(), "observed", "absent")
    frame = frame.merge(calendar[["datetime", "is_holiday"]], on="datetime", how="left", validate="many_to_one")
    if frame.is_holiday.isna().any():
        raise ValueError("Calendar does not cover the frozen audit data")
    return frame.assign(**dict.fromkeys(VALUE_COLUMNS, np.nan))


def build_models(names: list[str], calendar: pd.DataFrame, covariates: pd.DataFrame) -> dict:
    from traffic_prediction.train import LightGBMForecaster
    factories = {
        "baseline_day_type": lambda: DayTypeProfileBaseline(calendar),
        "baseline_holiday_aware": lambda: HolidayAwarePreviousWeekBaseline(calendar),
        "lightgbm_no_weather": lambda: LightGBMForecaster(covariates, n_jobs=1),
    }
    return {name: factories[name]() for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/quality_52")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_DIR / "calendar.parquet")
    parser.add_argument("--output-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/validation_53")
    parser.add_argument("--models", nargs="+", choices=["baseline_day_type", "baseline_holiday_aware", "lightgbm_no_weather"],
                        default=["baseline_day_type", "baseline_holiday_aware", "lightgbm_no_weather"])
    parser.add_argument("--outage-days", nargs="*", type=int, default=[28, 90, 180, 365])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_metrics(args.output_dir)))
        return
    if len(set(args.models)) != len(args.models) or len(set(args.outage_days)) != len(args.outage_days):
        parser.error("Models and outage durations must be unique")
    if any(day < 1 for day in args.outage_days):
        parser.error("Outage durations must be positive")
    calendar = pd.read_parquet(args.calendar)
    data = load_audit(args.audit_dir, calendar)
    cutoffs = evaluate.load_cutoffs()
    if RECENT_CUTOFF in cutoffs:
        raise ValueError("Recent validation cutoff must be separate from development cutoffs")
    # Calendar-only covariates avoid historical weather issuance assumptions.
    arcs = data[["arc", "role"]].drop_duplicates()
    covariates = arcs.merge(calendar, how="cross").assign(**dict.fromkeys(VALUE_COLUMNS, np.nan))
    models = build_models(args.models, calendar, covariates)
    frames = [run_candidates(models, data, cutoffs, "development"),
              run_candidates(models, data, [RECENT_CUTOFF], "recent_check")]
    outages, skipped = run_outages(data, calendar, cutoffs, args.outage_days)
    if not outages.empty:
        frames.append(outages)
    hourly = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.output_dir / "hourly_predictions.parquet", index=False, compression="zstd")
    save_scores(hourly, args.output_dir)
    packages = {}
    for package in ["numpy", "pandas", "pyarrow", "holidays", "lightgbm", "scikit-learn"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {"schema_version": 1, "development_cutoffs": [str(day) for day in cutoffs],
              "recent_validation_cutoff": str(RECENT_CUTOFF), "outage_days": args.outage_days,
              "skipped_outages": skipped,
              "scoring": {"pooled_rmse": "sqrt(sum((prediction-truth)^2)/n_scored)",
                          "macro_rmse": "unweighted mean of nonempty arc/window RMSEs",
                          "organiser_aggregation": "unspecified in brief; targets reported separately",
                          "truth_policy": "independent, measured targets only; identical rows across candidates",
                          "final_test": "November 7-11 reserved, never evaluated",
                          "recent_check": "one fixed-settings diagnostic; not claimed independent of all team exploration"},
              "model_settings": {name: model.describe() if hasattr(model, "describe") else type(model).__name__
                                 for name, model in models.items()},
              "weather_inputs": "All weather covariates are explicitly missing; calendar/traffic-only diagnostic, not production E5",
              "outage_settings": {"methods": ["profile", "copy", "offset", "linear"], "min_pairs": 20,
                                  "min_pre_outage_calendar_days": 56, "input": "previous week, at or before cutoff"},
              "data": {"content_sha256": dataset.content_sha256(data), "rows": len(data),
                       "start": str(data.datetime.min()), "end": str(data.datetime.max()),
                       "observations_sha256": traffic.file_sha256(args.audit_dir / "observations.parquet"),
                       "audit_inputs": json.loads((args.audit_dir / "inputs.json").read_text(encoding="utf-8")),
                       "audit_report": json.loads((args.audit_dir / "report.json").read_text(encoding="utf-8")),
                       "calendar_sha256": traffic.file_sha256(args.calendar)},
              "code_sha256": {str(path.relative_to(traffic.REPO_ROOT)): traffic.file_sha256(path) for path in [
                  Path(__file__), Path(evaluate.__file__), traffic.REPO_ROOT / "src/traffic_prediction/models/proxy.py",
                  traffic.REPO_ROOT / "src/traffic_prediction/models/baselines.py", traffic.REPO_ROOT / "src/traffic_prediction/train.py",
                  traffic.REPO_ROOT / "src/traffic_prediction/features.py"]},
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
              "environment": {"python": platform.python_version(), **packages}}
    _json(args.output_dir / "run.json", report)
    print(json.dumps(verify_metrics(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
