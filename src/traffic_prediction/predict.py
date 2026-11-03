"""Reproducible November 2026 submission from incrementally refreshed traffic."""

from __future__ import annotations

import argparse
from http.client import HTTPException
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import time

import duckdb
import numpy as np
import pandas as pd

from traffic_prediction.constants import CUTOFF, SUBMISSION_DECIMALS, TARGET_TIMES
from traffic_prediction.data import calendar_events, disruptions, query, traffic, update, weather
from traffic_prediction.data.build import DEFAULT_DATASET_DIR
from traffic_prediction.data.validate import SUBMISSION_ARCS, SUBMISSION_COLUMNS, validate_submission
from traffic_prediction.models.base import ForecastModel
from traffic_prediction.models.postprocess import copy_closures_from_previous_year
from traffic_prediction.submission_format import test_format_and_export_output
from traffic_prediction.submission_log import write_run_log
from traffic_prediction.models.baselines import (
    DayTypeProfileBaseline, FourWeekMeanBaseline, HolidayAwarePreviousWeekBaseline,
    PreviousWeekBaseline,
)
from traffic_prediction.models.ensemble import E5_WEIGHTS, WeightedEnsemble
from traffic_prediction.models.timestral import TimestralForecaster
from traffic_prediction.train import LightGBMForecaster

DEFAULT_OUTPUT_DIR = traffic.REPO_ROOT / "outputs/submissions"


def load_covariates() -> pd.DataFrame:
    """Measurement-free covariates of the #22 dataset (calendar and weather through the target week)."""
    path = DEFAULT_DATASET_DIR / "covariates.parquet"
    if not path.exists():
        raise FileNotFoundError(f"The ensemble needs {path}: run python -m traffic_prediction.data.build "
                                f"--dataset --end {CUTOFF:%Y-%m-%d} first.")
    return pd.read_parquet(path)


def e5_ensemble(calendar: pd.DataFrame, covariates: pd.DataFrame | None = None) -> WeightedEnsemble:
    """Submitted model (#32): baseline, Timestral-5M and LightGBM, weighted per variable and day kind.

    Members use their retained backtest settings (#17, #44, #31). LightGBM needs the
    measurement-free covariates of the #22 dataset, built through the cutoff day.
    """
    covariates = load_covariates() if covariates is None else covariates
    return WeightedEnsemble({
        "baseline_holiday_aware": HolidayAwarePreviousWeekBaseline(calendar),
        "timestral_5m": TimestralForecaster(calendar, model_size="5M", multivariate=True, context_hours=32768),
        # One thread: once torch is loaded (Timestral), multithreaded LightGBM deadlocks in OpenMP
        # (macOS, several libomp copies). Deterministic forecasts are identical to n_jobs=4.
        "lightgbm_raw": LightGBMForecaster(covariates, n_jobs=1),
    }, E5_WEIGHTS, calendar)


MODELS = {
    "ensemble": e5_ensemble,
    "holiday-aware": HolidayAwarePreviousWeekBaseline,
    "previous-week": PreviousWeekBaseline,
    "four-week-mean": FourWeekMeanBaseline,
    "day-type-profile": DayTypeProfileBaseline,
}
SEED = 42


def build_submission(history: pd.DataFrame, model: ForecastModel,
                     events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit the chosen model on cutoff-limited data and enforce the export contract.

    Model.predict fits the existing baselines and constructs their lag/profile
    features. Preserve local end-of-hour labels pending organizer clarification.
    With `events` (data/raw/events_manual.csv), ceremony closures are copied from the
    previous year before clipping and rounding (#47); `model.sources_` marks those hours.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    history = history.loc[history["datetime"].le(CUTOFF)].copy()
    if history.empty:
        raise ValueError("No traffic history at or before the submission cutoff.")
    table = model.predict(history, TARGET_TIMES.copy())[SUBMISSION_COLUMNS].copy()
    if events is not None:
        replaced = copy_closures_from_previous_year(table, history, events)
        sources = getattr(model, "sources_", None)
        if isinstance(sources, pd.Series) and len(replaced):
            sources = sources.astype(object)
            sources.loc[replaced] = [f"closure_copy_{time.year - 1}" for _, time in replaced]
            model.sources_ = sources
    table["datetime"] = table["datetime"].dt.strftime("%Y-%m-%d %H:%M").astype(object)
    table["arc"] = table["arc"].astype(object)
    for column, upper in [("debit_horaire", None), ("taux_occupation", 100)]:
        values = table[column].astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions for {column}.")
        # Round after clipping so that bounds stay exact; 15 decimals carry no information.
        table[column] = values.clip(lower=0, upper=upper).round(SUBMISSION_DECIMALS)
    validate_submission(table)
    return table.sort_values(["arc", "datetime"], kind="stable").reset_index(drop=True)


def export_submission(table: pd.DataFrame, team_name: str, output_dir: str | Path) -> Path:
    """Validate before atomically publishing a deterministic UTF-8 CSV."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", team_name):
        raise ValueError("team_name must contain only letters, digits, underscores or hyphens.")
    validate_submission(table)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"output_{team_name}.csv"
    with TemporaryDirectory(dir=output_dir, prefix=".submission-") as staging:
        # Keep the organizer's exporter unchanged, including its CSV index.
        # A fresh index and stable row ordering make direct exports deterministic.
        table = table.sort_values(["arc", "datetime"], kind="stable").reset_index(drop=True)
        test_format_and_export_output(table, staging, team_name)
        temporary = Path(staging) / destination.name
        # Check the serialized representation as well as the in-memory table.
        validate_submission(pd.read_csv(temporary, sep=";", index_col=0, dtype={
            "arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float,
        }))
        temporary.replace(destination)
    return destination


def run_submission(
    team_name: str, *, data_dir: str | Path = query.DEFAULT_TRAFFIC_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = traffic.DEFAULT_CONFIG,
    raw_dir: str | Path = traffic.REPO_ROOT / "data/raw/recent",
    school_holidays_path: str | Path = calendar_events.DEFAULT_SCHOOL_PATH,
    events_path: str | Path = disruptions.DEFAULT_EVENTS_PATH,
    skip_update: bool = False, update_start: str | None = None,
    model_name: str = "holiday-aware",
) -> Path:
    """Update recent observations, build calendar features, fit, predict and export.

    No historical archives are downloaded. Frozen local inputs with skip_update
    produce identical bytes; a live refresh can incorporate source corrections.
    """
    started = time.perf_counter()
    if pd.__version__.split(".")[0] != "2":
        raise ValueError("Submission requires pandas 2.x; install the pinned requirements.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", team_name):
        raise ValueError("team_name must contain only letters, digits, underscores or hyphens.")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if not skip_update:
        update.update_traffic(
            update_start, CUTOFF.strftime("%Y-%m-%d"), output_dir=data_dir,
            raw_dir=raw_dir, config_path=config_path,
        )
    history = query.query_traffic(
        """SELECT arc, datetime, role, qualite_mesure, debit_horaire, taux_occupation
           FROM traffic WHERE datetime <= ? AND role = 'evalue'
           ORDER BY arc, datetime""", [CUTOFF], path=data_dir,
    )
    if history.empty or set(history["arc"]) != set(SUBMISSION_ARCS):
        raise ValueError("History must contain all three official evaluated arcs.")
    school_holidays_path = Path(school_holidays_path)
    if not school_holidays_path.exists():
        if skip_update:
            raise FileNotFoundError("Offline run requires a local school-holidays snapshot.")
        calendar_events.download_school_holidays(school_holidays_path)
    calendar = calendar_events.calendar_table(
        history["datetime"].min().normalize(), school_holidays_path=school_holidays_path,
    )
    events = disruptions.read_events(events_path, list(SUBMISSION_ARCS))
    if model_name == "ensemble":
        # LightGBM learns from the holiday flag and weather of past hours, as in the backtest dataset.
        covariates = load_covariates()
        history = history.merge(calendar[["datetime", "is_holiday"]], on="datetime", how="left",
                                validate="many_to_one")
        history = history.merge(covariates[["arc", "datetime", *weather.VALUE_COLUMNS]],
                                on=["arc", "datetime"], how="left", validate="one_to_one")
        model = e5_ensemble(calendar, covariates)
    else:
        model = MODELS[model_name](calendar)
    table = build_submission(history, model, events)
    path = export_submission(table, team_name, output_dir)
    write_run_log(path, history=history, table=table, model=model, model_name=model_name,
                  cutoff=CUTOFF, seed=SEED, started=started)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--model", choices=MODELS, default="ensemble",
                        help="ensemble: retained model (#32); holiday-aware: plan B without Timestral or dataset")
    parser.add_argument("--data-dir", type=Path, default=query.DEFAULT_TRAFFIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=traffic.DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=traffic.REPO_ROOT / "data/raw/recent")
    parser.add_argument("--school-holidays-file", type=Path, default=calendar_events.DEFAULT_SCHOOL_PATH)
    parser.add_argument("--events-file", type=Path, default=disruptions.DEFAULT_EVENTS_PATH,
                        help="Hand-checked events; ceremony closures are copied from the previous year (#47).")
    parser.add_argument("--skip-update", action="store_true", help="Use frozen local data without network access.")
    parser.add_argument("--update-start", help="First inclusive refresh date; required when bootstrapping recent data.")
    args = parser.parse_args()
    if args.skip_update and args.update_start:
        parser.error("--update-start cannot be combined with --skip-update")
    try:
        path = run_submission(
            args.team_name, data_dir=args.data_dir, output_dir=args.output_dir,
            config_path=args.config, raw_dir=args.raw_dir,
            school_holidays_path=args.school_holidays_file, events_path=args.events_file,
            skip_update=args.skip_update,
            update_start=args.update_start, model_name=args.model,
        )
    except (OSError, ValueError, RuntimeError, HTTPException, duckdb.Error) as error:
        parser.exit(1, f"Submission failed: {error}\n")
    print(f"Validated submission: {path} (360 rows; model={args.model}; seed={SEED}).")


if __name__ == "__main__":
    main()
