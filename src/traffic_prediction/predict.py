"""Reproducible November 2026 submission from incrementally refreshed traffic."""

from __future__ import annotations

import argparse
from http.client import HTTPException
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd

from traffic_prediction.constants import CUTOFF, SUBMISSION_DECIMALS, TARGET_TIMES
from traffic_prediction.data import calendar_events, query, traffic, update
from traffic_prediction.data.validate import SUBMISSION_ARCS, SUBMISSION_COLUMNS, validate_submission
from traffic_prediction.models.base import ForecastModel
from traffic_prediction.submission_format import test_format_and_export_output
from traffic_prediction.models.baselines import (
    DayTypeProfileBaseline, FourWeekMeanBaseline, HolidayAwarePreviousWeekBaseline,
    PreviousWeekBaseline,
)

DEFAULT_OUTPUT_DIR = traffic.REPO_ROOT / "outputs/submissions"
MODELS = {
    "holiday-aware": HolidayAwarePreviousWeekBaseline,
    "previous-week": PreviousWeekBaseline,
    "four-week-mean": FourWeekMeanBaseline,
    "day-type-profile": DayTypeProfileBaseline,
}
SEED = 42


def build_submission(history: pd.DataFrame, model: ForecastModel) -> pd.DataFrame:
    """Fit the chosen model on cutoff-limited data and enforce the export contract.

    Model.predict fits the existing baselines and constructs their lag/profile
    features. Preserve local end-of-hour labels pending organizer clarification.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    history = history.loc[history["datetime"].le(CUTOFF)].copy()
    if history.empty:
        raise ValueError("No traffic history at or before the submission cutoff.")
    table = model.predict(history, TARGET_TIMES.copy())[SUBMISSION_COLUMNS].copy()
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
    skip_update: bool = False, update_start: str | None = None,
    model_name: str = "holiday-aware",
) -> Path:
    """Update recent observations, build calendar features, fit, predict and export.

    No historical archives are downloaded. Frozen local inputs with skip_update
    produce identical bytes; a live refresh can incorporate source corrections.
    """
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
    table = build_submission(history, MODELS[model_name](calendar))
    return export_submission(table, team_name, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--model", choices=MODELS, default="holiday-aware")
    parser.add_argument("--data-dir", type=Path, default=query.DEFAULT_TRAFFIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=traffic.DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=traffic.REPO_ROOT / "data/raw/recent")
    parser.add_argument("--school-holidays-file", type=Path, default=calendar_events.DEFAULT_SCHOOL_PATH)
    parser.add_argument("--skip-update", action="store_true", help="Use frozen local data without network access.")
    parser.add_argument("--update-start", help="First inclusive refresh date; required when bootstrapping recent data.")
    args = parser.parse_args()
    if args.skip_update and args.update_start:
        parser.error("--update-start cannot be combined with --skip-update")
    try:
        path = run_submission(
            args.team_name, data_dir=args.data_dir, output_dir=args.output_dir,
            config_path=args.config, raw_dir=args.raw_dir,
            school_holidays_path=args.school_holidays_file, skip_update=args.skip_update,
            update_start=args.update_start, model_name=args.model,
        )
    except (OSError, ValueError, RuntimeError, HTTPException, duckdb.Error) as error:
        parser.exit(1, f"Submission failed: {error}\n")
    print(f"Validated submission: {path} (360 rows; model={args.model}; seed={SEED}).")


if __name__ == "__main__":
    main()
