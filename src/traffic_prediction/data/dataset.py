"""Modelling tables: processed traffic joined with its known covariates.

``dataset.parquet`` keeps every traffic row of the period, with the calendar, the
weather and the disruption flags of its hour. ``covariates.parquet`` holds the same
covariates for every arc and hour up to the last day to predict, without any
measurement: models look up their target hours there, in backtests (whose history
stops at the cutoff) and for the submission.

Disruption flags are a mask, not a training feature (issue #34); they are <NA> for
proxies, whose surroundings were never searched. No feature is derived here (#30).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from traffic_prediction.data import calendar_events, traffic, weather
from traffic_prediction.data.traffic import Arc

DATASET_FILE = "dataset.parquet"
COVARIATES_FILE = "covariates.parquet"
# Leading underscore: Parquet readers skip the file when reading the folder.
MANIFEST_FILE = "_manifest.json"
KEYS = ["arc", "datetime"]
CALENDAR_FEATURES = calendar_events.CALENDAR_COLUMNS[1:]
# weather.parquet "source", renamed: in the joined table, "source" alone is ambiguous.
WEATHER_FEATURES = [*weather.VALUE_COLUMNS, "weather_source"]
DISRUPTION_FLAGS = ["chantier_actif", "evenement_actif"]
COVARIATES = [*CALENDAR_FEATURES, *WEATHER_FEATURES, *DISRUPTION_FLAGS]
DATASET_COLUMNS = [*traffic.OUTPUT_COLUMNS, *COVARIATES]
COVARIATE_COLUMNS = ["arc", "role", "datetime", *COVARIATES]
COVARIATE_DTYPES = {
    "hour": "int8",
    "day_of_week": "int8",
    **{column: "bool" for column in CALENDAR_FEATURES if column.startswith("is_")},
    **dict.fromkeys(weather.VALUE_COLUMNS, "float64"),
    "weather_source": "object",
    **dict.fromkeys(DISRUPTION_FLAGS, "boolean"),
}
# How to fix each input, named in its error message.
FIX_HINTS = {
    "traffic": "check it with: python -m traffic_prediction.data.validate",
    "calendar": "rebuild it with: python -m traffic_prediction.data.build --calendar",
    "weather": "rebuild it with: python -m traffic_prediction.data.build --weather",
    "disruptions": "rebuild it with: python -m traffic_prediction.data.build --disruptions",
}


# --- Reading -------------------------------------------------------------------


def read_traffic(
    path: Path, arcs: list[Arc], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Processed traffic rows of inclusive Paris days, arcs in configuration order.

    Without dates, every stored day. A period beyond the stored days is refused:
    the dataset keeps every traffic row of the period, and only those.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; build or update the traffic first (see the README).")
    stored = pd.read_parquet(path, columns=traffic.OUTPUT_COLUMNS)
    if stored.empty:
        raise ValueError(f"No processed traffic in {path}; build or update it first.")
    stored_first, stored_last = traffic.period_bounds(stored["datetime"].min(), stored["datetime"].max())
    first, last = traffic.period_bounds(start or stored_first, end or stored_last)
    if first < stored_first or last > stored_last:
        raise ValueError(
            f"Traffic is stored from {stored_first:%Y-%m-%d} to {stored_last:%Y-%m-%d}; "
            "choose --start and --end within these days."
        )
    return sort_rows(stored[stored["datetime"].between(first, last)], arcs)


def read_hourly(path: Path, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Rows of a processed hourly table (calendar, weather, disruptions) from first to last."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; build it first (see the README).")
    table = pd.read_parquet(path)
    return table[table["datetime"].between(first, last)].reset_index(drop=True)


def sort_rows(table: pd.DataFrame, arcs: list[Arc]) -> pd.DataFrame:
    """Arcs in configuration order (evaluated first), then time."""
    order = {arc.name: position for position, arc in enumerate(arcs)}
    return table.sort_values(
        KEYS, key=lambda column: column.map(order) if column.name == "arc" else column,
        ignore_index=True,
    )


# --- Joins ---------------------------------------------------------------------


def covariate_grid(arcs: list[Arc], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Every (arc, hour) label from first to last, without measurements."""
    hours = pd.date_range(first, last, freq="h")
    index = pd.MultiIndex.from_product([[arc.name for arc in arcs], hours], names=KEYS)
    grid = index.to_frame(index=False)
    grid.insert(1, "role", grid["arc"].map({arc.name: arc.role for arc in arcs}))
    return grid


def add_covariates(
    rows: pd.DataFrame,
    calendar: pd.DataFrame,
    weather_hours: pd.DataFrame,
    disruption_hours: pd.DataFrame,
) -> pd.DataFrame:
    """Append the covariates of each (arc, datetime) row; rows and their order are unchanged.

    Calendar and weather join on ``datetime``: one hour serves every arc. Disruption
    flags join on (``arc``, ``datetime``); arcs without disruption rows get <NA>.
    """
    weather_columns = weather_hours[["datetime", *weather.VALUE_COLUMNS, "source"]].rename(
        columns={"source": "weather_source"}
    )
    flags = disruption_hours[[*KEYS, *DISRUPTION_FLAGS]].astype(
        {flag: "boolean" for flag in DISRUPTION_FLAGS}
    )
    joined = (
        rows.merge(calendar[calendar_events.CALENDAR_COLUMNS], on="datetime", how="left",
                   validate="many_to_one")
        .merge(weather_columns, on="datetime", how="left", validate="many_to_one")
        .merge(flags, on=KEYS, how="left", validate="one_to_one")
    )
    return joined[[*rows.columns, *COVARIATES]]


# --- Validation ----------------------------------------------------------------


def check_source(name: str, check: Callable[..., None], *args: object) -> None:
    """Run an input's own validator; the error names the input and how to fix it."""
    try:
        check(*args)
    except ValueError as error:
        raise ValueError(f"Invalid {name} input: {str(error).rstrip('.')}; {FIX_HINTS[name]}") from error


def check_calendar(calendar: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> None:
    """One calendar row per hour from first to last, sorted, with the #16 columns and no gap."""
    if list(calendar.columns) != calendar_events.CALENDAR_COLUMNS:
        raise ValueError(f"Unexpected calendar columns: {list(calendar.columns)}")
    hours = pd.date_range(first, last, freq="h")
    if not pd.DatetimeIndex(calendar["datetime"]).equals(hours):
        missing = hours.difference(calendar["datetime"])
        duplicated = int(calendar["datetime"].duplicated().sum())
        raise ValueError(f"Calendar must cover {first} to {last}, once per hour and sorted: "
                         f"{len(missing)} missing hours, {duplicated} duplicates")
    if calendar.isna().any().any():
        raise ValueError("Calendar has missing values")


def validate_tables(
    table: pd.DataFrame,
    covariates: pd.DataFrame,
    history: pd.DataFrame,
    arcs: list[Arc],
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> None:
    """Stop the build unless both outputs match their contract.

    dataset: the traffic rows of the period, unchanged and in the same order, each with
    complete covariates. covariates: one row per configured arc and hour, first to last.
    """
    for name, frame, columns in [("dataset", table, DATASET_COLUMNS),
                                 ("covariates", covariates, COVARIATE_COLUMNS)]:
        if list(frame.columns) != columns:
            raise ValueError(f"{name}: expected columns {columns}, got {list(frame.columns)}.")
    if len(table) != len(history):
        raise ValueError(f"dataset has {len(table)} rows, the traffic of the period {len(history)}.")
    if duplicated := int(table.duplicated(KEYS).sum()):
        raise ValueError(f"dataset: {duplicated} duplicate (arc, datetime) rows.")
    if not table[traffic.OUTPUT_COLUMNS].equals(history):
        raise ValueError("dataset: traffic columns differ from the processed traffic.")
    if not covariates[["arc", "role", "datetime"]].equals(covariate_grid(arcs, first, last)):
        raise ValueError(f"covariates must have one row per configured arc and hour from {first} to {last}.")
    check_covariates(table, "dataset")
    check_covariates(covariates, "covariates")


def check_covariates(frame: pd.DataFrame, name: str) -> None:
    """Calendar and weather at every row, flags on evaluated arcs only, expected dtypes."""
    unknown = frame[[*CALENDAR_FEATURES, *WEATHER_FEATURES]].isna().any(axis=1)
    if unknown.any():
        raise ValueError(f"{name}: {int(unknown.sum())} rows without calendar or weather, "
                         f"first at {frame.loc[unknown, 'datetime'].iloc[0]}.")
    evaluated = frame["role"].eq("evalue")
    flags = frame[DISRUPTION_FLAGS]
    if missing := int((evaluated & flags.isna().any(axis=1)).sum()):
        raise ValueError(f"{name}: {missing} rows of evaluated arcs without disruption flags.")
    if filled := int((~evaluated & flags.notna().any(axis=1)).sum()):
        raise ValueError(f"{name}: {filled} proxy rows with disruption flags; they must be <NA>.")
    for column, dtype in COVARIATE_DTYPES.items():
        if str(frame[column].dtype) != dtype:
            raise ValueError(f"{name}: {column} must be {dtype}, got {frame[column].dtype}.")


# --- Version manifest ----------------------------------------------------------


def content_sha256(table: pd.DataFrame) -> str:
    """SHA-256 of the column names, dtypes and values.

    The same data gives the same hash on every computer, whatever the Parquet
    encoding, with the pinned pandas version. <NA> and False hash differently.
    """
    digest = hashlib.sha256(json.dumps([[column, str(dtype)] for column, dtype in table.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def input_hashes(traffic_dir: Path, sources: dict[str, Path]) -> dict:
    """SHA-256 of each input file: traffic partitions by relative path, then the hourly tables."""
    partitions = sorted(
        path for path in traffic_dir.rglob("*.parquet")
        # Parquet readers skip hidden files and folders, such as an interrupted update.
        if not any(part.startswith((".", "_")) for part in path.relative_to(traffic_dir).parts)
    )
    return {
        "traffic": {path.relative_to(traffic_dir).as_posix(): traffic.file_sha256(path) for path in partitions},
        **{name: traffic.file_sha256(path) for name, path in sources.items()},
    }


def manifest(table: pd.DataFrame, covariates: pd.DataFrame, inputs: dict, content: dict[str, str]) -> dict:
    """Version record of a build.

    ``fingerprint`` identifies the content of both outputs: teammates compare its first
    8 characters. Input hashes only help find which source differs.
    """
    return {
        "fingerprint": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            **_span(table),
            "rows_per_arc": {arc: int(rows) for arc, rows in table.groupby("arc", sort=False).size().items()},
        },
        "covariates": _span(covariates),
        "content_sha256": content,
        "inputs_sha256": inputs,
        "pandas": pd.__version__,
    }


def _span(table: pd.DataFrame) -> dict:
    return {"first_hour": f"{table['datetime'].min():%Y-%m-%d %H:%M}",
            "last_hour": f"{table['datetime'].max():%Y-%m-%d %H:%M}", "rows": len(table)}
