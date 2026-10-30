"""Retrieve and process hourly traffic observations from Open Data Paris.

Pipeline: read raw rows -> keep configured arcs (``normalize_raw``) ->
drop duplicates and complete the hourly grid (``build_hourly_table``).
``normalize_raw`` accepts both archive files and recent API records.
Run this module to preview or download recent raw observations. Run
``traffic_prediction.data.build`` to process archives and selected JSONL
snapshots into Parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "historical"
RECORDS_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "comptages-routiers-permanents/records"
)

# Config list -> value of the output "role" column.
ARC_ROLES = {"evaluated_arcs": "evalue", "proxy_arcs": "proxy"}

# Open Data columns needed downstream; "dessin" (geometry) is never read.
RAW_COLUMNS = [
    "iu_ac",
    "libelle_nd_amont",
    "libelle_nd_aval",
    "t_1h",
    "q",
    "k",
    "etat_trafic",
    "etat_barre",
]
OUTPUT_COLUMNS = [
    "arc",
    "role",
    "iu_ac",
    "datetime",
    "debit_horaire",
    "taux_occupation",
    "etat_trafic",
    "etat_barre",
    "qualite_mesure",
]

# Codes from the Open Data notice; the API returns the labels as text.
ETAT_TRAFIC_CODES = {
    "Inconnu": 0,
    "Fluide": 1,
    "Pré-saturé": 2,
    "Saturé": 3,
    "Bloqué": 4,
}
ETAT_BARRE_CODES = {"Inconnu": 0, "Ouvert": 1, "Barré": 2, "Invalide": 3}

PARIS_TZ = "Europe/Paris"
CHUNK_ROWS = 200_000
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Arc:
    """A road segment, identified by its upstream and downstream nodes."""

    name: str
    role: str
    upstream_node: str
    downstream_node: str
    iu_ac: int


@dataclass
class ProcessingStats:
    """Counts reported at the end of processing, so nothing is dropped silently."""

    archives_read: list[Path] = field(default_factory=list)
    duplicate_archives: list[Path] = field(default_factory=list)
    raw_rows: int = 0
    arc_rows: int = 0
    outside_period: int = 0
    identical_duplicates: int = 0
    conflicting_duplicates: int = 0
    invalid_rows: int = 0
    arcs_without_measurements: list[str] = field(default_factory=list)
    recent_files_read: list[Path] = field(default_factory=list)


# --- Configuration -----------------------------------------------------------


def load_arcs(config_path: Path = DEFAULT_CONFIG) -> list[Arc]:
    """Read evaluated and proxy arcs from the ``traffic`` config section."""
    import yaml

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    section = config.get("traffic") or {}
    arcs = []
    for key, role in ARC_ROLES.items():
        for item in section.get(key) or []:
            try:
                arcs.append(
                    Arc(
                        name=str(item["name"]),
                        role=role,
                        upstream_node=str(item["upstream_node"]),
                        downstream_node=str(item["downstream_node"]),
                        iu_ac=int(item["iu_ac"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid arc in traffic.{key}: {item!r}"
                ) from error

    validate_arcs(arcs)
    return arcs


def validate_arcs(arcs: list[Arc]) -> None:
    """Require at least one evaluated arc and unique names, IDs, node pairs."""
    if not any(arc.role == "evalue" for arc in arcs):
        raise ValueError("The config must list at least one evaluated arc.")

    keys = {
        "name": [arc.name for arc in arcs],
        "iu_ac": [arc.iu_ac for arc in arcs],
        "node pair": [(arc.upstream_node, arc.downstream_node) for arc in arcs],
    }
    for label, values in keys.items():
        duplicates = {value for value in values if values.count(value) > 1}
        if duplicates:
            raise ValueError(f"Duplicate arc {label}: {sorted(duplicates)}")


# --- Raw rows -> output format (shared with recent API data) -----------------


def normalize_raw(raw: pd.DataFrame, arcs: list[Arc]) -> pd.DataFrame:
    """Keep configured arcs from Open Data columns and convert to output format.

    The result has one row per input row of a configured arc; it is not yet
    deduplicated or completed to a full hourly grid.
    """
    import pandas as pd

    rows = filter_arcs(raw, arcs)
    debit, debit_invalid = parse_measure(rows["q"])
    taux, taux_invalid = parse_measure(rows["k"], upper=100)
    # No vehicle counted while the loop is occupied: a sensor fault.
    zero_flow = debit.eq(0) & taux.gt(0)
    debit = debit.mask(zero_flow)

    normalized = pd.DataFrame(
        {
            "arc": rows["arc"],
            "role": rows["role"],
            "iu_ac": rows["expected_iu_ac"].astype("int32"),
            "datetime": parse_t_1h(rows["t_1h"]),
            "debit_horaire": debit,
            "taux_occupation": taux,
            "etat_trafic": parse_state(rows["etat_trafic"], ETAT_TRAFIC_CODES),
            "etat_barre": parse_state(rows["etat_barre"], ETAT_BARRE_CODES),
        }
    )
    normalized["qualite_mesure"] = measure_quality(
        normalized, debit_invalid | taux_invalid | zero_flow
    )
    return normalized.reset_index(drop=True)


def filter_arcs(raw: pd.DataFrame, arcs: list[Arc]) -> pd.DataFrame:
    """Keep rows whose (upstream, downstream) node pair is configured.

    Raises if a configured node pair appears with an unexpected iu_ac.
    """
    import pandas as pd

    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing Open Data columns: {sorted(missing)}")

    targets = pd.DataFrame(
        {
            "libelle_nd_amont": [arc.upstream_node for arc in arcs],
            "libelle_nd_aval": [arc.downstream_node for arc in arcs],
            "arc": [arc.name for arc in arcs],
            "role": [arc.role for arc in arcs],
            "expected_iu_ac": [arc.iu_ac for arc in arcs],
        }
    )
    # An inner merge keeps the input row order, needed to resolve duplicates.
    rows = raw[RAW_COLUMNS].merge(
        targets, on=["libelle_nd_amont", "libelle_nd_aval"], how="inner"
    )

    found = pd.to_numeric(rows["iu_ac"], errors="coerce")
    wrong = rows[found.ne(rows["expected_iu_ac"])]
    if not wrong.empty:
        first = wrong.iloc[0]
        raise ValueError(
            f"Arc {first['arc']!r}: nodes {first['libelle_nd_amont']!r} -> "
            f"{first['libelle_nd_aval']!r} have iu_ac {first['iu_ac']!r}, "
            f"expected {first['expected_iu_ac']}. Check configs/baseline.yaml."
        )
    return rows


def parse_t_1h(values: pd.Series) -> pd.Series:
    """Parse t_1h as naive Paris local time, the end of the measured hour.

    Archive values are already naive local time and are kept unchanged.
    Values with a time zone (recent API, UTC) are converted to Paris time.
    """
    import pandas as pd

    if isinstance(values.dtype, pd.DatetimeTZDtype):
        parsed = values.dt.tz_convert(PARIS_TZ).dt.tz_localize(None)
    elif pd.api.types.is_datetime64_dtype(values):
        parsed = values
    else:
        text = values.astype(str).str.strip()
        has_zone = text.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True)
        if not has_zone.any():
            parsed = pd.to_datetime(text, format="ISO8601")
        elif not has_zone.all():
            raise ValueError("t_1h mixes values with and without a time zone.")
        else:
            parsed = pd.to_datetime(text, format="ISO8601", utc=True)
            parsed = parsed.dt.tz_convert(PARIS_TZ).dt.tz_localize(None)

    if parsed.isna().any():
        raise ValueError("t_1h contains missing timestamps.")
    if parsed.ne(parsed.dt.floor("h")).any():
        raise ValueError("t_1h must contain whole-hour timestamps in Paris time.")
    return parsed


def parse_measure(
    values: pd.Series, upper: float | None = None
) -> tuple[pd.Series, pd.Series]:
    """Convert measurements to float and flag invalid supplied values.

    Invalid values are non-numeric, non-finite, negative, or above upper.
    Genuine missing values remain NaN without being marked invalid.
    """
    import numpy as np
    import pandas as pd

    present = values.notna() & values.astype(str).str.strip().ne("")
    number = pd.to_numeric(
        values.where(present), errors="coerce"
    ).astype("float64")

    invalid = (present & ~np.isfinite(number)) | number.lt(0)
    if upper is not None:
        invalid |= number.gt(upper)

    return number.mask(invalid), invalid


def parse_state(values: pd.Series, labels: dict[str, int]) -> pd.Series:
    """Map valid codes or labels to Int8; reject invalid supplied states."""
    import pandas as pd

    text = values.astype("string").str.strip()
    present = text.notna() & text.ne("")
    codes = pd.to_numeric(text, errors="coerce")
    by_label = {label.casefold(): code for label, code in labels.items()}
    from_text = text.str.casefold().map(by_label)
    parsed = codes.fillna(from_text)

    allowed = set(labels.values())
    invalid = present & ~parsed.isin(allowed)

    if invalid.any():
        examples = text.loc[invalid].drop_duplicates().head(5).tolist()
        field = values.name or "state"
        raise ValueError(
            f"Invalid state values for {field}: {examples}; "
            f"expected codes {sorted(allowed)} or documented text labels."
        )

    return parsed.astype("Int8")


def measure_quality(frame: pd.DataFrame, invalid: pd.Series) -> pd.Series:
    """Label each row: invalide, ok, partielle (q or k missing) or vide."""
    import numpy as np
    import pandas as pd

    has_debit = frame["debit_horaire"].notna()
    has_taux = frame["taux_occupation"].notna()
    labels = np.select(
        [invalid, has_debit & has_taux, has_debit | has_taux],
        ["invalide", "ok", "partielle"],
        default="vide",
    )
    return pd.Series(labels, index=frame.index, dtype=object)


# --- Normalized rows -> complete hourly table ---------------------------------


def period_bounds(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour labels of an inclusive date period (00:00 to 23:00)."""
    import pandas as pd

    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize() + pd.Timedelta(hours=23)
    if first > last:
        raise ValueError("The start date must not be after the end date.")
    return first, last


def build_hourly_table(
    rows: pd.DataFrame,
    arcs: list[Arc],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    stats: ProcessingStats | None = None,
) -> pd.DataFrame:
    """Restrict to the period, drop duplicate hours, and complete the grid."""
    stats = stats if stats is not None else ProcessingStats()
    first, last = period_bounds(start, end)

    inside = rows["datetime"].between(first, last)
    stats.outside_period += int((~inside).sum())
    rows = rows[inside]
    # Counted before deduplication: each invalid source row is reported.
    stats.invalid_rows += int(rows["qualite_mesure"].eq("invalide").sum())

    rows, identical, conflicting = drop_duplicate_hours(rows)
    stats.identical_duplicates += identical
    stats.conflicting_duplicates += conflicting

    table = complete_grid(rows, arcs, first, last)
    warn_arcs_without_measurements(table, rows, stats)
    return table


def drop_duplicate_hours(rows: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Keep one row per (arc, datetime): the last one in reading order.

    Returns the rows and the counts of identical and conflicting duplicates.
    """
    identical = rows.duplicated(keep="last")
    rows = rows[~identical]
    conflicting = rows.duplicated(["arc", "datetime"], keep="last")
    return rows[~conflicting], int(identical.sum()), int(conflicting.sum())


def complete_grid(
    rows: pd.DataFrame,
    arcs: list[Arc],
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> pd.DataFrame:
    """One row per arc and hour; hours missing from the source are "absente"."""
    import pandas as pd

    hours = pd.date_range(first, last, freq="h")
    index = pd.MultiIndex.from_product(
        [[arc.name for arc in arcs], hours], names=["arc", "datetime"]
    )
    table = rows.set_index(["arc", "datetime"]).reindex(index).reset_index()

    by_name = {arc.name: arc for arc in arcs}
    table["role"] = table["arc"].map(lambda name: by_name[name].role)
    table["iu_ac"] = (
        table["arc"].map(lambda name: by_name[name].iu_ac).astype("int32")
    )
    table["qualite_mesure"] = table["qualite_mesure"].fillna("absente")
    return table[OUTPUT_COLUMNS]


def warn_arcs_without_measurements(
    table: pd.DataFrame, rows: pd.DataFrame, stats: ProcessingStats
) -> None:
    """Warn about arcs with no flow or occupancy value over the whole period."""
    measured = table["qualite_mesure"].isin(["ok", "partielle"])
    for name, arc_measured in measured.groupby(table["arc"], sort=False):
        if arc_measured.any():
            continue
        source_rows = int((rows["arc"] == name).sum())
        detail = (
            f"{source_rows} source rows, all empty"
            if source_rows
            else "no source rows"
        )
        print(
            f"WARNING: arc {name!r} has no measurement in the period "
            f"({detail}); its hours are kept with NaN values.",
            file=sys.stderr,
            flush=True,
        )
        stats.arcs_without_measurements.append(name)


# --- Historical archives ------------------------------------------------------


def find_archives(
    raw_dir: Path, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[Path]:
    """Select requested years and a readable prior archive for January 1."""
    from traffic_prediction.data.historical import check_archive_support

    first, last = period_bounds(start, end)
    raw_dir = Path(raw_dir)

    archives = sorted(
        path
        for year in range(first.year, last.year + 1)
        for path in (raw_dir / str(year)).glob("*.zip")
        if path.is_file()
    )

    if first.month != 1 or first.day != 1:
        return archives

    previous_year = first.year - 1
    boundary_archives = []

    for path in sorted((raw_dir / str(previous_year)).glob("*.zip")):
        if not path.is_file():
            continue

        try:
            with zipfile.ZipFile(path) as archive:
                check_archive_support(archive)
        except NotImplementedError as error:
            print(
                f"WARNING: cannot read optional boundary archive {path}: {error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            boundary_archives.append(path)

    if not boundary_archives:
        print(
            f"WARNING: no readable archive for {previous_year} in {raw_dir}; "
            f"{first:%Y-%m-%d %H:%M} may be missing.",
            file=sys.stderr,
            flush=True,
        )

    return sorted(boundary_archives + archives)


def unique_archives(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Split archives into unique ones and exact copies (same SHA-256)."""
    seen: dict[str, Path] = {}
    unique, copies = [], []
    for path in sorted(paths):
        digest = file_sha256(path)
        if digest in seen:
            print(
                f"Skipping {path}: identical to {seen[digest]}", flush=True
            )
            copies.append(path)
        else:
            seen[digest] = path
            unique.append(path)
    return unique, copies


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_chunk_rows(chunk_rows: int) -> None:
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunk_rows must be a positive integer.")


def read_archive(
    path: Path, arcs: list[Arc], chunk_rows: int = CHUNK_ROWS
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield (normalized rows, raw row count) per chunk, without extracting.

    Ordinary ZIP members use filename order; Deflate64 uses archive order.
    """
    import pandas as pd
    from traffic_prediction.data.historical import iter_zip_members

    _validate_chunk_rows(chunk_rows)
    try:
        for member, file in iter_zip_members(path):
            if member.is_dir() or not member.filename.lower().endswith(".txt"):
                continue
            print(f"  {member.filename}", flush=True)
            chunks = pd.read_csv(
                file, sep=";", usecols=RAW_COLUMNS, dtype=str,
                encoding="utf-8", chunksize=chunk_rows,
            )
            for chunk in chunks:
                yield normalize_raw(chunk, arcs), len(chunk)
    except NotImplementedError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def read_recent(
    path: Path, arcs: list[Arc], chunk_rows: int = CHUNK_ROWS
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield (normalized rows, raw count) from a completed API JSONL file."""
    import pandas as pd

    _validate_chunk_rows(chunk_rows)

    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Expected a completed .jsonl file, not a partial download.")

    records = []
    required_columns = set(RAW_COLUMNS)

    with path.open("rb") as source:
        line_number = 0
        while line := source.readline(MAX_JSONL_LINE_BYTES + 1):
            line_number += 1
            context = f"{path}, line {line_number}"

            if len(line) > MAX_JSONL_LINE_BYTES:
                raise ValueError(f"{context}: JSONL record exceeds 4 MiB.")
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{context}: invalid JSON.") from error

            if not isinstance(record, dict):
                raise ValueError(f"{context}: expected a JSON object.")

            missing = required_columns - record.keys()
            if missing:
                raise ValueError(f"{context}: missing columns {sorted(missing)}.")

            try:
                parse_timestamp(record["t_1h"])
            except (argparse.ArgumentTypeError, AttributeError, TypeError) as error:
                raise ValueError(
                    f"{context}: t_1h must include an explicit timezone."
                ) from error

            records.append(tuple(record[column] for column in RAW_COLUMNS))

            if len(records) == chunk_rows:
                rows = normalize_raw(
                    pd.DataFrame.from_records(records, columns=RAW_COLUMNS), arcs
                )
                count = len(records)
                records.clear()
                yield rows, count

        if records:
            yield normalize_raw(
                pd.DataFrame.from_records(records, columns=RAW_COLUMNS), arcs
            ), len(records)


def process_sources(
    paths: Iterable[Path],
    arcs: list[Arc],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    chunk_rows: int = CHUNK_ROWS,
    *,
    recent_paths: Iterable[Path] = (),
) -> tuple[pd.DataFrame, ProcessingStats]:
    """Build one hourly table from archives followed by ordered API snapshots.

    The last supplied row wins for each arc/hour, including null values.
    Raw files remain unchanged; duplicates are counted in ProcessingStats.
    """
    import pandas as pd

    _validate_chunk_rows(chunk_rows)
    first, last = period_bounds(start, end)
    stats = ProcessingStats()
    recent_paths = [Path(path) for path in recent_paths]
    for path in recent_paths:
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a completed .jsonl file: {path}")
        if not path.is_file():
            raise ValueError(f"Recent file does not exist: {path}")

    stats.archives_read, stats.duplicate_archives = unique_archives(paths)
    sources = [
        (path, read_archive) for path in stats.archives_read
    ] + [
        (path, read_recent) for path in recent_paths
    ]

    parts = []
    for path, reader in sources:
        print(f"Reading {path}", flush=True)
        for rows, raw_count in reader(path, arcs, chunk_rows):
            stats.raw_rows += raw_count
            stats.arc_rows += len(rows)
            inside = rows["datetime"].between(first, last)
            stats.outside_period += int((~inside).sum())
            if inside.any():
                parts.append(rows.loc[inside].copy())
        if reader is read_recent:
            stats.recent_files_read.append(path)

    if parts:
        rows = pd.concat(parts, ignore_index=True)
    else:
        rows = normalize_raw(pd.DataFrame(columns=RAW_COLUMNS), arcs)

    return build_hourly_table(rows, arcs, start, end, stats), stats


def process_archives(
    paths: Iterable[Path],
    arcs: list[Arc],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    chunk_rows: int = CHUNK_ROWS,
) -> tuple[pd.DataFrame, ProcessingStats]:
    """Build the historical hourly table through the shared source pipeline."""
    return process_sources(paths, arcs, start, end, chunk_rows)


# --- Recent API ingestion -----------------------------------------------------


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp with an explicit timezone and convert to UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("Timezone is missing.")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            "Use an ISO timestamp with Z or an offset, "
            "for example 2026-10-29T00:00:00Z."
        ) from error


def build_date_filter(
    start: datetime | None, end: datetime | None
) -> str | None:
    """Filter observation timestamps using inclusive start and exclusive end."""
    if start is None and end is None:
        return None

    if start is None or end is None:
        raise ValueError("Provide both --start and --end.")

    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("Date boundaries must include a timezone.")

    if start >= end:
        raise ValueError("--start must be earlier than --end.")

    lower = start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    upper = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return f"t_1h >= date'{lower}' AND t_1h < date'{upper}'"


def build_observation_filter(
    start: datetime | None, end: datetime | None, road_id: int | None = None
) -> str | None:
    """Combine date bounds with an optional single-arc API filter."""
    date_filter = build_date_filter(start, end)
    if road_id is None:
        return date_filter
    if isinstance(road_id, bool) or not isinstance(road_id, int) or road_id < 1:
        raise ValueError("road_id must be a positive integer.")
    road_filter = f"iu_ac = '{road_id}'"
    return f"({date_filter}) AND {road_filter}" if date_filter else road_filter


def fetch_sample(
    limit: int = 3,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    road_id: int | None = None,
) -> dict:
    """Preview recent raw observations without writing files."""
    if not 1 <= limit <= 10:
        raise ValueError("Sample limit must be between 1 and 10.")

    parameters = {"limit": limit, "order_by": "t_1h DESC"}

    date_filter = build_observation_filter(start, end, road_id)
    if date_filter is not None:
        parameters["where"] = date_filter

    query = urlencode(parameters)
    request = Request(
        f"{RECORDS_URL}?{query}",
        headers={
            "User-Agent": "traffic-prediction/0.1",
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(
                f"Unexpected HTTP status: {response.status}"
            )
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("API response must be an object.")

    total = payload.get("total_count")
    records = payload.get("results")

    if (
        type(total) is not int
        or total < 0
        or not isinstance(records, list)
    ):
        raise ValueError(
            "API response has invalid total_count or results."
        )

    if len(records) != min(limit, total):
        raise ValueError("API returned an incomplete sample.")

    if any(not isinstance(record, dict) for record in records):
        raise ValueError("API returned an invalid observation.")

    return payload


def validate_observation(
    record: object, start: datetime, end: datetime, *, road_id: int | None = None
) -> None:
    """Validate record structure and requested bounds without changing values."""
    required = {"iu_ac", "t_1h", "q", "k"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("Export contains an invalid traffic observation.")

    if not isinstance(record["t_1h"], str):
        raise ValueError("Export contains an invalid timestamp.")

    try:
        timestamp = parse_timestamp(record["t_1h"])
    except argparse.ArgumentTypeError as error:
        raise ValueError("Export contains an invalid timestamp.") from error

    if not start <= timestamp < end:
        raise ValueError(
            "Export contains an observation outside the requested period."
        )
    if road_id is not None and str(record["iu_ac"]) != str(road_id):
        raise ValueError("Export contains an observation outside the requested road.")


def download_observations(
    start: datetime,
    end: datetime,
    output_dir: Path,
    *,
    attempts: int = 3,
    road_id: int | None = None,
) -> Path:
    """Save a complete raw JSONL snapshot, retrying temporary failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    date_filter = build_date_filter(start, end)
    if date_filter is None:
        raise ValueError("Downloads require start and end timestamps.")
    date_filter = build_observation_filter(start, end, road_id)

    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"traffic_{start:%Y%m%dT%H%M%S%fZ}_"
        f"{end:%Y%m%dT%H%M%S%fZ}_"
    )

    parameters = {
        "where": date_filter,
        "limit": -1,
        "timezone": "UTC",
    }
    export_url = RECORDS_URL.rsplit("/", 1)[0] + "/exports/jsonl"
    request = Request(
        f"{export_url}?{urlencode(parameters)}",
        headers={
            "User-Agent": "traffic-prediction/0.1",
            "Accept-Encoding": "identity",
        },
    )

    for attempt in range(1, attempts + 1):
        temporary_path = None

        try:
            expected = fetch_sample(
                1, start=start, end=end, road_id=road_id
            )["total_count"]

            print(
                f"Exporting {expected} observations "
                f"(attempt {attempt}/{attempts})...",
                flush=True,
            )

            with urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(
                        f"Unexpected HTTP status: {response.status}"
                    )

                length = response.headers.get("Content-Length")
                expected_bytes = int(length) if length is not None else None
                received_bytes = 0
                count = 0

                with NamedTemporaryFile(
                    mode="wb",
                    dir=output_dir,
                    prefix=prefix,
                    suffix=".jsonl.part",
                    delete=False,
                ) as output:
                    temporary_path = Path(output.name)
                    while line := response.readline(MAX_JSONL_LINE_BYTES + 1):
                        if len(line) > MAX_JSONL_LINE_BYTES:
                            raise ValueError(
                                "A JSONL record exceeds the 4 MiB limit."
                            )

                        has_record = bool(line.strip())
                        if has_record:
                            record = json.loads(line)
                            validate_observation(record, start, end, road_id=road_id)
                            count += 1

                        output.write(line)
                        received_bytes += len(line)

                        if has_record and count % 10000 == 0:
                            print(
                                f"Received {count} observations...",
                                flush=True,
                            )

                if (
                    expected_bytes is not None
                    and received_bytes != expected_bytes
                ):
                    raise ValueError(
                        "Export byte count does not match Content-Length."
                    )

            if count != expected:
                raise ValueError(
                    f"Incomplete export or changed source: "
                    f"expected {expected} observations, received {count}."
                )

            latest_count = fetch_sample(
                1, start=start, end=end, road_id=road_id
            )["total_count"]

            if latest_count != expected:
                raise ValueError(
                    "Source count changed during export; retry required."
                )

            destination = temporary_path.with_suffix("")
            temporary_path.replace(destination)

            print(
                f"Saved {count} observations: {destination}",
                flush=True,
            )
            return destination

        except (OSError, ValueError, HTTPException) as error:
            permanent_http_error = (
                isinstance(error, HTTPError)
                and 400 <= error.code < 500
                and error.code not in (408, 429)
            )

            if attempt == attempts or permanent_http_error:
                raise

            print(
                f"Extraction failed: {error}; retrying...",
                file=sys.stderr,
            )
            time.sleep(2 ** (attempt - 1))

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview or download recent raw traffic observations."
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--sample",
        action="store_true",
        help="Preview observations.",
    )
    actions.add_argument(
        "--download",
        action="store_true",
        help="Save a full raw export.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Preview size only.",
    )
    parser.add_argument(
        "--start",
        type=parse_timestamp,
        help="Inclusive timestamp.",
    )
    parser.add_argument(
        "--end",
        type=parse_timestamp,
        help="Exclusive timestamp.",
    )
    parser.add_argument(
        "--road-id",
        type=int,
        help="Download or preview one arc by iu_ac, as recommended in the brief.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "raw"
            / "recent"
        ),
    )
    args = parser.parse_args()

    if args.download and (
        args.start is None or args.end is None
    ):
        parser.error("--download requires both --start and --end.")

    try:
        if args.sample:
            payload = fetch_sample(
                args.limit,
                start=args.start,
                end=args.end,
                road_id=args.road_id,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            download_observations(
                args.start,
                args.end,
                args.output_dir,
                road_id=args.road_id,
            )

    except (OSError, ValueError, HTTPException) as error:
        parser.exit(1, f"Traffic ingestion failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Traffic ingestion interrupted; rerun to retry.\n",
        )


if __name__ == "__main__":
    main()
