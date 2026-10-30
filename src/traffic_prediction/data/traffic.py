"""Hourly traffic counts (flow and occupancy rate) from Open Data Paris.

Pipeline: read raw rows -> keep configured arcs (``normalize_raw``) ->
drop duplicates and complete the hourly grid (``build_hourly_table``).
``normalize_raw`` accepts both archive files and recent API records.
"""

import hashlib
import sys
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "historical"

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


# --- Configuration -----------------------------------------------------------


def load_arcs(config_path: Path = DEFAULT_CONFIG) -> list[Arc]:
    """Read evaluated and proxy arcs from the ``traffic`` config section."""
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
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert(PARIS_TZ).dt.tz_localize(None)
    if pd.api.types.is_datetime64_dtype(values):
        return values

    text = values.astype(str).str.strip()
    has_zone = text.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True)
    if not has_zone.any():
        return pd.to_datetime(text, format="ISO8601")
    if not has_zone.all():
        raise ValueError("t_1h mixes values with and without a time zone.")

    parsed = pd.to_datetime(text, format="ISO8601", utc=True)
    return parsed.dt.tz_convert(PARIS_TZ).dt.tz_localize(None)


def parse_measure(
    values: pd.Series, upper: float | None = None
) -> tuple[pd.Series, pd.Series]:
    """Convert to float; impossible values become NaN and are flagged.

    Impossible means not a number, negative, or above ``upper``.
    """
    present = values.notna() & values.astype(str).str.strip().ne("")
    number = pd.to_numeric(values.where(present), errors="coerce")
    invalid = (present & number.isna()) | number.lt(0)
    if upper is not None:
        invalid |= number.gt(upper)
    return number.mask(invalid).astype("float64"), invalid


def parse_state(values: pd.Series, labels: dict[str, int]) -> pd.Series:
    """Return integer state codes; text labels (API) are mapped to codes."""
    codes = pd.to_numeric(values, errors="coerce")
    by_label = {label.casefold(): code for label, code in labels.items()}
    from_text = values.astype(str).str.strip().str.casefold().map(by_label)
    return codes.fillna(from_text).astype("Int8")


def measure_quality(frame: pd.DataFrame, invalid: pd.Series) -> pd.Series:
    """Label each row: invalide, ok, partielle (q or k missing) or vide."""
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
    """Archives stored as <raw_dir>/<year>/*.zip for the years of the period."""
    first, last = period_bounds(start, end)
    return sorted(
        path
        for year in range(first.year, last.year + 1)
        for path in (Path(raw_dir) / str(year)).glob("*.zip")
    )


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


def read_archive(
    path: Path, arcs: list[Arc], chunk_rows: int = CHUNK_ROWS
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield (normalized rows, raw row count) per chunk, without extracting.

    Weekly text files are read in name order, which is chronological.
    """
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist() if name.lower().endswith(".txt")
        )
        for number, member in enumerate(members, start=1):
            print(f"  [{number}/{len(members)}] {member}", flush=True)
            try:
                with archive.open(member) as file:
                    chunks = pd.read_csv(
                        file,
                        sep=";",
                        usecols=RAW_COLUMNS,
                        dtype=str,
                        encoding="utf-8",
                        chunksize=chunk_rows,
                    )
                    for chunk in chunks:
                        yield normalize_raw(chunk, arcs), len(chunk)
            except NotImplementedError as error:
                raise ValueError(
                    f"Cannot read {path} ({member}): {error}"
                ) from error


def process_archives(
    paths: Iterable[Path],
    arcs: list[Arc],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    chunk_rows: int = CHUNK_ROWS,
) -> tuple[pd.DataFrame, ProcessingStats]:
    """Turn historical ZIP archives into the complete hourly table."""
    stats = ProcessingStats()
    stats.archives_read, stats.duplicate_archives = unique_archives(paths)

    parts = []
    for path in stats.archives_read:
        print(f"Reading {path}", flush=True)
        for rows, raw_count in read_archive(path, arcs, chunk_rows):
            stats.raw_rows += raw_count
            if not rows.empty:
                parts.append(rows)

    if parts:
        rows = pd.concat(parts, ignore_index=True)
    else:
        rows = normalize_raw(pd.DataFrame(columns=RAW_COLUMNS), arcs)
    stats.arc_rows = len(rows)

    return build_hourly_table(rows, arcs, start, end, stats), stats
