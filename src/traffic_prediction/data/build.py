"""Assemble all data sources into one hourly dataset per street.

One ``build_<source>`` function per source; traffic is the first one.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_prediction.data import traffic
from traffic_prediction.data.traffic import Arc, ProcessingStats

DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_TRAFFIC_DIR = traffic.REPO_ROOT / "data" / "processed" / "traffic"

TRAFFIC_SCHEMA = pa.schema(
    [
        ("arc", pa.string()),
        ("role", pa.string()),
        ("iu_ac", pa.int32()),
        ("datetime", pa.timestamp("ns")),
        ("debit_horaire", pa.float64()),
        ("taux_occupation", pa.float64()),
        ("etat_trafic", pa.int8()),
        ("etat_barre", pa.int8()),
        ("qualite_mesure", pa.string()),
    ]
)
QUALITY_LABELS = ["ok", "partielle", "vide", "invalide", "absente"]


def build_traffic(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    raw_dir: Path = traffic.DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_TRAFFIC_DIR,
    config_path: Path = traffic.DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, ProcessingStats]:
    """Process historical archives and write the traffic dataset to Parquet."""
    arcs = traffic.load_arcs(config_path)
    roles = pd.Series([arc.role for arc in arcs]).value_counts().to_dict()
    print(f"Traffic {start} to {end}: {len(arcs)} arcs {roles}", flush=True)

    archives = traffic.find_archives(raw_dir, start, end)
    if not archives:
        raise ValueError(f"No archives in {raw_dir} for {start} to {end}.")
    warn_missing_years(archives, raw_dir, start, end)

    table, stats = traffic.process_archives(archives, arcs, start, end)
    validate_traffic(table, arcs, start, end)
    write_partitions(table, output_dir)
    return table, stats


def warn_missing_years(
    archives: list[Path], raw_dir: Path, start: str, end: str
) -> None:
    """Warn about years of the period without any archive."""
    first, last = traffic.period_bounds(start, end)
    found = {int(path.parent.name) for path in archives}
    for year in range(first.year, last.year + 1):
        if year not in found:
            print(
                f"WARNING: no archive for {year} in {raw_dir}; "
                'its hours will be "absente".',
                file=sys.stderr,
                flush=True,
            )


def validate_traffic(
    table: pd.DataFrame, arcs: list[Arc], start: str, end: str
) -> None:
    """Check columns, one row per arc and hour, and value ranges."""
    if list(table.columns) != traffic.OUTPUT_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(table.columns)}")

    first, last = traffic.period_bounds(start, end)
    expected_rows = len(arcs) * len(pd.date_range(first, last, freq="h"))
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(table)}.")

    measured = table["debit_horaire"].notna() | table["taux_occupation"].notna()
    taux = table["taux_occupation"]
    problems = {
        "duplicate (arc, datetime)": table.duplicated(["arc", "datetime"]),
        "unknown arc": ~table["arc"].isin([arc.name for arc in arcs]),
        "unknown qualite_mesure": ~table["qualite_mesure"].isin(QUALITY_LABELS),
        "negative debit_horaire": table["debit_horaire"].lt(0),
        "taux_occupation outside 0-100": taux.notna() & ~taux.between(0, 100),
        "etat_trafic outside 0-4": table["etat_trafic"].gt(4).fillna(False),
        "etat_barre outside 0-3": table["etat_barre"].gt(3).fillna(False),
        "measured hour marked absente": measured
        & table["qualite_mesure"].eq("absente"),
    }
    for label, mask in problems.items():
        if mask.any():
            raise ValueError(f"Invalid traffic table, {label}: {mask.sum()} rows.")


def write_partitions(table: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one zstd-compressed Parquet file per year: year=YYYY/part-0.parquet.

    Years in the table are replaced; other years already on disk are kept.
    """
    paths = []
    for year, part in table.groupby(table["datetime"].dt.year):
        folder = Path(output_dir) / f"year={year}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "part-0.parquet"
        # Dot prefix: Parquet readers ignore the file if a write is interrupted.
        temporary = folder / ".part-0.parquet.tmp"

        arrow = pa.Table.from_pandas(
            part, schema=TRAFFIC_SCHEMA, preserve_index=False
        )
        pq.write_table(arrow, temporary, compression="zstd")
        temporary.replace(path)
        paths.append(path)
        print(f"Saved {path} ({len(part):,} rows)", flush=True)

    check_written(paths, len(table))
    return paths


def check_written(paths: list[Path], expected_rows: int) -> None:
    """Re-read file metadata: total row count and schema must match."""
    rows = 0
    for path in paths:
        if not pq.read_schema(path).equals(TRAFFIC_SCHEMA):
            raise ValueError(f"Unexpected Parquet schema in {path}.")
        rows += pq.read_metadata(path).num_rows
    if rows != expected_rows:
        raise ValueError(f"Wrote {rows} rows, expected {expected_rows}.")


# --- Summary ------------------------------------------------------------------


def print_traffic_summary(table: pd.DataFrame, stats: ProcessingStats) -> None:
    """Print what was read, dropped or added, and missing values per arc."""
    print("\n=== Traffic summary ===")
    print(f"Archives read: {[str(path) for path in stats.archives_read]}")
    print(
        "Identical archives skipped: "
        f"{[str(path) for path in stats.duplicate_archives]}"
    )
    counts = {
        "Raw rows read (all sensors)": stats.raw_rows,
        "Rows of configured arcs": stats.arc_rows,
        "Rows outside the period": stats.outside_period,
        "Identical duplicate rows dropped": stats.identical_duplicates,
        "Conflicting duplicate rows dropped (last kept)": (
            stats.conflicting_duplicates
        ),
        "Rows with an invalid value (set to NaN)": stats.invalid_rows,
    }
    for label, value in counts.items():
        print(f"{label}: {value:,}")
    print(
        "Arcs without any measurement: "
        f"{stats.arcs_without_measurements or 'none'}"
    )
    print(
        f"Period: {table['datetime'].min()} to {table['datetime'].max()}, "
        f"{len(table):,} rows"
    )

    print("\nPer arc (missing values in %):")
    print(arc_overview(table).to_string())
    print("\nRows per qualite_mesure:")
    quality = pd.crosstab(table["arc"], table["qualite_mesure"]).reindex(
        index=table["arc"].unique(), columns=QUALITY_LABELS, fill_value=0
    )
    print(quality.to_string())

    months = table["datetime"].dt.strftime("%Y-%m").rename("month")
    for column in ["debit_horaire", "taux_occupation"]:
        print(f"\nMissing {column} per month (%):")
        print(missing_share(table, column, months).to_string())


def arc_overview(table: pd.DataFrame) -> pd.DataFrame:
    """Rows, first and last measured hour, and missing shares per arc."""
    measured = table[table["qualite_mesure"].isin(["ok", "partielle"])]
    groups = table.groupby("arc", sort=False)
    overview = pd.DataFrame(
        {
            "role": groups["role"].first(),
            "iu_ac": groups["iu_ac"].first(),
            "rows": groups.size(),
            "first_measure": measured.groupby("arc")["datetime"].min(),
            "last_measure": measured.groupby("arc")["datetime"].max(),
            "debit_horaire": groups["debit_horaire"].agg(percent_missing),
            "taux_occupation": groups["taux_occupation"].agg(percent_missing),
        }
    )
    # Keep the configuration order (evaluated arcs first).
    return overview.reindex(table["arc"].unique())


def missing_share(
    table: pd.DataFrame, column: str, by: pd.Series
) -> pd.DataFrame:
    """Percent of missing values of ``column``: one row per ``by``, one column per arc."""
    shares = table.groupby([by, table["arc"]], sort=False)[column].agg(
        percent_missing
    )
    arcs = table["arc"].unique()
    return shares.unstack("arc").reindex(columns=arcs).sort_index()


def percent_missing(values: pd.Series) -> float:
    """Share of NaN values, in percent, rounded to one decimal."""
    return round(100 * values.isna().mean(), 1)


# --- Command line ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the hourly traffic dataset from historical archives."
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        metavar="YYYY-MM-DD",
        help=f"First day, from 00:00 (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        metavar="YYYY-MM-DD",
        help=f"Last day, until 23:00 (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=traffic.DEFAULT_RAW_DIR,
        help="Archive folder with <year>/*.zip; default: data/raw/historical.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAFFIC_DIR,
        help="Parquet destination; default: data/processed/traffic.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=traffic.DEFAULT_CONFIG,
        help="Arc configuration; default: configs/baseline.yaml.",
    )
    args = parser.parse_args()

    try:
        table, stats = build_traffic(
            args.start, args.end, args.raw_dir, args.output_dir, args.config
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Traffic processing failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Traffic processing interrupted; rerun to retry.\n")

    print_traffic_summary(table, stats)


if __name__ == "__main__":
    main()
