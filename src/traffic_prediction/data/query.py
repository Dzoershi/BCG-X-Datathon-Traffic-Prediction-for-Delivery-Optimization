"""Query processed local traffic Parquet files with an in-memory DuckDB view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from traffic_prediction.data.traffic import OUTPUT_COLUMNS, REPO_ROOT

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_TRAFFIC_DIR = REPO_ROOT / "data" / "processed" / "traffic"


def connect_traffic(
    path: str | Path = DEFAULT_TRAFFIC_DIR,
) -> duckdb.DuckDBPyConnection:
    """Open a connection with a lazy ``traffic`` view over local Parquet files.

    Accept a single .parquet file or a directory searched recursively. Hive
    folders such as year=2025/month=01 become filterable SQL columns. Files
    are registered without materializing the dataset as a table or DataFrame.
    Use the returned connection as a context manager, or close it explicitly.
    Reconnect after adding or removing files to refresh the file list.
    """
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file or directory: {path}")
        files = [path]
    elif path.is_dir():
        files = sorted(file for file in path.rglob("*.parquet") if file.is_file())
    else:
        raise FileNotFoundError(f"Processed traffic data not found: {path}")

    if not files:
        raise FileNotFoundError(
            f"No Parquet files in {path}; build the processed traffic dataset first."
        )

    connection = duckdb.connect(":memory:")
    try:
        connection.read_parquet(
            [str(file) for file in files],
            hive_partitioning=True,
            union_by_name=False,
        ).create_view("traffic")
        columns = {row[0] for row in connection.execute("DESCRIBE traffic").fetchall()}
        missing = set(OUTPUT_COLUMNS) - columns
        if missing:
            raise ValueError(f"Missing processed traffic columns: {sorted(missing)}")
    except Exception:
        connection.close()
        raise

    return connection


def query_traffic(
    sql: str,
    parameters: Sequence[object] | Mapping[str, object] | None = None,
    *,
    path: str | Path = DEFAULT_TRAFFIC_DIR,
) -> pd.DataFrame:
    """Run SQL against ``traffic`` and return only its result as a DataFrame.

    Parameters can be positional (``?``) or named (``$name``). Date filters
    compare the stored timezone-free Paris timestamps; use explicit >= and <
    boundaries. The connection closes after fetching, including on failure.
    For several queries or joins to registered DataFrames, use connect_traffic.
    """
    with connect_traffic(path) as connection:
        return connection.execute(sql, parameters).fetchdf()
