"""Offline SQL integration tests using the actual processed Parquet schema."""

from datetime import datetime

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from traffic_prediction.data import query
from traffic_prediction.data.build import TRAFFIC_SCHEMA
from traffic_prediction.data.traffic import OUTPUT_COLUMNS


def observation(timestamp, flow, *, road=5671, arc="Convention", role="evalue"):
    return {
        "arc": arc,
        "role": role,
        "iu_ac": road,
        "datetime": datetime.fromisoformat(timestamp),
        "debit_horaire": flow,
        "taux_occupation": 3.0 if flow is not None else None,
        "etat_trafic": 1,
        "etat_barre": 3,
        "qualite_mesure": "ok" if flow is not None else "vide",
    }


@pytest.fixture
def parquet_dir(tmp_path):
    # Apostrophes and spaces exercise path handling without SQL interpolation.
    root = tmp_path / "team's traffic"
    partitions = {
        (2024, 12): [
            observation("2024-12-31 23:00", 100),
            observation("2024-12-31 23:00", None, road=4264, arc="Champs-Elysées"),
        ],
        (2025, 1): [
            observation("2025-01-01 00:00", 200),
            observation("2025-01-01 00:00", 900, road=42, arc="Proxy", role="proxy"),
        ],
        (2025, 2): [observation("2025-02-01 00:00", 300)],
    }
    for (year, month), rows in partitions.items():
        folder = root / f"year={year}" / f"month={month:02d}"
        folder.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=TRAFFIC_SCHEMA),
            folder / "part-0.parquet", compression="zstd",
        )
    return root


def test_registers_lazy_view_across_years_and_months(parquet_dir):
    with query.connect_traffic(parquet_dir) as connection:
        assert connection.execute("SELECT count(*) FROM traffic").fetchone() == (5,)
        assert connection.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name='traffic'"
        ).fetchone() == ("VIEW",)
        assert set(OUTPUT_COLUMNS).issubset(
            row[0] for row in connection.execute("DESCRIBE traffic").fetchall()
        )
        assert connection.execute(
            """SELECT DISTINCT year, CAST(month AS INTEGER) AS month
               FROM traffic ORDER BY year, month"""
        ).fetchall() == [(2024, 12), (2025, 1), (2025, 2)]

    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_parameterized_dates_roads_attributes_and_projection(parquet_dir):
    result = query.query_traffic(
        """SELECT datetime, debit_horaire FROM traffic
           WHERE datetime >= ? AND datetime < ? AND iu_ac IN (?, ?)
             AND role = ? AND qualite_mesure = ?
           ORDER BY datetime""",
        [datetime(2025, 1, 1), datetime(2025, 2, 1), 5671, 4264, "evalue", "ok"],
        path=parquet_dir,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["datetime", "debit_horaire"]
    assert result["debit_horaire"].tolist() == [200.0]
    assert result["datetime"].tolist() == [pd.Timestamp("2025-01-01")]


def test_aggregation_preserves_missing_values_and_separates_proxies(parquet_dir):
    result = query.query_traffic(
        """SELECT iu_ac, avg(debit_horaire) AS mean_flow,
                  count(debit_horaire) AS measured_hours
           FROM traffic WHERE role = 'evalue' GROUP BY iu_ac ORDER BY iu_ac""",
        path=parquet_dir,
    ).set_index("iu_ac")
    assert result.index.tolist() == [4264, 5671]
    assert pd.isna(result.loc[4264, "mean_flow"])
    assert result.loc[4264, "measured_hours"] == 0
    assert result.loc[5671, "mean_flow"] == 200


def test_join_with_registered_notebook_dataframe(parquet_dir):
    with query.connect_traffic(parquet_dir) as connection:
        connection.register("roads", pd.DataFrame({"iu_ac": [5671], "zone": ["west"]}))
        result = connection.execute(
            """SELECT zone, sum(debit_horaire) AS total
               FROM traffic JOIN roads USING (iu_ac) GROUP BY zone"""
        ).fetchdf()
    assert result.to_dict("records") == [{"zone": "west", "total": 600.0}]


def test_single_file_and_named_parameters(parquet_dir):
    path = parquet_dir / "year=2024" / "month=12" / "part-0.parquet"
    result = query.query_traffic(
        "SELECT debit_horaire FROM traffic WHERE iu_ac = $road",
        {"road": 5671}, path=path,
    )
    assert result["debit_horaire"].tolist() == [100.0]


def test_queries_the_existing_build_output(tmp_path):
    from traffic_prediction.data import build, traffic
    from test_traffic_processing import CONVENTION, convention_row, raw_rows

    table = traffic.build_hourly_table(
        traffic.normalize_raw(
            raw_rows(convention_row("2025-01-01 01:00:00")), [CONVENTION]
        ),
        [CONVENTION], "2025-01-01", "2025-01-01",
    )
    build.validate_traffic(table, [CONVENTION], "2025-01-01", "2025-01-01")
    build.write_partitions(table, tmp_path / "processed")

    result = query.query_traffic(
        """SELECT year, count(*) AS hours, count(debit_horaire) AS measured
           FROM traffic GROUP BY year""",
        path=tmp_path / "processed",
    )
    assert result.to_dict("records") == [{"year": 2025, "hours": 24, "measured": 1}]


def test_scan_keeps_projection_and_partition_filters(parquet_dir):
    with query.connect_traffic(parquet_dir) as connection:
        plan = connection.execute(
            """EXPLAIN SELECT debit_horaire FROM traffic
               WHERE year = 2025 AND iu_ac = 5671"""
        ).fetchone()[1]
    assert "PARQUET_SCAN" in plan or "READ_PARQUET" in plan
    assert "Projections:" in plan
    assert "File Filters:" in plan


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_missing_data_has_clear_error(tmp_path, kind):
    path = tmp_path / kind
    if kind == "empty":
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="not found|No Parquet"):
        query.connect_traffic(path)


def test_rejects_non_parquet_file(tmp_path):
    path = tmp_path / "traffic.csv"
    path.write_text("iu_ac,q\n5671,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a .parquet"):
        query.connect_traffic(path)


def test_rejects_incompatible_processed_schema(tmp_path):
    path = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"iu_ac": [5671]}), path)
    with pytest.raises(ValueError, match="Missing processed traffic columns"):
        query.connect_traffic(path)


def test_empty_parquet_remains_queryable(tmp_path):
    path = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=TRAFFIC_SCHEMA), path)
    result = query.query_traffic("SELECT * FROM traffic", path=path)
    assert result.empty
    assert set(OUTPUT_COLUMNS).issubset(result.columns)


def test_query_failure_closes_connection(parquet_dir, monkeypatch):
    connections = []
    original = query.duckdb.connect

    def record_connection(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(query.duckdb, "connect", record_connection)
    with pytest.raises(duckdb.ParserException):
        query.query_traffic("SELECT FROM", path=parquet_dir)
    with pytest.raises(duckdb.ConnectionException):
        connections[0].execute("SELECT 1")
