"""Dataset integrity, reproducibility, submission contracts, and full data flow."""

import io
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from traffic_prediction.data import build, historical, query, traffic, update, validate
from test_traffic_processing import (
    ARCS, CONVENTION, archive_line, convention_row, raw_rows, recent_record,
    write_archive, write_config,
)


def save_table(table, root, filename="part-0.parquet"):
    path = root / "year=2025" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(table, schema=build.TRAFFIC_SCHEMA, preserve_index=False),
        path, compression="zstd",
    )
    return path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml")
    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw_rows(convention_row("2025-01-01 01:00:00")), ARCS),
        ARCS, "2025-01-01", "2025-01-01",
    )
    path = save_table(table, root)
    return SimpleNamespace(root=root, config=config, table=table, path=path)


def inspect(state, **kwargs):
    return validate.validate_dataset(state.root, config_path=state.config, **kwargs)


def test_coverage_distinguishes_grid_source_rows_and_missing_measurements(dataset):
    report = inspect(dataset)
    assert report["coverage"]["validated_rows"] == 48
    assert report["coverage"]["expected_rows"] == 48
    assert report["coverage"]["stored_last_hour"] == "2025-01-01T23:00:00"
    assert report["coverage"]["stored_last_source_hour"] == "2025-01-01T01:00:00"
    summary = {arc["arc"]: arc for arc in report["quality"]}
    assert summary["Convention"]["measured_q"] == 1
    assert summary["Convention"]["source_rows"] == 1
    assert summary["Proxy"]["missing_q"] == 24
    assert summary["Proxy"]["quality_counts"]["absente"] == 24
    assert report["files"][0]["path"] == "year=2025/part-0.parquet"


def test_manifest_survives_copying_data_and_config_comments(dataset, tmp_path):
    report = inspect(dataset)
    path = validate.write_manifest(report, dataset.root / "_validation.json")
    copy = tmp_path / "another computer"
    shutil.copytree(dataset.root, copy)
    dataset.config.write_text(dataset.config.read_text(encoding="utf-8") + "\n# local note\n", encoding="utf-8")
    other = validate.validate_dataset(copy, config_path=dataset.config)
    validate.check_manifest(other, path)
    assert report["fingerprint"] == other["fingerprint"]
    assert all(not Path(entry["path"]).is_absolute() for entry in other["files"])


def test_saved_manifest_detects_changed_but_still_valid_values(dataset):
    path = validate.write_manifest(inspect(dataset), dataset.root / "_validation.json")
    dataset.table.loc[1, "debit_horaire"] = 125.0
    save_table(dataset.table, dataset.root)
    with pytest.raises(ValueError, match="differ from the saved manifest"):
        validate.check_manifest(inspect(dataset), path)


@pytest.mark.parametrize("mutation", [
    "duplicate", "missing_row", "bad_role", "bad_hour", "nonfinite",
    "quality_ok", "quality_partial", "quality_empty", "quality_invalid", "absent_state",
])
def test_dataset_constraint_errors_are_rejected(dataset, mutation):
    table = dataset.table.copy()
    if mutation == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_row":
        table = table.iloc[1:]
    elif mutation == "bad_role":
        table.loc[0, "role"] = "proxy"
    elif mutation == "bad_hour":
        table.loc[0, "datetime"] += pd.Timedelta(minutes=30)
    elif mutation == "nonfinite":
        table.loc[1, "debit_horaire"] = float("inf")
    elif mutation == "quality_ok":
        table.loc[0, "qualite_mesure"] = "ok"
    elif mutation == "quality_partial":
        table.loc[1, "qualite_mesure"] = "partielle"
    elif mutation == "quality_empty":
        table.loc[1, "qualite_mesure"] = "vide"
    elif mutation == "quality_invalid":
        table.loc[1, "qualite_mesure"] = "invalide"
    else:
        table.loc[0, "etat_trafic"] = 1
    save_table(table, dataset.root)
    with pytest.raises(ValueError, match="Expected|Invalid traffic table"):
        inspect(dataset)


def test_missing_fields_and_corrupted_files_are_rejected(dataset):
    pq.write_table(pa.table({"iu_ac": [5671]}), dataset.path)
    with pytest.raises(ValueError, match="Unexpected processed traffic schema"):
        inspect(dataset)
    dataset.path.write_bytes(b"truncated parquet")
    with pytest.raises((ValueError, OSError)):
        inspect(dataset)


def test_gap_between_partitions_is_not_treated_as_complete(dataset):
    later = dataset.table.copy()
    later["datetime"] += pd.Timedelta(days=2)
    save_table(later, dataset.root, "part-extra.parquet")
    with pytest.raises(ValueError, match="Expected 144 rows"):
        inspect(dataset)
    report = inspect(dataset, start="2025-01-01", end="2025-01-01")
    assert report["coverage"]["stored_rows"] == 96
    assert report["coverage"]["validated_rows"] == 48


@pytest.mark.parametrize("partition", ["year", "month"])
def test_partition_labels_must_match_observation_timestamps(dataset, partition):
    if partition == "year":
        folder = dataset.root / "year=2024"
    else:
        folder = dataset.root / "year=2025" / "month=02"
    folder.mkdir(parents=True, exist_ok=True)
    dataset.path.replace(folder / "part-0.parquet")
    with pytest.raises(ValueError, match=f"{partition} partition disagrees"):
        inspect(dataset)


def test_all_missing_targets_are_reported_without_imputation(dataset):
    table = dataset.table.copy()
    table[["debit_horaire", "taux_occupation"]] = float("nan")
    table.loc[1, "qualite_mesure"] = "vide"
    save_table(table, dataset.root)
    report = inspect(dataset)
    assert report["quality"][0]["source_rows"] == 1
    assert report["quality"][0]["measured_q"] == 0
    assert report["quality"][0]["measured_k"] == 0


def test_active_update_lock_prevents_validation(dataset):
    (dataset.root / ".traffic-update.lock").write_text("another writer", encoding="utf-8")
    with pytest.raises(ValueError, match="update lock"):
        inspect(dataset)


def test_changed_files_during_validation_are_detected(dataset, monkeypatch):
    original = build.validate_traffic

    def validate_then_change(*args):
        original(*args)
        table = dataset.table.copy()
        table.loc[1, "debit_horaire"] = 125.0
        save_table(table, dataset.root)

    monkeypatch.setattr(build, "validate_traffic", validate_then_change)
    with pytest.raises(ValueError, match="changed during validation"):
        inspect(dataset)


def test_failed_manifest_publication_preserves_previous_report(dataset, monkeypatch):
    report = inspect(dataset)
    path = validate.write_manifest(report, dataset.root / "_validation.json")
    before = path.read_bytes()

    def fail(*args):
        raise OSError("interrupted metadata publication")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        validate.write_manifest(report, path)
    assert path.read_bytes() == before
    assert not list(dataset.root.glob("*.tmp"))


def test_cli_saves_checks_and_preserves_manifest_after_invalid_data(dataset, monkeypatch, capsys):
    args = ["validate", "--data-dir", str(dataset.root), "--config", str(dataset.config)]
    monkeypatch.setattr(sys, "argv", args + ["--write-manifest"])
    validate.main()
    manifest = dataset.root / "_validation.json"
    before = manifest.read_bytes()
    monkeypatch.setattr(sys, "argv", args + ["--check-manifest"])
    validate.main()
    assert "Saved manifest verified" in capsys.readouterr().out
    save_table(dataset.table.iloc[1:], dataset.root)
    monkeypatch.setattr(sys, "argv", args + ["--write-manifest"])
    with pytest.raises(SystemExit) as error:
        validate.main()
    assert error.value.code == 1
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("payload", ["{}", "[]", '{"manifest_version": 99}', "{"])
def test_malformed_manifest_is_rejected(dataset, payload):
    path = dataset.root / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        validate.check_manifest(inspect(dataset), path)


def submission():
    hours = pd.date_range("2026-11-07", "2026-11-11 23:00", freq="h")
    return pd.DataFrame([
        (arc, hour.strftime("%Y-%m-%d %H:%M"), 100.0, 5.0)
        for arc in validate.SUBMISSION_ARCS for hour in hours
    ], columns=validate.SUBMISSION_COLUMNS)


def test_official_submission_contract_and_csv_cli(tmp_path, monkeypatch, capsys):
    table = submission()
    table.loc[0, "debit_horaire"] = 0.0  # Do not apply the provisional observed-data rule.
    validate.validate_submission(table)
    path = tmp_path / "output_team.csv"
    table.to_csv(path, sep=";", index=False)
    monkeypatch.setattr(sys, "argv", ["validate", "--submission", str(path)])
    validate.main()
    assert "Valid submission:" in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["team.csv", "output_.csv"])
def test_submission_cli_checks_the_official_filename(tmp_path, monkeypatch, capsys, filename):
    path = tmp_path / filename
    submission().to_csv(path, sep=";", index=False)
    monkeypatch.setattr(sys, "argv", ["validate", "--submission", str(path)])
    with pytest.raises(SystemExit) as error:
        validate.main()
    assert error.value.code == 1
    assert "output_<group_name>.csv" in capsys.readouterr().err


@pytest.mark.parametrize("mutation", [
    "extra_column", "missing_column", "missing_row", "duplicate", "proxy", "wrong_period",
    "bad_format", "nonhour", "missing_q", "infinite_k", "negative_q", "k_above_100", "integer_q",
])
def test_invalid_prediction_outputs_are_rejected(mutation):
    table = submission()
    if mutation == "extra_column": table["year"] = 2026
    elif mutation == "missing_column": table = table.drop(columns="debit_horaire")
    elif mutation == "missing_row": table = table.iloc[1:]
    elif mutation == "duplicate": table.iloc[0] = table.iloc[1]
    elif mutation == "proxy": table.loc[0, "arc"] = "Proxy"
    elif mutation == "wrong_period": table.loc[0, "datetime"] = "2025-11-07 00:00"
    elif mutation == "bad_format": table.loc[0, "datetime"] = "2026-11-07T00:00:00Z"
    elif mutation == "nonhour": table.loc[0, "datetime"] = "2026-11-07 00:30"
    elif mutation == "missing_q": table.loc[0, "debit_horaire"] = float("nan")
    elif mutation == "infinite_k": table.loc[0, "taux_occupation"] = float("inf")
    elif mutation == "negative_q": table.loc[0, "debit_horaire"] = -1.0
    elif mutation == "k_above_100": table.loc[0, "taux_occupation"] = 101.0
    else: table["debit_horaire"] = table["debit_horaire"].astype("int64")
    with pytest.raises(ValueError):
        validate.validate_submission(table)


class HttpResponse(io.BytesIO):
    status = 200

    def __init__(self, payload, *, interrupted=False):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self.interrupted = interrupted
        self.line_reads = 0

    def read(self, size=-1):
        if self.interrupted:
            if self.tell():
                raise TimeoutError("interrupted ZIP response")
            return super().read(min(size, 8))
        return super().read(size)

    def readline(self, size=-1):
        if self.interrupted and self.line_reads:
            raise TimeoutError("interrupted JSONL response")
        self.line_reads += 1
        return super().readline(size)


def test_full_http_ingestion_build_query_update_and_recovery(tmp_path, monkeypatch):
    nodes = (CONVENTION.iu_ac, CONVENTION.upstream_node, CONVENTION.downstream_node)
    zip_path = write_archive(tmp_path / "fixture.zip", {"boundary.txt": [
        archive_line(*nodes, "2024-12-31 23:00:00", "100", "5", "1"),
        archive_line(*nodes, "2025-01-01 00:00:00", "150", "5", "1"),
        archive_line(*nodes, "2025-01-01 01:00:00", "200", "5", "1"),
        archive_line(*nodes, "2025-01-01 01:00:00", "200", "5", "1"),
    ]})
    zip_bytes = zip_path.read_bytes()
    state = SimpleNamespace(zip_requests=0, abort_api=False, failed_exports=0, roads=[])
    config = write_config(tmp_path / "config.yaml")
    metadata = {"attachments": [{"id": "opendata_txt_2024_zip", "title": "opendata_txt_2024.zip"}]}

    def historical_http(request, **kwargs):
        if request.full_url == historical.DATASET_URL:
            return HttpResponse(json.dumps(metadata).encode())
        assert request.full_url.endswith("/attachments/opendata_txt_2024_zip/")
        state.zip_requests += 1
        return HttpResponse(zip_bytes, interrupted=state.zip_requests == 1)

    records = [recent_record(q="250"), recent_record(t_1h="2025-01-01T01:00:00Z", q="300")]

    def recent_http(request, **kwargs):
        parsed = urlparse(request.full_url)
        parameters = parse_qs(parsed.query)
        where = parameters["where"][0]
        road = int(re.search(r"iu_ac = '([0-9]+)'", where).group(1))
        bounds = [traffic.parse_timestamp(value) for value in re.findall(r"date'([^']+)'", where)]
        matching = [record for record in records if (
            int(record["iu_ac"]) == road
            and bounds[0] <= traffic.parse_timestamp(record["t_1h"]) < bounds[1]
        )]
        state.roads.append(road)
        if parsed.path.endswith("/exports/jsonl"):
            payload = b"".join(json.dumps(record).encode() + b"\n" for record in matching)
            state.failed_exports += int(state.abort_api)
            return HttpResponse(payload, interrupted=state.abort_api)
        assert parsed.path.endswith("/records")
        limit = int(parameters["limit"][0])
        return HttpResponse(json.dumps({"total_count": len(matching), "results": matching[:limit]}).encode())

    monkeypatch.setattr(historical, "urlopen", historical_http)
    monkeypatch.setattr(traffic, "urlopen", recent_http)
    monkeypatch.setattr(traffic.time, "sleep", lambda delay: None)
    monkeypatch.setattr(update, "_now", lambda: pd.Timestamp("2025-01-03", tz=traffic.PARIS_TZ))

    archive = historical.select_archives(historical.discover_archives(), range(2024, 2025))[0]
    raw = tmp_path / "raw" / "historical"
    downloaded = historical.download_archive(archive, raw)
    assert state.zip_requests == 2  # First stream failed; the retry restarted safely.
    assert downloaded.read_bytes() == zip_bytes
    historical.download_archive(archive, raw)
    assert state.zip_requests == 2  # A valid existing archive is reused.
    assert not list(raw.rglob("*.part"))

    output = tmp_path / "processed"
    _, stats = build.build_traffic("2024-12-31", "2025-01-01", raw, output, config)
    assert stats.identical_duplicates == 1
    old_year = output / "year=2024" / "part-0.parquet"
    preserved = old_year.read_bytes()
    first = update.update_traffic(
        "2025-01-01", "2025-01-01", output_dir=output,
        raw_dir=tmp_path / "raw" / "recent", config_path=config,
    )
    assert [path.parent.name for path in first.written] == ["year=2025"]
    assert old_year.read_bytes() == preserved
    assert set(state.roads) == {5671, 42}
    result = query.query_traffic(
        """SELECT datetime, debit_horaire FROM traffic
           WHERE iu_ac=5671 AND year=2025 AND debit_horaire IS NOT NULL
           ORDER BY datetime""", path=output,
    )
    assert result["debit_horaire"].tolist() == [150.0, 250.0, 300.0]

    report = validate.validate_dataset(output, config_path=config)
    manifest = validate.write_manifest(report, output / "_validation.json")
    assert report["coverage"]["validated_rows"] == 96
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*.parquet")}
    repeat = update.update_traffic(
        "2025-01-01", "2025-01-01", output_dir=output,
        raw_dir=tmp_path / "raw" / "recent", config_path=config,
    )
    assert repeat.written == []
    validate.check_manifest(validate.validate_dataset(output, config_path=config), manifest)

    # Reproduce the same workflow from the same raw archive in another location.
    other = tmp_path / "reproduced"
    build.build_traffic("2024-12-31", "2025-01-01", raw, other, config)
    update.update_traffic(
        "2025-01-01", "2025-01-01", output_dir=other,
        raw_dir=tmp_path / "raw" / "recent", config_path=config,
    )
    sql = "SELECT * FROM traffic ORDER BY arc, datetime"
    pd.testing.assert_frame_equal(
        query.query_traffic(sql, path=output), query.query_traffic(sql, path=other)
    )

    state.abort_api = True
    with pytest.raises(TimeoutError):
        update.update_traffic(
            "2025-01-01", "2025-01-01", output_dir=output,
            raw_dir=tmp_path / "raw" / "recent", config_path=config,
        )
    assert state.failed_exports == 3
    assert before == {path.relative_to(output): path.read_bytes() for path in output.rglob("*.parquet")}
    assert not (output / ".traffic-update.lock").exists()
    assert not list((tmp_path / "raw").rglob("*.part"))
    validate.check_manifest(validate.validate_dataset(output, config_path=config), manifest)
