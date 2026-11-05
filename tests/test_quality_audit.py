"""Raw provenance, independent targets, duplicate precedence and bounded audit I/O."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data import audit, traffic
from traffic_prediction.data.traffic import Arc

ARCS = [Arc("A", "evalue", "U", "V", 1), Arc("P", "proxy", "P", "Q", 2)]


def raw(time="2025-11-01 00:00:00", q="10", k="2", sensor="1", upstream="U", downstream="V"):
    return dict(zip(traffic.RAW_COLUMNS, [sensor, upstream, downstream, time, q, k, "1", "3"]))


def classified(records):
    frame = pd.DataFrame(records).assign(source_member="", source_row=np.arange(2, len(records) + 2))
    return audit.classify(audit._select(frame, ARCS).assign(source_file="fixture.jsonl"))


def sensor_snapshot(arcs=ARCS):
    return {"retrieved_at_utc": "2026-11-05T08:00:00Z", "entries": [
        {"arc": arc.name, "direction": direction, "expected_iu_ac": arc.iu_ac,
         "responses": {kind: {"url": "https://example.org/inventory", "total_count": 1,
                              "results": [{"iu_ac": str(sensor), "libelle_nd_amont": upstream,
                                           "libelle_nd_aval": downstream}]}
                       for kind in ("measurements", "reference")}}
        for arc in arcs
        for direction, upstream, downstream, sensor in [
            ("configured", arc.upstream_node, arc.downstream_node, arc.iu_ac),
            ("reverse", arc.downstream_node, arc.upstream_node, arc.iu_ac + 100),
        ]
    ]}


def test_each_target_is_classified_without_discarding_its_valid_partner():
    records = [raw(f"2025-11-01 {hour:02}:00:00", q, k) for hour, (q, k) in enumerate([
        ("10", "2"), (None, "5"), ("4", None), ("bad", "7"),
        ("6", "101"), ("0", "4"), ("0", "0"), (None, None),
    ])]
    rows = classified(records)
    assert rows.q_status.tolist() == ["valid", "missing", "valid", "invalid", "valid", "suspect_zero", "valid", "missing"]
    assert rows.k_status.tolist() == ["valid", "valid", "missing", "valid", "invalid", "valid", "valid", "missing"]
    assert rows.legacy_quality.tolist() == ["ok", "partielle", "partielle", "invalide", "invalide", "invalide", "ok", "vide"]
    assert rows.loc[5, "q_value"] == 0 and rows.loc[5, "k_value"] == 4
    assert rows.loc[3, "q"] == "bad" and rows.source_row.tolist() == list(range(2, 10))
    original = pd.DataFrame(records)
    assert original.loc[5, "q"] == "0" and original.loc[1, "q"] is None


@pytest.mark.parametrize("value,status", [("", "missing"), ("  ", "missing"), (None, "missing"),
                                         ("NaN", "invalid"), ("inf", "invalid"), ("-1", "invalid"),
                                         ("0", "valid"), ("12.5", "valid")])
def test_numeric_bounds_preserve_missing_and_valid_zero(value, status):
    result = classified([raw(q=value, k="0")]).iloc[0]
    assert result.q_status == status


def test_monthly_counts_reconcile_and_absence_is_not_an_explicit_null():
    rows = classified([raw("2025-11-01 00:00:00"), raw("2025-11-01 01:00:00", q=None, k="5"),
                       raw("2025-11-01 02:00:00", q="bad", k="7"),
                       raw("2025-11-01 03:00:00", q="0", k="4")])
    first, last = traffic.period_bounds("2025-11-01", "2025-11-01")
    grid, duplicates = audit.hourly_grid(rows, ARCS, first, last)
    assert duplicates == 0 and len(grid) == 48
    monthly = audit.monthly_report(grid).set_index(["arc", "variable"])
    q, k = monthly.loc[("A", "debit_horaire")], monthly.loc[("A", "taux_occupation")]
    assert (q.expected_hours, q.source_hours, q.absent_source_hours) == (24, 4, 20)
    assert (q.null_source_values, q.invalid_source_values, q.suspect_zero_flow, q.valid_target_values) == (1, 1, 1, 1)
    assert (k.valid_target_values, k.legacy_pair_ok, k.excluded_by_pair_requirement) == (4, 1, 3)
    assert k.valid_in_partial_rows == 1 and k.valid_in_invalid_pairs == 2
    assert k.latest_valid_hour == pd.Timestamp("2025-11-01 03:00")
    for _, row in monthly.iterrows():
        assert row.expected_hours == (row.absent_source_hours + row.null_source_values
                                      + row.invalid_source_values + row.suspect_zero_flow + row.valid_target_values)


def test_last_supplied_source_row_wins_including_explicit_nulls():
    a = classified([raw()])
    b = classified([raw(q=None, k="8")]).assign(source_file="correction.jsonl")
    first, last = traffic.period_bounds("2025-11-01", "2025-11-01")
    grid, removed = audit.hourly_grid(pd.concat([a, b], ignore_index=True), ARCS, first, last)
    row = grid.loc[grid.arc.eq("A") & grid.datetime.eq(first)].iloc[0]
    assert removed == 1 and pd.isna(row.q_value) and row.k_value == 8
    assert row.source_file == "correction.jsonl" and row.legacy_quality == "partielle"


def test_zero_context_uses_exact_hours_and_separate_runs():
    records = [raw(f"2025-11-01 {hour:02}:00:00", "0" if hour in (1, 2, 4) else "10", "5")
               for hour in (0, 1, 2, 4, 5)]
    first, last = traffic.period_bounds("2025-11-01", "2025-11-01")
    grid, _ = audit.hourly_grid(classified(records), ARCS, first, last)
    context = audit.zero_context(grid)
    assert context.run_hours.tolist() == [2, 2, 1]
    assert context.previous_q_value.iloc[0] == 10
    assert pd.isna(context.previous_q_value.iloc[2])
    assert context.previous_legacy_quality.iloc[2] == "absente"


def test_incomplete_audit_period_does_not_report_a_partial_backtest():
    first, last = traffic.period_bounds("2025-11-08", "2025-11-09")
    grid, _ = audit.hourly_grid(classified([raw("2025-11-08 00:00:00")]), ARCS, first, last)
    report = audit.eligibility_report(grid, [pd.Timestamp("2025-11-06")])
    assert report.empty and list(report.columns) == audit.ELIGIBILITY_COLUMNS


def test_direction_and_unexpected_identifiers_are_audited_without_silent_filtering():
    rows = classified([raw(sensor="99"), raw(sensor="7", upstream="V", downstream="U")])
    assert len(rows) == 1 and rows.unexpected_sensor_id.item()
    assert rows.iu_ac.item() == "99" and rows.expected_iu_ac.item() == 1
    with pytest.raises(ValueError, match="expected 1"):
        traffic.normalize_raw(pd.DataFrame([raw(sensor="99")]), ARCS)


def test_archive_chunks_retain_member_and_original_row_number(tmp_path):
    path = tmp_path / "raw.zip"
    records = [raw(), raw(sensor="9", upstream="V", downstream="U"), raw("2025-11-01 01:00:00", k=None)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("week.txt", pd.DataFrame(records).to_csv(sep=";", index=False))
    before = path.read_bytes()
    selected, total = audit._read_selected(path, ARCS, chunk_rows=2)
    assert total == 3 and len(selected) == 2
    assert selected.source_member.tolist() == ["week.txt", "week.txt"]
    assert selected.source_row.tolist() == [2, 4]
    assert path.read_bytes() == before


def test_archive_keeps_supplied_nonfinite_text_distinct_from_empty_fields(tmp_path):
    path = tmp_path / "literal.zip"
    records = [raw(q="NaN"), raw("2025-11-01 01:00:00", q="")]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("week.txt", pd.DataFrame(records).to_csv(sep=";", index=False))
    selected, _ = audit._read_selected(path, ARCS, 2)
    assert selected.q.tolist() == ["NaN", ""]
    assert audit.classify(selected).q_status.tolist() == ["invalid", "missing"]


def test_cache_reuses_verified_rows_and_recovers_from_cache_corruption(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw("2025-11-01T00:00:00Z")) + "\n", encoding="utf-8")
    digest = traffic.file_sha256(path)
    cache = tmp_path / "cache"
    calls = []
    original = audit._read_selected

    def read(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(audit, "_read_selected", read)
    first, receipt = audit.extract_source(path, digest, ARCS, cache, 2)
    second, _ = audit.extract_source(path, digest, ARCS, cache, 2)
    assert len(calls) == 1 and receipt["raw_rows"] == 1
    pd.testing.assert_frame_equal(first, second)
    next(cache.glob("*.parquet")).write_bytes(b"broken")
    third, _ = audit.extract_source(path, digest, ARCS, cache, 2)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first, third)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        audit.extract_source(path, digest, ARCS, cache, 2)


@pytest.mark.parametrize("damage", ["id", "nodes", "missing"])
def test_inventory_does_not_certify_changed_or_unverified_mapping(damage):
    snapshot = sensor_snapshot()
    if damage == "missing":
        snapshot["entries"] = snapshot["entries"][1:]
        with pytest.raises(ValueError, match="one configured-direction"):
            audit.inventory_report(snapshot, ARCS)
    else:
        field = "iu_ac" if damage == "id" else "libelle_nd_aval"
        snapshot["entries"][0]["responses"]["measurements"]["results"][0][field] = "99"
        result = audit.inventory_report(snapshot, ARCS)
        assert not result.snapshot_ids_match.iloc[0] and result.snapshot_ids_match.iloc[1]


def test_truncated_inventory_snapshot_is_rejected():
    snapshot = sensor_snapshot()
    snapshot["entries"][0]["responses"]["measurements"]["total_count"] = 2
    with pytest.raises(ValueError, match="incomplete response"):
        audit.inventory_report(snapshot, ARCS)


def test_sensor_capture_uses_both_node_directions_not_id_only(tmp_path, monkeypatch):
    urls = []

    def get(url):
        urls.append(url)
        return {"total_count": 0, "results": []}

    monkeypatch.setattr(audit, "_get", get)
    path = tmp_path / "sensors.json"
    audit.capture_sensors(path, ARCS[:1])
    assert len(urls) == 4
    from urllib.parse import parse_qs, urlparse
    filters = [parse_qs(urlparse(url).query)["where"][0] for url in urls]
    assert any("libelle_nd_amont='U'" in value and "libelle_nd_aval='V'" in value for value in filters)
    assert any("libelle_nd_amont='V'" in value and "libelle_nd_aval='U'" in value for value in filters)
    assert all("iu_ac=" not in value for value in filters)
    assert path.exists()
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="existing raw inventory"):
        audit.capture_sensors(path, ARCS[:1])
    assert len(urls) == 4 and path.read_bytes() == before


def test_empty_raw_source_is_reported_explicitly(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    selected, total = audit._read_selected(path, ARCS, 2)
    assert total == 0 and selected.empty
    assert audit.classify(selected.assign(source_file=str(path))).empty


def test_incomplete_sensor_api_response_is_explicit(monkeypatch):
    monkeypatch.setattr(audit, "urlopen", lambda *args, **kwargs: io.BytesIO(
        json.dumps({"total_count": 2, "results": [{"iu_ac": "1"}]}).encode()))
    with pytest.raises(ValueError, match="incomplete"):
        audit._get("https://example.org/inventory")


def test_end_to_end_offline_manifest_and_cli(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(raw(f"2025-11-08T{hour:02}:00:00Z", q=None if hour else "10"))
                              for hour in range(3)) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config_text = "traffic:\n  evaluated_arcs:\n    - {name: A, upstream_node: U, downstream_node: V, iu_ac: 1}\n"
    config.write_bytes(config_text.replace("\n", "\r\n").encode("utf-8"))
    sensors = tmp_path / "sensors.json"
    sensors.write_text(json.dumps(sensor_snapshot(ARCS[:1])), encoding="utf-8")
    inputs = {"config": str(config), "start": "2025-11-01", "end": "2025-11-12",
              "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
              "sensor_snapshot": {"path": str(sensors), "sha256": traffic.file_sha256(sensors)},
              "sources": [{"path": str(path), "sha256": traffic.file_sha256(path)},
                          {"path": str(path), "sha256": traffic.file_sha256(path)}]}
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps(inputs), encoding="utf-8")
    out = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["audit", "--inputs", str(manifest), "--output-dir", str(out),
                                     "--cache-dir", str(tmp_path / "cache"), "--chunk-rows", "2"])
    audit.main()
    report = json.loads((out / "report.json").read_text())
    assert report["rows"] == 12 * 24 and report["identical_files_skipped"] == [str(path)]
    assert report["sensor_inventory_matches"]
    monthly = pd.read_csv(out / "monthly.csv")
    k = monthly.loc[monthly.variable.eq("taux_occupation")].iloc[0]
    assert k.valid_target_values == 3 and k.excluded_by_pair_requirement == 2
    effects = pd.read_csv(out / "eligibility.csv")
    effect = effects.loc[effects.cutoff.eq("2025-11-06") & effects.variable.eq("taux_occupation")].iloc[0]
    assert effect.expected_hours == 120 and effect.target_scored == 3 and effect.legacy_scored == 1
    observations = pd.read_parquet(out / "observations.parquet")
    assert observations.source_file.notna().sum() == 3
    assert observations.loc[observations.source_file.notna(), "datetime"].min() == pd.Timestamp("2025-11-08 01:00")
    original = (out / "summary.csv").read_bytes()
    # A Git checkout's line endings must not invalidate the pinned configuration.
    config.write_bytes(config_text.encode("utf-8"))
    audit.main()
    assert (out / "summary.csv").read_bytes() == original
    inputs["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(inputs), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration checksum"):
        audit.run_audit(manifest, out, chunk_rows=2, cache_dir=tmp_path / "cache")
