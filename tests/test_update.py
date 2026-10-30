"""Offline incremental updates, including correction and failure recovery."""

import json
import sys
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from traffic_prediction.data import build, query, traffic, update
from test_traffic_processing import (
    ARCS, CONVENTION, convention_row, raw_rows, recent_record, write_config,
)


def seed(output_dir, start="2025-01-01", end="2025-01-01"):
    rows = traffic.normalize_raw(raw_rows(
        convention_row("2024-12-31 23:00:00", q="50"),
        convention_row("2025-01-01 01:00:00", q="100"),
    ), ARCS)
    table = traffic.build_hourly_table(rows, ARCS, start, end)
    build.write_partitions(table, output_dir)


def contents(output_dir):
    return {p.relative_to(output_dir).as_posix(): p.read_bytes() for p in output_dir.rglob("*.parquet")}


def read_hours(output_dir):
    return query.query_traffic(
        "SELECT * EXCLUDE (year) FROM traffic ORDER BY arc, datetime", path=output_dir
    ).set_index(["arc", "datetime"])


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    state = SimpleNamespace(
        output=tmp_path / "processed", raw=tmp_path / "raw",
        config=write_config(tmp_path / "config.yaml"), records=[], calls=[],
    )
    monkeypatch.setattr(update, "_now", lambda: pd.Timestamp("2025-01-05 12:30", tz=traffic.PARIS_TZ))

    def download(start, end, output_dir, *, road_id):
        state.calls.append((start, end, road_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"snapshot_{len(state.calls)}.jsonl"
        records = [record for record in state.records if (
            str(record["iu_ac"]) == str(road_id)
            and start <= traffic.parse_timestamp(record["t_1h"]) < end
        )]
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return path

    monkeypatch.setattr(traffic, "download_observations", download)
    return state


def run(state, start="2025-01-01", end="2025-01-01", **kwargs):
    return update.update_traffic(
        start, end, output_dir=state.output, raw_dir=state.raw,
        config_path=state.config, **kwargs,
    )


def test_new_observations_preserve_history_and_only_rewrite_affected_year(scenario):
    seed(scenario.output, "2024-12-31", "2025-01-01")
    before = contents(scenario.output)
    scenario.records.append(recent_record(t_1h="2025-01-02T00:00:00Z", q="200"))
    result = run(scenario, "2025-01-02", "2025-01-02")

    assert [p.parent.name for p in result.written] == ["year=2025"]
    assert contents(scenario.output)["year=2024/part-0.parquet"] == before["year=2024/part-0.parquet"]
    hours = read_hours(scenario.output)
    assert hours.loc[("Convention", pd.Timestamp("2025-01-01 01:00")), "debit_horaire"] == 100
    assert hours.loc[("Convention", pd.Timestamp("2025-01-02 01:00")), "debit_horaire"] == 200
    assert not hours.index.duplicated().any()
    assert [call[2] for call in scenario.calls] == [5671, 42]


def test_corrections_and_repeated_updates_are_idempotent(scenario):
    seed(scenario.output)
    scenario.records.append(recent_record(q="250"))
    first = run(scenario)
    before = contents(scenario.output)
    path = first.written[0]
    modified = path.stat().st_mtime_ns

    second = run(scenario)

    assert second.written == []
    assert second.unchanged == [path]
    assert contents(scenario.output) == before
    assert path.stat().st_mtime_ns == modified
    hours = read_hours(scenario.output)
    assert len(hours) == 48
    assert hours.loc[("Convention", pd.Timestamp("2025-01-01 01:00")), "debit_horaire"] == 250


def test_explicit_null_correction_replaces_measurements_without_imputation(scenario):
    seed(scenario.output)
    scenario.records.append(recent_record(q=None, k=None))
    run(scenario)
    hour = read_hours(scenario.output).loc[("Convention", pd.Timestamp("2025-01-01 01:00"))]
    assert pd.isna(hour["debit_horaire"])
    assert pd.isna(hour["taux_occupation"])
    assert hour["qualite_mesure"] == "vide"


def test_missing_api_rows_do_not_erase_existing_observations(scenario):
    seed(scenario.output)
    scenario.records.append(recent_record(t_1h="2025-01-01T01:00:00Z", q="200"))
    run(scenario)
    hours = read_hours(scenario.output)
    assert hours.loc[("Convention", pd.Timestamp("2025-01-01 01:00")), "debit_horaire"] == 100
    assert hours.loc[("Convention", pd.Timestamp("2025-01-01 02:00")), "debit_horaire"] == 200
    assert hours.loc["Proxy", ["debit_horaire", "taux_occupation"]].isna().all().all()


def test_empty_exports_do_not_extend_coverage_or_replace_files(scenario):
    seed(scenario.output)
    before = contents(scenario.output)
    coverage = update.dataset_coverage(scenario.output)
    result = run(scenario, "2025-01-02", "2025-01-02")
    assert result.written == []
    assert contents(scenario.output) == before
    assert update.dataset_coverage(scenario.output) == coverage


def test_default_refresh_uses_source_coverage_not_synthetic_future_hours(scenario):
    seed(scenario.output, "2025-01-01", "2025-01-10")
    result = run(scenario, None, None, refresh_days=2)
    assert result.previous_coverage.last_hour == pd.Timestamp("2025-01-10 23:00")
    assert result.previous_coverage.last_source_hour == pd.Timestamp("2025-01-01 01:00")
    assert result.start == pd.Timestamp("2024-12-31")
    assert scenario.calls[0][0].astimezone(timezone.utc) == traffic.parse_timestamp("2024-12-30T23:00:00Z")
    assert pd.Timestamp(scenario.calls[0][1]) == pd.Timestamp("2025-01-05 12:30", tz=traffic.PARIS_TZ)


def test_bootstrap_requires_an_explicit_start(scenario):
    with pytest.raises(ValueError, match="provide --start"):
        run(scenario, None, None)
    assert scenario.calls == []
    assert not (scenario.output / ".traffic-update.lock").exists()


def test_bootstrap_writes_complete_grid(scenario):
    scenario.records.append(recent_record())
    run(scenario)
    hours = read_hours(scenario.output)
    assert len(hours) == 48
    assert hours["qualite_mesure"].eq("absente").sum() == 47


@pytest.mark.parametrize("days", [0, -1, True])
def test_invalid_refresh_interval_fails_before_extraction(scenario, days):
    with pytest.raises(ValueError, match="positive integer"):
        run(scenario, refresh_days=days)
    assert scenario.calls == []


def test_download_failure_does_not_modify_processed_data(scenario, monkeypatch):
    seed(scenario.output)
    before = contents(scenario.output)
    original = traffic.download_observations

    def interrupted(*args, road_id, **kwargs):
        if road_id == 42:
            raise TimeoutError("network interrupted")
        return original(*args, road_id=road_id, **kwargs)

    monkeypatch.setattr(traffic, "download_observations", interrupted)
    with pytest.raises(TimeoutError):
        run(scenario)
    assert contents(scenario.output) == before
    assert not (scenario.output / ".traffic-update.lock").exists()


@pytest.mark.parametrize("stage", ["write", "validate"])
def test_staging_failures_preserve_all_partitions(scenario, monkeypatch, stage):
    seed(scenario.output, "2024-12-31", "2025-01-01")
    before = contents(scenario.output)
    scenario.records.extend([
        recent_record(t_1h="2024-12-31T22:00:00Z", q="150"),
        recent_record(q="250"),
    ])
    if stage == "write":
        original = update.pq.write_table
        calls = []

        def fail_second(table, path, **kwargs):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("disk full")
            return original(table, path, **kwargs)

        monkeypatch.setattr(update.pq, "write_table", fail_second)
    else:
        def fail_validation(*args):
            raise ValueError("invalid staged file")

        monkeypatch.setattr(build, "check_written", fail_validation)
    with pytest.raises((OSError, ValueError)):
        run(scenario, "2024-12-31", "2025-01-01")
    assert contents(scenario.output) == before
    assert not list(scenario.output.glob(".traffic-update*"))


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_commit_failure_rolls_back_completed_replacements(scenario, monkeypatch, failure):
    seed(scenario.output, "2024-12-31", "2025-01-01")
    before = contents(scenario.output)
    scenario.records.extend([
        recent_record(t_1h="2024-12-31T22:00:00Z", q="150"), recent_record(q="250"),
    ])
    original = Path.replace

    def fail_second(self, target):
        if self.suffix == ".new" and Path(target).parent.name == "year=2025":
            raise failure("interrupted commit")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(failure):
        run(scenario, "2024-12-31", "2025-01-01")
    assert contents(scenario.output) == before
    assert not list(scenario.output.glob(".traffic-update*"))


def test_existing_lock_is_not_removed_by_a_second_update(scenario):
    scenario.output.mkdir()
    lock = scenario.output / ".traffic-update.lock"
    lock.write_text("other process", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Another update"):
        run(scenario)
    assert lock.read_text(encoding="utf-8") == "other process"
    assert scenario.calls == []


@pytest.mark.parametrize("day,hours", [("2025-03-30", 23), ("2025-10-26", 25)])
def test_api_bounds_respect_paris_daylight_saving(scenario, monkeypatch, day, hours):
    monkeypatch.setattr(update, "_now", lambda: pd.Timestamp("2025-12-01", tz=traffic.PARIS_TZ))
    run(scenario, day, day)
    first, last, _ = scenario.calls[0]
    elapsed = last.astimezone(timezone.utc) - first.astimezone(timezone.utc)
    assert elapsed.total_seconds() == hours * 3600


def test_unsupported_partition_layout_fails_before_extraction(scenario):
    seed(scenario.output)
    path = scenario.output / "year=2025" / "part-0.parquet"
    nested = path.parent / "month=01"
    nested.mkdir()
    path.replace(nested / path.name)
    before = contents(scenario.output)
    with pytest.raises(ValueError, match="Updates require the build layout"):
        run(scenario)
    assert scenario.calls == []
    assert contents(scenario.output) == before


def test_one_command_updates_the_dataset(scenario, monkeypatch, capsys):
    scenario.records.append(recent_record())
    monkeypatch.setattr(sys, "argv", [
        "update", "--start", "2025-01-01", "--end", "2025-01-01",
        "--output-dir", str(scenario.output), "--raw-dir", str(scenario.raw),
        "--config", str(scenario.config),
    ])
    update.main()
    assert len(read_hours(scenario.output)) == 48
    assert "Written partitions:" in capsys.readouterr().out


def test_cli_reports_network_protocol_failure_without_a_traceback(scenario, monkeypatch, capsys):
    from http.client import HTTPException

    def fail(*args, **kwargs):
        raise HTTPException("incomplete response")

    monkeypatch.setattr(traffic, "download_observations", fail)
    monkeypatch.setattr(sys, "argv", [
        "update", "--start", "2025-01-01", "--end", "2025-01-01",
        "--output-dir", str(scenario.output), "--config", str(scenario.config),
    ])
    with pytest.raises(SystemExit) as error:
        update.main()
    assert error.value.code == 1
    assert "Traffic update failed: incomplete response" in capsys.readouterr().err
    assert not (scenario.output / ".traffic-update.lock").exists()
