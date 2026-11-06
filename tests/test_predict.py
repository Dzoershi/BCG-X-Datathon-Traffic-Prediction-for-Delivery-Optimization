"""Offline submission tests using synthetic processed traffic only."""

from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import predict
from traffic_prediction.data import build, traffic, validate
from traffic_prediction.data.validate import SUBMISSION_ARCS, validate_submission

# Pinned copy of the provisional (unapproved) review, so tests do not depend on the
# delivery-day decision recorded in configs/disruptions_review.json.
PROVISIONAL_REVIEW = Path(__file__).parent / "fixtures" / "disruptions_review_provisional.json"

SCHOOL_PATH = Path(__file__).parent / "fixtures/paris_school_holidays.json"


@pytest.fixture
def history():
    # 11/11/2025 is the reference day of the Champs-Elysées ceremony closure (#47).
    times = pd.date_range("2025-11-11", "2025-11-11 23:00", freq="h").append(
        pd.date_range("2026-10-01", "2026-11-11 23:00", freq="h"))
    return pd.concat([
        pd.DataFrame(dict(arc=arc, datetime=times, role="evalue", qualite_mesure="ok",
                          debit_horaire=100.0 + times.hour, taux_occupation=10.0))
        for arc in SUBMISSION_ARCS
    ], ignore_index=True)


@pytest.fixture
def local_data(tmp_path, history):
    arcs = traffic.load_arcs(traffic.DEFAULT_CONFIG)
    frames = []
    for arc in arcs:
        frame = history.loc[history.arc.eq(arc.name)].copy()
        if frame.empty:
            continue
        frame["iu_ac"] = arc.iu_ac
        frame["etat_trafic"] = pd.Series(1, index=frame.index, dtype="Int64")
        frame["etat_barre"] = pd.Series(0, index=frame.index, dtype="Int64")
        frames.append(frame)
    build.write_partitions(pd.concat(frames), tmp_path / "traffic")
    return dict(data_dir=tmp_path / "traffic", output_dir=tmp_path / "submissions",
                school_holidays_path=SCHOOL_PATH, closure_review_path=PROVISIONAL_REVIEW)


def test_cutoff_clipping_types_and_order(history):
    class Model:
        def predict(self, data, times):
            assert data.datetime.max() == predict.CUTOFF
            assert len(times) == 120
            index = pd.MultiIndex.from_product([SUBMISSION_ARCS[::-1], times[::-1]],
                                              names=["arc", "datetime"])
            table = index.to_frame(index=False)
            table["debit_horaire"] = -5
            table["taux_occupation"] = 101
            return table

    table = predict.build_submission(history, Model())
    validate_submission(table)
    assert table.arc.dtype == object and table.datetime.dtype == object
    assert table.debit_horaire.eq(0.0).all() and table.taux_occupation.eq(100.0).all()
    assert table.equals(table.sort_values(["arc", "datetime"]).reset_index(drop=True))


def test_exported_values_have_two_decimals(history, tmp_path):
    class Model:
        def predict(self, data, times):
            index = pd.MultiIndex.from_product([SUBMISSION_ARCS, times], names=["arc", "datetime"])
            table = index.to_frame(index=False)
            table["debit_horaire"] = 123.456789123
            table["taux_occupation"] = 7.0 / 3.0
            return table

    table = predict.build_submission(history, Model())
    assert table.debit_horaire.eq(123.46).all() and table.taux_occupation.eq(2.33).all()
    path = predict.export_submission(table, "rounded", tmp_path)
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        for value in line.split(";")[-2:]:
            assert len(value.partition(".")[2]) <= 2, line


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_rejected(history, bad):
    class Model:
        def predict(self, data, times):
            return pd.DataFrame(dict(arc=[SUBMISSION_ARCS[0]], datetime=[times[0]],
                                     debit_horaire=[bad], taux_occupation=[5.0]))
    with pytest.raises(ValueError, match="Non-finite"):
        predict.build_submission(history, Model())


# The ensemble needs Timestral weights and the #22 dataset: tested on stubs in test_ensemble.py.
@pytest.mark.parametrize("model_name", [name for name in predict.MODELS if name != "ensemble"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_pipeline_roundtrip_reproducible_and_no_network(local_data, monkeypatch, model_name, line_ending):
    # Exercise both organizer-export defaults, including Windows on Unix hosts.
    monkeypatch.setattr(os, "linesep", line_ending)
    def forbidden(*args, **kwargs):
        pytest.fail("Offline submission attempted network access")
    monkeypatch.setattr(predict.update, "update_traffic", forbidden)
    monkeypatch.setattr(predict.calendar_events, "download_school_holidays", forbidden)
    path = predict.run_submission("test_team", skip_update=True, model_name=model_name, **local_data)
    original = path.read_bytes()
    log_path = path.with_name(f"{path.stem}_run.json")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["csv_sha256"] == hashlib.sha256(original).hexdigest()
    assert log["model_choice"] == model_name
    assert log["cutoff"] == predict.CUTOFF.isoformat()
    assert log["closure_correction_applied"] is False
    assert log["closure_review"][0]["status"] == "provisional"
    assert log["closure_review"][0]["action"] == "none"
    assert log["closure_adjustments"] == []
    assert log["runtime_seconds"] >= 0
    assert len(log["sources"]) == 360
    assert all(row["source"] is not None for row in log["sources"])
    assert not any(row["source"] == "closure_copy_2025" for row in log["sources"])
    assert set(log["last_observed_hour_per_arc"].values()) == {predict.CUTOFF.isoformat()}
    assert path.name == "output_test_team.csv"
    assert original.startswith(
        b";arc;datetime;debit_horaire;taux_occupation" + line_ending.encode("ascii")
    )
    table = pd.read_csv(path, sep=";", index_col=0)
    validate_submission(table)
    predict.run_submission("test_team", skip_update=True, model_name=model_name, **local_data)
    assert path.read_bytes() == original
    repeated = json.loads(log_path.read_text(encoding="utf-8"))
    for field in ("csv_sha256", "traffic_fingerprint", "sources", "model_revisions"):
        assert repeated[field] == log[field]
    assert set(path.parent.iterdir()) == {path, path.with_name(f"{path.stem}_run.json")}


@pytest.mark.parametrize("cutoff_day", ["2026-11-05", "2026-11-04"])
def test_refresh_precedes_read_and_uses_submission_cutoff(local_data, monkeypatch, cutoff_day):
    monkeypatch.setattr(predict, "CUTOFF", pd.Timestamp(f"{cutoff_day} 23:00"))
    calls = []
    original_query = predict.query.query_traffic

    def refresh(start, end, **kwargs):
        calls.append("refresh")
        assert start == "2026-10-01" and end == cutoff_day
        assert kwargs["output_dir"] == local_data["data_dir"]

    def query(*args, **kwargs):
        assert calls == ["refresh"]
        assert args[1] == [predict.CUTOFF]
        calls.append("query")
        return original_query(*args, **kwargs)

    monkeypatch.setattr(predict.update, "update_traffic", refresh)
    monkeypatch.setattr(predict.query, "query_traffic", query)
    predict.run_submission("team", update_start="2026-10-01", **local_data)
    assert calls == ["refresh", "query"]


def test_failed_refresh_preserves_output(local_data, monkeypatch):
    path = predict.run_submission("team", skip_update=True, **local_data)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("network unavailable")
    monkeypatch.setattr(predict.update, "update_traffic", fail)
    with pytest.raises(OSError, match="network unavailable"):
        predict.run_submission("team", **local_data)
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "team name"])
def test_invalid_team_rejected_before_update(tmp_path, name, monkeypatch):
    monkeypatch.setattr(predict.update, "update_traffic", lambda *a, **k: pytest.fail("updated"))
    with pytest.raises(ValueError, match="team_name"):
        predict.run_submission(name, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli(local_data, monkeypatch, capsys):
    # The default ensemble needs Timestral weights and the #22 dataset (see test_ensemble.py).
    monkeypatch.setattr(sys, "argv", ["predict", "--team-name", "cli", "--skip-update", "--model", "holiday-aware",
        "--data-dir", str(local_data["data_dir"]), "--output-dir", str(local_data["output_dir"]),
        "--school-holidays-file", str(SCHOOL_PATH)])
    predict.main()
    assert "Validated submission:" in capsys.readouterr().out
    assert (local_data["output_dir"] / "output_cli.csv").exists()
    monkeypatch.setattr(sys, "argv", ["validate", "--submission",
                                     str(local_data["output_dir"] / "output_cli.csv")])
    validate.main()
    assert "Valid submission:" in capsys.readouterr().out


def test_validator_rejects_corrupt_organizer_index(local_data, monkeypatch, capsys):
    path = predict.run_submission("team", skip_update=True, **local_data)
    table = pd.read_csv(path, sep=";", index_col=0)
    table.index = table.index + 1
    table.to_csv(path, sep=";")
    monkeypatch.setattr(sys, "argv", ["validate", "--submission", str(path)])
    with pytest.raises(SystemExit) as error:
        validate.main()
    assert error.value.code == 1
    assert "index must be consecutive" in capsys.readouterr().err


def test_export_failure_keeps_existing_file_and_cleans_temporary(local_data, monkeypatch):
    path = predict.run_submission("team", skip_update=True, **local_data)
    original = path.read_bytes()
    table = pd.read_csv(path, sep=";", index_col=0)

    def fail(*args, **kwargs):
        raise OSError("write failed")
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="write failed"):
        predict.export_submission(table, "team", path.parent)
    assert path.read_bytes() == original
    assert set(path.parent.iterdir()) == {path, path.with_name(f"{path.stem}_run.json")}


def test_missing_arc_rejected(local_data):
    root = local_data["data_dir"]
    for partition in list(root.rglob("*.parquet")):  # One partition per year.
        table = pd.read_parquet(partition)
        build.write_partitions(table[table.arc.ne(SUBMISSION_ARCS[0])], root)
    with pytest.raises(ValueError, match="all three"):
        predict.run_submission("team", skip_update=True, **local_data)
    assert not local_data["output_dir"].exists()


def test_offline_missing_calendar_does_not_download(local_data, tmp_path, monkeypatch):
    local_data["school_holidays_path"] = tmp_path / "missing.json"
    monkeypatch.setattr(predict.calendar_events, "download_school_holidays",
                        lambda *a: pytest.fail("download attempted"))
    with pytest.raises(FileNotFoundError, match="local school-holidays"):
        predict.run_submission("team", skip_update=True, **local_data)


def test_organizer_function_checks_object_types_before_publication(local_data):
    path = predict.run_submission("team", skip_update=True, **local_data)
    original = path.read_bytes()
    table = pd.read_csv(path, sep=";", index_col=0)
    table["arc"] = table["arc"].astype("string")
    with pytest.raises(AssertionError, match="Column arc does not have type"):
        predict.export_submission(table, "team", path.parent)
    assert path.read_bytes() == original
    assert set(path.parent.iterdir()) == {path, path.with_name(f"{path.stem}_run.json")}


def test_requires_pandas_2_before_refresh(monkeypatch):
    monkeypatch.setattr(pd, "__version__", "3.0.0")
    monkeypatch.setattr(predict.update, "update_traffic", lambda *a, **k: pytest.fail("updated"))
    with pytest.raises(ValueError, match="pandas 2.x"):
        predict.run_submission("team")


@pytest.mark.parametrize("failure", ["metadata", "write", "replace"])
def test_run_log_failure_warns_and_keeps_submission(local_data, monkeypatch, capsys, failure):
    from traffic_prediction import submission_log

    def fail(*args, **kwargs):
        raise OSError("run log unavailable")

    if failure == "metadata":
        monkeypatch.setattr(submission_log, "timestral_package", fail)
    elif failure == "write":
        monkeypatch.setattr(Path, "write_text", fail)
    else:
        original = Path.replace

        def replace(path, target):
            if str(target).endswith("_run.json"):
                fail()
            return original(path, target)

        monkeypatch.setattr(Path, "replace", replace)
    path = predict.run_submission("team", skip_update=True, **local_data)
    validate_submission(pd.read_csv(path, sep=";", index_col=0))
    assert "Warning: submission run log" in capsys.readouterr().err
    assert list(path.parent.iterdir()) == [path]
