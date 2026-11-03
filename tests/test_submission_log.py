"""Provenance tests with synthetic inputs, without weights or network access."""

import hashlib
import importlib.metadata
import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import submission_log
from traffic_prediction.models.chronos import MODEL_REVISION
from traffic_prediction.models.timestral import MODEL_REVISIONS, SOURCE_REVISION


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(submission_log, "git_value", lambda *args: "" if args[0] == "status" else "abc123")
    monkeypatch.setattr(submission_log, "timestral_package", lambda: {"version": "test", "commit": "installed"})
    history = pd.DataFrame({
        "arc": ["A", "A", "A", "B"],
        "datetime": pd.to_datetime(["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-05"]),
        "role": "evalue", "qualite_mesure": ["ok", "missing", "ok", "missing"],
        "debit_horaire": [10.0, np.nan, 12.0, np.nan], "taux_occupation": 5.0,
    })
    table = pd.DataFrame({"arc": ["A", "B"], "datetime": ["2026-11-07 00:00"] * 2})
    index = pd.MultiIndex.from_frame(table.assign(datetime=pd.to_datetime(table.datetime)))
    model = SimpleNamespace(
        sources_=pd.Series(["closure_copy_2025", "ensemble_weekend"], index=index),
        member_sources_=pd.DataFrame({"timestral_5m": ["native", "fallback_profile"]}, index=index),
        weights={"example": {"timestral_5m": 1.0}},
    )
    csv_path = tmp_path / "output_team.csv"
    csv_path.write_bytes(b"published CSV\r\n")
    return dict(csv_path=csv_path, history=history, table=table, model=model,
                model_name="ensemble", cutoff=pd.Timestamp("2026-11-05 23:00"),
                seed=42, started=time.perf_counter())


def read_log(inputs):
    submission_log.write_run_log(**inputs)
    return json.loads(inputs["csv_path"].with_name("output_team_run.json").read_text(encoding="utf-8"))


def test_revisions_sources_and_last_valid_observation(inputs):
    # Sources need to follow CSV keys even when the model emits a different order.
    inputs["model"].sources_ = inputs["model"].sources_.iloc[::-1]
    inputs["model"].member_sources_ = inputs["model"].member_sources_.iloc[::-1]
    log = read_log(inputs)
    assert log["git_commit"] == "abc123" and log["git_dirty"] is False
    assert log["last_observed_hour_per_arc"] == {"A": "2026-11-04T00:00:00", "B": None}
    assert log["csv_sha256"] == hashlib.sha256(inputs["csv_path"].read_bytes()).hexdigest()
    assert log["weights"] == inputs["model"].weights
    assert log["sources"][0]["source"] == "closure_copy_2025"
    assert log["sources"][1]["member_sources"] == {"timestral_5m": "fallback_profile"}
    assert log["model_revisions"]["chronos_2"]["revision"] == MODEL_REVISION
    assert log["model_revisions"]["timestral"]["5M"] == MODEL_REVISIONS["5M"]
    assert log["model_revisions"]["timestral"]["209M"] == MODEL_REVISIONS["209M"]
    assert log["model_revisions"]["timestral_package_required_commit"] == SOURCE_REVISION
    assert log["model_revisions"]["timestral_package_installed"]["commit"] == "installed"


def test_fingerprint_tracks_only_cutoff_limited_traffic(inputs):
    original = read_log(inputs)
    inputs["history"] = inputs["history"].iloc[::-1].copy()
    inputs["history"].loc[inputs["history"].datetime.gt(inputs["cutoff"]), "debit_horaire"] = 999
    assert read_log(inputs)["traffic_fingerprint"] == original["traffic_fingerprint"]
    inputs["history"].loc[inputs["history"].datetime.lt(inputs["cutoff"]), "debit_horaire"] = 123
    assert read_log(inputs)["traffic_fingerprint"] != original["traffic_fingerprint"]


def test_no_closure_and_no_source_metadata(inputs):
    inputs["model"] = SimpleNamespace()
    log = read_log(inputs)
    assert log["closure_correction_applied"] is False
    assert all(row["source"] == "SimpleNamespace" for row in log["sources"])


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 5)])
def test_git_unavailable_is_explicit_null(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(submission_log.subprocess, "run", fail)
    assert submission_log.git_value("rev-parse", "HEAD") is None


def test_installed_package_commit_and_missing_metadata(monkeypatch):
    package = SimpleNamespace(version="1.0", read_text=lambda name: json.dumps(
        {"vcs_info": {"commit_id": "actual-installed-commit"}}))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: package)
    assert submission_log.timestral_package() == {"version": "1.0", "commit": "actual-installed-commit"}
    package.read_text = lambda name: None
    assert submission_log.timestral_package() == {"version": "1.0", "commit": None}

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    assert submission_log.timestral_package() == {"version": None, "commit": None}
