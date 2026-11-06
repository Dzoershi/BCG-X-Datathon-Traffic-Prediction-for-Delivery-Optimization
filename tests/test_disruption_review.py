"""Offline evidence, review authorization and historical-only assessment checks (#57)."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import predict
from traffic_prediction.constants import TARGET_TIMES
from traffic_prediction.data import disruption_review as review, disruptions, traffic
from traffic_prediction.models.postprocess import closure_hours

# Pinned copy of the provisional (unapproved) review, so tests do not depend on the
# delivery-day decision recorded in configs/disruptions_review.json.
PROVISIONAL_REVIEW = Path(__file__).parent / "fixtures" / "disruptions_review_provisional.json"


@pytest.fixture
def arcs():
    return [arc for arc in traffic.load_arcs() if arc.role == "evalue"]


@pytest.fixture
def event():
    return pd.DataFrame({"arc": ["Champs-Elysées"], "type": ["ceremonie"],
        "date_debut": [pd.Timestamp("2026-11-11 07:00")], "date_fin": [pd.Timestamp("2026-11-11 14:00")]})


def approved():
    data = review.read_review(PROVISIONAL_REVIEW)
    data["events"][0].update(status="confirmed", evidence_kind="traffic_order", hours_verified=True,
        direction_verified=True, action="copy_prior_year", source_url="https://example.org/official-order")
    return data


def test_default_and_missing_reviews_preserve_predictions(event):
    for data in (review.read_review(PROVISIONAL_REVIEW), {"schema_version": 1, "events": []}):
        selected, decisions = review.reviewed_events(event, data, TARGET_TIMES)
        assert selected.empty and decisions[0]["action"] == "none"
        assert len(decisions[0]["candidate_labels"]) == 7


@pytest.mark.parametrize("field,value", [("status", "provisional"), ("evidence_kind", "monument_hours"),
    ("hours_verified", False), ("direction_verified", False), ("iu_ac", 4265),
    ("upstream_node", "Av_Champs_Elysees-Berri"), ("reviewer", ""), ("rationale", ""),
    ("source_url", "file:///order.pdf"), ("checked_at", "unknown")])
def test_incomplete_or_wrong_direction_approvals_fail(event, field, value):
    data = approved()
    data["events"][0][field] = value
    with pytest.raises(ValueError, match="confirmed traffic order"):
        review.reviewed_events(event, data, TARGET_TIMES)


def test_exact_interval_match_and_duplicate_review_rejection(event):
    data = approved()
    selected, decisions = review.reviewed_events(event, data, TARGET_TIMES)
    assert len(selected) == 1 and decisions[0]["action"] == "copy_prior_year"
    data["events"][0]["date_fin"] = "2026-11-11 15:00"
    assert review.reviewed_events(event, data, TARGET_TIMES)[0].empty
    data["events"].append(copy.deepcopy(data["events"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        review.reviewed_events(event, data, TARGET_TIMES)


def test_partial_clock_hours_do_not_copy_the_previous_hour(event):
    event["date_debut"] = pd.Timestamp("2026-11-11 07:30")
    event["date_fin"] = pd.Timestamp("2026-11-11 14:15")
    labels = closure_hours(event, TARGET_TIMES)[0][1]
    assert labels[0].hour == 8 and labels[-1].hour == 15 and len(labels) == 8


def references(arcs):
    return [{"iu_ac": str(arc.iu_ac), "libelle_nd_amont": arc.upstream_node,
             "libelle_nd_aval": arc.downstream_node,
             "geo_shape": {"type": "LineString", "coordinates": [[2.3, 48.8], [2.301, 48.801]]}}
            for arc in arcs]


@pytest.mark.parametrize("damage", ["duplicate", "missing", "reverse"])
def test_official_direction_identity_is_required(arcs, damage):
    records = references(arcs)
    if damage == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif damage == "missing":
        records.pop()
    else:
        records[0]["libelle_nd_amont"], records[0]["libelle_nd_aval"] = records[0]["libelle_nd_aval"], records[0]["libelle_nd_amont"]
    with pytest.raises(ValueError, match="directed arc"):
        review.verified_lines(records, arcs)


def test_capture_and_offline_replay_preserve_hashes(tmp_path, monkeypatch, arcs):
    monkeypatch.setattr(disruptions, "fetch_json", lambda url, params:
        references(arcs) if disruptions.REFERENCE_DATASET in url else [])
    snapshot = review.capture(tmp_path, arcs)
    works, extra, sources = review.replay(snapshot, arcs)
    assert works.empty and extra.empty and len(sources) == 9
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert all(s["url"].startswith("https://") and s["retrieved_at"] for s in sources)
    def forbidden(*args, **kwargs):
        pytest.fail("Replay attempted network access")
    monkeypatch.setattr(disruptions, "fetch_json", forbidden)
    review.replay(snapshot, arcs)
    (snapshot / manifest["sources"][1]["file"]).write_text("[{}]")
    with pytest.raises(ValueError, match="checksum"):
        review.replay(snapshot, arcs)


def test_approval_decisions_and_adjusted_values_are_in_submission_log(tmp_path, event):
    from traffic_prediction.submission_log import write_run_log
    import time
    arcs = list(predict.SUBMISSION_ARCS)
    ref = pd.date_range("2025-11-11 08:00", periods=7, freq="h")
    history = pd.DataFrame({"arc": arcs[0], "datetime": ref, "role": "evalue", "qualite_mesure": "ok",
                            "debit_horaire": np.arange(7.) + 10, "taux_occupation": 1.})
    class Model:
        def predict(self, history, times):
            keys = pd.MultiIndex.from_product([arcs, times], names=["arc", "datetime"])
            self.sources_ = pd.Series("model", index=keys)
            return keys.to_frame(index=False).assign(debit_horaire=500., taux_occupation=20.)
    model = Model()
    table = predict.build_submission(history, model, event, closure_review=approved())
    path = predict.export_submission(table, "test", tmp_path)
    write_run_log(path, history=history, table=table, model=model, model_name="test", cutoff=predict.CUTOFF,
                  seed=42, started=time.perf_counter())
    data = json.loads(path.with_name("output_test_run.json").read_text(encoding="utf-8"))
    assert data["closure_correction_applied"] and len(data["closure_adjustments"]) == 7
    assert data["closure_review"][0]["review"]["source_url"] == "https://example.org/official-order"
    assert data["closure_adjustments"][0]["reference_datetime"].startswith("2025-11-11")


def test_historical_assessment_cannot_use_final_test_truth():
    with pytest.raises(ValueError, match="Final-test"):
        review.assess_history(pd.DataFrame({"datetime": [TARGET_TIMES[0]]}), pd.DataFrame())


def test_historical_copy_changes_only_seven_hours_without_using_target_truth():
    champs = "Champs-Elysées"
    times = pd.date_range("2025-11-08", periods=120, freq="h")
    reference = pd.date_range("2024-11-11 07:00", periods=9, freq="h")
    raw = pd.DataFrame({"arc": champs, "datetime": times.append(reference), "legacy_quality": "ok",
        "q_value": np.r_[np.full(120, 50.), np.full(9, 10.)], "k_value": 5.,
        "q_status": "valid", "k_status": "valid"})
    predictions = pd.concat([pd.DataFrame({"model": "baseline_holiday_aware", "cutoff": pd.Timestamp("2025-11-06"),
        "scenario": "observed", "arc": champs, "datetime": times, "variable": variable,
        "truth": truth, "truth_valid": True, "prediction": value})
        for variable, truth, value in (("debit_horaire", 50., 100.), ("taux_occupation", 5., 10.))], ignore_index=True)
    original = predictions.copy(deep=True)
    detail, scores = review.assess_history(raw, predictions)
    pd.testing.assert_frame_equal(predictions, original)
    assert detail.adjusted.sum() == 14  # Seven hours, separately for the two variables.
    assert detail.loc[detail.adjusted, "datetime"].dt.hour.min() == 8
    assert detail.loc[detail.adjusted, "reference_datetime"].max() == pd.Timestamp("2024-11-11 14:00")
    assert detail.loc[~detail.adjusted, "adjusted_prediction"].equals(detail.loc[~detail.adjusted, "prediction"])
    cell = scores.loc[scores.scope.eq("champs_closure") & scores.variable.eq("debit_horaire")]
    assert cell.set_index("policy").rmse.to_dict() == {"raw_model": 50., "copy_prior_year": 40.}
    damaged = raw.copy()
    damaged.loc[damaged.datetime.eq(times[0]), "q_status"] = "invalid"
    with pytest.raises(ValueError, match="truth/masks"):
        review.assess_history(damaged, predictions)
