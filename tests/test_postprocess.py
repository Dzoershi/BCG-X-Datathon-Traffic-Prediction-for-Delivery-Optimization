"""Champs-Elysées 11/11 closure copied from 11/11/2025 (issue #47), on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import predict
from traffic_prediction.constants import TARGET_TIMES
from traffic_prediction.data import disruptions
from traffic_prediction.data.validate import SUBMISSION_ARCS, validate_submission
from traffic_prediction.models.postprocess import closure_hours, copy_closures_from_previous_year

CHAMPS = "Champs-Elysées"
CLOSED = pd.date_range("2026-11-11 08:00", "2026-11-11 14:00", freq="h")


def events(*rows):
    columns = disruptions.EVENT_COLUMNS
    table = pd.DataFrame([dict(zip(columns, row)) for row in rows], columns=columns)
    for column in ("date_debut", "date_fin"):
        table[column] = pd.to_datetime(table[column])
    return table


CEREMONY = ("2026-11-11 07:00", "2026-11-11 14:00", CHAMPS, "ceremonie", "Cérémonie", "https://example.org")


@pytest.fixture
def history():
    """Every arc at a constant 500 / 20, except Champs-Elysées on 11 and 12 November 2025."""
    times = pd.date_range("2025-11-01", "2026-11-05 23:00", freq="h")
    frame = pd.concat([pd.DataFrame(dict(arc=arc, datetime=times, role="evalue", qualite_mesure="ok",
                                         debit_horaire=500.0, taux_occupation=20.0))
                       for arc in SUBMISSION_ARCS], ignore_index=True)
    champs = frame["arc"].eq(CHAMPS)
    on_11 = champs & frame["datetime"].dt.normalize().eq(pd.Timestamp("2025-11-11"))
    frame.loc[on_11, "debit_horaire"] = 10.0 + frame.loc[on_11, "datetime"].dt.hour
    frame.loc[on_11, "taux_occupation"] = 0.123456 * frame.loc[on_11, "datetime"].dt.hour
    on_12 = champs & frame["datetime"].dt.normalize().eq(pd.Timestamp("2025-11-12"))
    frame.loc[on_12, ["debit_horaire", "taux_occupation"]] = [999.0, 99.0]  # Same weekday as 11/11/2026.
    return frame


class Constant:
    def predict(self, history, times):
        index = pd.MultiIndex.from_product([SUBMISSION_ARCS, times], names=["arc", "datetime"])
        self.sources_ = pd.Series("model", index=index)
        table = index.to_frame(index=False)
        table["debit_horaire"], table["taux_occupation"] = 600.0, 25.0
        return table


def test_closure_hours_follow_positive_end_hour_overlap():
    found = closure_hours(events(CEREMONY), TARGET_TIMES)
    assert len(found) == 1 and found[0][0] == CHAMPS and found[0][1].equals(CLOSED)


def test_only_single_arc_ceremonies_within_the_window():
    ignored = events(
        ("2026-11-11 07:00", "2026-11-11 14:00", "toutes", "ceremonie", "X", "https://example.org"),
        ("2026-11-11 07:00", "2026-11-11 14:00", CHAMPS, "manifestation", "X", "https://example.org"),
        ("2025-11-11 07:00", "2025-11-11 14:00", CHAMPS, "ceremonie", "X", "https://example.org"),
    )
    assert closure_hours(ignored, TARGET_TIMES) == []


def test_repository_events_file_holds_the_2026_ceremony():
    table = disruptions.read_events(disruptions.DEFAULT_EVENTS_PATH, list(SUBMISSION_ARCS))
    assert [(arc, list(labels)) for arc, labels in closure_hours(table, TARGET_TIMES)] == [(CHAMPS, list(CLOSED))]


def test_copy_by_date_on_closure_hours_only(history):
    table = Constant().predict(history, TARGET_TIMES)
    replaced = copy_closures_from_previous_year(table, history, events(CEREMONY))
    assert list(replaced) == [(CHAMPS, time) for time in CLOSED]
    closed = table["arc"].eq(CHAMPS) & table["datetime"].isin(CLOSED)
    hours = table.loc[closed, "datetime"].dt.hour
    assert np.array_equal(table.loc[closed, "debit_horaire"], 10.0 + hours)  # 11/11/2025, not 12/11.
    assert np.allclose(table.loc[closed, "taux_occupation"], 0.123456 * hours)
    assert table.loc[~closed, "debit_horaire"].eq(600.0).all()
    assert table.loc[~closed, "taux_occupation"].eq(25.0).all()


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_missing_reference_hours_stop_the_submission(history, damage):
    hour = (history["arc"].eq(CHAMPS) & history["datetime"].eq(pd.Timestamp("2025-11-11 09:00")))
    history = history.loc[~hour] if damage == "missing" else history.assign(
        qualite_mesure=np.where(hour, "invalide", history["qualite_mesure"]))
    table = Constant().predict(history, TARGET_TIMES)
    with pytest.raises(ValueError, match="Closure on Champs-Elysées.*2025-11-11 09:00"):
        copy_closures_from_previous_year(table, history, events(CEREMONY))


def test_submission_marks_sources_and_rounds_copied_values(history):
    from traffic_prediction.data.disruption_review import read_review
    review = read_review()
    review["events"][0].update(status="confirmed", evidence_kind="traffic_order", direction_verified=True,
                               hours_verified=True, action="copy_prior_year", source_url="https://example.org/order")
    model = Constant()
    table = predict.build_submission(history, model, events(CEREMONY), closure_review=review)
    validate_submission(table)
    closed = table["arc"].eq(CHAMPS) & table["datetime"].isin(CLOSED.strftime("%Y-%m-%d %H:%M"))
    assert closed.sum() == len(CLOSED)
    assert table.loc[closed & table["datetime"].eq("2026-11-11 08:00"), "taux_occupation"].item() == 0.99
    assert table.loc[~closed, "debit_horaire"].eq(600.0).all()
    labels = model.sources_.loc[CHAMPS].loc[CLOSED]
    assert labels.eq("closure_copy_2025").all()
    assert model.sources_.drop(index=[(CHAMPS, time) for time in CLOSED]).eq("model").all()
    assert len(model.closure_adjustments_) == 7
    assert model.closure_adjustments_[0]["before"]["debit_horaire"] == 600
    assert model.closure_adjustments_[0]["after_before_rounding"]["debit_horaire"] == 18


def test_without_events_nothing_is_replaced(history):
    table = predict.build_submission(history, Constant())
    assert table["debit_horaire"].eq(600.0).all()
