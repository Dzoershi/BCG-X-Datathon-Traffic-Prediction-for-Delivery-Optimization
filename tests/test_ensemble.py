"""Day-kind weighted ensemble (#32) on stub members, without model weights or data."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import predict
from traffic_prediction.data.validate import SUBMISSION_ARCS, validate_submission
from traffic_prediction.evaluate import TARGETS
from traffic_prediction.models.ensemble import DAY_KINDS, E5_WEIGHTS, WeightedEnsemble, day_kinds

# Saturday 7 to Wednesday 11 November 2026: weekend, two working days, then a holiday.
TIMES = pd.date_range("2026-11-07", "2026-11-11 23:00", freq="h")


@pytest.fixture
def calendar():
    times = pd.date_range("2026-10-01", "2026-11-11 23:00", freq="h")
    return pd.DataFrame({"datetime": times, "is_holiday": times.normalize() == pd.Timestamp("2026-11-11")})


@pytest.fixture
def history():
    times = pd.date_range("2026-10-01", "2026-11-05 23:00", freq="h")
    return pd.concat([pd.DataFrame(dict(arc=arc, datetime=times, role="evalue", qualite_mesure="ok",
                                        debit_horaire=100.0, taux_occupation=10.0))
                      for arc in SUBMISSION_ARCS], ignore_index=True)


class Constant:
    """Stub member: the same flow and occupancy at every requested hour, in reverse order."""

    def __init__(self, flow, occupancy):
        self.flow, self.occupancy, self.calls = flow, occupancy, 0

    def predict(self, history, times):
        self.calls += 1
        index = pd.MultiIndex.from_product([SUBMISSION_ARCS[::-1], times[::-1]], names=["arc", "datetime"])
        table = index.to_frame(index=False)
        table["debit_horaire"], table["taux_occupation"] = self.flow, self.occupancy
        self.sources_ = pd.Series("native", index=index)
        return table


def weights(flow, occupancy):
    return {"debit_horaire": {kind: flow[kind] for kind in DAY_KINDS},
            "taux_occupation": {kind: occupancy[kind] for kind in DAY_KINDS}}


def test_e5_weights_are_complete_and_sum_to_one():
    assert set(E5_WEIGHTS) == set(TARGETS)
    for by_kind in E5_WEIGHTS.values():
        assert set(by_kind) == set(DAY_KINDS)
        for vector in by_kind.values():
            assert set(vector) <= {"baseline_holiday_aware", "timestral_5m", "lightgbm_raw"}
            assert np.isclose(sum(vector.values()), 1) and min(vector.values()) >= 0


def test_day_kinds_of_the_submission_week(calendar):
    kinds = pd.Series(day_kinds(TIMES, calendar), index=TIMES).groupby(TIMES.normalize()).unique()
    assert [list(value) for value in kinds] == [["weekend"], ["weekend"], ["ouvre"], ["ouvre"], ["ferie"]]
    with pytest.raises(ValueError, match="cover"):
        day_kinds(TIMES, calendar.iloc[:-1])


def test_weights_per_variable_and_day_kind(history, calendar):
    members = {"a": Constant(100.0, 10.0), "b": Constant(200.0, 20.0), "unused": Constant(0.0, 0.0)}
    model = WeightedEnsemble(members, weights(
        {"ouvre": {"a": 1.0}, "weekend": {"a": 0.5, "b": 0.5}, "ferie": {"b": 1.0, "unused": 0.0}},
        {"ouvre": {"b": 1.0}, "weekend": {"a": 0.25, "b": 0.75}, "ferie": {"a": 1.0}},
    ), calendar)
    table = model.predict(history, TIMES).set_index("datetime")
    assert len(table) == 3 * len(TIMES)
    day = table.index.normalize()
    expected = {"2026-11-07": (150.0, 17.5), "2026-11-09": (100.0, 20.0), "2026-11-11": (200.0, 10.0)}
    for date, (flow, occupancy) in expected.items():
        rows = table.loc[day == pd.Timestamp(date)]
        assert rows["debit_horaire"].eq(flow).all() and rows["taux_occupation"].eq(occupancy).all()
    assert members["unused"].calls == 0, "a member without positive weight must not run"
    assert set(model.sources_) == {"ensemble_weekend", "ensemble_ouvre", "ensemble_ferie"}
    assert list(model.member_sources_.columns) == ["a", "b"]
    assert model.member_sources_.eq("native").all().all()


def test_invalid_weights_are_refused(calendar):
    members = {"a": Constant(1.0, 1.0), "b": Constant(1.0, 1.0)}
    full = {kind: {"a": 1.0} for kind in DAY_KINDS}
    for bad, message in [
        ({"debit_horaire": full}, "every variable"),
        ({"debit_horaire": {"ouvre": {"a": 1.0}}, "taux_occupation": full}, "every day kind"),
        (weights({**full, "ferie": {"a": 0.6, "b": 0.6}}, full), "sum to 1"),
        (weights({**full, "ferie": {"a": 1.5, "b": -0.5}}, full), "nonnegative"),
        (weights({**full, "ferie": {"c": 1.0}}, full), "unknown members"),
    ]:
        with pytest.raises(ValueError, match=message):
            WeightedEnsemble(members, bad, calendar)


def test_members_must_forecast_the_same_rows(history, calendar):
    class Short(Constant):
        def predict(self, history, times):
            return super().predict(history, times).iloc[1:]

    model = WeightedEnsemble({"a": Constant(1.0, 1.0), "b": Short(1.0, 1.0)}, weights(
        {kind: {"a": 0.5, "b": 0.5} for kind in DAY_KINDS}, {kind: {"a": 1.0} for kind in DAY_KINDS}), calendar)
    with pytest.raises(ValueError, match="differ"):
        model.predict(history, TIMES)


def test_member_failure_is_not_hidden(history, calendar):
    class Broken(Constant):
        def predict(self, history, times):
            raise RuntimeError("weights not installed")

    model = WeightedEnsemble({"a": Constant(1.0, 1.0), "b": Broken(1.0, 1.0)}, weights(
        {kind: {"a": 0.5, "b": 0.5} for kind in DAY_KINDS}, {kind: {"a": 1.0} for kind in DAY_KINDS}), calendar)
    with pytest.raises(RuntimeError, match="not installed"):
        model.predict(history, TIMES)


def test_ensemble_goes_through_the_submission_contract(history, calendar):
    model = WeightedEnsemble({"a": Constant(-50.0, 120.0), "b": Constant(10.0, 5.0)}, weights(
        {kind: {"a": 0.5, "b": 0.5} for kind in DAY_KINDS}, {kind: {"a": 1.0} for kind in DAY_KINDS}), calendar)
    table = predict.build_submission(history, model)
    validate_submission(table)
    assert table["debit_horaire"].eq(0.0).all() and table["taux_occupation"].eq(100.0).all()


def test_submitted_model_is_the_ensemble(calendar):
    covariates = pd.DataFrame({"datetime": calendar["datetime"]})
    model = predict.e5_ensemble(calendar, covariates)
    assert model.weights is E5_WEIGHTS
    assert set(model.members) == {"baseline_holiday_aware", "timestral_5m", "lightgbm_raw"}
    assert model.members["timestral_5m"].model_size == "5M"
    assert model.members["lightgbm_raw"].parameters["n_jobs"] == 1
