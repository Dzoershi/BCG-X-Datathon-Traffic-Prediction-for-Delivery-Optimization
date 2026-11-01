"""Global training contract, target transforms, outage policy and native smoke tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data.calendar_events import calendar_table
from traffic_prediction.evaluate import KEYS, TARGETS, backtest, target_window
from traffic_prediction.models.baselines import HolidayAwarePreviousWeekBaseline
from traffic_prediction.train import LightGBMForecaster

CUTOFF = pd.Timestamp("2025-11-06")
LAST = CUTOFF + pd.Timedelta(hours=23)


@pytest.fixture(scope="module")
def tables():
    calendar = calendar_table("2025-09-01", "2025-11-20", school_holidays_path=
                              Path(__file__).parent / "fixtures/paris_school_holidays.json")
    frames = []
    for arc, offset, role in [("A", 0, "evalue"), ("B", 100, "evalue"),
                               ("empty", 0, "evalue"), ("proxy", 10000, "proxy")]:
        frame = calendar.copy().assign(
            arc=arc, role=role, qualite_mesure="ok",
            debit_horaire=100.0 + offset + calendar.hour.astype(float) * 10,
            taux_occupation=5.0 + offset / 100 + calendar.hour / 10,
            precipitation_mm=0.0, temperature_c=10.0, wind_speed_kmh=15.0,
        )
        if arc == "empty":
            frame[list(TARGETS)] = np.nan
            frame["qualite_mesure"] = "absente"
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    covariates = data.drop(columns=["role", "qualite_mesure", *TARGETS])
    return data, covariates


def past(tables):
    return tables[0].loc[tables[0].datetime.le(LAST)].copy()


class FakeRegressor:
    def __init__(self, mode=None):
        self.mode = mode

    def fit(self, X, y, **kwargs):
        self.X, self.y, self.fit_kwargs = X.copy(), np.array(y, copy=True), kwargs
        self.booster_ = SimpleNamespace(feature_importance=lambda **kwargs: np.arange(len(X.columns)))
        return self

    def predict(self, X):
        self.future = X.copy()
        if self.mode == "shape":
            return np.zeros((len(X), 1))
        value = self.y.mean() if self.mode is None else self.mode
        return np.full(len(X), value)


@pytest.fixture
def fake(monkeypatch):
    instances = []

    def factory(self):
        instance = FakeRegressor()
        instances.append(instance)
        return instance

    monkeypatch.setattr(LightGBMForecaster, "_new_model", factory)
    return instances


@pytest.mark.parametrize("log_flow", [False, True])
def test_two_global_models_labels_categorical_arc_and_trace(tables, fake, log_flow):
    history, covariates = past(tables), tables[1]
    before = history.copy(deep=True)
    model = LightGBMForecaster(covariates, log_flow=log_flow)
    output = model.predict(history, target_window(CUTOFF))
    assert list(output) == [*KEYS, *TARGETS] and len(output) == 360
    assert not output.duplicated(KEYS).any() and set(output.arc) == {"A", "B", "empty"}
    assert len(fake) == 2 and set(model.models_) == set(TARGETS)
    training = history.loc[history.role.eq("evalue") & history.qualite_mesure.eq("ok")]
    assert model.training_rows_ == len(training)
    for index, target in enumerate(TARGETS):
        expected = training[target].to_numpy(dtype=float)
        if target == "debit_horaire" and log_flow:
            expected = np.log1p(expected)
        np.testing.assert_array_equal(fake[index].y, expected)
        assert set(fake[index].X.arc.unique()) == {"A", "B"}
        assert isinstance(fake[index].X.arc.dtype, pd.CategoricalDtype)
        assert fake[index].X.arc.dtype == fake[index].future.arc.dtype
        assert fake[index].fit_kwargs == {"categorical_feature": ["arc"]}
        assert list(fake[index].X) == model.features_.feature_columns
        value = np.expm1(expected.mean()) if log_flow and target == "debit_horaire" else expected.mean()
        np.testing.assert_allclose(output.loc[output.arc.isin(["A", "B"]), target], value)
    assert len(model.feature_importances_) == 2 * 27
    assert model.sources_.loc[["A", "B"]].eq("lightgbm").all()
    assert model.sources_.loc["empty"].str.startswith("fallback_").all()
    pd.testing.assert_frame_equal(history, before)


def test_outage_predictions_exactly_match_explicit_baseline(tables, fake):
    history = past(tables)
    history.loc[history.arc.eq("B") & history.datetime.gt("2025-10-01"), "qualite_mesure"] = "absente"
    model = LightGBMForecaster(tables[1])
    times = target_window(CUTOFF)
    result = model.predict(history, times).set_index(KEYS)
    calendar = tables[1].loc[tables[1].arc.eq("A"), ["datetime", "day_of_week", "is_holiday"]]
    expected = HolidayAwarePreviousWeekBaseline(calendar).predict(history, times).set_index(KEYS)
    pd.testing.assert_frame_equal(result.loc[["B", "empty"]], expected.loc[["B", "empty"]])
    assert model.sources_.loc["B"].str.startswith("fallback_").all()


def test_quality_invalid_pairs_and_proxies_are_excluded_from_training(tables, fake):
    history = past(tables)
    selected = history.index[history.arc.eq("A")][:5]
    history.loc[selected[0], "qualite_mesure"] = "invalide"
    history.loc[selected[1], "debit_horaire"] = np.nan
    history.loc[selected[2], "debit_horaire"] = np.inf
    history.loc[selected[3], "debit_horaire"] = -1
    history.loc[selected[4], "taux_occupation"] = 101
    model = LightGBMForecaster(tables[1])
    model.predict(history, target_window(CUTOFF))
    assert model.training_rows_ == len(history.loc[history.arc.isin(["A", "B"])]) - 5
    assert all(np.isfinite(instance.y).all() for instance in fake)


def test_refit_replaces_models_and_all_fallback_requires_no_lightgbm(tables, fake):
    history = past(tables)
    model = LightGBMForecaster(tables[1])
    model.predict(history, target_window(CUTOFF))
    first = model.models_.copy()
    model.predict(history, target_window(CUTOFF))
    assert len(fake) == 4 and all(model.models_[target] is not first[target] for target in TARGETS)
    history.loc[history.datetime.gt("2025-10-01"), "qualite_mesure"] = "absente"
    model.predict(history, target_window(CUTOFF))
    assert len(fake) == 4 and not model.models_ and model.feature_importances_.empty
    assert model.features_ is None and model.training_rows_ == 0
    assert model.sources_.str.startswith("fallback_").all()


@pytest.mark.parametrize("mode", [np.nan, np.inf, "shape"])
def test_invalid_native_outputs_raise(tables, monkeypatch, mode):
    monkeypatch.setattr(LightGBMForecaster, "_new_model", lambda self: FakeRegressor(mode))
    with pytest.raises(ValueError, match="invalid predictions"):
        LightGBMForecaster(tables[1]).predict(past(tables), target_window(CUTOFF))


def test_inverse_log_overflow_raises(tables, monkeypatch):
    monkeypatch.setattr(LightGBMForecaster, "_new_model", lambda self: FakeRegressor(1000))
    with pytest.raises(ValueError, match="overflow"):
        LightGBMForecaster(tables[1], log_flow=True).predict(past(tables), target_window(CUTOFF))


def test_physical_prediction_bounds(tables, monkeypatch):
    calls = iter([-10.0, 120.0])
    monkeypatch.setattr(LightGBMForecaster, "_new_model", lambda self: FakeRegressor(next(calls)))
    result = LightGBMForecaster(tables[1]).predict(past(tables), target_window(CUTOFF))
    assert result.loc[result.arc.eq("A"), "debit_horaire"].eq(0).all()
    assert result.loc[result.arc.eq("A"), "taux_occupation"].eq(100).all()


@pytest.mark.parametrize("times", [[], ["2025-11-06 23:00"], ["2025-11-08 00:01"],
                                    ["2025-11-08", "2025-11-08"],
                                    ["2025-11-09", "2025-11-08"], ["2025-11-08T00:00Z"]])
def test_invalid_target_hours(tables, times):
    with pytest.raises(ValueError):
        LightGBMForecaster(tables[1]).predict(past(tables), pd.DatetimeIndex(times))


@pytest.mark.parametrize("setting", [{"n_estimators": 0}, {"max_depth": 0}, {"n_jobs": -1},
                                      {"min_child_samples": False}, {"learning_rate": 0},
                                      {"learning_rate": np.nan}, {"reg_lambda": -1}, {"lags": (24,)},
                                      {"log_flow": "yes"}])
def test_invalid_configuration(setting):
    with pytest.raises(ValueError):
        LightGBMForecaster(**setting)


def test_backtest_cuts_future_values_and_does_not_use_evaluation_labels(tables, fake):
    data, covariates = tables
    captured = []

    class Spy(LightGBMForecaster):
        def predict(self, history, times):
            assert history.datetime.max() == LAST
            output = super().predict(history, times)
            captured.append(output)
            return output

    backtest(Spy(covariates), data, [CUTOFF], experiments_path=None)
    poisoned = data.copy()
    poisoned.loc[poisoned.datetime.gt(LAST), list(TARGETS)] = [99999, 99]
    backtest(Spy(covariates), poisoned, [CUTOFF], experiments_path=None)
    pd.testing.assert_frame_equal(captured[0], captured[1])
    assert all(len(instance.X) == len(fake[0].X) for instance in fake)


def test_native_lightgbm_deterministic_fit_and_importances(tables):
    pytest.importorskip("lightgbm")
    model = LightGBMForecaster(tables[1], n_estimators=12, max_depth=3, min_child_samples=5, n_jobs=1)
    history, times = past(tables), target_window(CUTOFF)
    first = model.predict(history, times)
    second = model.predict(history, times)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first[list(TARGETS)]).all().all()
    assert model.feature_importances_.gain.ge(0).all()
    assert model.feature_importances_.split.ge(0).all()
    assert model.feature_importances_.gain.gt(0).any()
    assert not first.arc.eq("proxy").any()


def test_official_submission_contract(fake):
    from traffic_prediction.data.validate import SUBMISSION_ARCS, validate_submission
    from traffic_prediction.predict import build_submission

    calendar = calendar_table("2026-09-01", "2026-11-11", school_holidays_path=
                              Path(__file__).parent / "fixtures/paris_school_holidays.json")
    covariates = pd.concat([calendar.assign(arc=arc) for arc in SUBMISSION_ARCS], ignore_index=True)
    history = covariates.loc[covariates.datetime.le("2026-11-05 23:00")].assign(
        role="evalue", qualite_mesure="ok", debit_horaire=100.0, taux_occupation=5.0,
        precipitation_mm=0.0, temperature_c=10.0, wind_speed_kmh=15.0,
    )
    submission = build_submission(history, LightGBMForecaster(covariates))
    validate_submission(submission)
    assert len(submission) == 360 and len(fake) == 2
    assert submission.datetime.min() == "2026-11-07 00:00"
    assert submission.datetime.max() == "2026-11-11 23:00"
