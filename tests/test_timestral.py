"""Offline Timestral contract, tensor alignment, cutoff isolation and outage tests."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data.calendar_events import calendar_table
from traffic_prediction.evaluate import KEYS, TARGETS, backtest, target_window
from traffic_prediction.models.baselines import HolidayAwarePreviousWeekBaseline
from traffic_prediction.models.chronos import CALENDAR_COVARIATES, calendar_features
from traffic_prediction.models.timestral import MODEL_REVISIONS, TimestralForecaster

CUTOFF = pd.Timestamp("2025-11-06")
LAST = CUTOFF + pd.Timedelta(hours=23)


class ArrayTensor:
    """Only the tensor conversion interface used by the adapter; no torch needed."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakePipeline:
    def __init__(self, steps=False, invalid=None):
        self.calls, self.steps, self.invalid = [], steps, invalid

    def forecast(self, context, prediction_length, known_covariates=None, **options):
        self.calls.append((context.copy(), None if known_covariates is None else known_covariates.copy(),
                           prediction_length, options))
        median = np.repeat(np.nanmean(context, axis=-1)[..., None], prediction_length, axis=-1)
        if self.steps:
            median += np.arange(prediction_length)
        if self.invalid == "shape":
            median = median[..., :-1]
        elif self.invalid is not None:
            median[..., 0] = self.invalid
        return SimpleNamespace(median=ArrayTensor(median))


@pytest.fixture(scope="module")
def calendar():
    return calendar_table("2025-09-01", "2025-11-20", school_holidays_path=
                          Path(__file__).parent / "fixtures/paris_school_holidays.json")


@pytest.fixture
def data():
    times = pd.date_range("2025-09-01", "2025-11-20 23:00", freq="h")
    return pd.concat([
        pd.DataFrame({"arc": arc, "role": role, "datetime": times, "qualite_mesure": "ok",
                      "debit_horaire": 100.0 + offset + times.hour,
                      "taux_occupation": 5.0 + offset / 1000 + times.dayofweek / 10})
        for arc, role, offset in [("A", "evalue", 0), ("B", "evalue", 500), ("P", "proxy", 5000)]
    ], ignore_index=True)


def history(data):
    return data.loc[data.datetime.le(LAST)].copy()


@pytest.mark.parametrize("multivariate", [False, True])
@pytest.mark.parametrize("covariates", [(), CALENDAR_COVARIATES])
def test_contract_and_variable_order(data, calendar, multivariate, covariates):
    pipeline = FakePipeline()
    model = TimestralForecaster(calendar, pipeline=pipeline, multivariate=multivariate,
                               calendar_covariates=covariates)
    past = history(data)
    before = past.copy(deep=True)
    result = model.predict(past, target_window(CUTOFF))
    assert list(result) == [*KEYS, *TARGETS]
    assert len(result) == 240 and not result.duplicated(KEYS).any()
    assert set(result.arc) == {"A", "B"} and result.datetime.dt.tz is None
    assert (result[list(TARGETS)].dtypes == "float64").all()
    for arc in ["A", "B"]:
        expected = past.loc[past.arc.eq(arc), list(TARGETS)].mean().to_numpy()
        np.testing.assert_allclose(result.loc[result.arc.eq(arc), list(TARGETS)],
                                   np.broadcast_to(expected, (120, 2)))
    context, known, length, options = pipeline.calls[0]
    assert context.shape[:2] == ((2, 2) if multivariate else (4, 1))
    assert length == 144 and options == {"symmetric_averaging": True, "clip_nonnegative": True}
    if covariates:
        assert known.shape == (len(context), len(covariates), context.shape[-1] + 144)
    else:
        assert known is None
    assert model.sources_.eq("timestral").all()
    pd.testing.assert_frame_equal(past, before)


def test_keeps_last_120_hours_after_the_friday_gap(data, calendar):
    pipeline = FakePipeline(steps=True)
    result = TimestralForecaster(calendar, pipeline=pipeline).predict(history(data), target_window(CUTOFF))
    base = history(data).loc[data.arc.eq("A"), "debit_horaire"].mean()
    np.testing.assert_allclose(result.loc[result.arc.eq("A"), "debit_horaire"], base + np.arange(24, 144))


def test_missing_and_non_ok_values_stay_masked_on_a_bounded_grid(data, calendar):
    past = history(data)
    partial, absent = LAST - pd.Timedelta(hours=6), LAST - pd.Timedelta(hours=10)
    past.loc[past.arc.eq("A") & past.datetime.eq(partial), "qualite_mesure"] = "partielle"
    past = past.loc[~(past.arc.eq("A") & past.datetime.eq(absent))]
    pipeline = FakePipeline()
    TimestralForecaster(calendar, pipeline=pipeline, context_hours=240).predict(past, target_window(CUTOFF))
    context, known, _, _ = pipeline.calls[0]
    assert context.shape == (4, 1, 240)
    assert np.isnan(context[:2, :, -7]).all() and np.isnan(context[:2, :, -11]).all()
    assert np.isfinite(context[:2, :, -8]).all()
    assert known.shape == (4, 6, 384)


def test_calendar_alignment_includes_the_gap_and_holiday(data, calendar):
    pipeline = FakePipeline()
    TimestralForecaster(calendar, pipeline=pipeline, context_hours=240).predict(history(data), target_window(CUTOFF))
    _, known, _, _ = pipeline.calls[0]
    times = pd.date_range(LAST - pd.Timedelta(hours=239), target_window(CUTOFF)[-1], freq="h")
    expected = calendar_features(calendar).reindex(times)[list(CALENDAR_COVARIATES)].to_numpy(dtype="float32").T
    np.testing.assert_array_equal(known[0], expected)
    holiday = times.get_loc(pd.Timestamp("2025-11-11 08:00"))
    assert known[0, CALENDAR_COVARIATES.index("is_holiday"), holiday] == 1
    assert known[0, CALENDAR_COVARIATES.index("day_type"), holiday] == 6


def test_outage_fallback_is_explicit_and_matches_baseline(data, calendar):
    past = history(data)
    past.loc[past.arc.eq("B") & past.datetime.gt(LAST - pd.Timedelta(days=28)), "qualite_mesure"] = "vide"
    pipeline = FakePipeline()
    model = TimestralForecaster(calendar, pipeline=pipeline)
    actual = model.predict(past, target_window(CUTOFF)).set_index(KEYS)
    expected = HolidayAwarePreviousWeekBaseline(calendar).predict(past, target_window(CUTOFF)).set_index(KEYS)
    assert pipeline.calls[0][0].shape[0] == 2  # Only A's two variables reach Timestral.
    assert model.sources_.loc["A"].eq("timestral").all()
    assert model.sources_.loc["B"].str.startswith("fallback_").all()
    pd.testing.assert_frame_equal(actual.loc["B"], expected.loc["B"])


def test_module_and_all_fallbacks_work_without_model_dependencies(monkeypatch, data, calendar):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "timestral", None)
    monkeypatch.delitem(sys.modules, "traffic_prediction.models.timestral", raising=False)
    module = importlib.import_module("traffic_prediction.models.timestral")
    past = history(data)
    past.loc[past.datetime.gt(LAST - pd.Timedelta(days=28)), "qualite_mesure"] = "vide"
    model = module.TimestralForecaster(calendar)
    assert len(model.predict(past, target_window(CUTOFF))) == 240
    assert model.sources_.str.startswith("fallback_").all()
    with pytest.raises(ImportError, match="pinned timestral"):
        model._pipeline()


def test_backtest_isolates_cutoffs_and_repeated_calls_reset_sources(data, calendar):
    class Spy(TimestralForecaster):
        def predict(self, past, times):
            assert past.datetime.max() == times[0] - pd.Timedelta(hours=25)
            result = super().predict(past, times)
            self.outputs.append(result.copy())
            return result

    changed = data.copy()
    changed.loc[changed.datetime.gt(LAST), list(TARGETS)] = 1e9
    outputs = []
    for frame in [data, changed]:
        model = Spy(calendar, pipeline=FakePipeline())
        model.outputs = []
        backtest(model, frame, [CUTOFF - pd.Timedelta(days=7), CUTOFF], experiments_path=None)
        assert model.sources_.eq("timestral").all()
        outputs.append(pd.concat(model.outputs, ignore_index=True))
    pd.testing.assert_frame_equal(*outputs)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, "shape"])
def test_rejects_nonfinite_or_misaligned_native_output(data, calendar, invalid):
    with pytest.raises(ValueError, match="invalid median"):
        TimestralForecaster(calendar, pipeline=FakePipeline(invalid=invalid)).predict(history(data), target_window(CUTOFF))


def test_clips_finite_forecasts_to_submission_bounds(data, calendar):
    class Extreme(FakePipeline):
        def forecast(self, context, **options):
            result = super().forecast(context, **options)
            result.median.array[:] = -50
            result.median.array[1::2] = 150
            return result
    actual = TimestralForecaster(calendar, pipeline=Extreme()).predict(history(data), target_window(CUTOFF))
    assert actual.debit_horaire.eq(0).all() and actual.taux_occupation.eq(100).all()


@pytest.mark.parametrize("options", [{"context_hours": 0}, {"context_hours": 32769},
    {"context_hours": True}, {"context_hours": 168.5}, {"model_size": "unknown"},
    {"calendar_covariates": ("debit_horaire",)}, {"calendar_covariates": ("day_type", "day_type")}])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        TimestralForecaster(**options)


@pytest.mark.parametrize("kind", ["empty", "duplicate", "timezone", "partial_hour", "reverse", "past"])
def test_rejects_invalid_target_times(data, calendar, kind):
    times = target_window(CUTOFF)
    times = {"empty": times[:0], "duplicate": times.append(times[:1]),
             "timezone": times.tz_localize("Europe/Paris"), "partial_hour": times + pd.Timedelta(minutes=1),
             "reverse": times[::-1], "past": times - pd.Timedelta(days=7)}[kind]
    with pytest.raises(ValueError):
        TimestralForecaster(calendar, pipeline=FakePipeline()).predict(history(data), times)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_rejects_incomplete_or_duplicate_calendar(data, calendar, kind):
    calendar = (calendar.loc[calendar.datetime.le(LAST)] if kind == "missing"
                else pd.concat([calendar, calendar.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="Calendar"):
        TimestralForecaster(calendar, pipeline=FakePipeline()).predict(history(data), target_window(CUTOFF))


def test_native_loader_pins_weights_and_caches_backend(monkeypatch):
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None,
                         cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "timestral", SimpleNamespace(
        TimestralForecaster=SimpleNamespace(from_pretrained=load)))
    model = TimestralForecaster(model_size="28M")
    assert model._pipeline() is model._pipeline() and len(calls) == 1
    args, options = calls[0]
    assert args == ("mistralai/Timestral-28M",)
    assert options["revision"] == MODEL_REVISIONS["28M"] and options["device"] == "cpu"
    assert options["max_context"] == 8192


def test_model_integrates_with_the_official_submission_builder(data):
    from traffic_prediction.data.validate import validate_submission
    from traffic_prediction.predict import build_submission

    data["datetime"] += pd.DateOffset(years=1)
    data["arc"] = data["arc"].map({"A": "Champs-Elysées", "B": "Convention", "P": "Saint-Pères"})
    data["role"] = "evalue"
    model = TimestralForecaster(pipeline=FakePipeline(), calendar_covariates=())
    table = build_submission(data, model)
    validate_submission(table)
    assert len(table) == 360
    assert table.datetime.min() == "2026-11-07 00:00"
    assert table.datetime.max() == "2026-11-11 23:00"
