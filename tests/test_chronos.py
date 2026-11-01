"""Offline Chronos-2 tests: a fake pipeline replaces the model, so neither torch nor chronos is needed."""

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data.calendar_events import calendar_table
from traffic_prediction.data.validate import SUBMISSION_ARCS, SUBMISSION_COLUMNS, validate_submission
from traffic_prediction.evaluate import TARGETS, backtest, target_window
from traffic_prediction.models.baselines import HolidayAwarePreviousWeekBaseline
from traffic_prediction.models.chronos import (
    CALENDAR_COVARIATES, MIN_RECENT_OK_HOURS, QUANTILES, RECENT_HOURS, Chronos2Forecaster,
    forecast_horizon, has_recent_history, weather_features,
)

SCHOOL_PATH = Path(__file__).parent / "fixtures/paris_school_holidays.json"
CUTOFF = pd.Timestamp("2025-11-06")  # Window: Saturday 8 to Wednesday 12 November 2025.
LAST = CUTOFF + pd.Timedelta(hours=23)  # Last history hour: Thursday 23:00.


class FakePipeline:
    """Stands in for Chronos2Pipeline.predict_df: same arguments and output columns.

    Every quantile is the series' mean over its context (NaN hours skipped), plus
    ``spread * (level - 0.5) ** 2``, so that the mean of the quantiles differs from
    the median. ``values`` overrides the forecast of a variable. Calls are recorded.
    """

    def __init__(self, values=None, spread=0.0):
        self.values, self.spread, self.calls = values or {}, spread, []

    def predict_df(self, df, future_df=None, id_column="item_id", timestamp_column="timestamp",
                   target="target", prediction_length=None, quantile_levels=(), freq=None, **kwargs):
        self.calls.append(dict(df=df.copy(), future_df=None if future_df is None else future_df.copy(),
                               target=target, prediction_length=prediction_length, freq=freq, **kwargs))
        parts = []
        for series, group in df.groupby(id_column, sort=False):
            times = pd.date_range(group[timestamp_column].max() + pd.Timedelta(hours=1),
                                  periods=prediction_length, freq=freq)
            for name in target:
                variable = series.rsplit("|", 1)[1] if name == "target" else name
                level = self.values.get(variable, group[name].mean())
                parts.append(pd.DataFrame({
                    id_column: series, timestamp_column: times, "target_name": name, "predictions": level,
                    **{str(q): level + self.spread * (q - 0.5) ** 2 for q in quantile_levels},
                }))
        return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def calendar():
    return calendar_table("2025-06-01", "2025-11-20", school_holidays_path=SCHOOL_PATH)


@pytest.fixture
def data():
    """Two evaluated arcs and a proxy, hourly from June to November 2025."""
    times = pd.date_range("2025-06-01", "2025-11-20 23:00", freq="h")
    flow = 300.0 + 10 * times.hour + times.dayofweek
    frames = [pd.DataFrame(dict(arc=arc, role=role, datetime=times, qualite_mesure="ok",
                                debit_horaire=flow + offset, taux_occupation=5.0 + offset / 100))
              for arc, role, offset in [("A", "evalue", 0), ("B", "evalue", 1000), ("proxy", "proxy", 5000)]]
    return pd.concat(frames, ignore_index=True)


def history_until(data, cutoff=CUTOFF):
    return data.loc[data.datetime < cutoff + pd.Timedelta(days=1)].copy()


def forecaster(calendar, pipeline=None, **options):
    return Chronos2Forecaster(calendar, pipeline=pipeline or FakePipeline(), **options)


def test_module_imports_and_predicts_without_torch_or_chronos(monkeypatch, data, calendar):
    monkeypatch.setitem(sys.modules, "torch", None)  # Any import of torch or chronos now fails.
    monkeypatch.setitem(sys.modules, "chronos", None)
    monkeypatch.delitem(sys.modules, "traffic_prediction.models.chronos", raising=False)
    module = importlib.import_module("traffic_prediction.models.chronos")
    model = module.Chronos2Forecaster(calendar, pipeline=FakePipeline())
    assert len(model.predict(history_until(data), target_window(CUTOFF))) == 240
    with pytest.raises(ImportError):
        module.Chronos2Forecaster(calendar)._pipeline()  # Only loading the weights needs them.


def test_output_contract(data, calendar):
    times = target_window(CUTOFF)
    result = forecaster(calendar).predict(history_until(data), times)
    assert list(result.columns) == ["arc", "datetime", *TARGETS]
    assert len(result) == 2 * 120 and not result.duplicated(["arc", "datetime"]).any()
    assert set(result.arc) == {"A", "B"}  # Proxies are never forecast.
    for arc in ("A", "B"):
        assert result.loc[result.arc.eq(arc), "datetime"].tolist() == list(times)
    assert result.datetime.dt.tz is None
    assert result[list(TARGETS)].notna().all().all()
    assert (result[list(TARGETS)].dtypes == "float64").all()


def test_defaults_are_the_retained_run(data, calendar):
    """Run 4: both targets jointly, the six calendar covariates, the median, no weather."""
    pipeline = FakePipeline(spread=100.0)
    model = forecaster(calendar, pipeline)
    result = model.predict(history_until(data), target_window(CUTOFF)).set_index(["arc", "datetime"])
    call = pipeline.calls[0]
    assert call["target"] == list(TARGETS) and set(call["df"]["id"]) == {"A", "B"}
    assert list(call["future_df"].columns) == ["id", "timestamp", *CALENDAR_COVARIATES]
    assert not call["cross_learning"] and call["context_length"] == 8192
    raw = model.raw_forecast_.set_index(["id", "timestamp", "target_name"])["0.5"]
    assert result.loc[("A", target_window(CUTOFF)[0]), "debit_horaire"] == raw[("A", target_window(CUTOFF)[0], "debit_horaire")]


def test_horizon_covers_the_friday_gap_then_keeps_the_120_targets(data, calendar):
    pipeline = FakePipeline()
    forecaster(calendar, pipeline, multivariate=False).predict(history_until(data), target_window(CUTOFF))
    call = pipeline.calls[0]
    assert call["target"] == ["target"]  # Univariate: one series per arc and variable.
    assert set(call["df"]["id"]) == {f"{arc}|{variable}" for arc in "AB" for variable in TARGETS}
    assert call["prediction_length"] == 144 and call["freq"] == "h"
    assert call["df"]["timestamp"].max() == LAST
    assert forecast_horizon(LAST, target_window(CUTOFF))[0] == CUTOFF + pd.Timedelta(days=1)  # Friday 00:00.
    with pytest.raises(ValueError, match="follow"):
        forecast_horizon(LAST, target_window(CUTOFF - pd.Timedelta(days=7)))


def test_backtest_uses_no_data_after_the_cutoff(data, calendar):
    """Spy as in test_evaluate: inputs stop on Thursday 23:00, and later values cannot change forecasts."""
    class Recorder(Chronos2Forecaster):
        def predict(self, history, target_times):
            assert history.datetime.max() == target_times[0] - pd.Timedelta(hours=25)
            assert "proxy" in set(history.arc)
            self.outputs.append(super().predict(history, target_times))
            return self.outputs[-1]

    days = [CUTOFF - pd.Timedelta(days=7), CUTOFF]
    altered = data.copy()
    altered.loc[altered.datetime > LAST, list(TARGETS)] *= 3
    outputs = []
    for frame in (data, altered):
        pipeline = FakePipeline()
        model = Recorder(calendar, pipeline=pipeline, calendar_covariates=CALENDAR_COVARIATES)
        model.outputs = []
        result = backtest(model, frame, days, experiments_path=None)
        assert result.scores.rmse.notna().all()
        for day, call in zip(days, pipeline.calls):
            assert call["df"]["timestamp"].max() == day + pd.Timedelta(hours=23)
            assert call["future_df"]["timestamp"].min() == day + pd.Timedelta(days=1)
        outputs.append(pd.concat(model.outputs, ignore_index=True))
    pd.testing.assert_frame_equal(*outputs)


def test_context_keeps_only_ok_hours_on_a_regular_grid(data, calendar):
    history = history_until(data)
    a = history.arc.eq("A")
    partial, absent = pd.Timestamp("2025-11-03 08:00"), pd.Timestamp("2025-11-03 09:00")
    history.loc[a & history.datetime.eq(partial), "qualite_mesure"] = "partielle"  # Values kept, not ok.
    history = history.loc[~(a & history.datetime.eq(absent))]  # Row missing from the history.
    pipeline = FakePipeline()
    forecaster(calendar, pipeline, multivariate=True).predict(history, target_window(CUTOFF))
    context = pipeline.calls[0]["df"].set_index(["id", "timestamp"])
    assert context.loc[("A", partial), list(TARGETS)].isna().all()
    assert context.loc[("A", absent), list(TARGETS)].isna().all()
    assert context.loc[("A", partial - pd.Timedelta(hours=1)), list(TARGETS)].notna().all()
    times = context.loc["A"].index
    assert times.equals(pd.date_range(times[0], LAST, freq="h", name="timestamp"))


def test_context_is_limited_to_context_hours(data, calendar):
    pipeline = FakePipeline()
    forecaster(calendar, pipeline, context_hours=1000).predict(history_until(data), target_window(CUTOFF))
    call = pipeline.calls[0]
    assert call["context_length"] == 1000
    assert call["df"].groupby("id").size().eq(1000).all()


def test_forecasts_are_clipped_to_physical_ranges(data, calendar):
    pipeline = FakePipeline(values={"debit_horaire": -50.0, "taux_occupation": 150.0})
    result = forecaster(calendar, pipeline).predict(history_until(data), target_window(CUTOFF))
    assert result.debit_horaire.eq(0).all() and result.taux_occupation.eq(100).all()


def test_arc_without_recent_history_takes_the_baseline(data, calendar):
    """Like Champs-Elysées: B has no measurement during the two months before the cutoff."""
    gap = data.arc.eq("B") & data.datetime.ge(pd.Timestamp("2025-09-01"))
    data.loc[gap, "qualite_mesure"] = "vide"
    data.loc[gap, list(TARGETS)] = np.nan
    history, times = history_until(data), target_window(CUTOFF)
    pipeline = FakePipeline()
    model = forecaster(calendar, pipeline)
    result = model.predict(history, times).set_index(["arc", "datetime"])
    assert set(pipeline.calls[0]["df"]["id"]) == {"A"}
    assert model.sources_.loc["A"].eq("chronos").all()
    assert model.sources_.loc["B"].eq("fallback_day_type_profile").all()
    baseline = HolidayAwarePreviousWeekBaseline(calendar).predict(history, times).set_index(["arc", "datetime"])
    pd.testing.assert_frame_equal(result.loc["B"], baseline.loc["B"])


@pytest.mark.parametrize("measured,expected", [(MIN_RECENT_OK_HOURS, True), (MIN_RECENT_OK_HOURS - 1, False)])
def test_recent_history_needs_one_week_of_ok_hours_in_the_last_28_days(measured, expected):
    times = pd.date_range(end=LAST, periods=RECENT_HOURS + 100, freq="h")
    index = pd.MultiIndex.from_product([["A"], times], names=["arc", "datetime"])
    values = pd.DataFrame(np.nan, index=index, columns=list(TARGETS))
    values.iloc[:100] = 1.0  # Older than 28 days: does not count.
    values.iloc[-measured:] = 1.0
    assert has_recent_history(values, LAST)["A"] == expected


def test_no_recent_history_at_all_never_calls_the_model(data, calendar):
    data.loc[data.datetime.ge(pd.Timestamp("2025-10-01")), "qualite_mesure"] = "vide"

    class Unavailable:
        def predict_df(self, *args, **kwargs):
            raise AssertionError("Chronos-2 must not be called without recent history")

    model = Chronos2Forecaster(calendar, pipeline=Unavailable())
    assert len(model.predict(history_until(data), target_window(CUTOFF))) == 240
    assert model.raw_forecast_ is None
    assert model.sources_.str.startswith("fallback_").all()


def test_calendar_covariates_are_known_on_the_whole_horizon(data, calendar):
    pipeline = FakePipeline()
    model = forecaster(calendar, pipeline, calendar_covariates=CALENDAR_COVARIATES, multivariate=True)
    model.predict(history_until(data), target_window(CUTOFF))
    call = pipeline.calls[0]
    assert call["target"] == list(TARGETS) and set(call["df"]["id"]) == {"A", "B"}
    assert set(CALENDAR_COVARIATES) <= set(call["df"].columns)  # Past values too.
    future = call["future_df"]
    assert list(future.columns) == ["id", "timestamp", *CALENDAR_COVARIATES]
    assert future.groupby("id").size().eq(144).all()
    assert (future[list(CALENDAR_COVARIATES)].dtypes == "float32").all()  # Never booleans.
    days = future.set_index(["id", "timestamp"]).loc["A"]
    holiday, monday = days.loc[pd.Timestamp("2025-11-11 08:00")], days.loc[pd.Timestamp("2025-11-10 08:00")]
    assert holiday[["is_holiday", "is_sunday_or_holiday"]].eq(1).all() and holiday["day_type"] == 6
    assert monday[["is_day_before_holiday", "is_bridge"]].eq(1).all() and monday["day_type"] == 0


def test_weather_covariates_come_from_the_given_table(data, calendar):
    times = pd.date_range("2025-06-01", "2025-11-20 23:00", freq="h")
    weather = pd.DataFrame({"datetime": times, "precipitation_mm": 0.5, "temperature_c": 12.0,
                            "wind_speed_kmh": 8.0, "source": "historical_forecast"})
    pipeline = FakePipeline()
    model = forecaster(calendar, pipeline, weather=weather, weather_covariates=("temperature_c", "rain_3h"))
    model.predict(history_until(data), target_window(CUTOFF))
    future = pipeline.calls[0]["future_df"]
    assert future["temperature_c"].eq(12).all() and future["rain_3h"].eq(1.5).all()


def test_rain_3h_sums_the_label_and_the_two_previous_hours():
    weather = pd.DataFrame({"datetime": pd.date_range("2025-11-01", periods=5, freq="h"),
                            "precipitation_mm": [1.0, 0.0, 2.0, 0.0, 0.0]})
    assert weather_features(weather)["rain_3h"].tolist() == [1.0, 1.0, 3.0, 2.0, 2.0]


def test_covariates_must_cover_the_context_and_the_horizon(data, calendar):
    short = calendar[calendar.datetime < pd.Timestamp("2025-11-10")]
    with pytest.raises(ValueError, match="does not cover"):
        forecaster(short, calendar_covariates=("is_holiday",)).predict(history_until(data), target_window(CUTOFF))


def test_point_forecast_is_the_median_or_the_mean_of_quantiles(data, calendar):
    history, times = history_until(data), target_window(CUTOFF)
    median = forecaster(calendar, FakePipeline(spread=100.0)).predict(history, times)
    mean = forecaster(calendar, FakePipeline(spread=100.0), point="mean").predict(history, times)
    shift = 100.0 * np.mean([(q - 0.5) ** 2 for q in QUANTILES])
    np.testing.assert_allclose(mean[list(TARGETS)] - median[list(TARGETS)], shift)


def test_point_by_day_type_takes_one_quantile_per_day_kind(data, calendar):
    """Window: Sat 8 and Sun 9 (weekend), Mon 10 and Wed 12 (working days), Tue 11 November (holiday)."""
    gap = data.arc.eq("B") & data.datetime.ge(pd.Timestamp("2025-09-01"))  # B takes the fallback.
    data.loc[gap, "qualite_mesure"] = "vide"
    history, times = history_until(data), target_window(CUTOFF)
    median = forecaster(calendar, FakePipeline(spread=100.0)).predict(history, times)
    by_kind = forecaster(calendar, FakePipeline(spread=100.0),
                         point_by_day_type={"ouvre": 0.5, "weekend": 0.2, "ferie": 0.1}).predict(history, times)
    shift = (by_kind[list(TARGETS)] - median[list(TARGETS)]).set_index(median["arc"] + " " + median["datetime"].dt.strftime("%a"))
    # The fake adds 100 * (level - 0.5) ** 2: 9 at 0.2, 16 at 0.1, nothing at the median.
    expected = {"Sat": 9.0, "Sun": 9.0, "Mon": 0.0, "Tue": 16.0, "Wed": 0.0}
    for day, value in expected.items():
        np.testing.assert_allclose(shift.loc[f"A {day}"], value)
        np.testing.assert_allclose(shift.loc[f"B {day}"], 0.0)  # Fallback hours are untouched.


@pytest.mark.parametrize("options", [
    dict(point="max"), dict(context_hours=0), dict(context_hours=8193),
    dict(point_by_day_type={"dimanche": 0.2}), dict(point_by_day_type={"weekend": 0.33}),
])
def test_invalid_options_fail(calendar, options):
    with pytest.raises(ValueError):
        Chronos2Forecaster(calendar, **options)


def test_fills_a_valid_official_submission():
    """360 rows in the official format; Champs-Elysées, without recent measurements, takes the fallback."""
    cutoff = pd.Timestamp("2026-11-05")
    times = pd.date_range("2026-09-01", cutoff + pd.Timedelta(hours=23), freq="h")
    history = pd.concat([pd.DataFrame(dict(arc=arc, role="evalue", datetime=times, qualite_mesure="ok",
                                           debit_horaire=300.0 + times.hour, taux_occupation=5.0))
                         for arc in SUBMISSION_ARCS], ignore_index=True)
    champs = history.arc.eq("Champs-Elysées") & history.datetime.ge(pd.Timestamp("2026-10-01"))
    history.loc[champs, "qualite_mesure"] = "vide"
    calendar = calendar_table("2026-09-01", "2026-11-11", school_holidays_path=SCHOOL_PATH)
    model = Chronos2Forecaster(calendar, pipeline=FakePipeline(),
                               calendar_covariates=CALENDAR_COVARIATES, multivariate=True)
    forecast = model.predict(history, target_window(cutoff))
    submission = forecast.assign(datetime=forecast["datetime"].dt.strftime("%Y-%m-%d %H:%M"))
    validate_submission(submission[SUBMISSION_COLUMNS])
    assert model.sources_.loc["Champs-Elysées"].str.startswith("fallback_").all()
    assert model.sources_.drop("Champs-Elysées", level="arc").eq("chronos").all()


def test_fake_pipeline_receives_only_arguments_of_the_real_one(data, calendar):
    """Runs only where chronos is installed (not in the CI)."""
    chronos = pytest.importorskip("chronos")

    class Signature:
        def predict_df(self, df, **kwargs):
            self.names = set(kwargs)
            raise LookupError

    spy = Signature()
    with pytest.raises(LookupError):
        forecaster(calendar, spy, calendar_covariates=CALENDAR_COVARIATES).predict(
            history_until(data), target_window(CUTOFF))
    assert spy.names <= set(inspect.signature(chronos.Chronos2Pipeline.predict_df).parameters)
