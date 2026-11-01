"""Offline baseline tests on synthetic traffic: contract, leakage, fallbacks and holidays."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data.calendar_events import calendar_table
from traffic_prediction.data.validate import SUBMISSION_ARCS, SUBMISSION_COLUMNS, validate_submission
from traffic_prediction.evaluate import TARGETS, backtest, target_window
from traffic_prediction.models import baselines
from traffic_prediction.models.baselines import (
    DayTypeProfileBaseline, FourWeekMeanBaseline, HolidayAwarePreviousWeekBaseline,
    PreviousWeekBaseline, day_types,
)

SCHOOL_PATH = Path(__file__).parent / "fixtures/paris_school_holidays.json"
CUTOFF = pd.Timestamp("2025-11-06")  # Window: Saturday 8 to Wednesday 12 November 2025.
# Public holidays of the synthetic period, including Whit Monday.
HOLIDAYS = pd.to_datetime(["2025-06-09", "2025-07-14", "2025-08-15", "2025-11-01", "2025-11-11"])
MODELS = [PreviousWeekBaseline, FourWeekMeanBaseline, DayTypeProfileBaseline,
          HolidayAwarePreviousWeekBaseline]
WEEK = pd.Timedelta(weeks=1)


@pytest.fixture(scope="module")
def calendar():
    return calendar_table("2025-06-01", "2025-11-20", school_holidays_path=SCHOOL_PATH)


@pytest.fixture
def data():
    """Two evaluated arcs and a proxy; flow encodes the day type, the hour and the date."""
    times = pd.date_range("2025-06-01", "2025-11-20 23:00", freq="h")
    day = times.normalize()
    level = np.select([day.isin(HOLIDAYS), times.dayofweek == 6, times.dayofweek == 5],
                      [200, 100, 300], default=500 + 10 * times.dayofweek)
    # The date term makes every week different, so that lags can be told apart.
    flow = level + times.hour + 0.01 * (day - day[0]).days
    frames = [pd.DataFrame(dict(arc=arc, role=role, datetime=times, qualite_mesure="ok",
                                debit_horaire=flow + offset, taux_occupation=(flow + offset) / 100))
              for arc, role, offset in [("A", "evalue", 0), ("B", "evalue", 1000), ("proxy", "proxy", 5000)]]
    return pd.concat(frames, ignore_index=True)


def history_until(data, cutoff=CUTOFF):
    return data.loc[data.datetime < cutoff + pd.Timedelta(days=1)].copy()


def hide(data, rows):
    """Turn the selected rows into a sensor gap."""
    data.loc[rows, "qualite_mesure"] = "vide"
    data.loc[rows, list(TARGETS)] = np.nan


def value(data, arc, time):
    return data.set_index(["arc", "datetime"]).loc[(arc, time), list(TARGETS)].to_numpy(dtype=float)


def forecast(model, history):
    return model.predict(history, target_window(CUTOFF)).set_index(["arc", "datetime"])


@pytest.mark.parametrize("model_class", MODELS)
def test_output_contract(data, calendar, model_class):
    times = target_window(CUTOFF)
    result = model_class(calendar).predict(history_until(data), times)
    assert list(result.columns) == ["arc", "datetime", *TARGETS]
    assert len(result) == 2 * 120 and not result.duplicated(["arc", "datetime"]).any()
    assert set(result.arc) == {"A", "B"}  # Proxies are never forecast.
    assert set(result.datetime) == set(times) and result.datetime.dt.tz is None
    assert result[list(TARGETS)].notna().all().all()


@pytest.mark.parametrize("model_class", MODELS)
def test_backtest_scores_without_using_future_data(data, calendar, model_class):
    """Spy as in test_evaluate, then check that post-cutoff values cannot change forecasts."""
    class Recorder(model_class):
        def predict(self, history, target_times):
            assert history.datetime.max() == CUTOFF + pd.Timedelta(hours=23)
            self.output = super().predict(history, target_times)
            return self.output

    altered = data.copy()
    altered.loc[altered.datetime > CUTOFF + pd.Timedelta(hours=23), list(TARGETS)] *= 3
    outputs = []
    for frame in (data, altered):
        model = Recorder(calendar)
        result = backtest(model, frame, [CUTOFF], experiments_path=None)
        assert result.scores.rmse.notna().all()
        outputs.append(model.output)
    pd.testing.assert_frame_equal(*outputs)


def test_previous_week_falls_back_to_two_weeks_then_to_the_profile(data, calendar):
    times = target_window(CUTOFF)
    one_gap, two_gaps = times[10], times[11]
    hide(data, data.arc.eq("A") & data.datetime.isin([one_gap - WEEK, two_gaps - WEEK, two_gaps - 2 * WEEK]))
    history = history_until(data)
    model = PreviousWeekBaseline(calendar)
    result = forecast(model, history)
    assert model.sources_[("A", times[0])] == "previous_week"
    np.testing.assert_allclose(result.loc[("A", times[0])], value(data, "A", times[0] - WEEK))
    assert model.sources_[("A", one_gap)] == "two_weeks_before"
    np.testing.assert_allclose(result.loc[("A", one_gap)], value(data, "A", one_gap - 2 * WEEK))
    assert model.sources_[("A", two_gaps)] == "day_type_profile"
    profile = forecast(DayTypeProfileBaseline(calendar), history)
    np.testing.assert_allclose(result.loc[("A", two_gaps)], profile.loc[("A", two_gaps)])


def test_readme_backtest_baseline_handles_sensor_outages(data, calendar):
    """Valid truth still scores when weekly sources and recent arc history are absent."""
    times = target_window(CUTOFF)
    gap = times[10]  # Saturday: exercise the ordinary-day weekly fallbacks.
    hide(data, data.arc.eq("A") & data.datetime.isin([gap - WEEK, gap - 2 * WEEK]))
    # Keep B's target truth, but remove its recent source measurements.
    hide(data, data.arc.eq("B") & data.datetime.between(
        pd.Timestamp("2025-09-01"), CUTOFF + pd.Timedelta(hours=23)))
    model = HolidayAwarePreviousWeekBaseline(calendar)

    result = backtest(model, data, [CUTOFF], experiments_path=None)

    assert result.scores.rmse.notna().all()
    assert (result.scores.n_scored == result.scores.n_expected).all()
    assert model.sources_[("A", gap)] == "day_type_profile"
    assert model.sources_.loc["B"].eq("day_type_profile").all()
    # The unfilled internal reference still reports incomplete coverage.
    assert (result.scores.baseline_n_scored < result.scores.n_scored).all()
    assert result.scores.normalized_score.isna().all()


def test_four_week_mean_averages_the_available_weeks(data, calendar):
    times = target_window(CUTOFF)
    hide(data, data.arc.eq("A") & data.datetime.eq(times[5] - WEEK))
    result = forecast(FourWeekMeanBaseline(calendar), history_until(data))
    for time, weeks in [(times[0], [1, 2, 3, 4]), (times[5], [2, 3, 4])]:
        expected = np.mean([value(data, "A", time - n * WEEK) for n in weeks], axis=0)
        np.testing.assert_allclose(result.loc[("A", time)], expected)


@pytest.mark.parametrize("model_class", MODELS)
def test_arc_without_recent_history_uses_its_day_type_profile(data, calendar, model_class):
    """Like Champs-Elysées: no measurement during the two months before the cutoff."""
    hide(data, data.arc.eq("B") & data.datetime.ge(pd.Timestamp("2025-09-01")))
    model = model_class(calendar)
    result = forecast(model, history_until(data))
    assert model.sources_.loc["B"].eq("day_type_profile").all()
    assert result.loc["B"].notna().all().all()


def test_day_types_count_holidays_as_sundays(calendar):
    assert set(calendar.loc[calendar.is_holiday, "datetime"].dt.normalize()) == set(HOLIDAYS)
    times = pd.to_datetime(["2025-11-10 08:00", "2025-11-11 08:00", "2025-07-14 00:00", "2025-11-09 23:00"])
    assert day_types(times, calendar).tolist() == [0, 6, 6, 6]


def test_day_type_profile_treats_holidays_as_sundays(data, calendar):
    history = history_until(data)
    result = forecast(DayTypeProfileBaseline(calendar), history)
    a = history[history.arc.eq("A")]
    holiday = a.datetime.dt.normalize().isin(HOLIDAYS)
    sunday_like = a.datetime.dt.dayofweek.eq(6) | holiday
    monday = a.datetime.dt.dayofweek.eq(0) & ~holiday  # Holiday Mondays count as Sundays.
    for time, rows in [(pd.Timestamp("2025-11-11 08:00"), sunday_like),
                       (pd.Timestamp("2025-11-10 08:00"), monday)]:
        expected = a.loc[rows & a.datetime.dt.hour.eq(8), list(TARGETS)].mean().to_numpy()
        np.testing.assert_allclose(result.loc[("A", time)], expected)


def test_holiday_aware_uses_the_last_known_sunday_on_holidays_only(data, calendar):
    history = history_until(data)
    model = HolidayAwarePreviousWeekBaseline(calendar)
    result = forecast(model, history)
    holiday = pd.Timestamp("2025-11-11 08:00")
    assert model.sources_[("A", holiday)] == "last_sunday"
    np.testing.assert_allclose(result.loc[("A", holiday)], value(data, "A", pd.Timestamp("2025-11-02 08:00")))
    previous_week = forecast(PreviousWeekBaseline(calendar), history)
    normal = result.index.get_level_values("datetime").normalize() != pd.Timestamp("2025-11-11")
    pd.testing.assert_frame_equal(result[normal], previous_week[normal])


def test_holiday_aware_falls_back_to_the_previous_sunday_then_to_the_profile(data, calendar):
    gaps = pd.to_datetime(["2025-11-02 08:00", "2025-11-02 09:00", "2025-10-26 09:00"])
    hide(data, data.arc.eq("A") & data.datetime.isin(gaps))
    history = history_until(data)
    model = HolidayAwarePreviousWeekBaseline(calendar)
    result = forecast(model, history)
    eight, nine = pd.Timestamp("2025-11-11 08:00"), pd.Timestamp("2025-11-11 09:00")
    assert model.sources_[("A", eight)] == "sunday_before"
    np.testing.assert_allclose(result.loc[("A", eight)], value(data, "A", pd.Timestamp("2025-10-26 08:00")))
    assert model.sources_[("A", nine)] == "day_type_profile"
    profile = forecast(DayTypeProfileBaseline(calendar), history)
    np.testing.assert_allclose(result.loc[("A", nine)], profile.loc[("A", nine)])


@pytest.mark.parametrize("last_known,hour,expected", [
    ("2025-11-06 23:00", 8, "2025-11-02 08:00"),   # Thursday cutoff.
    ("2025-11-02 10:00", 9, "2025-11-02 09:00"),   # This Sunday hour is already known.
    ("2025-11-02 10:00", 15, "2025-10-26 15:00"),  # Not known yet: previous Sunday.
])
def test_last_sunday(last_known, hour, expected):
    times = pd.DatetimeIndex([pd.Timestamp("2025-11-11") + pd.Timedelta(hours=hour)])
    assert baselines._last_sunday(times, pd.Timestamp(last_known))[0] == pd.Timestamp(expected)


@pytest.mark.parametrize("model_class", MODELS)
def test_forecasts_are_clipped_to_physical_ranges(data, calendar, model_class):
    data.loc[data.arc.eq("A"), list(TARGETS)] = [-5.0, 150.0]
    result = forecast(model_class(calendar), history_until(data)).loc["A"]
    assert result.debit_horaire.eq(0).all() and result.taux_occupation.eq(100).all()


def test_no_valid_history_fails_explicitly(data, calendar):
    data["qualite_mesure"] = "vide"
    with pytest.raises(ValueError, match="Not enough valid history"):
        PreviousWeekBaseline(calendar).predict(history_until(data), target_window(CUTOFF))


def test_calendar_must_cover_history_and_targets(data, calendar):
    short = calendar[calendar.datetime < pd.Timestamp("2025-11-10")]
    with pytest.raises(ValueError, match="Calendar does not cover"):
        DayTypeProfileBaseline(short).predict(history_until(data), target_window(CUTOFF))


def test_best_baseline_fills_a_valid_official_submission():
    """360 rows in the official format, with Champs-Elysées lacking recent measurements."""
    cutoff = pd.Timestamp("2026-11-05")
    times = pd.date_range("2026-09-01", cutoff + pd.Timedelta(hours=23), freq="h")
    history = pd.concat([pd.DataFrame(dict(arc=arc, role="evalue", datetime=times, qualite_mesure="ok",
                                           debit_horaire=300.0 + times.hour, taux_occupation=5.0))
                         for arc in SUBMISSION_ARCS], ignore_index=True)
    hide(history, history.arc.eq("Champs-Elysées") & history.datetime.ge(pd.Timestamp("2026-10-01")))
    calendar = calendar_table("2026-09-01", "2026-11-11", school_holidays_path=SCHOOL_PATH)
    forecast = HolidayAwarePreviousWeekBaseline(calendar).predict(history, target_window(cutoff))
    submission = forecast.assign(datetime=forecast["datetime"].dt.strftime("%Y-%m-%d %H:%M"))
    validate_submission(submission[SUBMISSION_COLUMNS])


def test_calendar_is_read_from_the_built_parquet_by_default(data, calendar, tmp_path, monkeypatch):
    path = tmp_path / "calendar.parquet"
    calendar.to_parquet(path, index=False)
    read = baselines.load_calendar
    monkeypatch.setattr(baselines, "load_calendar", lambda: read(path))
    assert len(DayTypeProfileBaseline().predict(history_until(data), target_window(CUTOFF))) == 240
