"""Feature values and availability, checked independently of a fitted model."""

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data.calendar_events import calendar_table
from traffic_prediction.data.weather import VALUE_COLUMNS
from traffic_prediction.evaluate import backtest, target_window
from traffic_prediction.features import CALENDAR_FLAGS, DEFAULT_LAGS, TabularFeatures
from traffic_prediction.models.baselines import HolidayAwarePreviousWeekBaseline

CUTOFF = pd.Timestamp("2025-11-06 23:00")
KEYS = ["arc", "datetime"]
TARGETS = ["debit_horaire", "taux_occupation"]


@pytest.fixture(scope="module")
def inputs():
    calendar = calendar_table(
        "2025-10-01", "2025-11-20",
        school_holidays_path="tests/fixtures/paris_school_holidays.json",
    )
    frames = []
    for offset, arc in enumerate(("A", "B", "empty", "proxy")):
        frame = calendar.copy()
        frame["arc"] = arc
        frame["role"] = "proxy" if arc == "proxy" else "evalue"
        frame["debit_horaire"] = 100.0 + np.arange(len(frame)) + offset * 1000
        frame["taux_occupation"] = 5.0 + frame["hour"] / 10 + offset
        frame["qualite_mesure"] = "ok"
        if arc == "empty":
            frame[TARGETS] = np.nan
            frame["qualite_mesure"] = "absente"
        frame["precipitation_mm"] = frame["hour"] / 10
        frame["temperature_c"] = frame["datetime"].dt.day.astype(float)
        frame["wind_speed_kmh"] = frame["hour"].astype(float)
        frame["weather_source"] = "historical_forecast"
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    covariates = history[[*KEYS, *CALENDAR_FLAGS, *VALUE_COLUMNS, "weather_source"]].copy()
    return history, covariates


def requests(times=None, arcs=("A", "B", "empty")):
    times = target_window(CUTOFF.normalize()) if times is None else pd.DatetimeIndex(times)
    return pd.MultiIndex.from_product([arcs, times], names=KEYS).to_frame(index=False)


def test_schema_categories_cycles_and_no_input_mutation(inputs):
    history, covariates = inputs
    original = history.copy(deep=True)
    original_covariates = covariates.copy(deep=True)
    rows = requests(["2025-11-08 00:00", "2025-11-11 18:00", "2025-11-12 23:00"])
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    table = builder.transform(rows, covariates)
    assert list(table) == [*KEYS, *builder.feature_columns[1:]]
    assert table["arc"].cat.categories.tolist() == ["A", "B", "empty"]
    pd.testing.assert_frame_equal(table[KEYS].astype({"arc": object}), rows)
    for position, time in enumerate(rows.datetime):
        for name, value, period in (("hour", time.hour, 24),
                                     ("day_of_week", time.dayofweek, 7),
                                     ("month", time.month - 1, 12)):
            assert table.iloc[position][f"{name}_sin"] == pytest.approx(np.sin(2 * np.pi * value / period))
            assert table.iloc[position][f"{name}_cos"] == pytest.approx(np.cos(2 * np.pi * value / period))
    assert table.loc[table.datetime.eq("2025-11-11 18:00"), "is_holiday"].eq(1).all()
    assert all(pd.api.types.is_numeric_dtype(table[column]) for column in builder.feature_columns[1:])
    assert not {"role", "qualite_mesure", "etat_trafic", "etat_barre", "weather_source",
                "chantier_actif", "evenement_actif", *TARGETS} & set(builder.feature_columns)
    pd.testing.assert_frame_equal(history, original)
    pd.testing.assert_frame_equal(covariates, original_covariates)


def test_exact_lags_and_lagged_weather_with_shuffled_input(inputs):
    history, covariates = inputs
    rows = requests().sample(frac=1, random_state=42).reset_index(drop=True)
    table = TabularFeatures().fit(history.sample(frac=1, random_state=7), cutoff=CUTOFF).transform(
        rows, covariates.sample(frac=1, random_state=8))
    lookup = history.set_index(KEYS)
    for lag in DEFAULT_LAGS:
        keys = pd.MultiIndex.from_arrays([rows.arc, rows.datetime - pd.Timedelta(hours=lag)], names=KEYS)
        expected = lookup.reindex(keys)
        for target in TARGETS:
            np.testing.assert_allclose(table[f"{target}_lag_{lag}h"], expected[target], equal_nan=True)
    keys = pd.MultiIndex.from_arrays([rows.arc, rows.datetime - pd.Timedelta(days=7)], names=KEYS)
    for column in VALUE_COLUMNS:
        np.testing.assert_allclose(table[f"{column}_lag_168h"], lookup.reindex(keys)[column])


@pytest.mark.parametrize("bad", ["invalide", "absente", "partielle", "missing", "infinite", "negative", "occupancy"])
def test_bad_measurements_are_not_lags_or_profile_members(inputs, bad):
    history, covariates = inputs
    history = history.copy()
    source = pd.Timestamp("2025-11-02 08:00")
    selected = history.arc.eq("A") & history.datetime.eq(source)
    if bad in ("invalide", "absente", "partielle"):
        history.loc[selected, "qualite_mesure"] = bad
    else:
        column, value = {"missing": ("debit_horaire", np.nan),
                         "infinite": ("debit_horaire", np.inf),
                         "negative": ("debit_horaire", -1),
                         "occupancy": ("taux_occupation", 101)}[bad]
        history.loc[selected, column] = value
    row = requests([source + pd.Timedelta(days=7)], arcs=("A",))
    result = TabularFeatures().fit(history, cutoff=CUTOFF).transform(row, covariates).iloc[0]
    assert pd.isna(result.debit_horaire_lag_168h)
    assert pd.isna(result.taux_occupation_lag_168h)
    expected = history.loc[history.arc.eq("A") & history.datetime.le(CUTOFF)
                           & history.hour.eq(8) & (history.day_of_week.eq(6) | history.is_holiday)
                           & ~selected, "debit_horaire"].mean()
    assert result.debit_horaire_profile == pytest.approx(expected)


def test_missing_hour_is_not_replaced_by_adjacent_row(inputs):
    history, covariates = inputs
    source = pd.Timestamp("2025-11-01 08:00")
    history = history.loc[~(history.arc.eq("A") & history.datetime.eq(source))]
    rows = requests([source + pd.Timedelta(days=7)], arcs=("A",))
    table = TabularFeatures().fit(history, cutoff=CUTOFF).transform(rows, covariates)
    assert table.debit_horaire_lag_168h.isna().all()
    assert table.precipitation_mm_lag_168h.isna().all()


@pytest.mark.parametrize("time", ["2025-10-01 00:00", "2025-10-20 08:00", "2025-11-01 08:00", "2025-11-11 08:00"])
def test_profile_matches_direct_past_only_calculation(inputs, time):
    history, covariates = inputs
    rows = requests([time], arcs=("A",))
    result = TabularFeatures().fit(history, cutoff=CUTOFF).transform(rows, covariates).iloc[0]
    time = pd.Timestamp(time)
    holiday = covariates.loc[covariates.arc.eq("A") & covariates.datetime.eq(time), "is_holiday"].item()
    wanted_day = 6 if holiday else time.dayofweek
    source_day = history.datetime.dt.dayofweek.where(~history.is_holiday, 6)
    available = time - pd.Timedelta(days=7) if time <= CUTOFF else CUTOFF
    selected = history.loc[history.arc.eq("A") & history.datetime.le(available)
                           & source_day.eq(wanted_day)
                           & history.datetime.dt.hour.eq(time.hour)]
    assert result.profile_count == len(selected)
    for target in TARGETS:
        expected = selected[target].mean()
        assert (pd.isna(result[f"{target}_profile"]) if pd.isna(expected)
                else result[f"{target}_profile"] == pytest.approx(expected))


def test_no_future_traffic_or_weather_can_influence_features(inputs):
    history, covariates = inputs
    rows = requests()
    expected = TabularFeatures().fit(history, cutoff=CUTOFF).transform(rows, covariates)
    poisoned = history.copy()
    future = poisoned.datetime.gt(CUTOFF)
    poisoned.loc[future, TARGETS + VALUE_COLUMNS] = 999999.0
    poisoned.loc[future, "qualite_mesure"] = "invalide"
    poisoned_covariates = covariates.copy()
    poisoned_covariates.loc[poisoned_covariates.datetime.gt(CUTOFF), VALUE_COLUMNS] = 999999.0
    actual = TabularFeatures().fit(poisoned, cutoff=CUTOFF).transform(rows, poisoned_covariates)
    pd.testing.assert_frame_equal(actual, expected)
    truncated = history.loc[history.datetime.le(CUTOFF)]
    pd.testing.assert_frame_equal(
        TabularFeatures().fit(truncated, cutoff=CUTOFF).transform(rows, covariates), expected)


def test_training_profiles_exclude_self_and_later_training_labels(inputs):
    history, covariates = inputs
    time = pd.Timestamp("2025-10-20 08:00")
    rows = requests([time], arcs=("A",))
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    expected = builder.transform(rows, covariates)
    poisoned = history.copy()
    poisoned.loc[poisoned.datetime.gt(time - pd.Timedelta(days=7)), TARGETS + VALUE_COLUMNS] = 9000.0
    actual = TabularFeatures().fit(poisoned, cutoff=CUTOFF).transform(rows, covariates)
    pd.testing.assert_frame_equal(actual, expected)


def test_proxies_do_not_influence_any_feature(inputs):
    history, covariates = inputs
    expected = TabularFeatures().fit(history, cutoff=CUTOFF).transform(requests(), covariates)
    history = history.copy()
    history.loc[history.role.eq("proxy"), TARGETS + VALUE_COLUMNS] = 1e6
    pd.testing.assert_frame_equal(
        TabularFeatures().fit(history, cutoff=CUTOFF).transform(requests(), covariates), expected)


def test_current_measurements_in_requested_rows_are_never_features(inputs):
    history, covariates = inputs
    rows = history.loc[history.arc.eq("A") & history.datetime.gt(CUTOFF)].copy()
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    expected = builder.transform(rows, covariates)
    rows[TARGETS] = -12345
    pd.testing.assert_frame_equal(builder.transform(rows, covariates), expected)


def test_short_or_empty_valid_history_is_explicitly_missing(inputs):
    history, covariates = inputs
    history = history.loc[history.datetime.between("2025-11-06", CUTOFF)]
    table = TabularFeatures().fit(history, cutoff=CUTOFF).transform(requests(), covariates)
    assert table.filter(like="_lag_").isna().all().all()
    assert table.loc[table.arc.eq("empty"), ["debit_horaire_profile", "taux_occupation_profile"]].isna().all().all()
    assert table.loc[table.arc.eq("empty"), "profile_count"].eq(0).all()


def test_refit_discards_later_window_state(inputs):
    history, covariates = inputs
    old = CUTOFF - pd.Timedelta(days=7)
    rows = requests(["2025-11-01 10:00"])
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    builder.fit(history, cutoff=old)
    expected = TabularFeatures().fit(history, cutoff=old).transform(rows, covariates)
    pd.testing.assert_frame_equal(builder.transform(rows, covariates), expected)


def test_source_hours_after_cutoff_remain_missing_even_at_long_horizon(inputs):
    history, covariates = inputs
    rows = requests([CUTOFF + pd.Timedelta(hours=169)])
    table = TabularFeatures().fit(history, cutoff=CUTOFF).transform(rows, covariates)
    assert table.debit_horaire_lag_168h.isna().all()
    assert table.temperature_c_lag_168h.isna().all()


def test_known_weather_requires_issuance_and_masks_late_or_unknown_values(inputs):
    history, covariates = inputs
    rows = requests(["2025-11-11 08:00", "2025-11-11 09:00", "2025-11-11 10:00"], arcs=("A",))
    builder = TabularFeatures(weather="known").fit(history, cutoff=CUTOFF)
    with pytest.raises(ValueError, match="weather_available_at"):
        builder.transform(rows, covariates)
    covariates = covariates.assign(weather_available_at=CUTOFF)
    covariates.loc[covariates.datetime.eq("2025-11-11 09:00"), "weather_available_at"] = CUTOFF + pd.Timedelta(hours=1)
    covariates.loc[covariates.datetime.eq("2025-11-11 10:00"), "weather_available_at"] = pd.NaT
    table = builder.transform(rows, covariates)
    assert table.temperature_c.iloc[0] == 11
    assert table[VALUE_COLUMNS].iloc[1:].isna().all().all()
    poisoned = covariates.copy()
    unavailable = poisoned.weather_available_at.gt(CUTOFF) | poisoned.weather_available_at.isna()
    poisoned.loc[unavailable, VALUE_COLUMNS] = 1e9
    pd.testing.assert_frame_equal(builder.transform(rows, poisoned), table)


def test_training_weather_also_obeys_row_availability(inputs):
    history, covariates = inputs
    time = pd.Timestamp("2025-10-20 08:00")
    rows = requests([time], arcs=("A",))
    builder = TabularFeatures(weather="known").fit(history, cutoff=CUTOFF)
    covariates = covariates.assign(weather_available_at=time)
    assert builder.transform(rows, covariates)[VALUE_COLUMNS].isna().all().all()
    covariates["weather_available_at"] = time - pd.Timedelta(days=7)
    assert builder.transform(rows, covariates).temperature_c.item() == 20


@pytest.mark.parametrize("lags", [(), (24,), (167,), (168, 168), (168.0,), (True,)])
def test_unsafe_lags_rejected(lags):
    with pytest.raises(ValueError, match="at least 168"):
        TabularFeatures(lags)


@pytest.mark.parametrize("cutoff", ["NaT", "2025-11-06T23:00Z", "2025-11-06 23:01"])
def test_bad_cutoff_rejected(inputs, cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        TabularFeatures().fit(inputs[0], cutoff=cutoff)


@pytest.mark.parametrize("problem", ["duplicate_history", "duplicate_rows", "duplicate_covariates", "calendar_gap", "proxy", "timezone"])
def test_invalid_input_rejected(inputs, problem):
    history, covariates = (frame.copy() for frame in inputs)
    rows = requests()
    if problem == "duplicate_history":
        history = pd.concat([history, history.iloc[:1]])
    elif problem == "duplicate_rows":
        rows = pd.concat([rows, rows.iloc[:1]])
    elif problem == "duplicate_covariates":
        covariates = pd.concat([covariates, covariates.iloc[:1]])
    elif problem == "calendar_gap":
        covariates = covariates.loc[~covariates.datetime.eq(rows.datetime.iloc[0])]
    elif problem == "proxy":
        rows["arc"] = "proxy"
    else:
        rows["datetime"] = rows.datetime.dt.tz_localize("Europe/Paris")
    with pytest.raises(ValueError):
        TabularFeatures().fit(history, cutoff=CUTOFF).transform(rows, covariates)


def test_empty_requests_and_timestamp_resolutions(inputs):
    history, covariates = (frame.copy() for frame in inputs)
    rows = requests()
    history["datetime"] = history.datetime.astype("datetime64[us]")
    covariates["datetime"] = covariates.datetime.astype("datetime64[ms]")
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    assert len(builder.transform(rows, covariates)) == len(rows)
    assert builder.transform(rows.iloc[:0], covariates).empty


def test_unfitted_transform_rejected(inputs):
    with pytest.raises(ValueError, match="fit"):
        TabularFeatures().transform(requests(), inputs[1])


def test_nullable_measurements_and_missing_keys(inputs):
    history, covariates = (frame.copy() for frame in inputs)
    history[TARGETS + VALUE_COLUMNS] = history[TARGETS + VALUE_COLUMNS].astype("Float64")
    history["is_holiday"] = history.is_holiday.astype("boolean")
    builder = TabularFeatures().fit(history, cutoff=CUTOFF)
    table = builder.transform(requests(), covariates)
    assert all(pd.api.types.is_numeric_dtype(table[column]) for column in builder.feature_columns[1:])
    with pytest.raises(ValueError, match="missing columns"):
        builder.transform(requests().drop(columns="arc"), covariates)


def test_shared_backtest_can_build_training_and_forecast_features(inputs):
    history, covariates = inputs
    calendar = history.loc[history.arc.eq("A"), ["datetime", "day_of_week", "is_holiday"]]

    class FeatureSpy:
        tables = []

        def predict(self, available_history, times):
            builder = TabularFeatures().fit(available_history, cutoff=available_history.datetime.max())
            training = available_history.loc[available_history.role.eq("evalue"), KEYS]
            train_features = builder.transform(training, covariates)
            forecast_features = builder.transform(requests(times), covariates)
            assert len(train_features) == len(training)
            assert forecast_features.datetime.min() > builder.cutoff_
            assert list(train_features) == list(forecast_features)
            self.tables.append(forecast_features)
            return HolidayAwarePreviousWeekBaseline(calendar).predict(available_history, times)

    model = FeatureSpy()
    backtest(model, history, ["2025-11-06", "2025-11-13"], experiments_path=None)
    assert [len(table) for table in model.tables] == [360, 360]
