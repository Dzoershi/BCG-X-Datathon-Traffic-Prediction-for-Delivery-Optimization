"""Common truth, pooled errors, source attribution and chronological outage controls."""

import json

import numpy as np
import pandas as pd
import pytest

from traffic_prediction import evaluate, validation
from traffic_prediction.models.proxy import ProxyOutageForecaster

CUTOFF = pd.Timestamp("2025-11-06")


@pytest.fixture
def inputs():
    times = pd.date_range("2025-08-01", "2025-11-12 23:00", freq="h")
    proxy_q = 10 + times.hour.to_numpy() + np.arange(len(times)) / 2400
    proxy_k = proxy_q / 10
    frames = []
    for arc, role, q, k in [("A", "evalue", 2 * proxy_q + 10, proxy_k / 2 + 2.5),
                             ("P", "proxy", proxy_q, proxy_k)]:
        frames.append(pd.DataFrame({"arc": arc, "role": role, "datetime": times, "qualite_mesure": "ok",
                                    "debit_horaire": q, "taux_occupation": k}))
    calendar = pd.DataFrame({"datetime": times, "day_of_week": times.dayofweek,
                             "is_holiday": times.normalize() == pd.Timestamp("2025-11-11")})
    return pd.concat(frames, ignore_index=True), calendar


class Constant:
    def predict(self, history, times):
        arcs = history.loc[history.role.eq("evalue"), "arc"].unique()
        return pd.MultiIndex.from_product([arcs, times], names=evaluate.KEYS).to_frame(index=False).assign(
            debit_horaire=20.0, taux_occupation=5.0)


def hourly(inputs):
    data, _ = inputs
    return validation.run_candidates({"first": Constant(), "second": Constant()}, data, [CUTOFF], "development")


def test_target_validity_keeps_good_partner_but_never_proxy_truth():
    frame = pd.DataFrame({"debit_horaire": pd.Series([10, None, 5, 0, 9, 8], dtype="Float64"),
                          "taux_occupation": pd.Series([None, 4, None, 3, 2, 1], dtype="Float64"),
                          "qualite_mesure": ["partielle", "invalide", "invalide", "ok", "invalide", "ok"],
                          "debit_horaire_source": ["observed"] * 5 + ["proxy_estimate"]})
    assert evaluate.valid_truth(frame, "debit_horaire").tolist() == [True, False, True, False, False, False]
    assert evaluate.valid_truth(frame, "taux_occupation").tolist() == [False, True, False, True, False, True]


def test_backtest_publishes_pooled_errors_and_hourly_sources(inputs):
    data, _ = inputs
    result = evaluate.backtest(Constant(), data, [CUTOFF], experiments_path=None)
    assert len(result.hourly) == 240
    assert result.hourly.is_fallback.isna().all()  # Unknown instrumentation is not called native.
    assert set(result.hourly.day_kind) == {"working", "weekend", "holiday"}
    for variable in evaluate.TARGETS:
        selected = result.hourly.query("variable == @variable and truth_valid")
        expected = np.sqrt(np.mean((selected.prediction - selected.truth) ** 2))
        row = result.summary.query("variable == @variable and subset == 'all'").iloc[0]
        assert row.pooled_rmse == pytest.approx(expected)
        assert row.macro_rmse == row.rmse


def test_partial_truth_is_scored_per_target_with_paired_compatibility(inputs):
    data, _ = inputs
    at = evaluate.target_window(CUTOFF)[0]
    selected = data.arc.eq("A") & data.datetime.eq(at)
    data.loc[selected, "qualite_mesure"] = "partielle"
    data.loc[selected, "debit_horaire"] = np.nan
    independent = evaluate.backtest(Constant(), data, [CUTOFF], experiments_path=None)
    paired = evaluate.backtest(Constant(), data, [CUTOFF], experiments_path=None, truth_policy="paired")
    assert independent.hourly.query("variable == 'taux_occupation'").truth_valid.sum() == 120
    assert paired.hourly.query("variable == 'taux_occupation'").truth_valid.sum() == 119
    assert independent.hourly.query("variable == 'debit_horaire'").truth_valid.sum() == 119


def test_pooled_and_macro_rankings_can_reverse_with_unequal_coverage(inputs, tmp_path):
    table = hourly(inputs)
    other = table.assign(arc="B")
    table = pd.concat([table, other], ignore_index=True)
    table["truth"] = 0.0
    table["truth_valid"] = table.arc.eq("B") | table.datetime.eq(evaluate.target_window(CUTOFF)[0])
    table["prediction"] = np.where(table.model.eq("first"), np.where(table.arc.eq("A"), 0, 4),
                                    np.where(table.arc.eq("A"), 6, 1))
    validation.save_scores(table, tmp_path)
    summary = pd.read_csv(tmp_path / "summary.csv").query("variable == 'debit_horaire'").set_index("model")
    assert summary.loc["first", "macro_rmse"] < summary.loc["second", "macro_rmse"]
    assert summary.loc["first", "pooled_rmse"] > summary.loc["second", "pooled_rmse"]
    assert set(summary.n_expected) == {240} and set(summary.n_scored) == {121}


@pytest.mark.parametrize("fault", ["missing_row", "different_truth", "different_mask", "nan_prediction",
                                    "proxy_truth", "out_of_bounds", "duplicate", "shared_missing_hour", "final_test"])
def test_comparisons_reject_alignment_and_truth_shortcuts(inputs, fault):
    table = hourly(inputs)
    first = table.model.eq("first")
    at = table.index[first][0]
    if fault == "missing_row": table = table.drop(index=at)
    elif fault == "different_truth": table.loc[at, "truth"] += 1
    elif fault == "different_mask": table.loc[at, "truth_valid"] = False
    elif fault == "nan_prediction": table.loc[at, "prediction"] = np.nan
    elif fault == "proxy_truth": table.loc[:, "truth_source"] = "proxy_estimate"
    elif fault == "out_of_bounds": table.loc[at, "prediction"] = -1
    elif fault == "duplicate": table = pd.concat([table, table.iloc[[at]]], ignore_index=True)
    elif fault == "shared_missing_hour": table = table.loc[table.datetime.ne(table.datetime.min())]
    elif fault == "final_test": table["datetime"] += pd.Timedelta(days=365)
    with pytest.raises(ValueError): validation.check_alignment(table)


def test_missing_truth_keeps_expected_grid_and_separate_source_counts(inputs):
    table = hourly(inputs)
    table.loc[table.datetime.eq(table.datetime.min()), ["truth_valid", "truth"]] = [False, np.nan]
    table["is_fallback"] = pd.array([True] * len(table), dtype="boolean")
    table.loc[table.datetime.eq(table.datetime.max()), "is_fallback"] = pd.NA
    report = validation.metrics(table, validation.GROUPS["summary"])
    assert set(report.n_expected) == {120} and set(report.n_scored) == {119}
    assert set(report.n_ignored) == {1} and set(report.n_fallback) == {118}
    assert set(report.n_source_unknown) == {1}
    assert set(report.n_forecast_fallback) == {119} and set(report.n_forecast_unknown) == {1}


@pytest.mark.parametrize("field", ["rmse", "pooled_rmse", "macro_rmse", "n_scored", "sum_squared_error"])
def test_saved_hourly_errors_are_independently_verified_and_tampering_fails(inputs, tmp_path, field):
    table = hourly(inputs)
    table.to_parquet(tmp_path / "hourly_predictions.parquet", index=False)
    validation.save_scores(table, tmp_path)
    assert validation.verify_metrics(tmp_path)["comparison_alignment_verified"]
    summary = pd.read_csv(tmp_path / "summary.csv")
    summary.loc[0, field] += 1
    summary.to_csv(tmp_path / "summary.csv", index=False)
    with pytest.raises(ValueError, match="Verification failed"): validation.verify_metrics(tmp_path)


def test_no_truth_means_undefined_metric_never_zero(inputs, tmp_path):
    table = hourly(inputs)
    table["truth_valid"] = False
    table["truth"] = np.nan
    table.to_parquet(tmp_path / "hourly_predictions.parquet", index=False)
    validation.save_scores(table, tmp_path)
    assert pd.read_csv(tmp_path / "summary.csv").rmse.isna().all()
    validation.verify_metrics(tmp_path)


@pytest.mark.parametrize("day", ["2026-11-05", "2026-11-12"])
def test_final_test_is_rejected_before_any_model_call(inputs, day):
    data, _ = inputs
    class Forbidden:
        def predict(self, *args): raise AssertionError("Model must never be called")
    with pytest.raises(ValueError, match="final test"):
        validation.run_candidates({"forbidden": Forbidden()}, data, [pd.Timestamp(day)], "development")


def test_recent_validation_requires_complete_frozen_availability(inputs):
    data, _ = inputs
    with pytest.raises(ValueError, match="availability"):
        validation.validate_cutoffs(data.loc[data.datetime < "2025-11-10"], [CUTOFF])


@pytest.mark.parametrize("method", ["profile", "copy", "offset", "linear"])
def test_outage_hides_target_from_both_profile_and_calibration(inputs, method):
    data, calendar = inputs
    history = data.loc[data.datetime < CUTOFF + pd.Timedelta(days=1)].copy()
    original = history.copy(deep=True)
    model = ProxyOutageForecaster(calendar, "A", "P", 28, method)
    first = model.predict(history, evaluate.target_window(CUTOFF))
    poisoned = history.copy()
    hidden = poisoned.arc.eq("A") & poisoned.datetime.ge("2025-10-10")
    poisoned.loc[hidden, ["debit_horaire", "taux_occupation"]] = [999999, 99]
    second = model.predict(poisoned, evaluate.target_window(CUTOFF))
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(history, original)
    detail = json.loads(model.details_.loc[("A", evaluate.target_window(CUTOFF)[0]), "debit_horaire"])
    assert pd.Timestamp(detail["last_calibration_hour"]) < pd.Timestamp(detail["outage_start"])
    assert pd.Timestamp(detail["proxy_input_hour"]) <= pd.Timestamp(detail["cutoff"])
    if method != "profile":
        assert model.sources_.debit_horaire.str.startswith("proxy_").all()


@pytest.mark.parametrize("method", ["offset", "linear"])
def test_estimates_match_pre_outage_calibration(inputs, method):
    data, calendar = inputs
    history = data.loc[data.datetime < CUTOFF + pd.Timedelta(days=1)]
    times = evaluate.target_window(CUTOFF)
    proxy = history.query("arc == 'P'").set_index("datetime")
    model = ProxyOutageForecaster(calendar, "A", "P", 28, method)
    prediction = model.predict(history, times).set_index("datetime")
    slope = 2 if method == "linear" else 1
    intercept = 10 if method == "linear" else proxy.loc[proxy.index < "2025-10-10", "debit_horaire"].mean() + 10
    np.testing.assert_allclose(prediction.debit_horaire,
                               proxy.reindex(times - pd.Timedelta(days=7)).debit_horaire * slope + intercept)
    detail = json.loads(model.details_.iloc[0, 0])
    assert detail["slope"] == pytest.approx(slope) and detail["intercept"] == pytest.approx(intercept)


@pytest.mark.parametrize("fault", ["constant", "sparse", "missing"])
def test_proxy_failure_falls_back_per_variable_without_inventing_truth(inputs, fault):
    data, calendar = inputs
    history = data.loc[data.datetime < CUTOFF + pd.Timedelta(days=1)].copy()
    proxy = history.arc.eq("P")
    if fault == "constant": history.loc[proxy, "debit_horaire"] = 1
    elif fault == "sparse":
        mask = proxy & history.datetime.lt("2025-10-10")
        history.loc[history.loc[mask].index[19:], "debit_horaire"] = np.nan
    else: history.loc[proxy, "debit_horaire"] = np.nan
    model = ProxyOutageForecaster(calendar, "A", "P", 28, "linear")
    result = model.predict(history, evaluate.target_window(CUTOFF))
    assert model.sources_.debit_horaire.str.startswith("fallback_").all()
    assert model.sources_.taux_occupation.str.startswith("proxy_linear").all()
    assert np.isfinite(result[list(evaluate.TARGETS)]).all().all()


def test_future_proxy_changes_cannot_change_backtest_forecasts(inputs):
    data, calendar = inputs
    model = ProxyOutageForecaster(calendar, "A", "P", 28, "linear")
    first = evaluate.backtest(model, data, [CUTOFF], experiments_path=None).hourly
    poisoned = data.copy()
    poisoned.loc[poisoned.arc.eq("P") & poisoned.datetime.gt(CUTOFF + pd.Timedelta(hours=23)),
                 list(evaluate.TARGETS)] = [999999, 99]
    second = evaluate.backtest(model, poisoned, [CUTOFF], experiments_path=None).hourly
    pd.testing.assert_series_equal(first.prediction, second.prediction)


def test_calendar_only_cli_builds_native_lightgbm_and_reproducible_reports(inputs, tmp_path, monkeypatch):
    from traffic_prediction.features import CALENDAR_FLAGS
    data, calendar = inputs
    for column in CALENDAR_FLAGS:
        if column not in calendar:
            calendar[column] = False
    raw = data.rename(columns={"qualite_mesure": "legacy_quality", "debit_horaire": "q_value",
                               "taux_occupation": "k_value"}).assign(q_status="valid", k_status="valid", source_file="fixture")
    audit = tmp_path / "audit"
    audit.mkdir()
    raw.to_parquet(audit / "observations.parquet", index=False)
    for name in ["inputs", "report"]:
        (audit / f"{name}.json").write_text('{}', encoding="utf-8")
    calendar.to_parquet(tmp_path / "calendar.parquet", index=False)
    out = tmp_path / "results"
    monkeypatch.setattr(evaluate, "load_cutoffs", lambda: [CUTOFF - pd.Timedelta(days=7)])
    monkeypatch.setattr(validation, "RECENT_CUTOFF", CUTOFF)
    monkeypatch.setattr("sys.argv", ["validation", "--audit-dir", str(audit), "--calendar", str(tmp_path / "calendar.parquet"),
                                     "--output-dir", str(out), "--models", "lightgbm_no_weather", "--outage-days"])
    validation.main()
    hourly = pd.read_parquet(out / "hourly_predictions.parquet")
    assert hourly.forecast_source.eq("lightgbm").all() and not hourly.is_fallback.any()
    metadata = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert metadata["data"]["rows"] == len(data) and metadata["environment"]["lightgbm"]
    assert "All weather covariates" in metadata["weather_inputs"]
    assert validation.verify_metrics(out)["hourly_rows"] == 480


def test_ensemble_source_uses_positive_weights_and_target_specific_members():
    from types import SimpleNamespace
    times = evaluate.target_window(CUTOFF)[:2]
    index = pd.MultiIndex.from_product([["A"], times], names=evaluate.KEYS)
    model = SimpleNamespace(
        sources_=pd.Series(["ensemble_weekend"] * 2, index=index),
        member_sources_=pd.DataFrame({"baseline": ["day_type_profile"] * 2, "native": ["timestral"] * 2}, index=index),
        weights={"debit_horaire": {"weekend": {"baseline": 1.0, "native": 0.0}},
                 "taux_occupation": {"weekend": {"baseline": 0.0, "native": 1.0}}},
    )
    assert evaluate._forecast_sources(model, index, "debit_horaire").is_fallback.all()
    assert not evaluate._forecast_sources(model, index, "taux_occupation").is_fallback.any()


def test_unattributed_forecast_on_ignored_truth_is_explicit_and_verifiable(inputs, tmp_path):
    data, _ = inputs
    data.loc[data.datetime.eq(evaluate.target_window(CUTOFF)[0]), "qualite_mesure"] = "absente"
    class Incomplete(Constant):
        def predict(self, history, times):
            result = super().predict(history, times[1:])
            self.sources_ = pd.Series("native", index=pd.MultiIndex.from_frame(result[evaluate.KEYS]))
            return result
    hourly = validation.run_candidates({"partial": Incomplete()}, data, [CUTOFF], "development")
    assert hourly.loc[hourly.datetime.eq(hourly.datetime.min()), "forecast_source"].eq("unknown").all()
    hourly.to_parquet(tmp_path / "hourly_predictions.parquet", index=False)
    validation.save_scores(hourly, tmp_path)
    assert validation.verify_metrics(tmp_path)["comparison_alignment_verified"]
