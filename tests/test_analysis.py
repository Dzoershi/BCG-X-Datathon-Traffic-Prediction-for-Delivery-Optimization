"""Measured-only patterns, coverage, chronological examples and headless plot export."""
import numpy as np
import pandas as pd
import pytest

from traffic_prediction import analysis, evaluate


@pytest.fixture
def observed():
    times = pd.date_range("2025-11-08", periods=5, freq="h")
    raw = pd.DataFrame({"arc": "A", "role": "evalue", "datetime": times,
                        "source_file": ["f", "f", "f", "f", None],
                        "q_value": [10, np.nan, 0, 0, 20], "k_value": [2, 3, 0, 5, 7],
                        "q_status": ["valid", "missing", "valid", "suspect_zero", None],
                        "k_status": ["valid", "valid", "valid", "valid", None],
                        "legacy_quality": ["ok", "partielle", "ok", "invalide", "absente"]})
    calendar = pd.DataFrame({"datetime": times, "is_holiday": [False, False, False, True, False], "is_school_holiday": False})
    return raw, calendar


def test_pattern_counts_keep_zero_partial_missing_and_absent_distinct(observed):
    raw, calendar = observed
    before = raw.copy(deep=True)
    long = analysis.observations(raw, calendar)
    table = analysis.pattern_table(long, ["arc", "variable"])
    q = table.set_index("variable").loc["debit_horaire"]
    k = table.set_index("variable").loc["taux_occupation"]
    assert q.n_expected == 5 and q.n_valid == 2 and q.source_rows == 4 and q["mean"] == 5
    assert q.coverage_percent == 40 and k.n_valid == 4 and k["mean"] == 2.5
    assert long.loc[long.datetime.eq(raw.datetime.iloc[3]), "day_kind"].eq("holiday").all()
    assert long.loc[long.status.eq("absent"), "value"].isna().all()
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("fault", ["final_test", "duplicate", "calendar_gap"])
def test_unusable_snapshots_are_rejected(observed, fault):
    raw, calendar = observed
    if fault == "final_test": raw["datetime"] += pd.DateOffset(years=1)
    elif fault == "duplicate": raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    else: calendar = calendar.iloc[1:]
    with pytest.raises(ValueError): analysis.observations(raw, calendar)


@pytest.fixture
def hourly():
    times = pd.date_range("2025-10-01", "2025-11-12 23:00", freq="h")
    data = pd.DataFrame({"arc": "A", "role": "evalue", "datetime": times, "qualite_mesure": "ok",
                         "debit_horaire": 20., "taux_occupation": 2.})
    class Constant:
        def predict(self, history, future):
            return pd.DataFrame({"arc": "A", "datetime": future, "debit_horaire": 22., "taux_occupation": 5.})
    result = evaluate.backtest(Constant(), data, ["2025-11-06"], experiments_path=None).hourly
    return pd.concat([result.assign(model=name, phase="development", scenario="observed") for name in analysis.MODELS], ignore_index=True)


def test_error_profiles_and_examples_use_separate_units_and_observed_truth(hourly):
    table = analysis.errors(hourly)
    assert table.n_scored.eq(5).all() and table.n_ignored.eq(0).all()
    assert table.loc[table.variable.eq("debit_horaire"), "rmse"].eq(2).all()
    assert table.loc[table.variable.eq("taux_occupation"), "rmse"].eq(3).all()
    example, largest = analysis.examples(hourly)
    assert example["arc"] == "A" and example["flow_rmse"] == 2
    assert largest.groupby("variable").size().eq(10).all()
    poisoned = hourly.copy()
    poisoned.loc[0, "truth"] += 1
    with pytest.raises(ValueError, match="identical truth"): analysis.errors(poisoned)


def test_headless_figures_export_vector_and_raster_without_final_truth(observed, hourly, tmp_path):
    data = analysis.observations(*observed)
    tables = {"monthly": analysis.pattern_table(data, ["arc", "variable", "month"]),
              "hourly_patterns": analysis.pattern_table(data, ["arc", "variable", "day_kind", "hour"]),
              "school_patterns": analysis.pattern_table(data, ["arc", "variable", "day_kind", "hour", "is_school_holiday"]),
              "hourly_errors": analysis.errors(hourly)}
    chosen, _ = analysis.examples(hourly)
    paths = analysis.plots(data, hourly, tables, chosen, tmp_path)
    assert len(paths) == 18 and all(path.stat().st_size > 100 for path in paths)
    assert {path.suffix for path in paths} == {".svg", ".pdf", ".png"}
    assert "Observed" in (tmp_path / "05_error_example.svg").read_text(encoding="utf-8")
