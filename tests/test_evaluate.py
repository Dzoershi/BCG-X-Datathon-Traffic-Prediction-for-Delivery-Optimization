"""Offline backtest tests: window geometry, leakage, metrics and coverage."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.evaluate import (
    PreviousWeekModel, backtest, load_cutoffs, target_window,
)

CUTOFF = pd.Timestamp("2025-11-06")


@pytest.fixture
def data():
    times = pd.date_range("2025-10-25", "2025-11-20 23:00", freq="h")
    frames = []
    for arc, role in [("A", "evalue"), ("B", "evalue"), ("proxy", "proxy")]:
        frames.append(pd.DataFrame(dict(
            arc=arc, role=role, datetime=times, qualite_mesure="ok",
            debit_horaire=10.0, taux_occupation=20.0,
        )))
    result = pd.concat(frames, ignore_index=True)
    future = result.datetime.isin(target_window(CUTOFF))
    result.loc[future, ["debit_horaire", "taux_occupation"]] = [14.0, 26.0]
    return result


class ConstantModel:
    def predict(self, history, target_times):
        index = pd.MultiIndex.from_product(
            [history.loc[history.role.eq("evalue"), "arc"].unique(), target_times],
            names=["arc", "datetime"],
        )
        return index.to_frame(index=False).assign(debit_horaire=12.0, taux_occupation=23.0)


def run(data, model=None, cutoffs=None, **kwargs):
    return backtest(model or ConstantModel(), data, [CUTOFF] if cutoffs is None else cutoffs,
                    experiments_path=None, **kwargs)


def score(result, arc="A", variable="debit_horaire", subset="all"):
    rows = result.scores
    return rows.loc[rows.arc.eq(arc) & rows.variable.eq(variable) & rows.subset.eq(subset)].iloc[0]


def test_window_and_config():
    cutoffs = load_cutoffs()
    assert len(cutoffs) == 8
    assert any(day.year == 2026 for day in cutoffs)
    for day in cutoffs:
        times = target_window(day)
        assert len(times) == 120
        assert times[0] == day + pd.Timedelta(days=2)
        assert times[-1] == day + pd.Timedelta(days=6, hours=23)
        assert times[0].weekday() == 5
        assert times[-1].weekday() == 2
    # Keep 120 local labels even across the autumn DST change.
    assert len(target_window("2026-10-22")) == 120


@pytest.mark.parametrize("cutoff", ["2025-11-07", "2025-11-06 23:00", "2025-11-06T00:00Z", "NaT"])
def test_bad_cutoff(cutoff):
    with pytest.raises(ValueError, match="Thursday"):
        target_window(cutoff)


def test_spy_no_future_including_gap_and_independent_copies(data):
    original = data.copy(deep=True)
    days = [CUTOFF, CUTOFF + pd.Timedelta(days=7)]

    class Spy(ConstantModel):
        calls = 0

        def predict(self, history, target_times):
            day = days[self.calls]
            assert history.datetime.max() == day + pd.Timedelta(hours=23)
            assert (history.datetime < day + pd.Timedelta(days=1)).all()
            assert "proxy" in set(history.arc)
            assert target_times.equals(target_window(day))
            self.calls += 1
            history.loc[:, "debit_horaire"] = -999
            return super().predict(history, target_times)

    spy = Spy()
    run(data, spy, days)
    assert spy.calls == 2
    pd.testing.assert_frame_equal(data, original)


def test_known_metrics_alignment_and_holiday(data):
    class Shuffled(ConstantModel):
        def predict(self, history, target_times):
            return super().predict(history, target_times).sample(frac=1, random_state=3)

    result = run(data, Shuffled())
    assert set(result.scores.arc) == {"A", "B"}
    row = score(result)
    assert row.rmse == 2
    assert row.baseline_rmse == 4
    assert row.normalized_score == 0.5
    assert row.n_scored == 120 and row.n_ignored == 0
    assert score(result, variable="taux_occupation").rmse == 3
    # Saturday + Sunday + Tuesday Armistice holiday; Monday/Wednesday excluded.
    assert score(result, subset="atypical").n_expected == 72
    assert (result.per_arc.n_groups == 1).all()
    assert (result.summary.n_groups == 2).all()
    assert (result.summary.n_groups_scored == 2).all()
    assert set(result.summary.normalized_score) == {0.5}


@pytest.mark.parametrize("day,expected", [("2024-12-19", 72), ("2024-12-26", 72),
                                            ("2025-04-17", 72), ("2025-07-10", 72),
                                            ("2025-10-16", 48)])
def test_holiday_windows(data, day, expected):
    shift = pd.Timestamp(day) - CUTOFF
    data["datetime"] += shift
    assert score(run(data, cutoffs=[day]), subset="atypical").n_expected == expected


def test_missing_invalid_and_absent_arc_truth(data):
    times = target_window(CUTOFF)
    a = data.arc.eq("A")
    data = data.loc[~(a & data.datetime.eq(times[0]))].copy()  # Missing row.
    a = data.arc.eq("A")
    data.loc[a & data.datetime.eq(times[1]), "qualite_mesure"] = "invalide"
    data.loc[a & data.datetime.eq(times[2]), "debit_horaire"] = np.nan
    data.loc[a & data.datetime.eq(times[3]), "debit_horaire"] = np.inf
    data.loc[a & data.datetime.eq(times[4]), "debit_horaire"] = -1
    data.loc[a & data.datetime.eq(times[5]), "taux_occupation"] = 101
    # B has history, but no truth at all in this window.
    data = data.loc[~(data.arc.eq("B") & data.datetime.isin(times))]
    result = run(data)
    assert score(result).n_ignored == 5
    assert score(result, variable="taux_occupation").n_ignored == 3
    row = score(result, arc="B")
    assert row.n_ignored == 120 and row.n_scored == 0
    assert np.isnan(row.rmse) and np.isnan(row.normalized_score)
    summary = result.summary.query("variable == 'debit_horaire' and subset == 'all'").iloc[0]
    assert summary.n_groups == 2 and summary.n_groups_scored == 1
    assert summary.n_ignored == 125


def test_previous_week_is_a_usable_model_and_excludes_bad_source(data):
    result = run(data, PreviousWeekModel())
    assert score(result).rmse == 4
    assert score(result).normalized_score == 1
    source = target_window(CUTOFF)[0] - pd.Timedelta(days=7)
    data.loc[data.datetime.eq(source), "qualite_mesure"] = "partielle"
    result = run(data)
    assert score(result).baseline_n_scored == 119
    assert np.isnan(score(result).normalized_score)
    with pytest.raises(ValueError, match="Missing/nonfinite prediction"):
        run(data, PreviousWeekModel())


def test_zero_baseline_error_is_explicitly_undefined(data):
    data.loc[:, "debit_horaire"] = 14.0
    row = score(run(data))
    assert row.baseline_rmse == 0 and row.rmse == 2
    assert np.isnan(row.normalized_score)


def test_empty_history_does_not_hide_truth(data):
    data = data.loc[data.datetime >= target_window(CUTOFF)[0]].copy()

    class NoHistory:
        def predict(self, history, target_times):
            assert history.empty
            index = pd.MultiIndex.from_product([["A", "B"], target_times], names=["arc", "datetime"])
            return index.to_frame(index=False).assign(debit_horaire=14.0, taux_occupation=26.0)

    row = score(run(data, NoHistory()))
    assert row.rmse == 0 and row.baseline_n_scored == 0
    assert np.isnan(row.normalized_score)


@pytest.mark.parametrize("fault", ["missing_row", "nan", "infinity", "duplicate", "missing_column", "wrong_time"])
def test_bad_predictions_fail_instead_of_improving_score(data, fault):
    class Broken(ConstantModel):
        def predict(self, history, target_times):
            p = super().predict(history, target_times)
            if fault == "missing_row":
                return p.iloc[1:]
            if fault == "nan":
                p.loc[0, "debit_horaire"] = np.nan
            if fault == "infinity":
                p.loc[0, "debit_horaire"] = np.inf
            if fault == "duplicate":
                return pd.concat([p, p.iloc[:1]])
            if fault == "missing_column":
                return p.drop(columns="debit_horaire")
            if fault == "wrong_time":
                p.loc[0, "datetime"] -= pd.Timedelta(days=1)
            return p

    with pytest.raises(ValueError):
        run(data, Broken())


def test_missing_predictions_allowed_only_for_ignored_truth(data):
    times = target_window(CUTOFF)
    data.loc[data.datetime.eq(times[0]), "qualite_mesure"] = "absente"

    class Partial(ConstantModel):
        def predict(self, history, target_times):
            return super().predict(history, target_times[1:])

    assert score(run(data, Partial())).n_ignored == 1


@pytest.mark.parametrize("fault", ["duplicate", "timezone", "null_time", "fractional_hour", "role", "no_arcs"])
def test_bad_input(data, fault):
    if fault == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif fault == "timezone":
        data["datetime"] = data.datetime.dt.tz_localize("UTC")
    elif fault == "null_time":
        data.loc[0, "datetime"] = pd.NaT
    elif fault == "fractional_hour":
        data.loc[0, "datetime"] += pd.Timedelta(minutes=1)
    elif fault == "role":
        data.loc[0, "role"] = "proxy"
    else:
        data["role"] = "proxy"
    with pytest.raises(ValueError):
        run(data)


@pytest.mark.parametrize("cutoffs", [[], [CUTOFF, CUTOFF]])
def test_empty_duplicate_cutoffs(data, cutoffs):
    with pytest.raises(ValueError):
        run(data, cutoffs=cutoffs)


def test_macro_average_across_arcs_and_windows(data):
    # B has twice A's error in flow: mean RMSE is 3, pooled RMSE differs.
    data.loc[data.arc.eq("B") & data.datetime.isin(target_window(CUTOFF)), "debit_horaire"] = 16
    result = run(data)
    summary = result.summary.query("variable == 'debit_horaire' and subset == 'all'").iloc[0]
    assert summary.rmse == 3
    assert summary.normalized_score == pytest.approx((0.5 + 4 / 6) / 2)


def test_log_appends_one_strict_json_record_per_successful_run(data, tmp_path):
    path = tmp_path / "experiments.csv"
    for _ in range(2):
        backtest(ConstantModel(), data, [CUTOFF], author="tester", description='test, "quoted"',
                 experiments_path=path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["author"] == "tester"
    assert rows[0]["model"] == "ConstantModel"
    assert rows[0]["description"] == 'test, "quoted"'
    payload = json.loads(rows[0]["scores"])
    assert len(payload["scores"]) == 8
    assert payload["scores"][0]["cutoff"].startswith("2025-11-06")
    assert payload["summary"][0]["rmse"] == 2
    path.write_text("old,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Experiment CSV"):
        backtest(ConstantModel(), data, [CUTOFF], experiments_path=path)
    assert path.read_text(encoding="utf-8") == "old,header\n"


def test_default_cutoffs_are_loaded(data, monkeypatch):
    monkeypatch.setattr("traffic_prediction.evaluate.load_cutoffs", lambda: [CUTOFF])
    result = backtest(ConstantModel(), data, experiments_path=None)
    assert result.scores.cutoff.unique().tolist() == [CUTOFF]


def test_nullable_quality_missing_truth_and_json_null(data, tmp_path):
    data["qualite_mesure"] = data.qualite_mesure.astype("string")
    times = target_window(CUTOFF)
    data.loc[data.datetime.isin(times), "qualite_mesure"] = pd.NA
    path = tmp_path / "runs.csv"
    result = backtest(ConstantModel(), data, [CUTOFF], experiments_path=path)
    assert score(result).n_ignored == 120
    with path.open(encoding="utf-8", newline="") as stream:
        payload = json.loads(next(csv.DictReader(stream))["scores"])
    assert payload["scores"][0]["rmse"] is None
    assert payload["summary"][0]["n_groups_scored"] == 0


def test_local_processed_parquet_to_backtest(data, tmp_path):
    from traffic_prediction.data.build import TRAFFIC_SCHEMA
    from traffic_prediction.data.query import query_traffic
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = data.assign(iu_ac=data.arc.map({"A": 1, "B": 2, "proxy": 3}),
                       etat_trafic=1, etat_barre=3)
    path = tmp_path / "traffic.parquet"
    pq.write_table(pa.Table.from_pandas(data, schema=TRAFFIC_SCHEMA, preserve_index=False), path)
    loaded = query_traffic("SELECT * FROM traffic", path=path)
    assert score(run(loaded)).rmse == 2
