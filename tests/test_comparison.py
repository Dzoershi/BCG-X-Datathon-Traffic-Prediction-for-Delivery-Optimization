"""Fixed experiment registry, outage isolation and grouped ablation arithmetic."""
import numpy as np
import pandas as pd
import pytest

from traffic_prediction import comparison, evaluate


def test_registry_pins_groups_and_checkpoints_independently():
    factories = comparison.factories(pd.DataFrame(), pd.DataFrame())
    for group in comparison.FEATURE_GROUPS:
        assert group not in factories[f"lightgbm_without_{group}"]().feature_groups
    for size in ("5M", "28M", "209M"):
        assert factories[f"timestral_{size.lower()}_calendar"]().model_size == size
    assert not factories["timestral_5m_history"]().calendar_covariates


def test_outage_wrapper_hides_only_target_history_and_never_mutates_inputs():
    times = pd.date_range("2025-10-01", "2025-11-06 23:00", freq="h")
    history = pd.concat([pd.DataFrame({"arc": arc, "datetime": times, "qualite_mesure": "ok",
                                      "debit_horaire": 10.0, "taux_occupation": 2.0}) for arc in ("A", "P")])
    before = history.copy(deep=True)
    class Spy:
        def predict(self, past, future):
            hidden = past.arc.eq("A") & past.datetime.ge("2025-10-10")
            assert past.loc[hidden, list(evaluate.TARGETS)].isna().all().all()
            assert past.loc[~hidden, "debit_horaire"].eq(10).all()
            assert past.loc[hidden, "qualite_mesure"].eq("absente").all()
            self.sources_ = pd.Series(dtype=object)
            return "predicted"
    assert comparison.SensorOutage(Spy(), "A", 28).predict(history, evaluate.target_window("2025-11-06")) == "predicted"
    pd.testing.assert_frame_equal(history, before)


def test_ablation_difference_uses_matched_phase_scenario_and_target():
    table = pd.DataFrame([{"model": name, "phase": phase, "scenario": "observed", "variable": "debit_horaire",
                           "rmse": value, "n_scored": 100} for phase in ("development", "recent_check")
                          for name, value in [("lightgbm_all", 3.0), ("lightgbm_without_lags", 4.0)]])
    result = comparison.ablation_effects(table, pd.DataFrame())
    assert result.rmse_change_when_removed.eq(1).all() and result.removed_group.eq("lags").all()


def test_final_test_rejected_before_loading_foundation_weights():
    class Forbidden:
        def _pipeline(self): raise AssertionError("Must reject before loading weights")
    data = pd.DataFrame({"datetime": pd.date_range("2024-01-01", "2026-11-30", freq="h")})
    with pytest.raises(ValueError, match="final test"):
        comparison.measure_run(Forbidden(), data, [pd.Timestamp("2026-11-05")], pd.Timestamp("2026-10-29"))
