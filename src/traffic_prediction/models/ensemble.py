"""Day-kind weighted ensemble of retained forecasters (issue #32)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from traffic_prediction.evaluate import KEYS, TARGETS, _check_frame
from traffic_prediction.models.base import ForecastModel

DAY_KINDS = ("ouvre", "weekend", "ferie")
# E5, frozen weights chosen on all eight backtest windows by scripts/evaluate_combinations.py
# ("Poids figés" in docs/ensemble_results.md). Honest leave-one-window-out score of the same
# procedure: 88.02 veh/h and 1.705 points, pooled on Convention + Saint-Pères.
E5_WEIGHTS = {
    "debit_horaire": {
        "ouvre": {"timestral_5m": 0.9, "lightgbm_raw": 0.1},
        "weekend": {"baseline_holiday_aware": 0.5, "timestral_5m": 0.2, "lightgbm_raw": 0.3},
        "ferie": {"lightgbm_raw": 1.0},
    },
    "taux_occupation": {
        "ouvre": {"baseline_holiday_aware": 0.1, "timestral_5m": 0.9},
        "weekend": {"baseline_holiday_aware": 0.6, "timestral_5m": 0.4},
        "ferie": {"baseline_holiday_aware": 0.8, "timestral_5m": 0.2},
    },
}


def day_kinds(times: pd.DatetimeIndex, calendar: pd.DataFrame) -> np.ndarray:
    """"ferie" on public holidays, else "weekend" on Saturday and Sunday, else "ouvre".

    The label's calendar date decides, as in the backtest scores per day kind.
    """
    holiday = calendar.drop_duplicates("datetime").set_index("datetime")["is_holiday"].reindex(times)
    if holiday.isna().any():
        raise ValueError("calendar must cover every target hour")
    return np.where(holiday.astype(bool), "ferie", np.where(times.dayofweek >= 5, "weekend", "ouvre"))


class WeightedEnsemble:
    """Weighted mean of member forecasts, one weight vector per variable and day kind.

    Every member is fitted from the same cutoff-limited history; only members with a
    positive weight run. Weights are fixed, not refitted: they were chosen on the backtest.
    A member failure stops the forecast, so a submission never silently changes model.
    ``sources_`` gives the day kind of each hour; ``member_sources_`` the source reported
    by each member (native model or fallback).
    """

    def __init__(self, members: dict[str, ForecastModel], weights: dict, calendar: pd.DataFrame) -> None:
        if missing := {"datetime", "is_holiday"} - set(calendar.columns):
            raise ValueError(f"calendar: missing columns {sorted(missing)}")
        if set(weights) != set(TARGETS):
            raise ValueError(f"weights must give every variable {list(TARGETS)}")
        used = set()
        for variable, by_kind in weights.items():
            if set(by_kind) != set(DAY_KINDS):
                raise ValueError(f"{variable}: weights must give every day kind {list(DAY_KINDS)}")
            for kind, vector in by_kind.items():
                values = np.array(list(vector.values()), dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
                    raise ValueError(f"{variable} {kind}: weights must be nonnegative and sum to 1")
                used |= {name for name, weight in vector.items() if weight > 0}
        if missing := used - set(members):
            raise ValueError(f"weights refer to unknown members {sorted(missing)}")
        self.members = {name: model for name, model in members.items() if name in used}
        self.weights, self.calendar = weights, calendar
        self.sources_: pd.Series | None = None
        self.member_sources_ = pd.DataFrame()

    def describe(self) -> str:
        """Complete settings for the experiment journal: weights and each member's settings."""
        members = "; ".join(f"{name}: {model.describe() if hasattr(model, 'describe') else type(model).__name__}"
                            for name, model in self.members.items())
        return f"WeightedEnsemble per variable and day kind, weights {json.dumps(self.weights)}; members {members}"

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        self.sources_, self.member_sources_ = None, pd.DataFrame()
        times = pd.DatetimeIndex(target_times)
        kinds = pd.Series(day_kinds(times, self.calendar), index=times)
        forecasts, sources = {}, {}
        for name, model in self.members.items():
            table = model.predict(history, times.copy())
            _check_frame(table, [*KEYS, *TARGETS], f"{name} predictions")
            forecasts[name] = table.set_index(KEYS)[list(TARGETS)]
            if forecasts[name].index.has_duplicates:
                raise ValueError(f"{name} predictions: duplicate (arc, datetime) rows")
            member_sources = getattr(model, "sources_", None)
            sources[name] = (member_sources.reindex(forecasts[name].index) if member_sources is not None
                             else pd.Series(name, index=forecasts[name].index))
        index = next(iter(forecasts.values())).index.sort_values()
        for name, forecast in forecasts.items():
            if not forecast.index.sort_values().equals(index):
                raise ValueError(f"{name} predictions: arcs or hours differ from the other members")
        row_kinds = kinds.reindex(index.get_level_values("datetime")).to_numpy()
        result = pd.DataFrame(0.0, index=index, columns=list(TARGETS))
        for variable in TARGETS:
            for kind in DAY_KINDS:
                rows = row_kinds == kind
                for name, weight in self.weights[variable][kind].items():
                    if weight > 0:
                        result.loc[rows, variable] += weight * forecasts[name].reindex(index[rows])[variable].to_numpy()
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Ensemble predictions must be finite")
        result["debit_horaire"] = result["debit_horaire"].clip(lower=0)
        result["taux_occupation"] = result["taux_occupation"].clip(0, 100)
        self.sources_ = pd.Series("ensemble_" + row_kinds.astype(object), index=index)
        self.member_sources_ = pd.DataFrame({name: series.reindex(index) for name, series in sources.items()})
        return result.reset_index()
