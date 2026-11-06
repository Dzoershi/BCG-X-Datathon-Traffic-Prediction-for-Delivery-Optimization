"""Historical sensor-outage experiments for #53; never enabled in submission.

Implements the copy, offset and linear candidates described in the #51 study.
Calibration predates the simulated outage. Proxy inputs never follow the cutoff.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from traffic_prediction.evaluate import KEYS, TARGETS, valid_truth
from traffic_prediction.models.baselines import DayTypeProfileBaseline


class ProxyOutageForecaster:
    """Forecast via a known previous-week proxy, else a pre-outage target profile.

    ``profile`` is the common control. Every method hides the same target hours
    from both its profile and calibration; original truth is never overwritten.
    Fits are per target and require 20 measured pairs (and variance for linear).
    """

    def __init__(self, calendar: pd.DataFrame, target: str, proxy: str | None,
                 outage_days: int, method: str = "profile", min_pairs: int = 20):
        if method not in {"profile", "copy", "offset", "linear"}:
            raise ValueError("Unknown proxy method")
        if type(outage_days) is not int or outage_days < 1 or type(min_pairs) is not int or min_pairs < 2:
            raise ValueError("outage_days and min_pairs must be positive integers (min_pairs >= 2)")
        if method != "profile" and (not proxy or proxy == target):
            raise ValueError("A distinct proxy is required")
        self.calendar, self.target, self.proxy = calendar, target, proxy
        self.outage_days, self.method, self.min_pairs = outage_days, method, min_pairs
        self.sources_: pd.DataFrame | None = None
        self.details_: pd.DataFrame | None = None

    def describe(self) -> str:
        return json.dumps({"method": self.method, "target": self.target, "proxy": self.proxy,
                           "outage_days": self.outage_days, "min_pairs": self.min_pairs,
                           "calibration": "strictly pre-outage", "forecast": "known previous-week input"})

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        history = history.copy(deep=True)
        cutoff = history.datetime.max()
        if pd.isna(cutoff) or target_times.min() <= cutoff:
            raise ValueError("Forecast hours must follow a nonempty cutoff-limited history")
        start = cutoff.normalize() + pd.Timedelta(days=1 - self.outage_days)
        target = history.loc[history.arc.eq(self.target)].set_index("datetime")
        if target.empty:
            raise ValueError("Target is absent from history")
        proxy = history.loc[history.arc.eq(self.proxy)].set_index("datetime")
        hidden = history.arc.eq(self.target) & history.datetime.ge(start)
        history.loc[hidden, list(TARGETS)] = np.nan
        history.loc[hidden, "qualite_mesure"] = "absente"
        baseline = DayTypeProfileBaseline(self.calendar)
        result = baseline.predict(history, target_times).set_index(KEYS)
        self.sources_ = pd.DataFrame({variable: "fallback_" + baseline.sources_ for variable in TARGETS})
        self.details_ = pd.DataFrame("", index=result.index, columns=list(TARGETS))
        future_index = pd.MultiIndex.from_product([[self.target], target_times], names=KEYS)
        lag = target_times - pd.Timedelta(days=7)
        if (lag > cutoff).any():
            raise ValueError("A proxy forecast cannot use inputs after the cutoff")
        for variable in TARGETS:
            observed = target[variable].where(valid_truth(target, variable))
            known = observed.where(observed.index < start).reindex(lag).to_numpy(dtype=float)
            samples, slope, intercept, last_pair = 0, 1.0, 0.0, None
            if not proxy.empty:
                measured_proxy = proxy[variable].where(valid_truth(proxy, variable))
                pairs = pd.concat([observed.rename("y"), measured_proxy.rename("x")], axis=1)
                pairs = pairs.loc[pairs.index < start].dropna()
                samples = len(pairs)
                last_pair = str(pairs.index.max()) if samples else None
                if samples:
                    if self.method == "offset":
                        intercept = float((pairs.y - pairs.x).mean())
                    elif self.method == "linear" and pairs.x.var(ddof=0) > 0:
                        centered = pairs.x - pairs.x.mean()
                        slope = float((centered * (pairs.y - pairs.y.mean())).sum() / centered.pow(2).sum())
                        intercept = float(pairs.y.mean() - slope * pairs.x.mean())
                permitted = self.method == "copy" or (
                    self.method in {"offset", "linear"} and samples >= self.min_pairs
                    and (self.method != "linear" or pairs.x.var(ddof=0) > 0))
                estimates = measured_proxy.reindex(lag).to_numpy(dtype=float) * slope + intercept
            else:
                permitted, estimates = False, np.full(len(lag), np.nan)
            # Control uses the target profile at the forecast hour, as in #51.
            if self.method != "profile":
                use_observed = np.isfinite(known)
                use_proxy = ~use_observed & np.isfinite(estimates) & permitted
                result.loc[future_index[use_observed], variable] = known[use_observed]
                result.loc[future_index[use_proxy], variable] = estimates[use_proxy]
                self.sources_.loc[future_index[use_observed], variable] = "observed_previous_week"
                self.sources_.loc[future_index[use_proxy], variable] = f"proxy_{self.method}:{self.proxy}"
            self.details_.loc[future_index, variable] = [json.dumps({
                "outage_start": str(start), "cutoff": str(cutoff), "proxy_input_hour": str(hour),
                "calibration_pairs": samples, "last_calibration_hour": last_pair,
                "slope": slope, "intercept": intercept,
            }, allow_nan=False) for hour in lag]
        result["debit_horaire"] = result.debit_horaire.clip(lower=0)
        result["taux_occupation"] = result.taux_occupation.clip(0, 100)
        return result.reset_index()
