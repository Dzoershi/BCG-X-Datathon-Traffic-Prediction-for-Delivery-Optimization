"""Global LightGBM regressors using cutoff-aware tabular features (issue #31)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from traffic_prediction.data.build import DEFAULT_DATASET_DIR
from traffic_prediction.evaluate import KEYS, TARGETS, _check_frame
from traffic_prediction.features import DEFAULT_LAGS, TabularFeatures
from traffic_prediction.models.baselines import HolidayAwarePreviousWeekBaseline
from traffic_prediction.models.chronos import MIN_RECENT_OK_HOURS, RECENT_HOURS

SEED = 42
LIGHTGBM_RUNS = {
    "lightgbm_raw": {},
    "lightgbm_log": {"log_flow": True},
    "lightgbm_shallow": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    "lightgbm_shallow_log": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "log_flow": True},
    "lightgbm_deep": {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.03},
    "lightgbm_deep_log": {"n_estimators": 400, "max_depth": 8, "learning_rate": 0.03, "log_flow": True},
}


class LightGBMForecaster:
    """Fit one global model per target on all measured evaluated arcs.

    Each predict call resets fitted state and learns only from its history. Arc
    is categorical, NaN features are left to LightGBM, and no validation labels
    are supplied to fit. A direct forecast needs no recursive Friday predictions.
    Arcs with fewer than 168 valid hours in the last 28 days use the same explicit
    holiday-aware baseline fallback as the foundation models.

    ``covariates`` is #22's measurement-free table. Its calendar is always used;
    weather follows #30's conservative availability rules. LightGBM is imported
    only when a native forecast is needed, keeping offline/data CI lightweight.
    """

    def __init__(
        self, covariates: pd.DataFrame | None = None, *, n_estimators: int = 200,
        max_depth: int = 6, learning_rate: float = 0.05, log_flow: bool = False,
        min_child_samples: int = 30, reg_lambda: float = 1.0, n_jobs: int = 4,
        lags: tuple[int, ...] = DEFAULT_LAGS, weather: str = "lagged",
    ) -> None:
        for name, value in (("n_estimators", n_estimators), ("max_depth", max_depth),
                            ("min_child_samples", min_child_samples), ("n_jobs", n_jobs)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if not np.isfinite(reg_lambda) or reg_lambda < 0:
            raise ValueError("reg_lambda must be finite and nonnegative")
        if not isinstance(log_flow, bool):
            raise ValueError("log_flow must be a boolean")
        TabularFeatures(lags, weather=weather)  # Validate feature settings before any fit.
        self.covariates = covariates
        self.log_flow = log_flow
        self.lags, self.weather = tuple(lags), weather
        self.parameters = dict(
            objective="regression", n_estimators=n_estimators, max_depth=max_depth,
            num_leaves=min(31, 2 ** max_depth), learning_rate=learning_rate,
            min_child_samples=min_child_samples, reg_lambda=reg_lambda, n_jobs=n_jobs,
            random_state=SEED, deterministic=True, force_col_wise=True,
            importance_type="gain", verbosity=-1,
        )
        self.models_: dict = {}
        self.sources_: pd.Series | None = None
        self.feature_importances_ = pd.DataFrame()
        self.training_rows_ = 0
        self.features_: TabularFeatures | None = None

    def describe(self) -> str:
        """Complete settings for the experiment journal, including the fallback."""
        return (
            f"LightGBM global, separate targets, flow={'log1p' if self.log_flow else 'raw'}; "
            f"parameters {json.dumps(self.parameters, sort_keys=True)}; "
            f"lags {list(self.lags)} h; weather={self.weather}; training profiles gap=168 h; "
            f"fallback HolidayAwarePreviousWeekBaseline if fewer than {MIN_RECENT_OK_HOURS} "
            "valid hours in the last 28 days"
        )

    def _new_model(self):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**self.parameters)

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        """Refit on cutoff-limited assembled history and forecast the requested hours."""
        self.models_, self.sources_, self.features_ = {}, None, None
        self.feature_importances_, self.training_rows_ = pd.DataFrame(), 0
        _check_frame(history, [*KEYS, *TARGETS, "role", "qualite_mesure", "is_holiday"], "history")
        times = pd.DatetimeIndex(target_times)
        if (times.empty or times.tz is not None or times.hasnans or not times.is_unique
                or not times.is_monotonic_increasing or not times.equals(times.floor("h"))):
            raise ValueError("target_times must be unique increasing timezone-free whole Paris hours")
        cutoff = history["datetime"].max()
        if pd.isna(cutoff) or times.min() <= cutoff:
            raise ValueError("Target hours must all follow a nonempty history")
        if history.groupby("arc")["role"].nunique(dropna=False).gt(1).any():
            raise ValueError("history: an arc must have a consistent role")
        arcs = sorted(history.loc[history["role"].eq("evalue"), "arc"].unique())
        if not arcs:
            raise ValueError("history: no evaluated arcs")
        covariates = self.covariates
        if covariates is None:
            covariates = pd.read_parquet(DEFAULT_DATASET_DIR / "covariates.parquet")
        y = history[list(TARGETS)].to_numpy(dtype=float, na_value=np.nan)
        valid = (history["role"].eq("evalue") & history["qualite_mesure"].eq("ok")).fillna(False).to_numpy(dtype=bool)
        valid &= np.isfinite(y).all(axis=1) & (y[:, 0] >= 0) & (y[:, 1] >= 0) & (y[:, 1] <= 100)
        training = history.loc[valid].reset_index(drop=True)
        recent = training.loc[training["datetime"].gt(cutoff - pd.Timedelta(hours=RECENT_HOURS))]
        counts = recent.groupby("arc").size().reindex(arcs, fill_value=0)
        active = counts.index[counts.ge(MIN_RECENT_OK_HOURS)].tolist()
        index = pd.MultiIndex.from_product([arcs, times], names=KEYS)
        predictions = pd.DataFrame(np.nan, index=index, columns=list(TARGETS))
        sources = pd.Series(index=index, dtype=object)
        if active:
            builder = TabularFeatures(self.lags, weather=self.weather).fit(history, cutoff=cutoff)
            inputs = builder.transform(training, covariates)[builder.feature_columns]
            native_index = pd.MultiIndex.from_product([active, times], names=KEYS)
            future = builder.transform(native_index.to_frame(index=False), covariates)[builder.feature_columns]
            importance = []
            for target in TARGETS:
                labels = training[target].to_numpy(dtype=float)
                if target == "debit_horaire" and self.log_flow:
                    labels = np.log1p(labels)
                model = self._new_model()
                model.fit(inputs, labels, categorical_feature=["arc"])
                values = np.asarray(model.predict(future), dtype=float)
                if values.shape != (len(future),) or not np.isfinite(values).all():
                    raise ValueError(f"LightGBM returned invalid predictions for {target}")
                if target == "debit_horaire" and self.log_flow:
                    with np.errstate(over="ignore", invalid="ignore"):
                        values = np.expm1(values)
                    if not np.isfinite(values).all():
                        raise ValueError("LightGBM flow overflow after inverse log1p")
                predictions.loc[native_index, target] = values
                self.models_[target] = model
                importance.append(pd.DataFrame({
                    "variable": target, "feature": builder.feature_columns,
                    "gain": model.booster_.feature_importance(importance_type="gain"),
                    "split": model.booster_.feature_importance(importance_type="split"),
                }))
            self.features_, self.training_rows_ = builder, len(training)
            self.feature_importances_ = pd.concat(importance, ignore_index=True)
            sources.loc[native_index] = "lightgbm"
        if len(active) < len(arcs):
            calendar = covariates[["datetime", "is_holiday"]].drop_duplicates().copy()
            calendar["day_of_week"] = calendar["datetime"].dt.dayofweek
            baseline = HolidayAwarePreviousWeekBaseline(calendar)
            fallback_history = history.copy()
            fallback_history.loc[~valid, list(TARGETS)] = np.nan
            fallback = baseline.predict(fallback_history, times).set_index(KEYS)
            missing = index[~index.get_level_values("arc").isin(active)]
            predictions.loc[missing] = fallback.reindex(missing)[list(TARGETS)]
            sources.loc[missing] = "fallback_" + baseline.sources_.reindex(missing)
        if not np.isfinite(predictions.to_numpy(dtype=float)).all():
            raise ValueError("Not enough valid history to forecast every evaluated arc")
        predictions["debit_horaire"] = predictions["debit_horaire"].clip(lower=0)
        predictions["taux_occupation"] = predictions["taux_occupation"].clip(0, 100)
        self.sources_ = sources
        return predictions.reset_index()
