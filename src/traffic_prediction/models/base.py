"""Common forecasting contract used by the backtest."""

from typing import Protocol

import pandas as pd


class ForecastModel(Protocol):
    """Predict from cutoff-limited history, without accessing future observations.

    History includes evaluated and proxy arcs in the processed traffic schema.
    Return unique (arc, datetime) rows with debit_horaire and taux_occupation.
    Timestamps are timezone-free Paris local hours, as in processed data.
    Each call must fit/reset any learned state from this history alone.
    """

    def predict(
        self, history: pd.DataFrame, target_times: pd.DatetimeIndex
    ) -> pd.DataFrame: ...
