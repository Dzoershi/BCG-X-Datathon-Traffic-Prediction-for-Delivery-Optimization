"""Post-processing of the submitted forecast: closures copied from the previous year (issue #47)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_prediction.data.disruptions import ALL_ARCS
from traffic_prediction.evaluate import TARGETS

CLOSURE_TYPE = "ceremonie"


def closure_hours(events: pd.DataFrame, times: pd.DatetimeIndex) -> list[tuple[str, pd.DatetimeIndex]]:
    """Target labels from start to end, both included, of each ceremony on a single arc.

    Unlike ``disruptions.active_hours`` (08:00-14:00 for a 07:00-14:00 event), the start label
    is kept: copying it keeps the measured transition hours instead of guessing them.
    """
    closures = events.loc[events["type"].eq(CLOSURE_TYPE) & events["arc"].ne(ALL_ARCS)]
    found = []
    for row in closures.itertuples():
        labels = times[(times >= row.date_debut) & (times <= row.date_fin)]
        if len(labels):
            found.append((row.arc, labels))
    return found


def copy_closures_from_previous_year(table: pd.DataFrame, history: pd.DataFrame,
                                     events: pd.DataFrame) -> pd.MultiIndex:
    """Replace both variables on closure hours by the values measured one year earlier, same date.

    Copy by date, not by weekday: the ceremony is held on 11 November whatever the day.
    `table` has one row per (arc, datetime) with timestamps; it is modified in place.
    Returns the replaced (arc, datetime) pairs. Missing or invalid reference hours stop
    the submission rather than leaving model values on a closed avenue.
    """
    times = pd.DatetimeIndex(table["datetime"].unique()).sort_values()
    values = history[list(TARGETS)].to_numpy(dtype=float, na_value=np.nan)
    valid = (history["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool) & np.isfinite(values).all(axis=1)
             & (values[:, 0] >= 0) & (values[:, 1] >= 0) & (values[:, 1] <= 100))
    observed = history.loc[valid].set_index(["arc", "datetime"])[list(TARGETS)]
    replaced = []
    for arc, labels in closure_hours(events, times):
        reference = pd.MultiIndex.from_arrays([[arc] * len(labels), labels - pd.DateOffset(years=1)])
        if missing := [time for (_, time) in reference if (arc, time) not in observed.index]:
            raise ValueError(f"Closure on {arc}: no valid measurement to copy at {', '.join(map(str, missing))}.")
        rows = table["arc"].eq(arc) & table["datetime"].isin(labels)
        copied = observed.loc[reference].set_axis(labels).reindex(table.loc[rows, "datetime"])
        table.loc[rows, list(TARGETS)] = copied.to_numpy()
        replaced += [(arc, time) for time in labels]
    return pd.MultiIndex.from_tuples(replaced, names=["arc", "datetime"])
