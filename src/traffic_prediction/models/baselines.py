"""Baseline forecasts that every other model must beat (issue #17).

The four models follow the ``ForecastModel`` contract (``models/base.py``)
and are rebuilt from the cutoff-limited history at every call. They only use
evaluated arcs and ``qualite_mesure == "ok"`` hours; evaluation can separately
score each measured target even when its partner is unavailable (#53);
proxies are ignored.

Each model first applies its own rule, then falls back, in order, to the
arc's day-type profile, the arc's hourly profile over all days, and the
day-type profile pooled over all evaluated arcs. Every evaluated arc and
target hour thus gets a forecast, even without recent history: Champs-Elysées,
without measurements after 2025-11-19 11:00, uses its day-type profile.
Forecasts are clipped to flow >= 0 and occupancy in [0, 100].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR
from traffic_prediction.evaluate import KEYS, TARGETS

DEFAULT_CALENDAR = DEFAULT_CALENDAR_DIR / "calendar.parquet"
SUNDAY = 6
# Profile fallbacks, from the most to the least specific grouping.
PROFILE_LEVELS = [
    ("day_type_profile", ["arc", "day_type", "hour"]),
    ("hourly_profile", ["arc", "hour"]),
    ("all_arcs_profile", ["day_type", "hour"]),
]


def load_calendar(path: str | Path = DEFAULT_CALENDAR) -> pd.DataFrame:
    """Read the hourly calendar built by ``python -m traffic_prediction.data.build --calendar``."""
    return pd.read_parquet(path, columns=["datetime", "day_of_week", "is_holiday"])


def _calendar_days(times: pd.Series | pd.Index, calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar rows of the requested hours; fail if any hour is not covered."""
    days = calendar.set_index("datetime").reindex(pd.DatetimeIndex(times))
    if days["day_of_week"].isna().any():
        raise ValueError("Calendar does not cover every requested hour; rebuild it with --calendar")
    return days


def day_types(times: pd.Series | pd.Index, calendar: pd.DataFrame) -> np.ndarray:
    """Day of week (0 = Monday ... 6 = Sunday), public holidays counted as Sundays."""
    days = _calendar_days(times, calendar)
    return np.where(days["is_holiday"].astype(bool), SUNDAY, days["day_of_week"]).astype(int)


def _valid_history(history: pd.DataFrame) -> pd.DataFrame:
    """Targets of fully measured evaluated-arc hours, indexed by (arc, datetime)."""
    valid = (history["role"].eq("evalue") & history["qualite_mesure"].eq("ok")).fillna(False)
    return history.loc[valid.astype(bool), [*KEYS, *TARGETS]].set_index(KEYS)


def _values_at(valid: pd.DataFrame, index: pd.MultiIndex, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Values of each target's arc at the given source hours, aligned on the target index."""
    source = pd.MultiIndex.from_arrays([index.get_level_values("arc"), times], names=KEYS)
    return valid.reindex(source).set_axis(index)


def _lagged(valid: pd.DataFrame, index: pd.MultiIndex, weeks: int) -> pd.DataFrame:
    """Values of the same arc and local hour ``weeks`` weeks before each target."""
    return _values_at(valid, index, index.get_level_values("datetime") - pd.Timedelta(weeks=weeks))


def _last_sunday(times: pd.DatetimeIndex, last_known: pd.Timestamp) -> pd.DatetimeIndex:
    """Same local hour on the latest Sunday not after ``last_known``."""
    sunday = last_known.normalize() - pd.Timedelta(days=(last_known.dayofweek + 1) % 7)
    same_hour = sunday + pd.to_timedelta(times.hour, unit="h")
    return same_hour.where(same_hour <= last_known, same_hour - pd.Timedelta(weeks=1))


def _profiles(
    valid: pd.DataFrame, index: pd.MultiIndex, calendar: pd.DataFrame
) -> list[tuple[str, pd.DataFrame]]:
    """Mean history at each profile level, aligned on the target index."""
    history = valid.reset_index()
    targets = index.to_frame(index=False)
    for frame in (history, targets):
        frame["day_type"] = day_types(frame["datetime"], calendar)
        frame["hour"] = frame["datetime"].dt.hour
    candidates = []
    for name, keys in PROFILE_LEVELS:
        means = history.groupby(keys)[list(TARGETS)].mean()
        aligned = means.reindex(pd.MultiIndex.from_frame(targets[keys])).set_axis(index)
        candidates.append((name, aligned))
    return candidates


class _FallbackBaseline:
    """Apply the model rule, then the profile fallbacks, to every evaluated arc and hour.

    After each call, ``sources_`` gives the rule used for each (arc, datetime).
    """

    def __init__(self, calendar: pd.DataFrame | None = None) -> None:
        self.calendar = calendar
        self.sources_: pd.Series | None = None

    def predict(self, history: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
        calendar = load_calendar() if self.calendar is None else self.calendar
        valid = _valid_history(history)
        arcs = history.loc[history["role"].eq("evalue"), "arc"].unique()
        index = pd.MultiIndex.from_product([arcs, target_times], names=KEYS)
        forecast = pd.DataFrame(np.nan, index=index, columns=list(TARGETS))
        sources = pd.Series(None, index=index, dtype=object)
        rule = self._rule(valid, index, calendar, history["datetime"].max())
        for name, values in [*rule, *_profiles(valid, index, calendar)]:
            fill = forecast.isna().any(axis=1) & values.notna().all(axis=1)
            forecast.loc[fill] = values.loc[fill]
            sources.loc[fill] = name
        if forecast.isna().any().any():
            raise ValueError("Not enough valid history to forecast every evaluated arc and hour")
        self.sources_ = sources
        forecast["debit_horaire"] = forecast["debit_horaire"].clip(lower=0)
        forecast["taux_occupation"] = forecast["taux_occupation"].clip(0, 100)
        return forecast.reset_index()

    def _rule(
        self, valid: pd.DataFrame, index: pd.MultiIndex,
        calendar: pd.DataFrame, last_known: pd.Timestamp,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Model-specific candidates, tried before the profiles."""
        return []


class PreviousWeekBaseline(_FallbackBaseline):
    """Same arc and local hour 7 days before; if missing, 14 days before; then profiles.

    Why the previous week is always known: data stops on Thursday 23:00 and
    targets run from Saturday 00:00 to Wednesday 23:00, i.e. 25 to 144 hours
    ahead. A 168-hour lag thus always points at or before the Wednesday 23:00
    preceding the cutoff. Final run (cutoff Thursday 2026-11-05 23:00):

        target           previous-week value
        Sat 2026-11-07   Sat 2026-10-31
        Sun 2026-11-08   Sun 2026-11-01 (All Saints' Day, also a Sunday)
        Mon 2026-11-09   Mon 2026-11-02
        Tue 2026-11-10   Tue 2026-11-03
        Wed 2026-11-11   Wed 2026-11-04 (latest source: cutoff - 24 hours)

    Limit: the Wednesday 2026-11-11 public holiday copies an ordinary
    Wednesday, so weekday rush hours are predicted on a Sunday-like day
    (see ``HolidayAwarePreviousWeekBaseline``).
    """

    def _rule(
        self, valid: pd.DataFrame, index: pd.MultiIndex,
        calendar: pd.DataFrame, last_known: pd.Timestamp,
    ) -> list[tuple[str, pd.DataFrame]]:
        return [("previous_week", _lagged(valid, index, weeks=1)),
                ("two_weeks_before", _lagged(valid, index, weeks=2))]


class HolidayAwarePreviousWeekBaseline(_FallbackBaseline):
    """Previous week on normal days; on public holidays, the last known Sunday.

    Normal days: same rule as ``PreviousWeekBaseline`` (7, then 14 days before).
    Holidays: same local hour on the latest Sunday of the history (Sunday
    2026-11-01 for the final cutoff), then the Sunday before; then profiles,
    where holidays already count as Sundays.
    """

    def _rule(
        self, valid: pd.DataFrame, index: pd.MultiIndex,
        calendar: pd.DataFrame, last_known: pd.Timestamp,
    ) -> list[tuple[str, pd.DataFrame]]:
        times = index.get_level_values("datetime")
        holiday = _calendar_days(times, calendar)["is_holiday"].astype(bool).to_numpy()
        sunday = _last_sunday(times, last_known)
        candidates = []
        for name, values, rows in [
            ("previous_week", _lagged(valid, index, weeks=1), ~holiday),
            ("two_weeks_before", _lagged(valid, index, weeks=2), ~holiday),
            ("last_sunday", _values_at(valid, index, sunday), holiday),
            ("sunday_before", _values_at(valid, index, sunday - pd.Timedelta(weeks=1)), holiday),
        ]:
            values.loc[~rows] = np.nan
            candidates.append((name, values))
        return candidates


class FourWeekMeanBaseline(_FallbackBaseline):
    """Mean of the available values at the same arc and local hour 1 to 4 weeks before."""

    def _rule(
        self, valid: pd.DataFrame, index: pd.MultiIndex,
        calendar: pd.DataFrame, last_known: pd.Timestamp,
    ) -> list[tuple[str, pd.DataFrame]]:
        lagged = pd.concat([_lagged(valid, index, weeks) for weeks in range(1, 5)])
        mean = lagged.groupby(level=KEYS, sort=False).mean().reindex(index)
        return [("four_week_mean", mean)]


class DayTypeProfileBaseline(_FallbackBaseline):
    """Mean per (arc, day type, hour) over the whole history; holidays count as Sundays."""
