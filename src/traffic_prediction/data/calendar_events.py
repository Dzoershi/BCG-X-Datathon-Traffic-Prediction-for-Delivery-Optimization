"""Known hourly covariates in the processed traffic table's Paris convention."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import holidays
import pandas as pd

from traffic_prediction.data.traffic import PARIS_TZ, REPO_ROOT

SCHOOL_RECORDS_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/records"
)
DEFAULT_SCHOOL_PATH = REPO_ROOT / "data/raw/calendar/paris_school_holidays.json"
PREDICTION_END = "2026-11-11"
CALENDAR_COLUMNS = [
    "datetime", "hour", "day_of_week", "is_weekend", "is_holiday",
    "is_day_before_holiday", "is_day_after_holiday", "is_bridge",
    "is_school_holiday",
]


def download_school_holidays(path: str | Path = DEFAULT_SCHOOL_PATH) -> Path:
    """Snapshot the small official Paris/Zone C calendar, with pagination.

    Dataset catalog: https://www.data.gouv.fr/datasets/le-calendrier-scolaire
    Explicit invocation refreshes the snapshot; feature generation is offline.
    A failed download leaves the previous snapshot intact.
    """
    records = []
    while True:
        parameters = urlencode({
            "where": 'location = "Paris" AND zones = "Zone C"',
            "order_by": "start_date", "limit": 100, "offset": len(records),
        })
        with urlopen(f"{SCHOOL_RECORDS_URL}?{parameters}", timeout=30) as response:
            page = json.load(response)
        batch = page["results"]
        if not batch:
            raise ValueError("Incomplete or empty official school-calendar response")
        records.extend(batch)
        if len(records) >= page["total_count"]:
            break
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"source": SCHOOL_RECORDS_URL, "results": records},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _paris_date(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Missing school-calendar date")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert(PARIS_TZ).tz_localize(None)
    return timestamp.normalize()


def _school_days(path: str | Path, days: pd.DatetimeIndex) -> pd.Series:
    records = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    selected = [record for record in records
                if record["location"] == "Paris" and record["zones"] == "Zone C"
                and record["population"] in ("-", "Élèves")]
    years = {record["annee_scolaire"] for record in selected}
    required = {f"{day.year - (day.month < 9)}-{day.year + (day.month >= 9)}"
                for day in days}
    if missing := required - years:
        raise ValueError(f"School calendar missing academic years: {sorted(missing)}")
    result = pd.Series(False, index=days)
    for record in selected:
        start, end = _paris_date(record["start_date"]), _paris_date(record["end_date"])
        # Ignore records outside this build (some future source entries are placeholders).
        if start > days[-1] or end <= days[0]:
            continue
        if start >= end:
            raise ValueError(f"Invalid school-holiday interval: {record}")
        # Departure date included, return-to-school date excluded, at day resolution.
        result.loc[(days >= start) & (days < end)] = True
    return result


def calendar_table(
    start: object,
    end: object = PREDICTION_END,
    *,
    school_holidays_path: str | Path = DEFAULT_SCHOOL_PATH,
) -> pd.DataFrame:
    """Build inclusive local dates, with 24 end-of-hour labels per day.

    Features describe the label's calendar date; no one-hour shift is applied.
    Monday=0, Sunday=6. A bridge is a non-holiday Monday before a Tuesday
    holiday or Friday after a Thursday holiday. School holidays are day-level
    indicators, including the departure date and excluding the return date.
    """
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if any(pd.isna(day) or day.tz is not None or day != day.normalize()
           for day in (first, last)) or first > last:
        raise ValueError("Supply ordered timezone-free calendar dates at midnight")
    days = pd.date_range(first, last, freq="D")
    public = holidays.France(years=range(first.year - 1, last.year + 2), observed=False)
    daily = pd.DataFrame({"datetime": days})
    daily["day_of_week"] = days.dayofweek.astype("int8")
    daily["is_weekend"] = days.dayofweek >= 5
    daily["is_holiday"] = [day.date() in public for day in days]
    daily["is_day_before_holiday"] = [(day + pd.Timedelta(days=1)).date() in public for day in days]
    daily["is_day_after_holiday"] = [(day - pd.Timedelta(days=1)).date() in public for day in days]
    daily["is_bridge"] = ~daily["is_holiday"] & (
        (daily["day_of_week"].eq(0) & daily["is_day_before_holiday"])
        | (daily["day_of_week"].eq(4) & daily["is_day_after_holiday"])
    )
    daily["is_school_holiday"] = _school_days(school_holidays_path, days).to_numpy()
    hours = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
    table = daily.set_index("datetime").reindex(hours.normalize())
    table.index = hours
    table = table.rename_axis("datetime").reset_index()
    table["hour"] = hours.hour.astype("int8")
    return table[CALENDAR_COLUMNS]
