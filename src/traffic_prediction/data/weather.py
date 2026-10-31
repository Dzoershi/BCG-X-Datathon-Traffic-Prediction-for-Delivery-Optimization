"""Hourly Paris weather from Open-Meteo, in the processed traffic time convention.

Requests are made in UTC: with a time zone, Open-Meteo applies one fixed offset to
the whole response and ignores daylight saving changes. Each UTC timestamp is then
converted to a timezone-free Paris label, the end of the hour, as for traffic.
Precipitation is the sum of the preceding hour; temperature and wind are instant
values at the label.

Past days come from ``download_history`` and the days from today onward from
``download_forecast``; ``join_consecutive`` checks the seam and
``validate_weather`` the whole table.
"""

from __future__ import annotations

import json
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from traffic_prediction.data.calendar_events import PREDICTION_END
from traffic_prediction.data.traffic import PARIS_TZ, REPO_ROOT, period_bounds

# "historical_forecast" chains the first hours of each run of the forecast models,
# so training values match the live forecast; "archive" (ERA5 reanalysis) is kept
# for comparison.
HISTORY_URLS = {
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
}
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 48.856, 2.310  # Centre of the three evaluated arcs.
# Open-Meteo name -> (output column, unit expected in the response).
VARIABLES = {
    "precipitation": ("precipitation_mm", "mm"),
    "temperature_2m": ("temperature_c", "°C"),
    "wind_speed_10m": ("wind_speed_kmh", "km/h"),
}
VALUE_COLUMNS = [column for column, _ in VARIABLES.values()]
# source: "historical_forecast" or "archive" for past days, "forecast" from the cut day.
WEATHER_COLUMNS = ["datetime", *VALUE_COLUMNS, "source"]
# Wide bounds that only catch unit or parsing errors. Paris records: -23.9 °C and 42.6 °C.
PLAUSIBLE_RANGES = {
    "precipitation_mm": (0.0, 100.0),
    "temperature_c": (-30.0, 50.0),
    "wind_speed_kmh": (0.0, 200.0),
}
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "weather"
USER_AGENT = "traffic-prediction/0.1"


def fetch_json(url: str, parameters: dict, *, attempts: int = 3) -> dict:
    """GET an Open-Meteo JSON response, retrying temporary failures.

    Same policy as the traffic downloads: 60 s timeout, waits of 1 s then 2 s,
    no retry on a 4xx error except 408 and 429.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    request = Request(f"{url}?{urlencode(parameters)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (OSError, HTTPException, json.JSONDecodeError, UnicodeDecodeError) as error:
            permanent = (
                isinstance(error, HTTPError)
                and 400 <= error.code < 500
                and error.code not in (408, 429)
            )
            if permanent:
                raise ValueError(
                    f"Open-Meteo rejected the request ({error.code}): {_reason(error)}"
                ) from error
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"Weather request failed (attempt {attempt}/{attempts}): "
                f"{error}; retrying in {delay}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _reason(error: HTTPError) -> str:
    """Open-Meteo explains a rejected request in a JSON "reason" field."""
    try:
        return json.load(error)["reason"]
    except (OSError, ValueError, KeyError, TypeError):
        return str(error)


def save_raw(payload: dict, raw_dir: str | Path, stem: str) -> Path:
    """Keep the untouched response, named with its UTC retrieval time."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{stem}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)
    return path


def download_history(
    start: object,
    end: object,
    *,
    source: str = "historical_forecast",
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Download inclusive Paris days, from 00:00 to 23:00 labels, and save the raw JSON.

    The request starts one UTC day earlier: the 00:00 label covers the previous
    evening's last hour (2024-01-01 00:00 is 2023-12-31 23:00 UTC in winter).
    """
    if source not in HISTORY_URLS:
        raise ValueError(f"Unknown weather source {source!r}; use one of {sorted(HISTORY_URLS)}.")
    return _download(HISTORY_URLS[source], start, end, source=source, raw_dir=raw_dir)


def download_forecast(
    start: object | None = None,
    end: object = PREDICTION_END,
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Download the live forecast from ``start`` (default: today in Paris) to the last day to predict.

    The raw JSON name holds the retrieval time, to find the forecast used for a submission.
    Open-Meteo forecasts at most 16 days ahead and keeps 92 past days.
    """
    start = today_paris() if start is None else start
    return _download(FORECAST_URL, start, end, source="forecast", raw_dir=raw_dir)


def today_paris() -> pd.Timestamp:
    """Current Paris calendar date, timezone-free: the first day taken from the forecast."""
    return pd.Timestamp.now(tz=PARIS_TZ).tz_localize(None).normalize()


def join_consecutive(earlier: pd.DataFrame, later: pd.DataFrame) -> pd.DataFrame:
    """Stack two hourly tables; ``later`` must start the hour after ``earlier`` ends.

    Used for history then forecast. Each part is already a complete hourly grid,
    so a checked seam means no duplicate and no missing hour in the result.
    """
    expected = earlier["datetime"].iloc[-1] + pd.Timedelta(hours=1)
    if later["datetime"].iloc[0] != expected:
        raise ValueError(
            f"Weather must continue at {expected}, the hour after the previous part; "
            f"got {later['datetime'].iloc[0]}."
        )
    return pd.concat([earlier, later], ignore_index=True)


def validate_weather(table: pd.DataFrame, start: object, end: object) -> None:
    """Reject a table that would not join one-to-one with traffic hours of inclusive Paris days.

    Checks the columns, one row per hour of the grid (no duplicate, no gap, sorted),
    finite plausible values, one history source, and history rows before forecast rows.
    """
    if list(table.columns) != WEATHER_COLUMNS:
        raise ValueError(f"Unexpected weather columns: {list(table.columns)}")
    times = table["datetime"]
    if not pd.api.types.is_datetime64_dtype(times.dtype):
        raise ValueError("Weather datetime must be timezone-free Paris time.")
    first, last = _label_bounds(start, end)
    expected = pd.date_range(first, last, freq="h")
    if duplicated := int(times.duplicated().sum()):
        raise ValueError(f"Duplicate weather hours: {duplicated}.")
    missing = expected.difference(times)
    outside = pd.DatetimeIndex(times).difference(expected)
    if len(missing) or len(outside):
        raise ValueError(
            f"Weather must cover {first} to {last} hourly: {len(missing)} missing hours"
            f"{f' (first {missing[0]})' if len(missing) else ''}, {len(outside)} outside."
        )
    if not times.is_monotonic_increasing:
        raise ValueError("Weather rows must be sorted by datetime.")
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if not pd.api.types.is_float_dtype(table[column].dtype):
            raise ValueError(f"Weather {column} must be a float column.")
        if invalid := int((~table[column].between(low, high)).sum()):
            raise ValueError(f"Weather {column}: {invalid} values missing or outside {low} to {high}.")
    forecast = table["source"].eq("forecast")
    history_sources = set(table.loc[~forecast, "source"])
    if not history_sources <= set(HISTORY_URLS) or len(history_sources) > 1:
        raise ValueError(f"Weather history must come from one known source, got {sorted(history_sources)}.")
    if not forecast.is_monotonic_increasing:
        raise ValueError("Weather forecast rows must follow all history rows.")


def _download(url: str, start: object, end: object, *, source: str, raw_dir: str | Path) -> pd.DataFrame:
    first, last = _label_bounds(start, end)
    payload = fetch_json(url, _parameters(
        start_date=f"{first - pd.Timedelta(days=1):%Y-%m-%d}", end_date=f"{last:%Y-%m-%d}"
    ))
    save_raw(payload, raw_dir, f"{source}_{first:%Y-%m-%d}_{last:%Y-%m-%d}")
    return hourly_table(payload, first, last, source=source)


def _label_bounds(start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = period_bounds(start, end)
    if first.tz is not None or last.tz is not None:
        raise ValueError("Supply timezone-free Paris dates.")
    return first, last


def _parameters(**dates: str) -> dict:
    return {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(VARIABLES),
        **dates,
    }


def hourly_table(payload: dict, start: object, end: object, *, source: str) -> pd.DataFrame:
    """Convert an Open-Meteo hourly UTC response to the traffic grid of inclusive Paris days.

    Always 24 labels per day. The repeated October 02:00 label keeps the later hour,
    as traffic does; the missing March 02:00 label is filled by ``fill_spring_gap``.
    Any other missing hour or value is an error. ``source`` labels every row.
    """
    if payload.get("error"):
        raise ValueError(f"Open-Meteo error: {payload.get('reason')}")
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Weather hours must be requested in UTC, without a timezone parameter.")
    hourly, units = payload["hourly"], payload["hourly_units"]
    for name, (_, unit) in VARIABLES.items():
        if units.get(name) != unit:
            raise ValueError(f"Expected {name} in {unit}, got {units.get(name)!r}.")

    frame = pd.DataFrame(
        {column: pd.Series(hourly[name], dtype="float64") for name, (column, _) in VARIABLES.items()}
    )
    frame.index = paris_labels(hourly["time"])
    frame = frame[~frame.index.duplicated(keep="last")]

    first, last = _label_bounds(start, end)
    table = frame.reindex(pd.date_range(first, last, freq="h", name="datetime"))
    table = fill_spring_gap(table)
    missing = table.index[table.isna().any(axis=1)]
    if len(missing):
        raise ValueError(f"Missing weather for {len(missing)} hours, first {missing[0]}.")
    return table.reset_index().assign(source=source)[WEATHER_COLUMNS]


def paris_labels(utc_times: list[str]) -> pd.DatetimeIndex:
    """UTC end-of-hour timestamps to timezone-free Paris labels."""
    times = pd.DatetimeIndex(pd.to_datetime(utc_times, format="%Y-%m-%dT%H:%M"))
    return times.tz_localize("UTC").tz_convert(PARIS_TZ).tz_localize(None)


def fill_spring_gap(table: pd.DataFrame) -> pd.DataFrame:
    """Fill the 02:00 label that does not exist when clocks go forward in March.

    Clocks jump from 01:59 to 03:00, so that instant is 03:00: temperature and wind
    are copied from 03:00. No time elapses, so precipitation is 0 mm.
    """
    labels = pd.DatetimeIndex(table.index)
    gap = labels.tz_localize(PARIS_TZ, nonexistent="NaT", ambiguous=True).isna()
    if not gap.any():
        return table
    table = table.copy()
    following = table.shift(-1)
    for column in ("temperature_c", "wind_speed_kmh"):
        table.loc[gap, column] = following.loc[gap, column]
    table.loc[gap, "precipitation_mm"] = 0.0
    return table
