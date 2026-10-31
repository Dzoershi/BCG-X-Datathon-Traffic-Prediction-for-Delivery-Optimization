"""Offline weather checks: Open-Meteo responses are simulated, never downloaded.

The fake server encodes the UTC hour in each value (temperature = UTC hour,
precipitation = UTC hour / 10) and the API in the wind speed, so every Paris label
can be checked by hand.
"""

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from traffic_prediction.data import build, calendar_events, traffic, weather

WIND_BY_HOST = {
    "historical-forecast-api.open-meteo.com": 10.0,
    "api.open-meteo.com": 20.0,
    "archive-api.open-meteo.com": 30.0,
}


def payload(start_date, end_date, wind=10.0):
    """Open-Meteo-like hourly UTC response, from start_date 00:00 to end_date 23:00."""
    times = pd.date_range(start_date, f"{end_date} 23:00", freq="h")
    return {
        "utc_offset_seconds": 0,
        "hourly_units": {"time": "iso8601", "precipitation": "mm",
                         "temperature_2m": "°C", "wind_speed_10m": "km/h"},
        "hourly": {
            "time": [f"{time:%Y-%m-%dT%H:%M}" for time in times],
            "precipitation": [time.hour / 10 for time in times],
            "temperature_2m": [float(time.hour) for time in times],
            "wind_speed_10m": [wind] * len(times),
        },
    }


def response(start_date, end_date, wind=10.0):
    return io.BytesIO(json.dumps(payload(start_date, end_date, wind)).encode())


def fake_open_meteo(request, timeout):
    url = urlparse(request.full_url)
    query = parse_qs(url.query)
    return response(query["start_date"][0], query["end_date"][0], WIND_BY_HOST[url.netloc])


def requests_made(http):
    """(host, start_date, end_date) of each simulated request."""
    made = []
    for args, _ in http.call_args_list:
        url = urlparse(args[0].full_url)
        query = parse_qs(url.query)
        made.append((url.netloc, query["start_date"][0], query["end_date"][0]))
    return made


def utc_hour(label):
    return pd.Timestamp(label).tz_localize(traffic.PARIS_TZ).tz_convert("UTC").hour


@pytest.fixture(autouse=True)
def network():
    """Every test of this file runs offline against the fake server, without waiting."""
    with patch.object(weather, "urlopen", side_effect=fake_open_meteo) as http, \
            patch.object(weather.time, "sleep") as sleep:
        yield SimpleNamespace(http=http, sleep=sleep)


def table_for(day, source="historical_forecast"):
    """Weather of one Paris day, built from the fake response the API would send."""
    previous = pd.Timestamp(day) - pd.Timedelta(days=1)
    return weather.hourly_table(payload(previous, day), day, day, source=source)


# --- Request, raw copy and output format ---------------------------------------


def test_history_request_raw_copy_and_format(network, tmp_path):
    table = weather.download_history("2024-01-01", "2024-01-01", raw_dir=tmp_path)

    (request,), kwargs = network.http.call_args
    query = parse_qs(urlparse(request.full_url).query)
    assert urlparse(request.full_url).netloc == "historical-forecast-api.open-meteo.com"
    assert query == {"latitude": ["48.856"], "longitude": ["2.31"],
                     "hourly": ["precipitation,temperature_2m,wind_speed_10m"],
                     "start_date": ["2023-12-31"], "end_date": ["2024-01-01"]}
    assert kwargs == {"timeout": 60}
    assert request.get_header("User-agent") == weather.USER_AGENT

    (raw,) = tmp_path.glob("historical_forecast_2024-01-01_2024-01-01_*Z.json")
    assert json.loads(raw.read_text(encoding="utf-8")) == payload("2023-12-31", "2024-01-01")

    assert list(table.columns) == weather.WEATHER_COLUMNS
    assert table["datetime"].tolist() == pd.date_range("2024-01-01", "2024-01-01 23:00", freq="h").tolist()
    assert table["datetime"].dt.tz is None
    assert all(table[column].dtype == "float64" for column in weather.VALUE_COLUMNS)
    assert table["source"].eq("historical_forecast").all()
    # The 00:00 label covers 23:00-24:00 on December 31 (22:00-23:00 UTC), ending at 23:00 UTC.
    assert table.iloc[0][["temperature_c", "precipitation_mm"]].tolist() == [23.0, 2.3]


def test_archive_source_and_unknown_source(network, tmp_path):
    table = weather.download_history("2024-01-01", "2024-01-01", source="archive", raw_dir=tmp_path)
    assert requests_made(network.http) == [("archive-api.open-meteo.com", "2023-12-31", "2024-01-01")]
    assert table["wind_speed_kmh"].eq(30.0).all() and table["source"].eq("archive").all()
    with pytest.raises(ValueError, match="Unknown weather source"):
        weather.download_history("2024-01-01", "2024-01-01", source="metar", raw_dir=tmp_path)
    assert network.http.call_count == 1


def test_forecast_defaults_to_today_until_prediction_end(network, tmp_path):
    with patch.object(weather, "today_paris", return_value=pd.Timestamp("2026-11-06")):
        table = weather.download_forecast(raw_dir=tmp_path)
    assert requests_made(network.http) == [("api.open-meteo.com", "2026-11-05", calendar_events.PREDICTION_END)]
    assert table["datetime"].iloc[[0, -1]].tolist() == [pd.Timestamp("2026-11-06"),
                                                        pd.Timestamp("2026-11-11 23:00")]
    assert len(table) == 144 and table["source"].eq("forecast").all()
    assert len(list(tmp_path.glob("forecast_2026-11-06_2026-11-11_*Z.json"))) == 1


# --- Paris labels and clock changes --------------------------------------------


@pytest.mark.parametrize("label", ["2024-01-15 08:00", "2024-07-15 08:00", "2026-11-11 00:00"])
def test_values_follow_paris_end_of_hour_labels(label):
    row = table_for(pd.Timestamp(label).normalize()).set_index("datetime").loc[label]
    # Winter 08:00 Paris is 07:00 UTC, summer 08:00 Paris is 06:00 UTC.
    assert row["temperature_c"] == utc_hour(label)
    assert row["precipitation_mm"] == utc_hour(label) / 10


def test_spring_gap_copies_03h_temperature_and_wind_with_no_rain():
    day = table_for("2026-03-29").set_index("datetime")
    assert len(day) == 24 and not day.isna().any().any()
    # 01:00 Paris = 00:00 UTC; 03:00 Paris = 01:00 UTC; 04:00 Paris = 02:00 UTC.
    assert day.loc["2026-03-29 01:00", "temperature_c"] == 0.0
    assert day.loc["2026-03-29 03:00", "temperature_c"] == 1.0
    assert day.loc["2026-03-29 04:00", "temperature_c"] == 2.0
    gap = day.loc["2026-03-29 02:00"]
    assert gap["temperature_c"] == 1.0 and gap["wind_speed_kmh"] == 10.0
    assert gap["precipitation_mm"] == 0.0


def test_autumn_repeated_hour_keeps_the_later_utc_hour():
    day = table_for("2025-10-26").set_index("datetime")
    assert len(day) == 24 and day.index.is_unique
    # 02:00 Paris happens at 00:00 UTC (summer time) then 01:00 UTC (winter time).
    assert day.loc["2025-10-26 01:00", "temperature_c"] == 23.0
    assert day.loc["2025-10-26 02:00", "temperature_c"] == 1.0
    assert day.loc["2025-10-26 03:00", "temperature_c"] == 2.0


@pytest.mark.parametrize("day", ["2024-03-31", "2024-10-27", "2026-03-29", "2026-10-25"])
def test_joins_the_traffic_grid_one_to_one(day):
    arcs = traffic.load_arcs()[:1]
    history = traffic.build_hourly_table(pd.DataFrame(columns=traffic.OUTPUT_COLUMNS), arcs, day, day)
    joined = history.merge(table_for(day), on="datetime", validate="one_to_one")
    assert len(joined) == len(history) == 24
    assert not joined[weather.VALUE_COLUMNS].isna().any().any()


def end_at_21h_utc(data):
    """The 23:00 Paris label of November 7 is 22:00 UTC: drop 22:00 and 23:00 UTC."""
    for values in data["hourly"].values():
        del values[-2:]


@pytest.mark.parametrize("change,message", [
    (lambda p: p.update(error=True, reason="Invalid date"), "Open-Meteo error: Invalid date"),
    (lambda p: p.update(utc_offset_seconds=3600), "UTC"),
    (lambda p: p["hourly_units"].update(temperature_2m="°F"), "temperature_2m in °C"),
    (lambda p: p["hourly"]["temperature_2m"].__setitem__(30, None), "Missing weather for 1 hours"),
    (end_at_21h_utc, "Missing weather for 1 hours, first 2026-11-07 23:00"),
])
def test_inconsistent_responses_are_rejected(change, message):
    data = payload("2026-11-06", "2026-11-07")
    change(data)
    with pytest.raises(ValueError, match=message):
        weather.hourly_table(data, "2026-11-07", "2026-11-07", source="forecast")


# --- History and forecast seam -------------------------------------------------


def test_join_consecutive_rejects_gaps_and_overlaps():
    first, second = table_for("2026-11-05"), table_for("2026-11-06", "forecast")
    assert len(weather.join_consecutive(first, second)) == 48
    with pytest.raises(ValueError, match="2026-11-05 23:00"):
        weather.join_consecutive(first.iloc[:-1], second)
    with pytest.raises(ValueError, match="2026-11-06 00:00"):
        weather.join_consecutive(first, pd.concat([first.iloc[[-1]], second]))


def test_build_uses_history_until_yesterday_and_forecast_from_today(network, tmp_path):
    with patch.object(weather, "today_paris", return_value=pd.Timestamp("2026-11-05")):
        table = build.build_weather("2026-11-01", output_dir=tmp_path, raw_dir=tmp_path / "raw")

    assert requests_made(network.http) == [
        ("historical-forecast-api.open-meteo.com", "2026-10-31", "2026-11-04"),
        ("api.open-meteo.com", "2026-11-04", "2026-11-11"),
    ]
    assert table["datetime"].tolist() == pd.date_range("2026-11-01", "2026-11-11 23:00", freq="h").tolist()
    history = table["source"].eq("historical_forecast")
    assert table.loc[history, "datetime"].max() == pd.Timestamp("2026-11-04 23:00")
    assert table.loc[~history, "datetime"].min() == pd.Timestamp("2026-11-05 00:00")
    assert table.loc[~history, "source"].eq("forecast").all()
    assert table.loc[history, "wind_speed_kmh"].eq(10.0).all()
    assert table.loc[~history, "wind_speed_kmh"].eq(20.0).all()
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "weather.parquet"), table)


def test_refresh_keeps_history_fills_missing_days_and_replaces_forecast(network, tmp_path):
    with patch.object(weather, "today_paris", return_value=pd.Timestamp("2026-10-28")):
        before = build.build_weather("2026-10-25", output_dir=tmp_path, raw_dir=tmp_path / "raw")
    network.http.reset_mock()
    with patch.object(weather, "today_paris", return_value=pd.Timestamp("2026-11-06")):
        after = build.build_weather(output_dir=tmp_path, raw_dir=tmp_path / "raw", refresh_forecast=True)

    # Only the days after the stored history are downloaded, then the new forecast.
    assert requests_made(network.http) == [
        ("historical-forecast-api.open-meteo.com", "2026-10-27", "2026-11-05"),
        ("api.open-meteo.com", "2026-11-05", "2026-11-11"),
    ]
    kept = int(before["source"].eq("historical_forecast").sum())
    pd.testing.assert_frame_equal(after.iloc[:kept], before.iloc[:kept])
    assert after["datetime"].tolist() == pd.date_range("2026-10-25", "2026-11-11 23:00", freq="h").tolist()
    forecast = after["source"].eq("forecast")
    # On Friday November 6, history ends on Thursday 23:00, the model cutoff.
    assert after.loc[~forecast, "datetime"].max() == pd.Timestamp("2026-11-05 23:00")
    assert after.loc[forecast, "datetime"].min() == pd.Timestamp("2026-11-06 00:00")
    assert len(list((tmp_path / "raw").glob("forecast_*.json"))) == 2


def test_failed_refresh_keeps_the_previous_table(network, tmp_path):
    with patch.object(weather, "today_paris", return_value=pd.Timestamp("2026-11-05")):
        build.build_weather("2026-11-01", output_dir=tmp_path, raw_dir=tmp_path / "raw")
        before = (tmp_path / "weather.parquet").read_bytes()
        network.http.side_effect = URLError("offline")
        with pytest.raises(URLError):
            build.build_weather(output_dir=tmp_path, raw_dir=tmp_path / "raw", refresh_forecast=True)
    assert (tmp_path / "weather.parquet").read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_refresh_needs_a_stored_table_without_start(tmp_path):
    with pytest.raises(FileNotFoundError, match="run --weather once"):
        build.build_weather(output_dir=tmp_path, refresh_forecast=True)
    with pytest.raises(ValueError, match="omit start"):
        build.build_weather("2026-11-01", output_dir=tmp_path, refresh_forecast=True)


# --- Network failures -----------------------------------------------------------


@pytest.mark.parametrize("failure", [
    TimeoutError("slow"),
    io.BytesIO(b'{"incomplete":'),
    HTTPError("https://api.open-meteo.com", 429, "Too many requests", {}, None),
    HTTPError("https://api.open-meteo.com", 503, "Unavailable", {}, None),
])
def test_temporary_failure_is_retried(network, tmp_path, failure):
    network.http.side_effect = [failure, response("2026-11-05", "2026-11-06")]
    table = weather.download_forecast("2026-11-06", "2026-11-06", raw_dir=tmp_path)
    assert len(table) == 24
    assert network.http.call_count == 2
    assert network.sleep.call_args_list == [call(1)]


def test_network_gives_up_after_three_attempts(network, tmp_path):
    network.http.side_effect = URLError("offline")
    with pytest.raises(URLError):
        weather.download_forecast("2026-11-06", "2026-11-06", raw_dir=tmp_path)
    assert network.http.call_count == 3
    assert network.sleep.call_args_list == [call(1), call(2)]
    assert not list(tmp_path.iterdir())


def test_rejected_request_is_not_retried_and_explained(network, tmp_path):
    body = io.BytesIO(b'{"error": true, "reason": "end_date is out of allowed range"}')
    network.http.side_effect = HTTPError("https://api.open-meteo.com", 400, "Bad Request", {}, body)
    with pytest.raises(ValueError, match="400.*end_date is out of allowed range"):
        weather.download_forecast("2026-11-06", "2026-12-31", raw_dir=tmp_path)
    assert network.http.call_count == 1
    network.sleep.assert_not_called()


# --- Validation -----------------------------------------------------------------


def valid_table():
    return weather.join_consecutive(table_for("2026-11-05"), table_for("2026-11-06", "forecast"))


def set_row(column, value, row=3):
    return lambda table: table.assign(**{column: table[column].where(table.index != row, value)})


@pytest.mark.parametrize("corrupt,message", [
    (lambda t: t.assign(extra=1), "Unexpected weather columns"),
    (lambda t: t.assign(datetime=t["datetime"].dt.tz_localize("UTC")), "timezone-free"),
    (lambda t: pd.concat([t, t.iloc[[5]]]).sort_values("datetime", kind="stable"), "Duplicate weather hours: 1"),
    (lambda t: t.drop(index=10), "1 missing hours"),
    (lambda t: pd.concat([t, t.iloc[[-1]].assign(datetime=pd.Timestamp("2026-11-07"))]), "1 outside"),
    (lambda t: t.iloc[::-1], "sorted"),
    (set_row("temperature_c", float("nan")), "temperature_c: 1 values"),
    (set_row("temperature_c", 80.0), "temperature_c: 1 values"),
    (set_row("precipitation_mm", -0.1), "precipitation_mm: 1 values"),
    (set_row("wind_speed_kmh", float("inf")), "wind_speed_kmh: 1 values"),
    (lambda t: t.assign(precipitation_mm=t["precipitation_mm"].astype(str)), "float column"),
    (set_row("source", "archive"), "one known source"),
    (set_row("source", "observed"), "one known source"),
    (set_row("source", "forecast", row=0), "forecast rows must follow"),
])
def test_validation_rejects_inconsistent_tables(corrupt, message):
    table = valid_table()
    weather.validate_weather(table, "2026-11-05", "2026-11-06")
    with pytest.raises(ValueError, match=message):
        weather.validate_weather(corrupt(table), "2026-11-05", "2026-11-06")


# --- Command line ---------------------------------------------------------------


def test_cli_passes_weather_options():
    calls = []
    argv = ["build", "--weather", "--refresh-forecast", "--weather-source", "archive"]
    with patch.object(sys, "argv", argv), \
            patch.object(build, "build_weather", lambda *args, **kwargs: calls.append((args, kwargs))):
        build.main()
    assert calls == [((None, calendar_events.PREDICTION_END, build.DEFAULT_WEATHER_DIR),
                      {"traffic_dir": build.DEFAULT_TRAFFIC_DIR, "source": "archive",
                       "refresh_forecast": True})]


@pytest.mark.parametrize("arguments,message", [
    (["--refresh-forecast"], "--refresh-forecast requires --weather"),
    (["--calendar", "--weather"], "not both"),
    (["--weather", "--weather-source", "metar"], "invalid choice"),
])
def test_cli_rejects_inconsistent_options(capsys, arguments, message):
    with patch.object(sys, "argv", ["build", *arguments]), pytest.raises(SystemExit) as stop:
        build.main()
    assert stop.value.code == 2
    assert message in capsys.readouterr().err
