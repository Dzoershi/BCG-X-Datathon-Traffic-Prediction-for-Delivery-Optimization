"""Offline calendar checks against a small official Paris/Zone C snapshot."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from traffic_prediction.data import build, calendar_events as calendar, traffic

SCHOOL_PATH = Path(__file__).parent / "fixtures/paris_school_holidays.json"


def make_calendar(start="2024-01-01", end="2026-11-11", path=SCHOOL_PATH):
    return calendar.calendar_table(start, end, school_holidays_path=path)


@pytest.fixture
def table():
    return make_calendar().set_index("datetime")


@pytest.mark.parametrize("date,weekday,weekend,holiday", [
    ("2026-11-07", 5, True, False),
    ("2026-11-08", 6, True, False),
    ("2026-11-11", 2, False, True),
    ("2024-11-11", 0, False, True),
    ("2025-11-11", 1, False, True),
    ("2024-03-29", 4, False, False),  # Good Friday is not a Paris holiday.
])
def test_hand_checked_days(table, date, weekday, weekend, holiday):
    day = table.loc[date]
    assert len(day) == 24
    assert day["day_of_week"].eq(weekday).all()
    assert day["is_weekend"].eq(weekend).all()
    assert day["is_holiday"].eq(holiday).all()


def test_complete_hourly_grid_and_dtypes(table):
    expected = pd.date_range("2024-01-01", "2026-11-11 23:00", freq="h", name="datetime")
    pd.testing.assert_index_equal(table.index, expected)
    assert list(table.reset_index()) == calendar.CALENDAR_COLUMNS
    assert not table.isna().any().any()
    assert table.index.is_unique
    assert table["hour"].tolist() == list(range(24)) * (len(table) // 24)
    assert table["hour"].dtype == table["day_of_week"].dtype == "int8"
    assert all(table[column].dtype == bool for column in table if column.startswith("is_"))
    for date in ("2024-03-31", "2024-10-27", "2026-03-29", "2026-10-25"):
        assert len(table.loc[date]) == 24


@pytest.mark.parametrize("date,school", [
    ("2026-10-16", False), ("2026-10-17", True),
    ("2026-10-31", True), ("2026-11-01", True), ("2026-11-02", False),
    ("2026-11-07", False), ("2026-11-11", False),
    ("2024-01-07", True), ("2024-01-08", False),
    ("2024-08-30", True), ("2024-09-02", False),
    ("2026-02-21", True), ("2026-03-09", False),
    ("2026-05-15", True),  # Official school Ascension break.
])
def test_school_holidays_and_paris_date_boundaries(table, date, school):
    assert table.loc[date, "is_school_holiday"].eq(school).all()


def test_holiday_neighbors_and_bridges(table):
    assert table.loc["2025-11-10", "is_bridge"].all()
    assert table.loc["2025-11-10", "is_day_before_holiday"].all()
    assert table.loc["2025-05-30", "is_bridge"].all()
    assert table.loc["2025-05-30", "is_day_after_holiday"].all()
    assert not table.loc["2025-11-11", "is_bridge"].any()
    assert not table.loc["2026-11-10", "is_bridge"].any()
    assert not table.loc["2026-11-07":"2026-11-08", "is_bridge"].any()
    # Adjacent holidays must be computed outside the requested output range.
    one_day = make_calendar("2024-12-31", "2024-12-31")
    assert one_day["is_day_before_holiday"].all()
    assert make_calendar("2025-01-02", "2025-01-02")["is_day_after_holiday"].all()


@pytest.mark.parametrize("start,end", [
    ("2026-11-12", "2026-11-11"), ("NaT", "2026-11-11"),
    ("2026-11-07T00:00Z", "2026-11-11"), ("2026-11-07 01:00", "2026-11-11"),
])
def test_invalid_date_bounds(start, end):
    with pytest.raises(ValueError, match="calendar dates"):
        make_calendar(start, end)


def test_reject_missing_school_coverage(tmp_path):
    payload = json.loads(SCHOOL_PATH.read_text(encoding="utf-8"))
    payload["results"] = [r for r in payload["results"] if r["annee_scolaire"] != "2024-2025"]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="2024-2025"):
        make_calendar(path=path)


def test_other_academies_zones_and_teacher_dates_do_not_change_flags(tmp_path):
    payload = json.loads(SCHOOL_PATH.read_text(encoding="utf-8"))
    extra = deepcopy(payload["results"][0])
    extra.update(start_date="2026-11-07", end_date="2026-11-12", annee_scolaire="2026-2027")
    for change in ({"location": "Lyon"}, {"zones": "Zone A"}, {"population": "Enseignants"}):
        payload["results"].append(dict(extra, **change))
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    pd.testing.assert_frame_equal(make_calendar(path=path), make_calendar())


def test_build_infers_history_start_and_joins_without_missing_hours(tmp_path):
    arcs = traffic.load_arcs()[:1]
    history = traffic.build_hourly_table(pd.DataFrame(columns=traffic.OUTPUT_COLUMNS), arcs,
                                         "2024-03-30", "2024-04-01")
    history_dir = tmp_path / "traffic"
    build.write_partitions(history, history_dir)
    output_dir = tmp_path / "calendar"
    table = build.build_calendar(traffic_dir=history_dir, output_dir=output_dir,
                                 school_holidays_path=SCHOOL_PATH)
    assert table["datetime"].iloc[0] == history["datetime"].min()
    assert table["datetime"].iloc[-1] == pd.Timestamp("2026-11-11 23:00")
    stored = pd.read_parquet(output_dir / "calendar.parquet")
    pd.testing.assert_frame_equal(table, stored)
    joined = history.merge(stored, on="datetime", validate="many_to_one")
    assert len(joined) == len(history)
    assert not joined[calendar.CALENDAR_COLUMNS].isna().any().any()
    pd.testing.assert_frame_equal(table, build.build_calendar(
        traffic_dir=history_dir, output_dir=output_dir, school_holidays_path=SCHOOL_PATH))


def test_download_paginates_and_preserves_previous_snapshot_on_failure(tmp_path, monkeypatch):
    record = json.loads(SCHOOL_PATH.read_text(encoding="utf-8"))["results"][0]
    offsets = []

    def response(url, timeout):
        parameters = parse_qs(urlparse(url).query)
        assert 'location = "Paris" AND zones = "Zone C"' == parameters["where"][0]
        assert timeout == 30
        offsets.append(int(parameters["offset"][0]))
        return io.BytesIO(json.dumps({"total_count": 2, "results": [record]}).encode())

    monkeypatch.setattr(calendar, "urlopen", response)
    path = calendar.download_school_holidays(tmp_path / "school.json")
    assert offsets == [0, 1]
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(calendar, "urlopen", fail)
    with pytest.raises(URLError):
        calendar.download_school_holidays(path)
    assert path.read_bytes() == before


def test_build_calendar_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build", "--calendar", "--start", "2026-11-07",
                                     "--school-holidays-file", str(SCHOOL_PATH),
                                     "--output-dir", str(tmp_path)])
    build.main()
    assert len(pd.read_parquet(tmp_path / "calendar.parquet")) == 120


def test_traffic_cli_defaults_are_preserved(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["build"])
    monkeypatch.setattr(build, "build_traffic", lambda *args, **kwargs: (calls.append(args), None))
    monkeypatch.setattr(build, "print_traffic_summary", lambda *args: None)
    build.main()
    assert calls == [(build.DEFAULT_START, build.DEFAULT_END, traffic.DEFAULT_RAW_DIR,
                      build.DEFAULT_TRAFFIC_DIR, traffic.DEFAULT_CONFIG)]


def test_failed_build_keeps_previous_calendar(tmp_path):
    build.build_calendar("2026-11-07", output_dir=tmp_path, school_holidays_path=SCHOOL_PATH)
    path = tmp_path / "calendar.parquet"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="academic years"):
        build.build_calendar("2010-01-01", output_dir=tmp_path, school_holidays_path=SCHOOL_PATH)
    assert path.read_bytes() == before


def test_no_history_requires_explicit_start(tmp_path):
    with pytest.raises(FileNotFoundError, match="traffic"):
        build.build_calendar(traffic_dir=tmp_path / "traffic", output_dir=tmp_path / "calendar",
                             school_holidays_path=SCHOOL_PATH)
    assert not (tmp_path / "calendar").exists()


def test_empty_download_page_is_rejected_without_replacing_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "school.json"
    path.write_text("previous snapshot", encoding="utf-8")
    monkeypatch.setattr(calendar, "urlopen", lambda *args, **kwargs:
                        io.BytesIO(b'{"total_count": 2, "results": []}'))
    with pytest.raises(ValueError, match="Incomplete or empty"):
        calendar.download_school_holidays(path)
    assert path.read_text(encoding="utf-8") == "previous snapshot"
