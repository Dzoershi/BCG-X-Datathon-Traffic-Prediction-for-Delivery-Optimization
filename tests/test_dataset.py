"""Offline checks of the modelling dataset, on small synthetic processed sources.

Three traffic days frame the March 2026 clock change. The fake weather encodes each
label (temperature = hour + day / 100), so every join can be checked by hand.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from traffic_prediction.data import build, calendar_events, dataset, disruptions, traffic, weather

SCHOOL_PATH = Path(__file__).parent / "fixtures/paris_school_holidays.json"
ARCS = traffic.load_arcs()
EVALUATED = [arc.name for arc in ARCS if arc.role == "evalue"]
TRAFFIC_START, TRAFFIC_END = "2026-03-28", "2026-03-30"
SPRING_FORWARD = pd.Timestamp("2026-03-29 02:00")
FORECAST_START = pd.Timestamp("2026-11-01")
EVENT_HOUR = pd.Timestamp("2026-03-28 09:00")


# --- Synthetic sources ---------------------------------------------------------------


def hours_to_prediction_end() -> pd.DatetimeIndex:
    return pd.date_range(TRAFFIC_START, f"{calendar_events.PREDICTION_END} 23:00", freq="h")


def traffic_table() -> pd.DataFrame:
    """Processed traffic grid: hours 07-09 measured (flow = 100 + hour), the others absent."""
    table = traffic.build_hourly_table(pd.DataFrame(columns=traffic.OUTPUT_COLUMNS), ARCS,
                                       TRAFFIC_START, TRAFFIC_END)
    table = table.astype({"debit_horaire": "float64", "taux_occupation": "float64",
                          "etat_trafic": "Int8", "etat_barre": "Int8"})
    measured = table["datetime"].dt.hour.between(7, 9)
    table.loc[measured, "debit_horaire"] = 100.0 + table.loc[measured, "datetime"].dt.hour
    table.loc[measured, "taux_occupation"] = 5.0
    table.loc[measured, "qualite_mesure"] = "ok"
    return table


def calendar_table() -> pd.DataFrame:
    return calendar_events.calendar_table(TRAFFIC_START, calendar_events.PREDICTION_END,
                                          school_holidays_path=SCHOOL_PATH)


def weather_table() -> pd.DataFrame:
    hours = hours_to_prediction_end()
    return pd.DataFrame({
        "datetime": hours,
        "precipitation_mm": hours.hour / 10,
        "temperature_c": hours.hour + hours.day / 100,
        "wind_speed_kmh": hours.month.astype("float64"),
        "source": np.where(hours < FORECAST_START, "historical_forecast", "forecast"),
    })[weather.WEATHER_COLUMNS]


def disruptions_table() -> pd.DataFrame:
    """Works on Saint-Pères all the time, one event hour on the Champs-Elysées."""
    index = pd.MultiIndex.from_product([EVALUATED, hours_to_prediction_end()], names=dataset.KEYS)
    table = index.to_frame(index=False)
    table["chantier_actif"] = table["arc"].eq("Saint-Pères")
    table["evenement_actif"] = table["arc"].eq("Champs-Elysées") & table["datetime"].eq(EVENT_HOUR)
    table["description"] = ""
    return table[disruptions.DISRUPTION_COLUMNS]


def write_sources(root: Path, **replace: pd.DataFrame) -> dict[str, Path]:
    """Write the four processed inputs under root; keyword tables replace the defaults."""
    build.write_partitions(traffic_table(), root / "traffic")
    tables = {"calendar": calendar_table(), "weather": weather_table(),
              "disruptions": disruptions_table(), **replace}
    for name, table in tables.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        table.to_parquet(root / name / f"{name}.parquet", index=False)
    return {f"{name}_dir": root / name for name in ["traffic", *tables]}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build shared by the read-only tests."""
    root = tmp_path_factory.mktemp("sources")
    dirs = write_sources(root)
    table, covariates = build.build_dataset(output_dir=root / "dataset", **dirs)
    return SimpleNamespace(root=root, dirs=dirs, table=table, covariates=covariates)


def row(frame: pd.DataFrame, arc: str, label: str) -> pd.Series:
    return frame.set_index(dataset.KEYS).loc[(arc, pd.Timestamp(label))]


# --- Joins ---------------------------------------------------------------------------


def test_each_row_gets_the_calendar_and_weather_of_its_own_hour(built):
    saturday = row(built.table, "Convention", "2026-03-28 08:00")
    assert saturday["debit_horaire"] == 108.0
    assert saturday["temperature_c"] == pytest.approx(8.28)
    assert (saturday["hour"], saturday["day_of_week"], saturday["is_weekend"]) == (8, 5, True)
    # Midnight labels describe their own date, as in the calendar of issue #16.
    monday = row(built.table, "Saint-Pères", "2026-03-30 00:00")
    assert (monday["day_of_week"], monday["is_weekend"]) == (0, False)
    assert monday["temperature_c"] == pytest.approx(0.30)
    # One weather row serves every arc of the hour.
    assert built.table.groupby("datetime")["temperature_c"].nunique().eq(1).all()
    assert built.table["weather_source"].eq("historical_forecast").all()


def test_dataset_keeps_every_traffic_row_unchanged(built):
    stored = pd.read_parquet(built.dirs["traffic_dir"], columns=traffic.OUTPUT_COLUMNS)
    assert len(built.table) == len(stored) == len(ARCS) * 3 * 24
    assert not built.table.duplicated(dataset.KEYS).any()
    pd.testing.assert_frame_equal(built.table[traffic.OUTPUT_COLUMNS], dataset.sort_rows(stored, ARCS))
    assert list(built.table["arc"].unique()) == [arc.name for arc in ARCS]
    assert list(built.table.columns) == dataset.DATASET_COLUMNS


def test_spring_forward_hour_is_kept_with_its_covariates(built):
    rows = built.table[built.table["datetime"].eq(SPRING_FORWARD)]
    assert len(rows) == len(ARCS)
    assert rows["qualite_mesure"].eq("absente").all() and rows["debit_horaire"].isna().all()
    assert rows["hour"].eq(2).all() and rows["temperature_c"].notna().all()
    assert built.table.groupby(built.table["datetime"].dt.date).size().eq(len(ARCS) * 24).all()


def test_proxies_get_na_flags_and_evaluated_arcs_their_own(built):
    for frame in (built.table, built.covariates):
        proxy = frame["role"].eq("proxy")
        assert frame.loc[proxy, dataset.DISRUPTION_FLAGS].isna().all().all()
        assert frame.loc[~proxy, dataset.DISRUPTION_FLAGS].notna().all().all()
        assert all(frame[flag].dtype == "boolean" for flag in dataset.DISRUPTION_FLAGS)
    flagged = built.table[built.table["evenement_actif"].fillna(False)]
    assert list(zip(flagged["arc"], flagged["datetime"])) == [("Champs-Elysées", EVENT_HOUR)]
    works = built.table.groupby("arc", sort=False)["chantier_actif"].sum()
    assert works.to_dict() == {"Champs-Elysées": 0, "Convention": 0, "Saint-Pères": 72,
                               ARCS[3].name: 0, ARCS[4].name: 0}


def test_covariates_cover_every_arc_until_the_last_day_to_predict(built):
    covariates = built.covariates
    assert list(covariates.columns) == dataset.COVARIATE_COLUMNS
    assert len(covariates) == len(ARCS) * len(hours_to_prediction_end())
    assert covariates["datetime"].max() == pd.Timestamp("2026-11-11 23:00")
    # A model finds the covariates of its target hours, beyond the traffic history.
    holiday = row(covariates, "Convention", "2026-11-11 08:00")
    assert holiday["is_holiday"] and holiday["weather_source"] == "forecast"
    # Same covariates as the dataset on the traffic hours.
    shared = covariates.merge(built.table[dataset.KEYS], on=dataset.KEYS)
    pd.testing.assert_frame_equal(shared[dataset.COVARIATES], built.table[dataset.COVARIATES])


# --- Period ----------------------------------------------------------------------------


def test_period_options_select_traffic_days(built, tmp_path):
    table, covariates = build.build_dataset("2026-03-29", "2026-03-29", tmp_path, **built.dirs)
    assert len(table) == len(ARCS) * 24
    assert table["datetime"].min() == covariates["datetime"].min() == pd.Timestamp("2026-03-29")
    assert covariates["datetime"].max() == pd.Timestamp("2026-11-11 23:00")


@pytest.mark.parametrize("start, end", [("2026-03-27", None), (None, "2026-03-31")])
def test_period_beyond_the_stored_traffic_is_refused(built, tmp_path, start, end):
    with pytest.raises(ValueError, match="choose --start and --end within these days"):
        build.build_dataset(start, end, tmp_path / "dataset", **built.dirs)
    assert not (tmp_path / "dataset").exists()


# --- Incomplete inputs -------------------------------------------------------------------


def mixed_weather() -> pd.DataFrame:
    table = weather_table()
    table.loc[5, "source"] = "archive"
    return table


@pytest.mark.parametrize("name, broken, message", [
    ("weather", lambda: weather_table().iloc[:-24], r"Invalid weather input: .*24 missing hours.*--weather"),
    ("weather", mixed_weather, "Invalid weather input: Weather history must come from one known source"),
    ("calendar", lambda: calendar_table().loc[lambda t: t["datetime"].ne(SPRING_FORWARD)],
     r"Invalid calendar input: .*1 missing hours.*--calendar"),
    ("calendar", lambda: calendar_table().drop(columns="is_bridge"), "Invalid calendar input: Unexpected calendar columns"),
    ("disruptions", lambda: disruptions_table().loc[lambda t: t["arc"].ne("Convention")],
     r"Invalid disruptions input: .*--disruptions"),
])
def test_incomplete_input_stops_the_build_and_keeps_previous_files(tmp_path, name, broken, message):
    dirs = write_sources(tmp_path)
    output = tmp_path / "dataset"
    build.build_dataset(output_dir=output, **dirs)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    broken().to_parquet(dirs[f"{name}_dir"] / f"{name}.parquet", index=False)
    with pytest.raises(ValueError, match=message):
        build.build_dataset(output_dir=output, **dirs)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_incomplete_traffic_grid_is_refused(tmp_path):
    dirs = write_sources(tmp_path)
    build.write_partitions(traffic_table().drop(index=10), dirs["traffic_dir"])
    with pytest.raises(ValueError, match=r"Invalid traffic input: .*traffic_prediction.data.validate"):
        build.build_dataset(output_dir=tmp_path / "dataset", **dirs)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("source", ["traffic", "weather"])
def test_missing_input_file_is_named(tmp_path, source):
    dirs = write_sources(tmp_path)
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="missing"):
        build.build_dataset(output_dir=tmp_path / "dataset", **{**dirs, f"{source}_dir": missing})


@pytest.mark.parametrize("corrupt, message", [
    (lambda t: t.drop(index=3), "rows, the traffic of the period"),
    (lambda t: pd.concat([t, t.iloc[[3]]], ignore_index=True), "rows, the traffic of the period"),
    (lambda t: pd.concat([t.drop(index=3), t.iloc[[4]]], ignore_index=True), "duplicate"),
    (lambda t: t.assign(debit_horaire=t["debit_horaire"].fillna(0.0)), "traffic columns differ"),
    (lambda t: t.assign(temperature_c=t["temperature_c"].mask(t.index == 7)), "without calendar or weather"),
    (lambda t: t.assign(evenement_actif=t["evenement_actif"].fillna(False)), "proxy rows with disruption flags"),
    (lambda t: t.assign(chantier_actif=t["chantier_actif"].mask(t["arc"].eq("Convention"))),
     "evaluated arcs without disruption flags"),
    (lambda t: t.assign(hour=t["hour"].astype("int64")), "hour must be int8"),
    (lambda t: t.drop(columns="weather_source"), "expected columns"),
])
def test_dataset_contract_is_enforced(built, corrupt, message):
    first, last = built.covariates["datetime"].iloc[[0, -1]]
    with pytest.raises(ValueError, match=message):
        dataset.validate_tables(corrupt(built.table), built.covariates, built.table[traffic.OUTPUT_COLUMNS],
                                ARCS, first, last)


def test_covariates_contract_is_enforced(built):
    first, last = built.covariates["datetime"].iloc[[0, -1]]
    lost_hour = built.covariates.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="one row per configured arc and hour"):
        dataset.validate_tables(built.table, lost_hour, built.table[traffic.OUTPUT_COLUMNS], ARCS, first, last)


# --- Files and version manifest --------------------------------------------------------


def test_files_round_trip_and_manifest(built):
    folder = built.root / "dataset"
    assert sorted(path.name for path in folder.iterdir()) == [
        dataset.MANIFEST_FILE, dataset.COVARIATES_FILE, dataset.DATASET_FILE]
    pd.testing.assert_frame_equal(pd.read_parquet(folder / dataset.DATASET_FILE), built.table)
    pd.testing.assert_frame_equal(pd.read_parquet(folder / dataset.COVARIATES_FILE), built.covariates)
    report = json.loads((folder / dataset.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert report["content_sha256"] == {dataset.DATASET_FILE: dataset.content_sha256(built.table),
                                        dataset.COVARIATES_FILE: dataset.content_sha256(built.covariates)}
    assert report["dataset"]["rows"] == len(built.table)
    assert (report["dataset"]["first_hour"], report["dataset"]["last_hour"]) == ("2026-03-28 00:00", "2026-03-30 23:00")
    assert report["dataset"]["rows_per_arc"] == {arc.name: 72 for arc in ARCS}
    assert report["covariates"]["last_hour"] == "2026-11-11 23:00"
    assert list(report["inputs_sha256"]) == ["traffic", "calendar", "weather", "disruptions"]
    assert list(report["inputs_sha256"]["traffic"]) == ["year=2026/part-0.parquet"]


def test_rebuild_gives_the_same_fingerprint(built, tmp_path):
    build.build_dataset(output_dir=tmp_path, **built.dirs)
    first = json.loads((built.root / "dataset" / dataset.MANIFEST_FILE).read_text(encoding="utf-8"))
    second = json.loads((tmp_path / dataset.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert second["fingerprint"] == first["fingerprint"]


def test_content_hash_follows_values_not_parquet_encoding(built, tmp_path):
    table = built.table
    table.to_parquet(tmp_path / "snappy.parquet", compression="snappy", index=False)
    assert dataset.content_sha256(pd.read_parquet(tmp_path / "snappy.parquet")) == dataset.content_sha256(table)
    warmer = table.assign(temperature_c=table["temperature_c"] + 0.1)
    proxy_false = table.assign(chantier_actif=table["chantier_actif"].fillna(False))
    assert len({dataset.content_sha256(frame) for frame in (table, warmer, proxy_false)}) == 3


# --- Command line ----------------------------------------------------------------------


@pytest.mark.parametrize("extra, output_dir", [([], build.DEFAULT_DATASET_DIR),
                                                (["--output-dir", "out"], Path("out"))])
def test_cli_passes_dataset_options(extra, output_dir):
    calls = []
    argv = ["build", "--dataset", "--end", "2026-10-30", *extra]
    with patch.object(sys, "argv", argv), \
            patch.object(build, "build_dataset", lambda *args, **kwargs: calls.append((args, kwargs))):
        build.main()
    assert calls == [((None, "2026-10-30", output_dir),
                      {"traffic_dir": build.DEFAULT_TRAFFIC_DIR, "config_path": traffic.DEFAULT_CONFIG})]


@pytest.mark.parametrize("other", ["--calendar", "--weather", "--disruptions"])
def test_cli_rejects_dataset_with_another_source(capsys, other):
    with patch.object(sys, "argv", ["build", "--dataset", other]), pytest.raises(SystemExit) as stop:
        build.main()
    assert stop.value.code == 2
    assert "choose --dataset alone" in capsys.readouterr().err


def test_cli_reports_a_failed_build(capsys, tmp_path):
    argv = ["build", "--dataset", "--traffic-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as stop:
        build.main()
    assert stop.value.code == 1
    assert "Dataset processing failed" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
