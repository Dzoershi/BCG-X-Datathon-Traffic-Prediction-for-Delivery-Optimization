"""Offline checks of road works and events: Open Data Paris is simulated, never called.

Synthetic arcs and works sit at known distances in metres, converted to degrees with
the module's flat projection, so every filter and every hour label can be checked by hand.
"""

import io
import json
import math
import sys
from types import SimpleNamespace
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from traffic_prediction.data import build, calendar_events, disruptions, traffic

ARCS = [arc for arc in traffic.load_arcs() if arc.role == "evalue"]
NAMES = [arc.name for arc in ARCS]
# One east-west line of about 146 m per arc, kilometres apart from each other.
LINES = {
    "Champs-Elysées": [(2.3000, 48.8700), (2.3020, 48.8700)],
    "Convention": [(2.2900, 48.8400), (2.2920, 48.8400)],
    "Saint-Pères": [(2.3300, 48.8570), (2.3320, 48.8570)],
}
CITY = "Ville de Paris (Tvx sur espace ou édifice public)"
PRIVATE = disruptions.PRIVATE_BUILDING_WORKS


def offset(point, east=0.0, north=0.0):
    """``point`` moved by metres east and north."""
    lon, lat = point
    return lon + east / (math.cos(math.radians(lat)) * 111_320), lat + north / 110_540


def middle(arc):
    (lon_a, lat_a), (lon_b, lat_b) = LINES[arc]
    return (lon_a + lon_b) / 2, (lat_a + lat_b) / 2


def square(centre, side=10.0):
    half = side / 2
    ring = [offset(centre, -half, -half), offset(centre, half, -half),
            offset(centre, half, half), offset(centre, -half, half)]
    return {"type": "Polygon", "coordinates": [[list(point) for point in ring + ring[:1]]]}


def square_at(arc, distance):
    """10 m square whose nearest side is ``distance`` metres north of the arc; 0 crosses it."""
    return square(offset(middle(arc), north=distance + 5 if distance else 0))


def feature(geometry):
    return {"type": "Feature", "geometry": geometry, "properties": {}}


def emprise(identifier, geometry, start="2026-11-01", end="2026-11-30", *, roadway=True, category=CITY):
    """A "chantiers à Paris" record."""
    return {"num_emprise": identifier, "date_debut": start, "date_fin": end,
            "chantier_categorie": category, "moa_principal": "DVD", "chantier_synthese": "Voirie",
            "localisation_detail": ["EMPRISE_CHAUSSEE" if roadway else "EMPRISE_TROTTOIR"],
            "geo_shape": feature(geometry) if geometry else None}


def perturbant(identifier, geometry, start="2026-11-01", end="2026-11-30"):
    """A "chantiers perturbants" record."""
    return {"identifiant": identifier, "date_debut": start, "date_fin": end, "typologie": "Concessionnaire",
            "objet": "CREATION_RESEAUX", "voie": "rue Test", "niveau_perturbation": "Très perturbant",
            "geo_shape": feature(geometry)}


def reference():
    ids = {arc.name: arc.iu_ac for arc in ARCS}
    return [{"iu_ac": str(ids[name]),
             "geo_shape": feature({"type": "LineString", "coordinates": [list(point) for point in line]})}
            for name, line in LINES.items()]


def open_data():
    """Records served per dataset; the fake server ignores the distance filter."""
    return {
        disruptions.REFERENCE_DATASET: reference(),
        "historique-chantiers-a-paris-en-2024": [
            emprise("EC1", square_at("Champs-Elysées", 10), "2024-03-01", "2024-03-31", category=PRIVATE),
            emprise("EC2", square_at("Convention", 0), "2024-05-06", "2024-05-07"),
            emprise("EC3", square_at("Saint-Pères", 120), "2024-11-01", "2024-11-30"),
        ],
        "chantiers-a-paris": [
            emprise("EC3", square_at("Saint-Pères", 120), "2024-11-01", "2026-12-31"),
            emprise("EC4", square_at("Saint-Pères", 20), roadway=False),
            emprise("EC5", square_at("Champs-Elysées", 120)),
            emprise("EC6", square_at("Champs-Elysées", 500)),
            emprise("EC7", None),
        ],
        disruptions.PERTURBANTS: [perturbant("CP1", square_at("Saint-Pères", 30), "2026-11-09", "2026-11-20")],
    }


class FakeOpenData:
    def __init__(self):
        self.datasets = open_data()

    def __call__(self, request, timeout):
        dataset = urlparse(request.full_url).path.split("/")[-3]
        return io.BytesIO(json.dumps(self.datasets[dataset]).encode())


def requests_made(http):
    """(dataset, where) of each simulated request."""
    made = []
    for args, _ in http.call_args_list:
        url = urlparse(args[0].full_url)
        made.append((url.path.split("/")[-3], parse_qs(url.query)["where"][0]))
    return made


@pytest.fixture(autouse=True)
def network():
    """Every test of this file runs offline against the fake server, without waiting."""
    server = FakeOpenData()
    with patch.object(disruptions, "urlopen", side_effect=server) as http, \
            patch.object(disruptions.time, "sleep") as sleep:
        yield SimpleNamespace(http=http, sleep=sleep, server=server)


def events_file(tmp_path, *rows, header=",".join(disruptions.EVENT_COLUMNS)):
    path = tmp_path / "events.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


CEREMONY = "2026-11-11 07:00,2026-11-11 14:00,Champs-Elysées,ceremonie,Cérémonie,https://example.org/a"
STRIKE = "2026-11-05 00:00,2026-11-06 00:00,toutes,greve,Grève,https://example.org/b"
NO_EVENTS = pd.DataFrame(columns=disruptions.EVENT_COLUMNS)


def at(table, arc, label, column):
    return table.set_index(["arc", "datetime"]).at[(arc, pd.Timestamp(label)), column]


# --- Geometry --------------------------------------------------------------------


@pytest.mark.parametrize("geometry,expected", [
    ({"type": "Point", "coordinates": offset(middle("Convention"), north=30)}, 30.0),
    ({"type": "Point", "coordinates": offset(LINES["Convention"][1], east=40)}, 40.0),
    ({"type": "MultiPoint", "coordinates": [offset(middle("Convention"), north=n) for n in (90, 60)]}, 60.0),
    ({"type": "LineString", "coordinates": [offset(point, north=25) for point in LINES["Convention"]]}, 25.0),
    (square_at("Convention", 0), 0.0),
    (square(middle("Convention"), side=400), 0.0),
    ({"type": "MultiPolygon", "coordinates": [square_at("Convention", 300)["coordinates"],
                                              square_at("Convention", 12)["coordinates"]]}, 12.0),
])
def test_distance_to_the_arc_line(geometry, expected):
    assert disruptions.distance_m(geometry, LINES["Convention"]) == pytest.approx(expected, abs=0.01)


def test_unknown_geometry_is_rejected():
    with pytest.raises(ValueError, match="Unsupported geometry type 'GeometryCollection'"):
        disruptions.distance_m({"type": "GeometryCollection", "coordinates": []}, LINES["Convention"])


def test_search_circle_covers_the_arc_plus_the_near_distance():
    centre, radius = disruptions.search_circle(LINES["Convention"])
    assert centre == pytest.approx(middle("Convention"))
    half_length = disruptions.distance_m({"type": "Point", "coordinates": LINES["Convention"][0]},
                                         [centre, centre])
    assert radius == math.ceil(half_length + disruptions.NEAR_M + 50)


# --- Works on the arcs -------------------------------------------------------------


def test_works_are_filtered_on_the_arc_and_dates_converted():
    records = [
        emprise("ON", {"type": "Point", "coordinates": offset(middle("Convention"), north=49.9)}),
        emprise("NEAR", {"type": "Point", "coordinates": offset(middle("Convention"), north=50.1)}),
        emprise("EDGE", {"type": "Point", "coordinates": offset(middle("Convention"), north=199.9)}),
        emprise("FAR", {"type": "Point", "coordinates": offset(middle("Convention"), north=200.1)}),
    ]
    table = disruptions.works_table(records, "chantiers-a-paris", "Convention", LINES["Convention"])
    assert list(table.columns) == disruptions.WORKS_COLUMNS
    assert table["identifiant"].tolist() == ["ON", "NEAR", "EDGE"]
    assert table["sur_arc"].tolist() == [True, False, False]
    assert table["distance_m"].tolist() == [49.9, 50.1, 199.9]
    assert table["date_debut"].dtype == "datetime64[ns]"
    assert table["date_debut"].iloc[0] == pd.Timestamp("2026-11-01")
    assert table["date_fin"].iloc[0] == pd.Timestamp("2026-11-30")
    assert table[["chaussee", "categorie", "description"]].iloc[0].tolist() == [True, CITY, "Voirie (DVD)"]


def test_perturbant_fields():
    record = perturbant("CP1", square_at("Convention", 30))
    (row,) = disruptions.works_table([record], disruptions.PERTURBANTS, "Convention",
                                     LINES["Convention"]).to_dict("records")
    assert row["identifiant"] == "CP1" and row["chaussee"] and row["categorie"] == "Concessionnaire"
    assert row["description"] == "CREATION_RESEAUX - rue Test (Très perturbant)"


def test_records_without_geometry_or_valid_dates_are_skipped_with_a_warning(capsys):
    geometry = square_at("Convention", 0)
    records = [emprise("NO_GEOMETRY", None), emprise("DAY_FIRST", geometry, start="07/11/2026"),
               emprise("NO_END", geometry, end=None), emprise("REVERSED", geometry, "2026-11-30", "2026-11-01"),
               emprise("OK", geometry)]
    table = disruptions.works_table(records, "chantiers-a-paris", "Convention", LINES["Convention"])
    assert table["identifiant"].tolist() == ["OK"]
    assert "4 records without geometry or valid dates skipped" in capsys.readouterr().err


# --- Manual events -----------------------------------------------------------------


def test_events_are_read_with_paris_times(tmp_path):
    events = disruptions.read_events(events_file(tmp_path, CEREMONY, STRIKE), NAMES)
    assert events["date_debut"].tolist() == [pd.Timestamp("2026-11-11 07:00"), pd.Timestamp("2026-11-05")]
    assert events["date_debut"].dt.tz is None
    assert events["arc"].tolist() == ["Champs-Elysées", "toutes"]


@pytest.mark.parametrize("row,message", [
    ("11/11/2026 07:00,2026-11-11 14:00,Champs-Elysées,ceremonie,X,https://example.org",
     "line 3: date_debut '11/11/2026 07:00' is not YYYY-MM-DD HH:MM"),
    ("2026-11-11,2026-11-11 14:00,Champs-Elysées,ceremonie,X,https://example.org", "line 3: date_debut"),
    ("2026-11-11 14:00,2026-11-11 07:00,Champs-Elysées,ceremonie,X,https://example.org",
     "line 3: date_fin must be after date_debut"),
    ("2026-11-11 07:00,2026-11-11 14:00,Concorde,ceremonie,X,https://example.org", "line 3: arc must be"),
    ("2026-11-11 07:00,2026-11-11 14:00,Champs-Elysées,concert,X,https://example.org", "line 3: type must be"),
    ("2026-11-11 07:00,2026-11-11 14:00,Champs-Elysées,ceremonie, ,https://example.org",
     "line 3: description is empty"),
    ("2026-11-11 07:00,2026-11-11 14:00,Champs-Elysées,ceremonie,X,www.paris.fr", "line 3: source_url"),
    ("2026-11-11 07:00,2026-11-11 14:00,Champs-Elysées,ceremonie,X,", "line 3: source_url"),
])
def test_malformed_events_are_refused_with_their_line(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        disruptions.read_events(events_file(tmp_path, CEREMONY, row), NAMES)


def test_events_file_needs_the_exact_columns(tmp_path):
    with pytest.raises(ValueError, match="must have columns"):
        disruptions.read_events(events_file(tmp_path, header="debut,fin,arc,type,description,source"), NAMES)


def test_versioned_events_file_is_valid():
    events = disruptions.read_events(disruptions.DEFAULT_EVENTS_PATH, NAMES)
    eleventh = events[events["date_debut"].dt.normalize().eq(pd.Timestamp("2026-11-11"))]
    assert eleventh[["arc", "type"]].values.tolist() == [["Champs-Elysées", "ceremonie"]]
    assert "PROVISOIRES" in eleventh["description"].iloc[0]
    sundays = events.loc[events["type"].eq("pietonnisation"), "date_debut"]
    assert sundays.dt.dayofweek.eq(6).all() and sundays.dt.day.le(7).all()
    assert not sundays.dt.month.eq(1).any()
    marathons = events.loc[events["description"].str.startswith("Marathon de Paris"), "date_debut"]
    assert marathons.dt.strftime("%Y-%m-%d").tolist() == ["2024-04-07", "2025-04-13", "2026-04-12"]
    not_applied = events.loc[events["description"].str.contains(disruptions.NOT_APPLIED), "date_debut"]
    assert not_applied.dt.strftime("%Y-%m-%d").tolist() == ["2025-08-03", "2025-09-07", "2025-11-02"]


# --- Hourly table ------------------------------------------------------------------


def test_events_follow_end_of_hour_labels(tmp_path):
    events = disruptions.read_events(events_file(tmp_path, CEREMONY, STRIKE), NAMES)
    works = disruptions.works_table([], "chantiers-a-paris", "Convention", LINES["Convention"])
    table = disruptions.hourly_table(works, events, NAMES, "2026-11-05", "2026-11-11")
    disruptions.validate_disruptions(table, NAMES, "2026-11-05", "2026-11-11")
    ceremony = table[table["arc"].eq("Champs-Elysées") & table["evenement_actif"]
                     & table["datetime"].dt.day.eq(11)]
    # 07:00-14:00 covers the hours ending at 08:00 ... 14:00.
    assert ceremony["datetime"].dt.hour.tolist() == list(range(8, 15))
    assert ceremony["description"].iloc[0] == "Ceremonie : Cérémonie"
    for arc in NAMES:
        assert not at(table, arc, "2026-11-05 00:00", "evenement_actif")
        assert at(table, arc, "2026-11-05 01:00", "evenement_actif")
        assert at(table, arc, "2026-11-06 00:00", "evenement_actif")
        assert not at(table, arc, "2026-11-06 01:00", "evenement_actif")
    assert not at(table, "Convention", "2026-11-11 10:00", "evenement_actif")
    assert at(table, "Convention", "2026-11-11 10:00", "description") == ""


def test_events_marked_not_applied_are_kept_but_never_active(tmp_path):
    skipped = CEREMONY.replace("Cérémonie", "Cérémonie ; Non appliqué d'après le capteur")
    events = disruptions.read_events(events_file(tmp_path, skipped), NAMES)
    assert len(events) == 1
    works = disruptions.works_table([], "chantiers-a-paris", "Convention", LINES["Convention"])
    table = disruptions.hourly_table(works, events, NAMES, "2026-11-11", "2026-11-11")
    assert not table["evenement_actif"].any()
    assert table["description"].eq("").all()


def test_works_days_cover_whole_days_with_end_of_hour_labels():
    record = emprise("EC", square_at("Convention", 0), "2026-11-07", "2026-11-08")
    works = disruptions.works_table([record], "chantiers-a-paris", "Convention", LINES["Convention"])
    table = disruptions.hourly_table(works, NO_EVENTS, NAMES, "2026-11-06", "2026-11-09")
    active = table.loc[table["arc"].eq("Convention") & table["chantier_actif"], "datetime"]
    assert active.min() == pd.Timestamp("2026-11-07 01:00")
    assert active.max() == pd.Timestamp("2026-11-09 00:00")
    assert len(active) == 48
    assert not table.loc[table["arc"].ne("Convention"), "chantier_actif"].any()


@pytest.mark.parametrize("dataset,record,expected", [
    (disruptions.PERTURBANTS, perturbant("CP", square_at("Convention", 30)), True),
    ("chantiers-a-paris", emprise("CITY", square_at("Convention", 30)), True),
    ("chantiers-a-paris", emprise("NETWORK", square_at("Convention", 30), category="Opérateurs de réseau"), True),
    ("chantiers-a-paris", emprise("PRIVATE", square_at("Convention", 30), category=PRIVATE), False),
    ("chantiers-a-paris", emprise("SIDEWALK", square_at("Convention", 30), roadway=False), False),
    ("chantiers-a-paris", emprise("NEAR", square_at("Convention", 120)), False),
    (disruptions.PERTURBANTS, perturbant("CP_NEAR", square_at("Convention", 120)), False),
])
def test_chantier_actif_rule(dataset, record, expected):
    works = disruptions.works_table([record], dataset, "Convention", LINES["Convention"])
    assert len(works) == 1
    table = disruptions.hourly_table(works, NO_EVENTS, NAMES, "2026-11-07", "2026-11-07")
    assert at(table, "Convention", "2026-11-07 12:00", "chantier_actif") == expected


def test_descriptions_list_each_active_item_once():
    records = [emprise("A", square_at("Convention", 0)), emprise("B", square_at("Convention", 10))]
    works = disruptions.works_table(records, "chantiers-a-paris", "Convention", LINES["Convention"])
    table = disruptions.hourly_table(works, NO_EVENTS, NAMES, "2026-11-07", "2026-11-07")
    assert at(table, "Convention", "2026-11-07 12:00", "description") == "Chantier : Voirie (DVD)"


def small_table():
    works = disruptions.works_table([], "chantiers-a-paris", "Convention", LINES["Convention"])
    return disruptions.hourly_table(works, NO_EVENTS, NAMES, "2026-11-07", "2026-11-08")


@pytest.mark.parametrize("corrupt,message", [
    (lambda t: t.assign(extra=1), "Unexpected disruption columns"),
    (lambda t: t.assign(datetime=t["datetime"].dt.tz_localize("UTC")), "timezone-free"),
    (lambda t: t.assign(arc=t["arc"].where(t.index != 0, "Concorde")), "differ"),
    (lambda t: pd.concat([t, t.iloc[[5]]]), "Duplicate \\(arc, datetime\\) rows: 1"),
    (lambda t: t.drop(index=7), "must cover"),
    (lambda t: t.assign(chantier_actif=t["chantier_actif"].astype(int)), "chantier_actif must be boolean"),
    (lambda t: t.assign(description=t["description"].where(t.index != 3, None)), "must be a string"),
])
def test_validation_rejects_inconsistent_tables(corrupt, message):
    table = small_table()
    assert len(table) == 3 * 48
    disruptions.validate_disruptions(table, NAMES, "2026-11-07", "2026-11-08")
    with pytest.raises(ValueError, match=message):
        disruptions.validate_disruptions(corrupt(table), NAMES, "2026-11-07", "2026-11-08")


# --- Download ----------------------------------------------------------------------


def test_arc_lines_come_from_the_reference_in_config_order(network, tmp_path):
    lines = disruptions.download_arc_lines(ARCS, tmp_path)
    assert requests_made(network.http) == [(disruptions.REFERENCE_DATASET, "iu_ac in ('4264', '5671', '191')")]
    assert network.http.call_args.kwargs == {"timeout": 60}
    assert list(lines) == NAMES
    assert lines["Saint-Pères"] == LINES["Saint-Pères"]
    (raw,) = tmp_path.glob("referentiel-comptages-routiers_*Z.json")
    assert json.loads(raw.read_text(encoding="utf-8")) == reference()


def test_missing_or_invalid_reference_geometry_is_rejected(network, tmp_path):
    network.server.datasets[disruptions.REFERENCE_DATASET] = reference()[:2]
    with pytest.raises(ValueError, match="No reference geometry for arcs \\['Saint-Pères'\\]"):
        disruptions.download_arc_lines(ARCS, tmp_path)
    point = {"iu_ac": "191", "geo_shape": feature({"type": "Point", "coordinates": [2.33, 48.857]})}
    network.server.datasets[disruptions.REFERENCE_DATASET] = [*reference()[:2], point]
    with pytest.raises(ValueError, match="Arc 191 has no line geometry"):
        disruptions.download_arc_lines(ARCS, tmp_path)


def test_works_requests_raw_copies_and_current_rows_win(network, tmp_path):
    works = disruptions.download_works(LINES, tmp_path)
    made = requests_made(network.http)
    assert [dataset for dataset, _ in made] == list(disruptions.WORKS_DATASETS) * 3
    (lon, lat), radius = disruptions.search_circle(LINES["Champs-Elysées"])
    assert made[0][1] == f"within_distance(geo_shape, geom'POINT({lon:.6f} {lat:.6f})', {radius}m)"
    assert len(list(tmp_path.glob("*Z.json"))) == 9
    assert len(list(tmp_path.glob("chantiers-perturbants_saint-peres_*Z.json"))) == 1
    kept = works.set_index("identifiant")
    assert sorted(kept.index) == ["CP1", "EC1", "EC2", "EC3", "EC4", "EC5"]
    assert kept.at["EC3", "source"] == "chantiers-a-paris"
    assert kept.at["EC3", "date_fin"] == pd.Timestamp("2026-12-31")
    assert kept.at["EC1", "arc"] == "Champs-Elysées" and kept.at["CP1", "arc"] == "Saint-Pères"


@pytest.mark.parametrize("failure", [
    URLError("offline"),
    TimeoutError("slow"),
    io.BytesIO(b'[{"incomplete":'),
    HTTPError("https://opendata.paris.fr", 429, "Too many requests", {}, None),
    HTTPError("https://opendata.paris.fr", 503, "Unavailable", {}, None),
])
def test_temporary_failure_is_retried(network, failure):
    network.http.side_effect = [failure, io.BytesIO(b"[]")]
    assert disruptions.fetch_json(disruptions.EXPORT_URL.format(dataset="x"), {}) == []
    assert network.http.call_count == 2
    assert network.sleep.call_args_list == [call(1)]


def test_network_gives_up_after_three_attempts(network):
    network.http.side_effect = URLError("offline")
    with pytest.raises(URLError):
        disruptions.fetch_json(disruptions.EXPORT_URL.format(dataset="x"), {})
    assert network.sleep.call_args_list == [call(1), call(2)]


def test_rejected_request_is_not_retried(network):
    network.http.side_effect = HTTPError("https://opendata.paris.fr", 400, "Bad Request", {}, None)
    with pytest.raises(ValueError, match="rejected the request \\(400\\)"):
        disruptions.fetch_json(disruptions.EXPORT_URL.format(dataset="x"), {})
    assert network.http.call_count == 1
    network.sleep.assert_not_called()


# --- Build and command line --------------------------------------------------------


def test_build_writes_works_and_hourly_table(network, tmp_path, capsys):
    events = events_file(tmp_path, CEREMONY, STRIKE)
    table = build.build_disruptions(output_dir=tmp_path / "out", events_path=events, raw_dir=tmp_path / "raw")
    hours = len(pd.date_range("2024-01-01", "2026-11-11 23:00", freq="h"))
    assert len(table) == 3 * hours
    assert pd.read_parquet(tmp_path / "out" / "disruptions.parquet").equals(table)
    works = pd.read_parquet(tmp_path / "out" / "chantiers.parquet")
    assert list(works.columns) == disruptions.WORKS_COLUMNS and len(works) == 6

    # Private building works on the arc stay out of chantier_actif.
    assert not at(table, "Champs-Elysées", "2024-03-15 12:00", "chantier_actif")
    assert at(table, "Convention", "2024-05-06 01:00", "chantier_actif")
    assert not at(table, "Convention", "2024-05-08 01:00", "chantier_actif")
    assert not at(table, "Saint-Pères", "2026-11-09 00:00", "chantier_actif")
    assert at(table, "Saint-Pères", "2026-11-09 01:00", "chantier_actif")
    assert at(table, "Champs-Elysées", "2026-11-11 10:00", "evenement_actif")

    output = capsys.readouterr().out
    assert "=== Disruptions from 2026-11-07 to 2026-11-11 ===" in output
    assert "CREATION_RESEAUX - rue Test" in output and "- Ceremonie : Cérémonie" in output


def test_malformed_events_stop_the_build_before_any_download(network, tmp_path):
    events = events_file(tmp_path, CEREMONY.replace("ceremonie", "concert"))
    with pytest.raises(ValueError, match="type must be"):
        build.build_disruptions(output_dir=tmp_path / "out", events_path=events, raw_dir=tmp_path / "raw")
    assert network.http.call_count == 0
    assert not (tmp_path / "out").exists()


def test_cli_passes_disruptions_options(tmp_path):
    calls = []
    argv = ["build", "--disruptions", "--start", "2026-10-01", "--events-file", str(tmp_path / "e.csv")]
    with patch.object(sys, "argv", argv), \
            patch.object(build, "build_disruptions", lambda *args, **kwargs: calls.append((args, kwargs))):
        build.main()
    assert calls == [(("2026-10-01", calendar_events.PREDICTION_END, build.DEFAULT_DISRUPTIONS_DIR),
                      {"config_path": traffic.DEFAULT_CONFIG, "events_path": tmp_path / "e.csv"})]


@pytest.mark.parametrize("other", ["--calendar", "--weather"])
def test_cli_rejects_disruptions_with_another_source(capsys, other):
    with patch.object(sys, "argv", ["build", "--disruptions", other]), pytest.raises(SystemExit) as stop:
        build.main()
    assert stop.value.code == 2
    assert "choose --disruptions alone" in capsys.readouterr().err
