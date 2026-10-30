"""Offline tests for traffic processing, on small synthetic data only."""

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from traffic_prediction.data import build, traffic
from traffic_prediction.data.traffic import Arc

CONVENTION = Arc(
    "Convention", "evalue", "Lecourbe-Convention", "Convention-Blomet", 5671
)
PROXY = Arc("Proxy", "proxy", "Node-A", "Node-B", 42)
ARCS = [CONVENTION, PROXY]

HEADER = (
    "iu_ac;libelle;iu_nd_amont;libelle_nd_amont;iu_nd_aval;libelle_nd_aval;"
    "t_1h;q;k;etat_trafic;etat_barre;dessin"
)
# Same quoting as the real files: the geometry contains escaped quotes.
DESSIN = '"""<PLINE COURBE=""""1""""><PT X=""""1"""" Y=""""2""""/></PLINE>"""'


def raw_rows(*rows: tuple) -> pd.DataFrame:
    """Raw Open Data frame from (iu_ac, upstream, downstream, t_1h, q, k, etats)."""
    return pd.DataFrame(rows, columns=traffic.RAW_COLUMNS)


def convention_row(t_1h, q="100", k="5.0", etat_trafic="1", etat_barre="3"):
    return ("5671", CONVENTION.upstream_node, CONVENTION.downstream_node,
            t_1h, q, k, etat_trafic, etat_barre)


def archive_line(iu_ac, upstream, downstream, t_1h, q="", k="", etat="0"):
    return ";".join(
        [str(iu_ac), "Libelle", "1", upstream, "2", downstream,
         t_1h, q, k, etat, "3", DESSIN]
    )


def write_archive(path: Path, files: dict[str, list[str]]) -> Path:
    """Zip weekly text files with the Open Data header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, lines in files.items():
            archive.writestr(name, "\n".join([HEADER, *lines]) + "\n")
    return path


def write_config(path: Path) -> Path:
    path.write_text(
        "traffic:\n"
        "  evaluated_arcs:\n"
        "    - {name: Convention, upstream_node: Lecourbe-Convention,"
        " downstream_node: Convention-Blomet, iu_ac: 5671}\n"
        "  proxy_arcs:\n"
        "    - {name: Proxy, upstream_node: Node-A,"
        " downstream_node: Node-B, iu_ac: 42}\n",
        encoding="utf-8",
    )
    return path


# --- Configuration -----------------------------------------------------------


def test_baseline_config_lists_the_evaluated_and_proxy_arcs():
    arcs = traffic.load_arcs()

    evaluated = {arc.name: arc.iu_ac for arc in arcs if arc.role == "evalue"}
    assert evaluated == {
        "Champs-Elysées": 4264, "Convention": 5671, "Saint-Pères": 191
    }
    assert {arc.iu_ac for arc in arcs if arc.role == "proxy"} == {4272, 4265}


def test_config_arcs_are_read_with_their_role(tmp_path):
    assert traffic.load_arcs(write_config(tmp_path / "c.yaml")) == ARCS


@pytest.mark.parametrize(
    "content",
    [
        "traffic:\n  proxy_arcs:\n    - {name: P, upstream_node: A,"
        " downstream_node: B, iu_ac: 1}\n",
        "traffic:\n  evaluated_arcs:\n    - {name: X, upstream_node: A}\n",
        "traffic:\n  evaluated_arcs:\n"
        "    - {name: X, upstream_node: A, downstream_node: B, iu_ac: 1}\n"
        "    - {name: Y, upstream_node: C, downstream_node: D, iu_ac: 1}\n",
    ],
    ids=["no evaluated arc", "missing key", "duplicate iu_ac"],
)
def test_invalid_config_is_rejected(tmp_path, content):
    path = tmp_path / "c.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        traffic.load_arcs(path)


# --- Raw rows -> output format -----------------------------------------------


def test_only_configured_node_pairs_in_their_direction_are_kept():
    raw = raw_rows(
        convention_row("2025-01-01 01:00:00"),
        ("5672", "Convention-Blomet", "Lecourbe-Convention",
         "2025-01-01 01:00:00", "1", "1", "1", "3"),
        ("799", "Other-A", "Other-B", "2025-01-01 01:00:00", "1", "1", "1", "3"),
    )

    rows = traffic.normalize_raw(raw, ARCS)

    assert rows["iu_ac"].tolist() == [5671]


def test_unexpected_iu_ac_for_a_configured_pair_stops_processing():
    raw = raw_rows(convention_row("2025-01-01 01:00:00"))
    raw["iu_ac"] = "1234"

    with pytest.raises(ValueError, match="expected 5671"):
        traffic.normalize_raw(raw, ARCS)


def test_columns_are_renamed_and_typed():
    raw = raw_rows(convention_row("2025-01-01 01:00:00", q="264.0", k="1.7"))

    rows = traffic.normalize_raw(raw, ARCS)

    assert list(rows.columns) == traffic.OUTPUT_COLUMNS
    row = rows.iloc[0]
    assert (row["arc"], row["role"], row["iu_ac"]) == ("Convention", "evalue", 5671)
    assert row["datetime"] == pd.Timestamp("2025-01-01 01:00")
    assert (row["debit_horaire"], row["taux_occupation"]) == (264.0, 1.7)
    assert (row["etat_trafic"], row["etat_barre"]) == (1, 3)
    assert rows["iu_ac"].dtype == "int32"
    assert rows["datetime"].dtype == "datetime64[ns]"
    assert rows["etat_trafic"].dtype == "Int8"


def test_quality_flags_missing_and_impossible_values():
    raw = raw_rows(
        convention_row("2025-01-01 01:00:00", q="10", k="1"),
        convention_row("2025-01-01 02:00:00", q="10", k=None),
        convention_row("2025-01-01 03:00:00", q=None, k=None, etat_trafic="0"),
        convention_row("2025-01-01 04:00:00", q="10", k="150"),
        convention_row("2025-01-01 05:00:00", q="-1", k="1"),
        convention_row("2025-01-01 06:00:00", q="abc", k="1"),
        convention_row("2025-01-01 07:00:00", q="0", k="5"),
        convention_row("2025-01-01 08:00:00", q="0", k="0"),
    )

    rows = traffic.normalize_raw(raw, ARCS)

    assert rows["qualite_mesure"].tolist() == [
        "ok", "partielle", "vide", "invalide", "invalide", "invalide",
        "invalide", "ok",
    ]
    assert rows["taux_occupation"].isna().tolist()[3]
    assert rows["debit_horaire"].isna().tolist()[4:7] == [True, True, True]
    # Zero flow with an occupied loop: q is dropped, k is kept.
    assert rows["taux_occupation"].iloc[6] == 5
    assert rows["debit_horaire"].iloc[7] == 0


def test_api_timestamps_and_text_states_follow_archive_conventions():
    raw = raw_rows(
        convention_row("2025-01-15T10:00:00+00:00", etat_trafic="Fluide",
                       etat_barre="Invalide"),
        convention_row("2025-07-15T10:00:00+00:00", etat_trafic="Inconnu",
                       etat_barre="Barré"),
        convention_row("2025-07-15T12:00:00+02:00"),
        convention_row("2025-07-15T13:00:00Z"),
    )

    rows = traffic.normalize_raw(raw, ARCS)

    # Paris is UTC+1 in winter and UTC+2 in summer; output is naive local time.
    assert rows["datetime"].tolist() == [
        pd.Timestamp("2025-01-15 11:00"),
        pd.Timestamp("2025-07-15 12:00"),
        pd.Timestamp("2025-07-15 12:00"),
        pd.Timestamp("2025-07-15 15:00"),
    ]
    assert rows["etat_trafic"].tolist()[:2] == [1, 0]
    assert rows["etat_barre"].tolist()[:2] == [3, 2]


def test_mixed_time_zone_formats_are_rejected():
    raw = raw_rows(
        convention_row("2025-01-01 01:00:00"),
        convention_row("2025-01-01T01:00:00+00:00"),
    )

    with pytest.raises(ValueError, match="time zone"):
        traffic.normalize_raw(raw, ARCS)


# --- Hourly table -------------------------------------------------------------


def test_duplicate_hours_are_dropped_and_counted():
    raw = raw_rows(
        convention_row("2025-01-01 01:00:00", q="10"),
        convention_row("2025-01-01 01:00:00", q="10"),
        convention_row("2025-01-01 02:00:00", q="20"),
        convention_row("2025-01-01 02:00:00", q="25"),
    )
    stats = traffic.ProcessingStats()

    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-01-01", "2025-01-01", stats
    )

    convention = table[table["arc"] == "Convention"].set_index("datetime")
    assert convention.loc["2025-01-01 01:00", "debit_horaire"] == 10
    assert convention.loc["2025-01-01 02:00", "debit_horaire"] == 25
    assert (stats.identical_duplicates, stats.conflicting_duplicates) == (1, 1)


def test_missing_hours_are_added_for_every_arc():
    raw = raw_rows(convention_row("2025-01-01 05:00:00"))

    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-01-01", "2025-01-02"
    )

    assert len(table) == 2 * 48
    assert not table.duplicated(["arc", "datetime"]).any()
    assert table["datetime"].min() == pd.Timestamp("2025-01-01 00:00")
    assert table["datetime"].max() == pd.Timestamp("2025-01-02 23:00")
    absent = table[table["qualite_mesure"] == "absente"]
    assert len(absent) == 2 * 48 - 1
    assert absent["debit_horaire"].isna().all()
    assert absent["etat_trafic"].isna().all()
    assert set(table.loc[table["arc"] == "Proxy", "iu_ac"]) == {42}
    assert set(table.loc[table["arc"] == "Proxy", "role"]) == {"proxy"}


def test_rows_outside_the_period_are_counted():
    raw = raw_rows(
        convention_row("2024-12-31 23:00:00"),
        convention_row("2025-01-01 00:00:00"),
        convention_row("2025-01-02 00:00:00"),
    )
    stats = traffic.ProcessingStats()

    traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-01-01", "2025-01-01", stats
    )

    assert stats.outside_period == 2


def test_arc_without_measurements_warns_without_stopping(capsys):
    raw = raw_rows(convention_row("2025-01-01 01:00:00"))
    stats = traffic.ProcessingStats()

    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-01-01", "2025-01-01", stats
    )

    assert stats.arcs_without_measurements == ["Proxy"]
    assert "WARNING: arc 'Proxy'" in capsys.readouterr().err
    assert (table.loc[table["arc"] == "Proxy", "qualite_mesure"] == "absente").all()


def test_archive_hours_are_kept_unchanged_on_daylight_saving_days():
    # Archives always have 24 naive hours per day, even on 2025-03-30.
    raw = raw_rows(
        *(convention_row(f"2025-03-30 {hour:02d}:00:00", q=str(100 + hour))
          for hour in range(24))
    )

    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-03-30", "2025-03-30"
    )

    convention = table[table["arc"] == "Convention"]
    assert len(convention) == 24
    assert convention["debit_horaire"].tolist() == list(range(100, 124))


def test_api_repeated_autumn_hour_becomes_one_conflicting_duplicate():
    # 00:00 and 01:00 UTC on 2025-10-26 are both 02:00 in Paris.
    raw = raw_rows(
        convention_row("2025-10-26T00:00:00+00:00", q="10"),
        convention_row("2025-10-26T01:00:00+00:00", q="20"),
    )
    stats = traffic.ProcessingStats()

    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-10-26", "2025-10-26", stats
    )

    convention = table[table["arc"] == "Convention"].set_index("datetime")
    assert len(convention) == 24
    assert convention.loc["2025-10-26 02:00", "debit_horaire"] == 20
    assert stats.conflicting_duplicates == 1


# --- Archives and Parquet -------------------------------------------------------


def make_raw_dir(root: Path) -> Path:
    """2025 archive: two weekly files sharing a boundary row, plus an exact copy."""
    convention = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    archive = write_archive(
        root / "2025" / "a.zip",
        {
            "trafic_capteurs_2025_W00_20250101_20250102.txt": [
                archive_line(*convention, "2025-01-01 01:00:00", "100", "2.5", "1"),
                archive_line(799, "Other-A", "Other-B", "2025-01-01 01:00:00", "5"),
                archive_line(*convention, "2025-01-01 02:00:00"),
            ],
            "trafic_capteurs_2025_W01_20250102_20250109.txt": [
                archive_line(*convention, "2025-01-01 02:00:00"),
                archive_line(*convention, "2025-01-01 04:00:00", "80", "1.5", "1"),
            ],
        },
    )
    shutil.copy(archive, root / "2025" / "b.zip")
    return root


def test_archives_are_streamed_filtered_and_deduplicated(tmp_path):
    raw_dir = make_raw_dir(tmp_path / "raw")

    table, stats = traffic.process_archives(
        traffic.find_archives(raw_dir, "2025-01-01", "2025-01-01"),
        ARCS, "2025-01-01", "2025-01-01", chunk_rows=2,
    )

    assert [path.name for path in stats.archives_read] == ["a.zip"]
    assert [path.name for path in stats.duplicate_archives] == ["b.zip"]
    assert (stats.raw_rows, stats.arc_rows) == (5, 4)
    assert stats.identical_duplicates == 1
    convention = table[table["arc"] == "Convention"].set_index("datetime")
    assert convention["qualite_mesure"].value_counts().to_dict() == {
        "absente": 21, "ok": 2, "vide": 1
    }
    assert convention.loc["2025-01-01 01:00", "debit_horaire"] == 100
    assert convention.loc["2025-01-01 04:00", "taux_occupation"] == 1.5


def test_unsupported_compression_gives_a_clear_error(tmp_path):
    path = write_archive(tmp_path / "2023.zip", {"week.txt": []})
    # Mark the member as Deflate64 (method 9), like the 2023 archive.
    data = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        start = data.index(signature) + offset
        data[start:start + 2] = (9).to_bytes(2, "little")
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="Cannot read"):
        list(traffic.read_archive(path, ARCS))


def test_build_traffic_writes_valid_parquet_per_year(tmp_path):
    convention = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    raw_dir = tmp_path / "raw"
    write_archive(raw_dir / "2024" / "a.zip", {"w.txt": [
        archive_line(*convention, "2024-12-31 23:00:00", "264", "1.7", "1")
    ]})
    write_archive(raw_dir / "2025" / "a.zip", {"w.txt": [
        archive_line(*convention, "2025-01-01 01:00:00", "200", "1.2", "1")
    ]})
    output_dir = tmp_path / "processed"
    kept = output_dir / "year=2023" / "part-0.parquet"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"other year")

    table, _ = build.build_traffic(
        "2024-12-31", "2025-01-01", raw_dir, output_dir,
        write_config(tmp_path / "c.yaml"),
    )

    for year in (2024, 2025):
        path = output_dir / f"year={year}" / "part-0.parquet"
        assert pq.read_schema(path).equals(build.TRAFFIC_SCHEMA)
        written = pd.read_parquet(path)
        expected = table[table["datetime"].dt.year == year].reset_index(drop=True)
        pd.testing.assert_frame_equal(written, expected)
    assert kept.read_bytes() == b"other year"
    assert not list(output_dir.rglob("*.tmp"))


@pytest.mark.parametrize(
    "change",
    [
        lambda table: table.assign(taux_occupation=150.0),
        lambda table: table.iloc[1:],
        lambda table: table.assign(qualite_mesure="inconnue"),
    ],
    ids=["occupation above 100", "missing row", "unknown quality label"],
)
def test_validation_rejects_tables_breaking_the_output_contract(change):
    raw = raw_rows(convention_row("2025-01-01 01:00:00"))
    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS), ARCS, "2025-01-01", "2025-01-01"
    )
    build.validate_traffic(table, ARCS, "2025-01-01", "2025-01-01")

    with pytest.raises(ValueError, match="Invalid traffic table|Expected"):
        build.validate_traffic(change(table), ARCS, "2025-01-01", "2025-01-01")
