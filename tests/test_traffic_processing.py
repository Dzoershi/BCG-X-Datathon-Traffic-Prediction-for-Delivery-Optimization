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


def _valid_traffic_table():
    raw = raw_rows(convention_row("2025-01-01 01:00:00"))
    return traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS),
        ARCS, "2025-01-01", "2025-01-01",
    )


def recent_record(**changes):
    record = dict(zip(traffic.RAW_COLUMNS, convention_row(
        "2025-01-01T00:00:00Z",
        etat_trafic="Fluide",
        etat_barre="Ouvert",
    )))
    record.update(changes)
    return record


def write_recent_snapshot(path, *records):
    import json

    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


# --- Configuration ------------------------------------------------------------


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


# --- Normalization and timestamps ---------------------------------------------


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


@pytest.mark.parametrize(
    "bad", ["inf", "-inf", float("inf"), float("-inf")]
)
@pytest.mark.parametrize(
    "field,column",
    [("q", "debit_horaire"), ("k", "taux_occupation")],
)
def test_non_finite_measurements_are_flagged(bad, field, column):
    raw = raw_rows(convention_row(
        "2025-01-01 01:00:00", **{field: bad}
    ))

    rows = traffic.normalize_raw(raw, ARCS)

    assert len(rows) == 1
    assert pd.isna(rows.loc[0, column])
    assert rows.loc[0, "qualite_mesure"] == "invalide"


@pytest.mark.parametrize(
    "missing", [None, "", " ", float("nan"), pd.NA]
)
def test_missing_measurements_are_not_marked_invalid(missing):
    values, invalid = traffic.parse_measure(
        pd.Series([missing], dtype="object")
    )

    assert pd.isna(values.iloc[0])
    assert not invalid.iloc[0]


def test_finite_measurement_boundaries_are_preserved():
    values, invalid = traffic.parse_measure(
        pd.Series(["0", "100"]), upper=100
    )

    assert values.tolist() == [0.0, 100.0]
    assert invalid.tolist() == [False, False]


@pytest.mark.parametrize("field", ["etat_trafic", "etat_barre"])
@pytest.mark.parametrize(
    "bad", [-1, 99, 1.5, "unknown", "inf", float("inf"), True, False]
)
def test_invalid_supplied_state_is_rejected(field, bad):
    raw = raw_rows(convention_row(
        "2025-01-01 01:00:00", **{field: bad}
    ))

    with pytest.raises(ValueError, match="Invalid state"):
        traffic.normalize_raw(raw, ARCS)


@pytest.mark.parametrize(
    "labels",
    [traffic.ETAT_TRAFIC_CODES, traffic.ETAT_BARRE_CODES],
)
def test_valid_state_codes_and_labels_are_preserved(labels):
    codes = list(labels.values())
    values = pd.Series(
        codes
        + [str(code) for code in codes]
        + [float(code) for code in codes]
        + [f" {label.upper()} " for label in labels]
    )

    result = traffic.parse_state(values, labels)

    assert result.tolist() == codes * 4
    assert str(result.dtype) == "Int8"


@pytest.mark.parametrize(
    "labels",
    [traffic.ETAT_TRAFIC_CODES, traffic.ETAT_BARRE_CODES],
)
def test_missing_states_stay_missing(labels):
    values = pd.Series(
        [None, "", " ", float("nan"), pd.NA], dtype="object"
    )

    result = traffic.parse_state(values, labels)

    assert result.isna().all()
    assert str(result.dtype) == "Int8"


@pytest.mark.parametrize("timestamp", [
    "2025-01-01 01:30:00",
    "2025-01-01 01:00:01",
    "2025-01-01 01:00:00.000001",
    "2025-01-01T00:30:00Z",
    "NaT",
])
def test_bad_hour_labels_are_rejected_before_grid_can_drop_them(timestamp):
    with pytest.raises(ValueError, match="whole-hour|missing timestamps"):
        traffic.normalize_raw(
            raw_rows(convention_row(timestamp)), [CONVENTION]
        )


# --- Hourly grid and duplicates -----------------------------------------------


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


# --- Archive selection and source readers -------------------------------------


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
    # Mark the member with an unsupported method; Deflate64 (9) is supported.
    data = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        start = data.index(signature) + offset
        data[start:start + 2] = (99).to_bytes(2, "little")
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="Cannot read"):
        list(traffic.read_archive(path, ARCS))


@pytest.mark.parametrize(
    "start,end,expected_years",
    [
        ("2025-01-01", "2025-01-01", ["2024", "2025"]),
        ("2025-01-02", "2025-01-02", ["2025"]),
        ("2024-12-31", "2025-01-01", ["2024", "2025"]),
    ],
)
def test_archive_selection_handles_year_boundaries(
    tmp_path, start, end, expected_years
):
    for year in [2023, 2024, 2025]:
        write_archive(tmp_path / str(year) / "data.zip", {"week.txt": []})

    paths = traffic.find_archives(tmp_path, start, end)

    assert [path.parent.name for path in paths] == expected_years


def test_previous_archive_supplies_first_midnight(tmp_path):
    nodes = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    write_archive(
        tmp_path / "2024" / "data.zip",
        {
            "week.txt": [
                archive_line(*nodes, "2024-12-31 23:00:00", "100", "2", "1"),
                archive_line(*nodes, "2025-01-01 00:00:00", "200", "3", "1"),
            ]
        },
    )
    write_archive(
        tmp_path / "2025" / "data.zip",
        {
            "week.txt": [
                archive_line(*nodes, "2025-01-01 01:00:00", "300", "4", "1"),
            ]
        },
    )

    table, stats = traffic.process_archives(
        traffic.find_archives(tmp_path, "2025-01-01", "2025-01-01"),
        [CONVENTION],
        "2025-01-01",
        "2025-01-01",
    )

    build.validate_traffic(table, [CONVENTION], "2025-01-01", "2025-01-01")
    hours = table.set_index("datetime")

    assert hours.loc["2025-01-01 00:00", "debit_horaire"] == 200
    assert hours.loc["2025-01-01 01:00", "debit_horaire"] == 300
    assert hours.loc["2025-01-01 00:00", "qualite_mesure"] == "ok"
    assert stats.outside_period == 1


def test_missing_previous_archive_warns_and_preserves_missing_hour(
    tmp_path, capsys
):
    nodes = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    write_archive(
        tmp_path / "2025" / "data.zip",
        {
            "week.txt": [
                archive_line(*nodes, "2025-01-01 01:00:00", "100", "2", "1"),
            ]
        },
    )

    paths = traffic.find_archives(tmp_path, "2025-01-01", "2025-01-01")

    assert "no readable archive for 2024" in capsys.readouterr().err

    table, _ = traffic.process_archives(
        paths, [CONVENTION], "2025-01-01", "2025-01-01"
    )
    midnight = table.set_index("datetime").loc["2025-01-01 00:00"]

    assert pd.isna(midnight["debit_horaire"])
    assert midnight["qualite_mesure"] == "absente"


def test_unsupported_optional_boundary_archive_warns(tmp_path, capsys):
    previous = write_archive(
        tmp_path / "2024" / "data.zip", {"week.txt": []}
    )

    # Mark the fixture with a compression method that remains unsupported.
    data = bytearray(previous.read_bytes())
    for signature, offset in [(b"PK\x03\x04", 8), (b"PK\x01\x02", 10)]:
        index = data.index(signature) + offset
        data[index:index + 2] = (99).to_bytes(2, "little")
    previous.write_bytes(data)

    current = write_archive(
        tmp_path / "2025" / "data.zip", {"week.txt": []}
    )

    paths = traffic.find_archives(tmp_path, "2025-01-01", "2025-01-01")

    assert paths == [current]
    assert "cannot read optional boundary archive" in capsys.readouterr().err


def test_recent_reader_batches_and_normalizes(tmp_path):
    import json

    records = [
        recent_record(),
        recent_record(t_1h="2025-01-01T01:00:00Z", q=None, k=None),
        recent_record(
            iu_ac="799",
            libelle_nd_amont="Other-A",
            libelle_nd_aval="Other-B",
        ),
        recent_record(t_1h="2025-01-01T02:00:00Z"),
        recent_record(t_1h="2025-01-01T03:00:00Z"),
    ]
    path = tmp_path / "traffic.jsonl"
    path.write_text(
        "\n" + "\n\n".join(json.dumps(row) for row in records),
        encoding="utf-8",
    )

    chunks = list(traffic.read_recent(path, ARCS, chunk_rows=2))

    assert [count for _, count in chunks] == [2, 2, 1]
    assert [len(rows) for rows, _ in chunks] == [2, 1, 1]

    rows = pd.concat([rows for rows, _ in chunks], ignore_index=True)

    assert rows["datetime"].tolist() == list(
        pd.date_range("2025-01-01 01:00", periods=4, freq="h")
    )
    assert rows["qualite_mesure"].tolist() == ["ok", "vide", "ok", "ok"]
    assert rows.loc[0, "etat_trafic"] == 1
    assert rows.loc[0, "etat_barre"] == 1
    assert pd.isna(rows.loc[1, "debit_horaire"])


@pytest.mark.parametrize("payload", [b"{", b"[]", b"{}", b"\xff"])
def test_recent_reader_rejects_malformed_records(tmp_path, payload):
    path = tmp_path / "traffic.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="line 1"):
        list(traffic.read_recent(path, ARCS))


def test_recent_reader_yields_before_reading_later_records(tmp_path):
    import json

    path = tmp_path / "traffic.jsonl"
    path.write_text(
        json.dumps(recent_record()) + "\n{",
        encoding="utf-8",
    )
    chunks = traffic.read_recent(path, ARCS, chunk_rows=1)

    rows, raw_count = next(chunks)
    assert raw_count == 1
    assert len(rows) == 1

    with pytest.raises(ValueError, match="line 2"):
        next(chunks)


def test_mixed_sources_use_explicit_order_and_count_conflicts(tmp_path):
    nodes = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    archive = write_archive(tmp_path / "history.zip", {"week.txt": [
        archive_line(*nodes, "2025-01-01 01:00:00", "100", "5", "1"),
        archive_line(*nodes, "2025-01-01 02:00:00", "200", "5", "1"),
    ]})
    first = write_recent_snapshot(
        tmp_path / "z_first.jsonl",
        recent_record(q="300"),
    )
    second = write_recent_snapshot(
        tmp_path / "a_second.jsonl",
        recent_record(q=None, k=None),
    )

    table, stats = traffic.process_sources(
        [archive], ARCS, "2025-01-01", "2025-01-01",
        chunk_rows=1, recent_paths=[first, second],
    )
    build.validate_traffic(table, ARCS, "2025-01-01", "2025-01-01")
    hours = table[table["arc"] == "Convention"].set_index("datetime")

    assert pd.isna(hours.loc["2025-01-01 01:00", "debit_horaire"])
    assert pd.isna(hours.loc["2025-01-01 01:00", "taux_occupation"])
    assert hours.loc["2025-01-01 01:00", "qualite_mesure"] == "vide"
    assert hours.loc["2025-01-01 02:00", "debit_horaire"] == 200
    assert stats.conflicting_duplicates == 2
    assert stats.archives_read == [archive]
    assert stats.recent_files_read == [first, second]
    assert (stats.raw_rows, stats.arc_rows) == (4, 4)
    assert table.loc[table["role"] == "proxy", "debit_horaire"].isna().all()


def test_recent_only_sources_deduplicate_and_filter_the_period(tmp_path):
    snapshot = write_recent_snapshot(
        tmp_path / "recent.jsonl",
        recent_record(),
        recent_record(),
        recent_record(t_1h="2025-01-02T00:00:00Z"),
    )

    table, stats = traffic.process_sources(
        [], ARCS, "2025-01-01", "2025-01-01",
        chunk_rows=1, recent_paths=[snapshot],
    )

    build.validate_traffic(table, ARCS, "2025-01-01", "2025-01-01")
    assert stats.identical_duplicates == 1
    assert stats.outside_period == 1
    assert (stats.raw_rows, stats.arc_rows) == (3, 3)
    assert len(table) == 48
    paths = build.write_partitions(table, tmp_path / "processed")
    restored = pq.ParquetFile(paths[0]).read().to_pandas()
    pd.testing.assert_frame_equal(restored, table)


@pytest.mark.parametrize("filename", ["missing.jsonl", "unfinished.jsonl.part"])
def test_source_processing_rejects_unusable_recent_files(tmp_path, filename):
    with pytest.raises(ValueError, match="Recent file|completed .jsonl"):
        traffic.process_sources(
            [], ARCS, "2025-01-01", "2025-01-01",
            recent_paths=[tmp_path / filename],
        )


@pytest.mark.parametrize("timestamp", ["2025-01-01 00:00:00", "bad", None, 123])
def test_recent_reader_requires_aware_timestamps(tmp_path, timestamp):
    snapshot = write_recent_snapshot(
        tmp_path / "recent.jsonl", recent_record(t_1h=timestamp)
    )
    with pytest.raises(ValueError, match="line 1.*timezone"):
        list(traffic.read_recent(snapshot, ARCS))


def test_recent_reader_rejects_oversized_records(tmp_path):
    path = tmp_path / "recent.jsonl"
    path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds 4 MiB"):
        list(traffic.read_recent(path, ARCS))


def test_deflate64_processing_reports_missing_decoder(tmp_path, monkeypatch):
    import sys
    from test_historical import make_deflate64_zip

    path = tmp_path / "deflate64.zip"
    path.write_bytes(make_deflate64_zip())
    monkeypatch.setitem(sys.modules, "stream_unzip", None)
    with pytest.raises(ValueError, match="Cannot read.*pip install stream-unzip"):
        list(traffic.read_archive(path, ARCS))


def test_ignored_deflate64_members_still_have_crc_checked(tmp_path):
    from test_historical import make_deflate64_zip

    payload = bytearray(
        make_deflate64_zip().replace(b"boundary.txt", b"boundary.bin")
    )
    payload[14] ^= 1
    path = tmp_path / "corrupt_ignored_member.zip"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="CRC32"):
        list(traffic.read_archive(path, ARCS))


# --- Output validation --------------------------------------------------------


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


@pytest.mark.parametrize(
    "column,bad",
    [
        ("etat_trafic", -1),
        ("etat_trafic", 5),
        ("etat_trafic", 1.5),
        ("etat_barre", -1),
        ("etat_barre", 4),
        ("etat_barre", 1.5),
    ],
)
def test_output_validation_rejects_invalid_state_codes(column, bad):
    raw = raw_rows(convention_row("2025-01-01 01:00:00"))
    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS),
        ARCS, "2025-01-01", "2025-01-01",
    )

    with pytest.raises(ValueError, match=column):
        build.validate_traffic(
            table.assign(**{column: bad}),
            ARCS, "2025-01-01", "2025-01-01",
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        pd.NaT,
        pd.Timestamp("2024-12-31 23:00:00"),
        pd.Timestamp("2025-01-02 00:00:00"),
        pd.Timestamp("2025-01-01 00:30:00"),
    ],
)
def test_validation_rejects_timestamps_outside_hourly_grid(timestamp):
    table = _valid_traffic_table()
    table.loc[0, "datetime"] = timestamp

    with pytest.raises(ValueError, match="datetime"):
        build.validate_traffic(
            table, ARCS, "2025-01-01", "2025-01-01"
        )


@pytest.mark.parametrize("mode", ["text", "utc"])
def test_validation_rejects_incorrect_datetime_format(mode):
    table = _valid_traffic_table()
    if mode == "text":
        table["datetime"] = table["datetime"].astype(str)
    else:
        table["datetime"] = table["datetime"].dt.tz_localize("UTC")

    with pytest.raises(ValueError, match="datetime"):
        build.validate_traffic(
            table, ARCS, "2025-01-01", "2025-01-01"
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("iu_ac", PROXY.iu_ac),
        ("iu_ac", pd.NA),
        ("role", "proxy"),
        ("role", None),
    ],
)
def test_validation_checks_id_and_role_for_each_arc(column, value):
    table = _valid_traffic_table()
    if column == "iu_ac":
        table[column] = table[column].astype("Int32")
    table.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        build.validate_traffic(
            table, ARCS, "2025-01-01", "2025-01-01"
        )


def test_validation_rejects_duplicate_hours():
    table = _valid_traffic_table()
    table.iloc[0] = table.iloc[1]

    with pytest.raises(ValueError, match="duplicate"):
        build.validate_traffic(
            table, ARCS, "2025-01-01", "2025-01-01"
        )


@pytest.mark.parametrize(
    "column", ["debit_horaire", "taux_occupation"]
)
def test_validation_rejects_infinity_in_output(column):
    table = _valid_traffic_table()
    table.loc[1, column] = float("inf")

    with pytest.raises(ValueError, match=column):
        build.validate_traffic(
            table, ARCS, "2025-01-01", "2025-01-01"
        )


@pytest.mark.parametrize(
    "missing", [{"q": None}, {"k": None}, {"q": None, "k": None}]
)
def test_validation_accepts_missing_measurements_and_proxies(missing):
    raw = raw_rows(convention_row(
        "2025-01-01 01:00:00", **missing
    ))
    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, ARCS),
        ARCS, "2025-01-01", "2025-01-01",
    )

    build.validate_traffic(
        table, ARCS, "2025-01-01", "2025-01-01"
    )
    proxy = table[table["role"] == "proxy"]
    assert len(proxy) == 24
    assert proxy[
        ["debit_horaire", "taux_occupation"]
    ].isna().all().all()


def test_validation_accepts_shuffled_rows():
    table = _valid_traffic_table().sample(
        frac=1, random_state=7
    )
    build.validate_traffic(
        table, ARCS, "2025-01-01", "2025-01-01"
    )


# --- Build and Parquet integration --------------------------------------------


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


def test_recent_download_can_be_processed_and_written_to_parquet(tmp_path, monkeypatch):
    import io
    import json

    record = dict(zip(traffic.RAW_COLUMNS, convention_row(
        "2025-01-01T00:00:00Z", etat_trafic="Fluide", etat_barre="Ouvert"
    )))
    raw_bytes = json.dumps(record).encode("utf-8") + b"\n"

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(raw_bytes))}

    monkeypatch.setattr(traffic, "fetch_sample", lambda *args, **kwargs: {"total_count": 1})
    monkeypatch.setattr(traffic, "urlopen", lambda *args, **kwargs: Response(raw_bytes))
    snapshot = traffic.download_observations(
        traffic.parse_timestamp("2025-01-01T00:00:00Z"),
        traffic.parse_timestamp("2025-01-01T01:00:00Z"),
        tmp_path / "raw",
    )
    assert snapshot.read_bytes() == raw_bytes

    raw = pd.DataFrame([json.loads(snapshot.read_bytes())])
    table = traffic.build_hourly_table(
        traffic.normalize_raw(raw, [CONVENTION]), [CONVENTION],
        "2025-01-01", "2025-01-01",
    )
    build.validate_traffic(table, [CONVENTION], "2025-01-01", "2025-01-01")
    paths = build.write_partitions(table, tmp_path / "processed")
    restored = pq.ParquetFile(paths[0]).read().to_pandas().set_index("datetime")

    assert len(restored) == 24
    measured = restored.loc[pd.Timestamp("2025-01-01 01:00")]
    assert measured["debit_horaire"] == 100.0
    assert measured["taux_occupation"] == 5.0
    assert measured["etat_trafic"] == 1
    assert measured["etat_barre"] == 1
    assert measured["qualite_mesure"] == "ok"


def test_build_recent_only_converts_timezone_and_writes_parquet(tmp_path):
    snapshot = write_recent_snapshot(
        tmp_path / "recent.jsonl",
        recent_record(t_1h="2024-12-31T23:00:00Z"),
        recent_record(t_1h="2025-01-01T23:00:00Z"),
    )
    table, stats = build.build_traffic(
        "2025-01-01", "2025-01-01",
        tmp_path / "no_archives", tmp_path / "processed",
        write_config(tmp_path / "c.yaml"),
        recent_paths=[snapshot],
    )

    hours = table[table["arc"] == "Convention"].set_index("datetime")
    assert hours.loc["2025-01-01 00:00", "debit_horaire"] == 100
    assert hours.loc["2025-01-01 00:00", "qualite_mesure"] == "ok"
    assert len(table) == 48
    assert stats.archives_read == []
    assert stats.recent_files_read == [snapshot]
    assert stats.outside_period == 1
    written = pq.ParquetFile(
        tmp_path / "processed" / "year=2025" / "part-0.parquet"
    ).read().to_pandas()
    pd.testing.assert_frame_equal(written, table)


def test_build_cli_reads_archives_then_snapshots_in_argument_order(
    tmp_path, monkeypatch, capsys
):
    import sys

    nodes = (5671, CONVENTION.upstream_node, CONVENTION.downstream_node)
    write_archive(tmp_path / "raw" / "2025" / "data.zip", {"week.txt": [
        archive_line(*nodes, "2025-01-01 01:00:00", "100", "5", "1"),
        archive_line(*nodes, "2025-01-01 02:00:00", "200", "5", "1"),
    ]})
    first = write_recent_snapshot(
        tmp_path / "z_first.jsonl", recent_record(q="300")
    )
    last = write_recent_snapshot(
        tmp_path / "a_last.jsonl", recent_record(q=None, k=None)
    )
    config = write_config(tmp_path / "c.yaml")
    monkeypatch.setattr(sys, "argv", [
        "build", "--start", "2025-01-01", "--end", "2025-01-01",
        "--raw-dir", str(tmp_path / "raw"),
        "--output-dir", str(tmp_path / "processed"),
        "--config", str(config),
        "--recent-file", str(first), "--recent-file", str(last),
    ])

    build.main()

    written = pq.ParquetFile(
        tmp_path / "processed" / "year=2025" / "part-0.parquet"
    ).read().to_pandas()
    hours = written[written["arc"] == "Convention"].set_index("datetime")
    assert pd.isna(hours.loc["2025-01-01 01:00", "debit_horaire"])
    assert hours.loc["2025-01-01 01:00", "qualite_mesure"] == "vide"
    assert hours.loc["2025-01-01 02:00", "debit_horaire"] == 200
    summary = capsys.readouterr().out
    assert "Recent files read (in order):" in summary
    assert "including nulls): 2" in summary


def test_build_without_any_sources_fails_before_writing(tmp_path):
    with pytest.raises(ValueError, match="no recent JSONL files supplied"):
        build.build_traffic(
            "2025-01-02", "2025-01-02",
            tmp_path / "no_archives", tmp_path / "processed",
            write_config(tmp_path / "c.yaml"),
        )
    assert not (tmp_path / "processed").exists()


def test_build_empty_snapshot_keeps_all_hours_missing(tmp_path):
    snapshot = write_recent_snapshot(tmp_path / "empty.jsonl")
    table, stats = build.build_traffic(
        "2025-01-02", "2025-01-02",
        tmp_path / "no_archives", tmp_path / "processed",
        write_config(tmp_path / "c.yaml"), recent_paths=[snapshot],
    )
    assert len(table) == 48
    assert table["qualite_mesure"].eq("absente").all()
    assert table[["debit_horaire", "taux_occupation"]].isna().all().all()
    assert stats.raw_rows == 0
    assert stats.recent_files_read == [snapshot]


def test_build_cli_reports_bad_jsonl_without_replacing_output(
    tmp_path, monkeypatch, capsys
):
    import sys

    snapshot = tmp_path / "broken.jsonl"
    snapshot.write_text("{", encoding="utf-8")
    output = tmp_path / "processed" / "year=2025" / "part-0.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing output")
    config = write_config(tmp_path / "c.yaml")
    monkeypatch.setattr(sys, "argv", [
        "build", "--start", "2025-01-02", "--end", "2025-01-02",
        "--raw-dir", str(tmp_path / "no_archives"),
        "--output-dir", str(tmp_path / "processed"),
        "--config", str(config), "--recent-file", str(snapshot),
    ])

    with pytest.raises(SystemExit) as error:
        build.main()

    assert error.value.code == 1
    assert output.read_bytes() == b"existing output"
    message = capsys.readouterr().err
    assert "Traffic processing failed:" in message
    assert "broken.jsonl, line 1: invalid JSON" in message


def test_deflate64_archive_processes_and_writes_parquet(tmp_path):
    from test_historical import make_deflate64_zip

    archive = tmp_path / "real_deflate64.zip"
    archive.write_bytes(make_deflate64_zip())
    table, stats = traffic.process_archives(
        [archive], [CONVENTION], "2025-01-01", "2025-01-01", chunk_rows=1
    )
    build.validate_traffic(table, [CONVENTION], "2025-01-01", "2025-01-01")
    hours = table.set_index("datetime")
    assert hours.loc["2025-01-01 00:00", "debit_horaire"] == 200
    assert hours.loc["2025-01-01 01:00", "debit_horaire"] == 300
    assert (stats.raw_rows, stats.arc_rows) == (2, 2)
    paths = build.write_partitions(table, tmp_path / "processed")
    pd.testing.assert_frame_equal(
        pq.ParquetFile(paths[0]).read().to_pandas(), table
    )


def test_deflate64_previous_archive_recovers_january_midnight(tmp_path):
    from test_historical import make_deflate64_zip

    previous = tmp_path / "raw" / "2024" / "boundary.zip"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(make_deflate64_zip())
    table, stats = build.build_traffic(
        "2025-01-01", "2025-01-01", tmp_path / "raw",
        tmp_path / "processed", write_config(tmp_path / "c.yaml"),
    )
    assert stats.archives_read == [previous]
    hours = table[table["arc"] == "Convention"].set_index("datetime")
    assert hours.loc["2025-01-01 00:00", "debit_horaire"] == 200
