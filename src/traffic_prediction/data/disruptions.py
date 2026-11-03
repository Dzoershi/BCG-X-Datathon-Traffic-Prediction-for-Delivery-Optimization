"""Road works and events on the evaluated arcs, in the processed traffic time convention.

Open Data Paris gives the arc geometry (traffic counting reference) and three works
datasets: "chantiers perturbants" (current and planned works with a disruption level),
"chantiers à Paris" (all current works areas) and its 2024 history. Works are kept
within ``NEAR_M`` of an arc, measured to the arc line; ``ON_ARC_M`` marks those on the
arc. A hand-checked CSV adds ceremonies, closures, demonstrations and strikes.

The hourly table serves to check the test period and to explain past forecast errors,
not as a training feature: 2025 history only contains works still running today, and
works that have not started yet appear only in "chantiers perturbants".
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from traffic_prediction.constants import TEST_START  
from traffic_prediction.data.traffic import REPO_ROOT, Arc, period_bounds
from traffic_prediction.data.weather import save_raw

EXPORT_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{dataset}/exports/json"
REFERENCE_DATASET = "referentiel-comptages-routiers"
# Dataset -> identifier field. Order matters: a works area found in the 2024 history
# and in the current list keeps the current row, whose dates may have been extended.
WORKS_DATASETS = {
    "historique-chantiers-a-paris-en-2024": "num_emprise",
    "chantiers-a-paris": "num_emprise",
    "chantiers-perturbants": "identifiant",
}
PERTURBANTS = "chantiers-perturbants"
# Private building works: mostly a skip or scaffolding on a parking space, on every
# arc almost all the time. Listed in the works table, never flagged in the hourly one.
PRIVATE_BUILDING_WORKS = "Tiers (travaux sur bâtiment)"
ON_ARC_M = 50.0  # About half the width of the Champs-Elysées, facade to facade.
NEAR_M = 200.0  # Neighbouring junctions.
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "disruptions"
DEFAULT_EVENTS_PATH = REPO_ROOT / "data" / "raw" / "events_manual.csv"
USER_AGENT = "traffic-prediction/0.1"

ALL_ARCS = "toutes"
EVENT_COLUMNS = ["date_debut", "date_fin", "arc", "type", "description", "source_url"]
EVENT_TYPES = {"ceremonie", "fermeture", "pietonnisation", "manifestation", "greve"}
EVENT_TIME_FORMAT = "%Y-%m-%d %H:%M"
# In a description: the event did not take place (e.g. the sensor shows normal traffic).
# The row is kept for the record but never sets evenement_actif.
NOT_APPLIED = "non appliqué"
WORKS_COLUMNS = [
    "arc", "source", "identifiant", "date_debut", "date_fin", "distance_m",
    "sur_arc", "chaussee", "categorie", "niveau_perturbation", "description",
]
DISRUPTION_COLUMNS = ["datetime", "arc", "chantier_actif", "evenement_actif", "description"]


# --- Download -------------------------------------------------------------------


def fetch_json(url: str, parameters: dict, *, attempts: int = 3) -> list | dict:
    """GET an Open Data Paris JSON export, with the retry policy of the other downloads.

    60 s timeout, waits of 1 s then 2 s, no retry on a 4xx error except 408 and 429.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    request = Request(f"{url}?{urlencode(parameters)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except (OSError, HTTPException, json.JSONDecodeError, UnicodeDecodeError) as error:
            if isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code not in (408, 429):
                raise ValueError(f"Open Data Paris rejected the request ({error.code}): {url}") from error
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(f"Disruptions request failed (attempt {attempt}/{attempts}): "
                  f"{error}; retrying in {delay}s...", file=sys.stderr, flush=True)
            time.sleep(delay)


def download_arc_lines(arcs: list[Arc], raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, list]:
    """Arc name -> line of (lon, lat) points from the traffic counting reference."""
    ids = ", ".join(f"'{arc.iu_ac}'" for arc in arcs)
    records = fetch_json(EXPORT_URL.format(dataset=REFERENCE_DATASET), {"where": f"iu_ac in ({ids})"})
    save_raw(records, raw_dir, REFERENCE_DATASET)
    return arc_lines(records, arcs)


def arc_lines(records: list[dict], arcs: list[Arc]) -> dict[str, list]:
    """Match reference records to the configured arcs by ``iu_ac``."""
    lines = {}
    for record in records:
        geometry = _geometry(record)
        if geometry is None or geometry["type"] != "LineString":
            raise ValueError(f"Arc {record.get('iu_ac')} has no line geometry in the reference.")
        lines[int(record["iu_ac"])] = [tuple(point[:2]) for point in geometry["coordinates"]]
    missing = [arc.name for arc in arcs if arc.iu_ac not in lines]
    if missing:
        raise ValueError(f"No reference geometry for arcs {missing}.")
    return {arc.name: lines[arc.iu_ac] for arc in arcs}


def download_works(lines: dict[str, list], raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Works of the three datasets within ``NEAR_M`` of each arc, raw JSON kept.

    The API keeps only works around the middle of the arc; the exact distance to the
    line is computed here.
    """
    parts = []
    for arc, line in lines.items():
        (lon, lat), radius = search_circle(line)
        where = f"within_distance(geo_shape, geom'POINT({lon:.6f} {lat:.6f})', {radius}m)"
        for dataset in WORKS_DATASETS:
            records = fetch_json(EXPORT_URL.format(dataset=dataset), {"where": where})
            save_raw(records, raw_dir, f"{dataset}_{_slug(arc)}")
            parts.append(works_table(records, dataset, arc, line))
    works = pd.concat([part for part in parts if len(part)] or [parts[0]], ignore_index=True)
    return works.drop_duplicates(["arc", "identifiant"], keep="last").reset_index(drop=True)


def search_circle(line: list) -> tuple[tuple[float, float], int]:
    """Centre of the arc and a radius in metres that covers every point within ``NEAR_M``."""
    lons, lats = zip(*line)
    centre = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
    reach = max(math.dist((0.0, 0.0), _project(point, centre)) for point in line)
    return centre, math.ceil(reach + NEAR_M + 50)


def _slug(name: str) -> str:
    """File-name part without accents: "Saint-Pères" -> "saint-peres"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


# --- Works on the arcs -------------------------------------------------------------


def works_table(records: list[dict], dataset: str, arc: str, line: list) -> pd.DataFrame:
    """Works areas of one dataset within ``NEAR_M`` of one arc, one row each.

    Dates are whole days, both included. ``chaussee`` is true for works on the roadway:
    every "chantier perturbant", and areas declared as ``EMPRISE_CHAUSSEE``.
    ``categorie`` is the works category ("chantier_categorie"), or the owner type
    ("typologie") for "chantiers perturbants". Records without geometry or valid dates
    cannot be placed and are skipped with a warning.
    """
    rows, skipped = [], 0
    for record in records:
        geometry = _geometry(record)
        start, end = _day(record.get("date_debut")), _day(record.get("date_fin"))
        if geometry is None or start is None or end is None or end < start:
            skipped += 1
            continue
        distance = distance_m(geometry, line)
        if distance > NEAR_M:
            continue
        perturbant = dataset == PERTURBANTS
        rows.append({
            "arc": arc,
            "source": dataset,
            "identifiant": str(record[WORKS_DATASETS[dataset]]),
            "date_debut": start,
            "date_fin": end,
            "distance_m": round(distance, 1),
            "sur_arc": distance <= ON_ARC_M,
            "chaussee": perturbant or "EMPRISE_CHAUSSEE" in (record.get("localisation_detail") or []),
            "categorie": record.get("chantier_categorie") or record.get("typologie"),
            "niveau_perturbation": record.get("niveau_perturbation"),
            "description": _works_description(record, perturbant),
        })
    if skipped:
        print(f"WARNING: {dataset}, {arc}: {skipped} records without geometry or valid dates skipped.",
              file=sys.stderr, flush=True)
    table = pd.DataFrame(rows, columns=WORKS_COLUMNS)
    return table.astype({"date_debut": "datetime64[ns]", "date_fin": "datetime64[ns]",
                         "distance_m": "float64", "sur_arc": bool, "chaussee": bool})


def relevant(works: pd.DataFrame) -> pd.Series:
    """Works on the roadway that are a "chantier perturbant" or not private building works."""
    kind = works["source"].eq(PERTURBANTS) | works["categorie"].ne(PRIVATE_BUILDING_WORKS)
    return works["chaussee"] & kind


def affects_arc(works: pd.DataFrame) -> pd.Series:
    """Works that set ``chantier_actif``: relevant and within ``ON_ARC_M`` of the arc."""
    return works["sur_arc"] & relevant(works)


def works_active(works: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    """Works running on at least one day of the inclusive period."""
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return works[(works["date_debut"] <= last) & (works["date_fin"] >= first)]


def _geometry(record: dict) -> dict | None:
    shape = record.get("geo_shape")
    return shape.get("geometry", shape) if shape else None


def _day(value: object) -> pd.Timestamp | None:
    """An API date ("2026-11-07") as a timezone-free day; None if missing or invalid."""
    try:
        return pd.to_datetime(value, format="%Y-%m-%d") if value else None
    except (TypeError, ValueError):
        return None


def _works_description(record: dict, perturbant: bool) -> str:
    if perturbant:
        level = f" ({record['niveau_perturbation']})" if record.get("niveau_perturbation") else ""
        return f"{record.get('objet') or 'Chantier'} - {record.get('voie') or '?'}{level}"
    return f"{record.get('chantier_synthese') or 'Chantier'} ({record.get('moa_principal') or '?'})"


# --- Geometry ----------------------------------------------------------------------


def distance_m(geometry: dict, line: list) -> float:
    """Shortest distance in metres between a GeoJSON geometry and an arc line (lon, lat).

    Zero when they cross or when an arc point lies inside a polygon. A local flat
    projection is accurate to well under a metre over a few hundred metres.
    """
    origin = line[0]
    arc = [_project(point, origin) for point in line]
    best = math.inf
    for ring, is_area in _parts(geometry):
        points = [_project(point[:2], origin) for point in ring]
        if is_area and any(_inside(point, points) for point in arc):
            return 0.0
        edges = list(zip(points, points[1:])) or [(points[0], points[0])]
        for a, b in zip(arc, arc[1:]):
            for c, d in edges:
                if _segments_cross(a, b, c, d):
                    return 0.0
                best = min(best, _point_segment(c, a, b), _point_segment(d, a, b),
                           _point_segment(a, c, d), _point_segment(b, c, d))
    return best


def _project(point: tuple, origin: tuple) -> tuple[float, float]:
    """(lon, lat) to metres east and north of ``origin``."""
    return ((point[0] - origin[0]) * math.cos(math.radians(origin[1])) * 111_320,
            (point[1] - origin[1]) * 110_540)


def _parts(geometry: dict) -> list[tuple[list, bool]]:
    """Point lists of a GeoJSON geometry; True for polygon outer rings."""
    kind, coordinates = geometry["type"], geometry["coordinates"]
    if kind == "Point":
        return [([coordinates], False)]
    if kind == "MultiPoint":
        return [([point], False) for point in coordinates]
    if kind == "LineString":
        return [(coordinates, False)]
    if kind == "MultiLineString":
        return [(line, False) for line in coordinates]
    if kind == "Polygon":
        polygons = [coordinates]
    elif kind == "MultiPolygon":
        polygons = coordinates
    else:
        raise ValueError(f"Unsupported geometry type {kind!r}.")
    return [(ring, index == 0) for polygon in polygons for index, ring in enumerate(polygon)]


def _point_segment(p: tuple, a: tuple, b: tuple) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    return _cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0


def _inside(point: tuple, ring: list) -> bool:
    """Ray casting: is ``point`` inside the closed ``ring``?"""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


# --- Manual events -----------------------------------------------------------------


def read_events(path: str | Path, arc_names: list[str]) -> pd.DataFrame:
    """Read the hand-checked events list; any malformed row stops the build.

    Times are Paris local time without timezone, "YYYY-MM-DD HH:MM"; the end is excluded.
    ``arc`` is an evaluated arc name or "toutes"; every row needs a description and an
    http(s) source.
    """
    events = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(events.columns) != EVENT_COLUMNS:
        raise ValueError(f"Events file {path} must have columns {EVENT_COLUMNS}, got {list(events.columns)}.")
    problems = []
    for column in ("date_debut", "date_fin"):
        parsed = pd.to_datetime(events[column], format=EVENT_TIME_FORMAT, errors="coerce")
        problems += [f"line {row + 2}: {column} {events.at[row, column]!r} is not YYYY-MM-DD HH:MM"
                     for row in events.index[parsed.isna()]]
        events[column] = parsed
    checks = {
        "date_fin must be after date_debut": events["date_fin"] <= events["date_debut"],
        f"arc must be an evaluated arc or {ALL_ARCS!r}": ~events["arc"].isin([*arc_names, ALL_ARCS]),
        f"type must be one of {sorted(EVENT_TYPES)}": ~events["type"].isin(EVENT_TYPES),
        "description is empty": events["description"].str.strip().eq(""),
        "source_url must be an http(s) link": ~events["source_url"].str.match(r"https?://\S+$"),
    }
    for message, invalid in checks.items():
        problems += [f"line {row + 2}: {message}" for row in events.index[invalid]]
    if problems:
        raise ValueError(f"Invalid events file {path}:\n" + "\n".join(sorted(problems)))
    return events


# --- Hourly table ------------------------------------------------------------------


def hourly_table(
    works: pd.DataFrame, events: pd.DataFrame, arc_names: list[str], start: object, end: object
) -> pd.DataFrame:
    """One row per arc and hour label of inclusive Paris days, as in the traffic table.

    ``chantier_actif``: works selected by ``affects_arc``; works days run from 00:00 to
    24:00. ``evenement_actif``: a manual event of the arc or of all arcs, except rows
    marked ``NOT_APPLIED``. ``description`` lists what is active, "" when nothing is.
    """
    first, last = period_bounds(start, end)
    labels = pd.date_range(first, last, freq="h", name="datetime")
    parts = []
    for arc in arc_names:
        on_arc = works[works["arc"].eq(arc) & affects_arc(works)]
        works_items = [(active_hours(labels, row.date_debut, row.date_fin + pd.Timedelta(days=1)),
                        f"Chantier : {row.description}") for row in on_arc.itertuples()]
        applied = ~events["description"].str.casefold().str.contains(NOT_APPLIED, regex=False)
        arc_events = events[events["arc"].isin([arc, ALL_ARCS]) & applied]
        event_items = [(active_hours(labels, row.date_debut, row.date_fin),
                        f"{row.type.capitalize()} : {row.description}") for row in arc_events.itertuples()]
        parts.append(pd.DataFrame({
            "datetime": labels,
            "arc": arc,
            "chantier_actif": _any(works_items, len(labels)),
            "evenement_actif": _any(event_items, len(labels)),
            "description": _descriptions(works_items + event_items, len(labels)),
        }))
    return pd.concat(parts, ignore_index=True)[DISRUPTION_COLUMNS]


def active_hours(labels: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Labels whose hour (label - 1 h, label] overlaps [start, end).

    An event from 07:00 to 14:00 is active on the labels 08:00 to 14:00.
    """
    return np.asarray((labels > start) & (labels - pd.Timedelta(hours=1) < end))


def _any(items: list, size: int) -> np.ndarray:
    return np.any([mask for mask, _ in items], axis=0) if items else np.zeros(size, dtype=bool)


def _descriptions(items: list, size: int) -> list[str]:
    if not items:
        return [""] * size
    active = np.array([mask for mask, _ in items])
    texts = [text for _, text in items]
    return ["; ".join(dict.fromkeys(texts[i] for i in np.flatnonzero(column))) for column in active.T]


def validate_disruptions(table: pd.DataFrame, arc_names: list[str], start: object, end: object) -> None:
    """Reject a table that would not join one-to-one with traffic on (arc, datetime)."""
    if list(table.columns) != DISRUPTION_COLUMNS:
        raise ValueError(f"Unexpected disruption columns: {list(table.columns)}")
    if not pd.api.types.is_datetime64_dtype(table["datetime"].dtype):
        raise ValueError("Disruption datetime must be timezone-free Paris time.")
    first, last = period_bounds(start, end)
    hours = pd.date_range(first, last, freq="h")
    if set(table["arc"]) != set(arc_names):
        raise ValueError(f"Disruption arcs {sorted(set(table['arc']))} differ from {sorted(arc_names)}.")
    if duplicated := int(table.duplicated(["arc", "datetime"]).sum()):
        raise ValueError(f"Duplicate (arc, datetime) rows: {duplicated}.")
    if len(table) != len(arc_names) * len(hours) or not table["datetime"].isin(hours).all():
        raise ValueError(f"Disruptions must cover {first} to {last} hourly for each arc.")
    for column in ("chantier_actif", "evenement_actif"):
        if table[column].dtype != bool:
            raise ValueError(f"Disruption {column} must be boolean.")
    if table["description"].isna().any():
        raise ValueError("Disruption description must be a string, empty when nothing is active.")
