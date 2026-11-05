"""Timestamped disruption evidence and conservative submission decisions (#57).

Refresh captures live reference/works exports; replay verifies their hashes offline.
Proximity establishes a candidate, never a directed road closure or a numeric effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

from traffic_prediction.constants import TARGET_TIMES
from traffic_prediction.data import disruptions, traffic
from traffic_prediction.data.audit import _json

DEFAULT_REVIEW = traffic.REPO_ROOT / "configs/disruptions_review.json"
DEFAULT_RAW = traffic.REPO_ROOT / "data/raw/disruption_57"
DEFAULT_OUTPUT = traffic.REPO_ROOT / "outputs/metrics/disruption_57"
CURRENT_DATASETS = ("chantiers-a-paris", "chantiers-perturbants")
EXTRA_DATASETS = ("circulation_evenement", "fermetures-voirie")


def read_review(path: str | Path = DEFAULT_REVIEW) -> dict:
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema_version") != 1 or not isinstance(review.get("events"), list):
        raise ValueError("Expected disruption review schema_version 1 and events list")
    required = {"arc", "date_debut", "date_fin", "status", "action", "rationale"}
    if any(not isinstance(entry, dict) or not required.issubset(entry) for entry in review["events"]):
        raise ValueError(f"Each restriction review needs {sorted(required)}")
    return review


def reviewed_events(events: pd.DataFrame, review: dict, times: pd.DatetimeIndex,
                    arcs: list[traffic.Arc] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Authorize only explicitly reviewed traffic orders for the exact directed arc/interval.

    Invalid approvals fail closed; missing/provisional reviews preserve the model forecast.
    Every considered ceremony receives a decision, including its candidate hour labels.
    """
    arcs = traffic.load_arcs() if arcs is None else arcs
    identities = {arc.name: arc for arc in arcs if arc.role == "evalue"}
    entries, keys = {}, set()
    for entry in review["events"]:
        key = (entry["arc"], pd.Timestamp(entry["date_debut"]), pd.Timestamp(entry["date_fin"]))
        if key in keys:
            raise ValueError("Duplicate disruption review interval")
        keys.add(key)
        if entry.get("status") not in {"confirmed", "provisional", "unknown"}:
            raise ValueError("Unknown restriction status")
        if entry.get("action") not in {"none", "copy_prior_year"}:
            raise ValueError("Unknown restriction action")
        if entry.get("action") == "copy_prior_year":
            arc = identities.get(entry["arc"])
            identity = arc is not None and all(entry.get(field) == getattr(arc, field) for field in
                                              ("iu_ac", "upstream_node", "downstream_node"))
            checked = pd.to_datetime(entry.get("checked_at"), utc=True, errors="coerce")
            authorized = (identity and entry.get("status") == "confirmed"
                          and entry.get("evidence_kind") == "traffic_order"
                          and entry.get("direction_verified") is True and entry.get("hours_verified") is True
                          and bool(entry.get("reviewer")) and bool(entry.get("rationale"))
                          and urlparse(entry.get("source_url", "")).scheme in {"http", "https"}
                          and pd.notna(checked) and key[2] > key[1])
            if not authorized:
                raise ValueError("Closure copy needs a confirmed traffic order, exact direction/hours and signed review")
        entries[key] = entry
    selected, decisions = [], []
    for index, event in events.iterrows():
        if event["type"] != "ceremonie" or event.arc == disruptions.ALL_ARCS:
            continue
        labels = times[disruptions.active_hours(times, event.date_debut, event.date_fin)]
        if not len(labels):
            continue
        key = (event.arc, event.date_debut, event.date_fin)
        entry = entries.get(key)
        action = entry["action"] if entry else "none"
        if action == "copy_prior_year":
            selected.append(index)
        decisions.append({"arc": event.arc, "date_debut": str(event.date_debut), "date_fin": str(event.date_fin),
                          "status": entry["status"] if entry else "unknown", "action": action,
                          "candidate_labels": [str(time) for time in labels],
                          "review": entry, "reason": entry["rationale"] if entry else "No exact reviewed restriction"})
    return events.loc[selected].copy(), decisions


def verified_lines(records: list[dict], arcs: list[traffic.Arc]) -> dict:
    """Fail on missing/duplicate IDs, reversed nodes or a changed official target."""
    for arc in arcs:
        matches = [row for row in records if str(row.get("iu_ac")) == str(arc.iu_ac)]
        if len(matches) != 1 or any(matches[0].get(field) != expected for field, expected in
            (("libelle_nd_amont", arc.upstream_node), ("libelle_nd_aval", arc.downstream_node))):
            raise ValueError(f"Reference does not uniquely match directed arc {arc.name} ({arc.iu_ac})")
    return disruptions.arc_lines(records, arcs)


def capture(raw_dir: Path, arcs: list[traffic.Arc], evidence: list[dict] | None = None) -> Path:
    """Create a new immutable capture directory. Failed captures cannot be replayed as complete."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = raw_dir / stamp
    destination.mkdir(parents=True, exist_ok=False)
    sources = []

    def fetch(dataset, parameters, arc=None):
        url = disruptions.EXPORT_URL.format(dataset=dataset)
        records = disruptions.fetch_json(url, parameters)
        if not isinstance(records, list):
            raise ValueError(f"Incomplete export for {dataset}: expected list")
        filename = f"{len(sources):02d}_{dataset}.json"
        _json(destination / filename, records)
        sources.append({"file": filename, "dataset": dataset, "arc": arc,
                        "url": f"{url}?{urlencode(parameters)}", "records": len(records),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": traffic.file_sha256(destination / filename)})
        return records

    ids = ", ".join(f"'{arc.iu_ac}'" for arc in arcs)
    records = fetch(disruptions.REFERENCE_DATASET, {"where": f"iu_ac in ({ids})"})
    lines = verified_lines(records, arcs)
    for arc, line in lines.items():
        (lon, lat), radius = disruptions.search_circle(line)
        where = f"within_distance(geo_shape, geom'POINT({lon:.6f} {lat:.6f})', {radius}m)"
        for dataset in CURRENT_DATASETS:
            fetch(dataset, {"where": where}, arc)
    # Event polylines have no geo_shape; capture both small feeds completely and filter locally.
    for dataset in EXTRA_DATASETS:
        fetch(dataset, {})
    documents = []
    for item in evidence or []:
        filename = f"evidence_{len(documents):02d}.bin"
        # Retain exact bytes, not the browser's extracted summary, outside Git.
        receipt = {**item, "status": "unavailable"}
        for attempt in range(3):
            try:
                with urlopen(Request(item["url"], headers={"User-Agent": disruptions.USER_AGENT}), timeout=60) as response:
                    with (destination / filename).open("wb") as output:
                        while block := response.read(65536):
                            output.write(block)
                receipt.update(status="captured", file=filename, sha256=traffic.file_sha256(destination / filename))
                receipt.pop("error", None)
                break
            except (OSError, HTTPException) as error:
                receipt["error"] = str(error)
                if attempt < 2:
                    time.sleep(2**attempt)
        documents.append({**receipt, "retrieved_at": datetime.now(timezone.utc).isoformat()})
    _json(destination / "manifest.json", {"schema_version": 1, "sources": sources, "documents": documents})
    return destination


def replay(snapshot: Path, arcs: list[traffic.Arc]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Rebuild nearby works from exact captured API results, retaining source-level provenance."""
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    for source in manifest.get("documents", []):
        if source["status"] != "captured":
            continue
        path = snapshot / source["file"]
        if path.parent != snapshot or traffic.file_sha256(path) != source["sha256"]:
            raise ValueError("Evidence document checksum/path mismatch")
    expected = {(disruptions.REFERENCE_DATASET, None)} | {
        (dataset, arc.name) for arc in arcs for dataset in CURRENT_DATASETS} | {
        (dataset, None) for dataset in EXTRA_DATASETS}
    if len(sources) != len(expected) or {(s["dataset"], s["arc"]) for s in sources} != expected:
        raise ValueError("Incomplete disruption snapshot")
    records = []
    for source in sources:
        path = snapshot / source["file"]
        if path.parent != snapshot or traffic.file_sha256(path) != source["sha256"]:
            raise ValueError("Disruption snapshot checksum/path mismatch")
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list) or len(content) != source["records"]:
            raise ValueError("Disruption snapshot record count mismatch")
        records.append(content)
    reference = next(r for s, r in zip(sources, records) if s["dataset"] == disruptions.REFERENCE_DATASET)
    lines = verified_lines(reference, arcs)
    parts = []
    for source, content in zip(sources, records):
        if source["arc"] is not None:
            part = disruptions.works_table(content, source["dataset"], source["arc"], lines[source["arc"]])
            parts.append(part.assign(source_url=source["url"], checked_at=source["retrieved_at"],
                                     raw_sha256=source["sha256"]))
    extra = []
    for source, content in zip(sources, records):
        if source["dataset"] not in EXTRA_DATASETS:
            continue
        for record in content:
            event = source["dataset"] == "circulation_evenement"
            if event:
                try:
                    points = [tuple(map(float, p.split()))[::-1] for p in record["polyline"].split(",")]
                    if len(points) == 1:  # Feed separates points with spaces, latitude then longitude.
                        values = list(map(float, record["polyline"].split()))
                        points = [(values[i + 1], values[i]) for i in range(0, len(values), 2)]
                    shape = {"type": "LineString", "coordinates": points}
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    raise ValueError("Malformed event polyline; review the captured source") from error
            else:
                shape = disruptions._geometry(record)
            start_field, end_field = ("starttime", "endtime") if event else ("date_debut_bar_fv", "date_fin_bar_fv")
            start, end = [pd.to_datetime(record.get(key), utc=True, errors="coerce") for key in (start_field, end_field)]
            if shape is None or pd.isna(start) or pd.isna(end) or end <= start:
                raise ValueError("Unplaceable source restriction; review captured geometry/dates")
            start, end = [value.tz_convert("Europe/Paris").tz_localize(None) for value in (start, end)]
            for arc, line in lines.items():
                distance = disruptions.distance_m(shape, line)
                if distance <= 500 and start < TARGET_TIMES.max() and end > TARGET_TIMES.min() - pd.Timedelta(hours=1):
                    extra.append({"arc": arc, "source": source["dataset"],
                        "identifiant": str(record.get("id" if event else "identifiant_fv")),
                        "date_debut": start, "date_fin": end, "distance_m": round(distance, 1),
                        "description": record.get("description" if event else "lieu_fv"),
                        "declared_direction": record.get("direction" if event else "sens_fv"),
                        "status": "source_declared_effect_unknown", "action": "none",
                        "source_url": source["url"], "checked_at": source["retrieved_at"],
                        "raw_sha256": source["sha256"], "direction_verified": False, "hours_verified": False})
    columns = ["arc", "source", "identifiant", "date_debut", "date_fin", "distance_m", "description",
               "declared_direction", "status", "action", "source_url", "checked_at", "raw_sha256",
               "direction_verified", "hours_verified"]
    return pd.concat(parts, ignore_index=True), pd.DataFrame(extra, columns=columns), sources


def report(snapshot: Path, output: Path, review_path: Path, events_path: Path) -> None:
    arcs = [arc for arc in traffic.load_arcs() if arc.role == "evalue"]
    works, extra, sources = replay(snapshot, arcs)
    identities = pd.DataFrame([vars(arc) for arc in arcs]).rename(columns={"name": "arc"})
    # Dates are inclusive whole days, not verified active clock times or a direction-specific closure.
    works = disruptions.works_active(works, TARGET_TIMES.min() - pd.Timedelta(hours=1), TARGET_TIMES.max())
    works = works.merge(identities, on="arc", validate="many_to_one")
    extra = extra.merge(identities, on="arc", validate="many_to_one")
    works["status"], works["action"] = "works_declared_effect_unknown", "none"
    works["direction_verified"], works["hours_verified"] = False, False
    works["time_precision"] = "inclusive dates only; possible hourly overlap, not continuous activity"
    events = disruptions.read_events(events_path, [arc.name for arc in arcs])
    review = read_review(review_path)
    _, decisions = reviewed_events(events, review, TARGET_TIMES, arcs)
    rows = []
    for item in works.itertuples():
        labels = TARGET_TIMES[disruptions.active_hours(
            TARGET_TIMES, item.date_debut, item.date_fin + pd.Timedelta(days=1))]
        for label in labels:
            rows.append({"arc": item.arc, "datetime": label, "kind": "works_candidate",
                         "identifier": item.identifiant, "status": item.status, "action": "none",
                         "source_url": item.source_url, "checked_at": item.checked_at,
                         "direction": f"{item.upstream_node} -> {item.downstream_node}",
                         "direction_verified": False, "hours_verified": False})
    for decision in decisions:
        arc = next(arc for arc in arcs if arc.name == decision["arc"])
        evidence = decision["review"] or {}
        for label in decision["candidate_labels"]:
            rows.append({"arc": arc.name, "datetime": pd.Timestamp(label), "kind": "ceremony_candidate",
                         "identifier": "November11", "status": decision["status"], "action": decision["action"],
                         "source_url": evidence.get("source_url"), "checked_at": evidence.get("checked_at"),
                         "direction": f"{arc.upstream_node} -> {arc.downstream_node}",
                         "direction_verified": evidence.get("direction_verified", False),
                         "hours_verified": evidence.get("hours_verified", False)})
    for item in extra.itertuples():
        for label in TARGET_TIMES[disruptions.active_hours(TARGET_TIMES, item.date_debut, item.date_fin)]:
            rows.append({"arc": item.arc, "datetime": label, "kind": "source_restriction_candidate",
                         "identifier": item.identifiant, "status": item.status, "action": "none",
                         "source_url": item.source_url, "checked_at": item.checked_at,
                         "direction": f"{item.upstream_node} -> {item.downstream_node}",
                         "direction_verified": False, "hours_verified": False})
    output.mkdir(parents=True, exist_ok=True)
    works.to_csv(output / "works.csv", index=False, lineterminator="\n")
    extra.to_csv(output / "other_restrictions.csv", index=False, lineterminator="\n")
    hours = pd.DataFrame(rows, columns=["arc", "datetime", "kind", "identifier", "status", "action",
        "source_url", "checked_at", "direction", "direction_verified", "hours_verified"])
    hours.to_csv(output / "candidate_hours.csv", index=False, lineterminator="\n")
    hours.groupby(["arc", "datetime", "kind", "status"], sort=True).size().rename("n_candidates").reset_index().to_csv(
        output / "candidate_hour_counts.csv", index=False, lineterminator="\n")
    identities.to_csv(output / "verified_arcs.csv", index=False, lineterminator="\n")
    _json(output / "decisions.json", decisions)
    _json(output / "run.json", {"snapshot": snapshot.name, "sources": sources,
        "review_sha256": traffic.file_sha256(review_path), "events_sha256": traffic.file_sha256(events_path),
        "code_sha256": traffic.file_sha256(Path(__file__)), "candidate_hour_rows": len(rows),
        "interpretation": "Source-declared works and provisional events are not verified road closures. No numeric adjustment without explicit approved review."})
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    pd.DataFrame(manifest.get("documents", [])).to_csv(output / "evidence.csv", index=False, lineterminator="\n")


def assess_history(raw: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory 2024-to-2025 transfer, using the frozen #55 forecasts and truth masks.

    The 2025 order was published November 4, before the November 6 cutoff.
    One annual transfer, already explored in #47, is not independent validation.
    """
    from traffic_prediction.models.postprocess import copy_closures_from_previous_year
    from traffic_prediction.evaluate import TARGETS
    cutoff = pd.Timestamp("2025-11-06")
    if raw.datetime.ge(TARGET_TIMES.min()).any():
        raise ValueError("Final-test observations are not allowed in historical assessment")
    selected = predictions.loc[predictions.cutoff.eq(cutoff) & predictions.scenario.eq("observed")
        & predictions.model.isin(["baseline_holiday_aware", "lightgbm_all", "timestral_209m_calendar"])].copy()
    if selected.empty or selected.duplicated(["model", "arc", "datetime", "variable"]).any():
        raise ValueError("Missing or duplicate historical predictions")
    history = raw.loc[raw.datetime.lt(cutoff + pd.Timedelta(days=1))].copy()
    history["qualite_mesure"] = history.legacy_quality
    for short, target in zip(("q", "k"), TARGETS):
        history[target] = history[f"{short}_value"].where(history[f"{short}_status"].eq("valid"))
    # Never use held-out measurements as the reference copied into predictions.
    champs = next(arc.name for arc in traffic.load_arcs() if arc.iu_ac == 4264)
    event = pd.DataFrame({"arc": [champs], "type": ["ceremonie"],
                          "date_debut": [pd.Timestamp("2025-11-11 07:00")],
                          "date_fin": [pd.Timestamp("2025-11-11 14:00")]})
    keys = ["arc", "datetime"]
    references = raw.set_index(keys)
    for variable, short in zip(TARGETS, ("q", "k")):
        part = selected.loc[selected.variable.eq(variable)]
        truth = references.reindex(pd.MultiIndex.from_frame(part[keys]))
        mask = truth[f"{short}_status"].eq("valid").to_numpy()
        if not np.array_equal(part.truth_valid.to_numpy(), mask) or not np.allclose(
            part.truth.to_numpy()[mask], truth[f"{short}_value"].to_numpy()[mask]):
            raise ValueError("Historical truth/masks differ from the frozen audit")
    adjusted, scores = [], []
    for name, part in selected.groupby("model", sort=True):
        table = part.pivot(index=keys, columns="variable", values="prediction").reset_index()
        changed = copy_closures_from_previous_year(table, history, event)
        changed_keys = set(changed)
        for variable in TARGETS:
            detail = part.loc[part.variable.eq(variable)].merge(table[keys + [variable]], on=keys,
                                                                             validate="one_to_one")
            detail = detail.rename(columns={variable: "adjusted_prediction"})
            detail["adjusted"] = [(arc, time) in changed_keys for arc, time in zip(detail.arc, detail.datetime)]
            detail["reference_datetime"] = detail.datetime.where(detail.adjusted) - pd.DateOffset(years=1)
            adjusted.append(detail)
            for scope, scope_mask in (("all_arcs_window", pd.Series(True, index=detail.index)),
                ("champs_window", detail.arc.eq(champs)),
                ("champs_nov11", detail.arc.eq(champs) & detail.datetime.dt.day.eq(11)),
                ("champs_closure", detail.adjusted)):
                mask = scope_mask & detail.truth_valid
                for policy, column in (("raw_model", "prediction"), ("copy_prior_year", "adjusted_prediction")):
                    error = detail.loc[mask, column] - detail.loc[mask, "truth"]
                    scores.append({"model": name, "variable": variable, "scope": scope, "policy": policy,
                        "n_expected": int(scope_mask.sum()), "n_scored": int(mask.sum()),
                        "n_adjusted": int((scope_mask & detail.adjusted).sum()),
                        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None})
    return pd.concat(adjusted, ignore_index=True), pd.DataFrame(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Capture new source exports before review")
    parser.add_argument("--snapshot", type=Path, help="Replay a specific immutable capture offline")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--events-file", type=Path, default=disruptions.DEFAULT_EVENTS_PATH)
    parser.add_argument("--assess-history", action="store_true", help="Reassess frozen #55 2025 forecasts with the 2024 measurements")
    args = parser.parse_args()
    if args.refresh == bool(args.snapshot):
        parser.error("Choose exactly one of --refresh or --snapshot")
    arcs = [arc for arc in traffic.load_arcs() if arc.role == "evalue"]
    snapshot = capture(args.raw_dir, arcs, read_review(args.review).get("sources", [])) if args.refresh else args.snapshot
    report(snapshot, args.output_dir, args.review, args.events_file)
    if args.assess_history:
        root = traffic.REPO_ROOT / "outputs/metrics"
        audit = root / "quality_52/observations.parquet"
        predictions = root / "complexity_55/hourly_predictions.parquet"
        metadata = json.loads((predictions.parent / "run.json").read_text(encoding="utf-8"))
        if metadata["audit_observations_sha256"] != traffic.file_sha256(audit):
            raise ValueError("Historical assessment snapshots differ")
        raw = pd.read_parquet(audit)
        detail, scores = assess_history(raw, pd.read_parquet(predictions))
        detail.to_csv(args.output_dir / "historical_predictions.csv", index=False, lineterminator="\n")
        scores.to_csv(args.output_dir / "historical_scores.csv", index=False, lineterminator="\n")
        past = raw.loc[raw.expected_iu_ac.eq(4264) & raw.datetime.dt.month.eq(11) & raw.datetime.dt.day.eq(11)
                       & raw.datetime.dt.year.isin([2024, 2025])]
        past[["arc", "datetime", "q_value", "k_value", "q_status", "k_status", "etat_barre",
              "source_file", "source_member", "source_row"]].to_csv(
            args.output_dir / "historical_observations.csv", index=False, lineterminator="\n")
        _json(args.output_dir / "historical_run.json", {"audit_sha256": traffic.file_sha256(audit),
            "predictions_sha256": traffic.file_sha256(predictions), "cutoff": "2025-11-06 23:00",
            "reference_date": "2024-11-11", "reference_labels": "08:00-14:00 Paris end-hours",
            "interpretation": "Exploratory single annual transfer; already investigated in #47, not an independent test or a promise for 2026."})
    print(f"Reviewed snapshot {snapshot}; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
