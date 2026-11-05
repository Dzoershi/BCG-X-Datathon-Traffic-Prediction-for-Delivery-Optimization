"""Reproduce target-specific traffic quality reports from fingerprinted raw files (#52).

The audit never rewrites production traffic or replaces observations with estimates.
Archives are streamed; only configured directed arcs are retained in a checked cache.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from traffic_prediction.data import traffic
from traffic_prediction.data.historical import iter_zip_members
from traffic_prediction.evaluate import KEYS, TARGETS, load_cutoffs, target_window

CACHE_VERSION = 2
REFERENCE_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/referentiel-comptages-routiers/records"
SELECTED_COLUMNS = [*traffic.RAW_COLUMNS, "arc", "role", "expected_iu_ac", "source_member", "source_row"]
ELIGIBILITY_COLUMNS = ["cutoff", "arc", "role", "variable", "expected_hours", "legacy_scored", "target_scored",
                       "scored_including_suspect_zeros", "legacy_training", "target_training"]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else traffic.REPO_ROOT / candidate


def _get(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        payload = json.load(response)
    if (not isinstance(payload, dict) or not isinstance(payload.get("results"), list)
            or type(payload.get("total_count")) is not int
            or len(payload["results"]) != payload["total_count"]):
        raise ValueError("Sensor inventory response is incomplete; refine the query or paginate")
    return payload


def capture_sensors(path: Path, arcs: list[traffic.Arc]) -> dict:
    """Snapshot both directions by node pair, independently of configured sensor IDs."""
    if path.exists():
        raise FileExistsError(f"Keep the existing raw inventory intact; choose a new snapshot path: {path}")
    entries = []
    for arc in arcs:
        for direction, upstream, downstream in [
            ("configured", arc.upstream_node, arc.downstream_node),
            ("reverse", arc.downstream_node, arc.upstream_node),
        ]:
            # ODSQL string literals: config values are data, not query expressions.
            quote = lambda value: "'" + value.replace("'", "''") + "'"
            where = f"libelle_nd_amont={quote(upstream)} AND libelle_nd_aval={quote(downstream)}"
            queries = {
                "measurements": traffic.RECORDS_URL + "?" + urlencode({
                    "select": "iu_ac,libelle_nd_amont,libelle_nd_aval,min(t_1h) as first_hour,max(t_1h) as last_hour,count(*) as rows",
                    "where": where, "group_by": "iu_ac,libelle_nd_amont,libelle_nd_aval", "limit": 100,
                }),
                "reference": REFERENCE_URL + "?" + urlencode({"where": where, "limit": 100}),
            }
            entries.append({"arc": arc.name, "direction": direction,
                            "expected_iu_ac": arc.iu_ac,
                            "responses": {name: {"url": url, **_get(url)} for name, url in queries.items()}})
    result = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "entries": entries}
    _json(path, result)
    return result


def _select(chunk: pd.DataFrame, arcs: list[traffic.Arc]) -> pd.DataFrame:
    if missing := set(traffic.RAW_COLUMNS) - set(chunk):
        raise ValueError(f"Raw audit input is missing {sorted(missing)}")
    pairs = pd.DataFrame({
        "libelle_nd_amont": [arc.upstream_node for arc in arcs],
        "libelle_nd_aval": [arc.downstream_node for arc in arcs],
        "arc": [arc.name for arc in arcs], "role": [arc.role for arc in arcs],
        "expected_iu_ac": [arc.iu_ac for arc in arcs],
    })
    selected = chunk.merge(pairs, on=["libelle_nd_amont", "libelle_nd_aval"], how="inner", validate="many_to_one")
    # Preserve supplied values in a common string representation; original bytes remain in the source file.
    selected[traffic.RAW_COLUMNS] = selected[traffic.RAW_COLUMNS].astype("string")
    return selected[SELECTED_COLUMNS]


def _read_selected(path: Path, arcs: list[traffic.Arc], chunk_rows: int) -> tuple[pd.DataFrame, int]:
    parts, total = [], 0
    if path.suffix.lower() == ".zip":
        for member, stream in iter_zip_members(path):
            if member.is_dir() or not member.filename.lower().endswith(".txt"):
                continue
            print(f"Reading {path.name}: {member.filename}", flush=True)
            offset = 0
            for chunk in pd.read_csv(stream, sep=";", usecols=traffic.RAW_COLUMNS, dtype=str, keep_default_na=False,
                                     encoding="utf-8", chunksize=chunk_rows):
                chunk["source_member"] = member.filename
                chunk["source_row"] = np.arange(offset + 2, offset + len(chunk) + 2)
                offset += len(chunk)
                total += len(chunk)
                selected = _select(chunk, arcs)
                if not selected.empty:
                    parts.append(selected)
    elif path.suffix.lower() == ".jsonl":
        records = []
        with path.open("rb") as stream:
            number = 0
            while line := stream.readline(traffic.MAX_JSONL_LINE_BYTES + 1):
                number += 1
                if len(line) > traffic.MAX_JSONL_LINE_BYTES:
                    raise ValueError(f"{path}:{number}: JSONL record exceeds 4 MiB")
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or set(traffic.RAW_COLUMNS) - set(record):
                    raise ValueError(f"{path}:{number}: incomplete raw observation")
                traffic.parse_timestamp(record["t_1h"])
                records.append({**{name: record[name] for name in traffic.RAW_COLUMNS},
                                "source_member": "", "source_row": number})
                total += 1
                if len(records) == chunk_rows:
                    parts.append(_select(pd.DataFrame(records), arcs))
                    records.clear()
        if records:
            parts.append(_select(pd.DataFrame(records), arcs))
    else:
        raise ValueError(f"Expected raw .zip or .jsonl, got {path}")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SELECTED_COLUMNS)), total


def extract_source(path: Path, digest: str, arcs: list[traffic.Arc], cache_dir: Path,
                   chunk_rows: int) -> tuple[pd.DataFrame, dict]:
    """Cache only selected raw rows; verify source, config/parser key and cache bytes."""
    if traffic.file_sha256(path) != digest:
        raise ValueError(f"Source checksum mismatch: {path}; use the recorded snapshot")
    configuration = json.dumps([asdict(arc) for arc in arcs], sort_keys=True)
    key = hashlib.sha256(f"{CACHE_VERSION}:{digest}:{configuration}".encode()).hexdigest()
    cached, receipt = cache_dir / f"{key}.parquet", cache_dir / f"{key}.json"
    if cached.exists() and receipt.exists():
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        if metadata.get("cache_sha256") == traffic.file_sha256(cached):
            return pd.read_parquet(cached), metadata
    rows, total = _read_selected(path, arcs, chunk_rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".parquet.tmp")
    rows.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(cached)
    metadata = {"raw_rows": total, "selected_rows": len(rows), "cache_sha256": traffic.file_sha256(cached)}
    _json(receipt, metadata)
    return rows, metadata


def classify(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-target physical validity plus a separate, explicitly uncertain zero-flow flag."""
    result = rows.copy()
    result["datetime"] = traffic.parse_t_1h(result["t_1h"])
    q, bad_q = traffic.parse_measure(result["q"])
    k, bad_k = traffic.parse_measure(result["k"], upper=100)
    suspect = q.eq(0) & k.gt(0)
    result["q_value"], result["k_value"] = q, k
    result["q_status"] = np.select([bad_q, suspect, q.notna()], ["invalid", "suspect_zero", "valid"], default="missing")
    result["k_status"] = np.select([bad_k, k.notna()], ["invalid", "valid"], default="missing")
    legacy = pd.DataFrame({"debit_horaire": q.mask(suspect), "taux_occupation": k})
    result["legacy_quality"] = traffic.measure_quality(legacy, bad_q | bad_k | suspect)
    result["unexpected_sensor_id"] = pd.to_numeric(result["iu_ac"], errors="coerce").ne(result["expected_iu_ac"])
    return result


def hourly_grid(rows: pd.DataFrame, arcs: list[traffic.Arc], first: pd.Timestamp,
                last: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Apply documented last-source-row precedence; retain absent hours explicitly."""
    rows = rows.loc[rows["datetime"].between(first, last)]
    duplicates = int(rows.duplicated(KEYS).sum())
    rows = rows.drop_duplicates(KEYS, keep="last")
    index = pd.MultiIndex.from_product([[arc.name for arc in arcs], pd.date_range(first, last, freq="h")], names=KEYS)
    grid = rows.set_index(KEYS).reindex(index).reset_index()
    grid["role"] = grid.arc.map({arc.name: arc.role for arc in arcs})
    grid["legacy_quality"] = grid["legacy_quality"].fillna("absente")
    grid["month"] = grid.datetime.dt.strftime("%Y-%m")
    return grid, duplicates


def monthly_report(grid: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (arc, month), group in grid.groupby(["arc", "month"], sort=False):
        source = group.source_file.notna()
        for short, target in zip(("q", "k"), TARGETS):
            status, value = group[f"{short}_status"], group[f"{short}_value"]
            valid = status.eq("valid")
            inclusive = valid | status.eq("suspect_zero")
            records.append({
                "arc": arc, "role": group.role.iloc[0], "variable": target, "month": month,
                "expected_hours": len(group), "source_hours": int(source.sum()),
                "absent_source_hours": int((~source).sum()),
                "null_source_values": int((source & status.eq("missing")).sum()),
                "invalid_source_values": int(status.eq("invalid").sum()),
                "suspect_zero_flow": int(status.eq("suspect_zero").sum()),
                "raw_zero_values": int(value.eq(0).sum()),
                "valid_target_values": int(valid.sum()), "valid_including_suspect_zeros": int(inclusive.sum()),
                "legacy_pair_ok": int(group.legacy_quality.eq("ok").sum()),
                "excluded_by_pair_requirement": int((valid & group.legacy_quality.ne("ok")).sum()),
                "valid_in_partial_rows": int((valid & group.legacy_quality.eq("partielle")).sum()),
                "valid_in_invalid_pairs": int((valid & group.legacy_quality.eq("invalide")).sum()),
                "latest_valid_hour": group.loc[valid, "datetime"].max(),
            })
    return pd.DataFrame(records)


def eligibility_report(grid: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Count training and truth eligibility effects, without changing models or scores (#53)."""
    records = []
    for cutoff in cutoffs:
        times = target_window(cutoff)
        if times.min() < grid.datetime.min() or times.max() > grid.datetime.max():
            continue  # Do not present a partial audited period as a complete forecast window.
        for arc, rows in grid.groupby("arc", sort=False):
            history = rows.datetime.le(cutoff + pd.Timedelta(hours=23))
            test = rows.datetime.isin(times)
            for short, target in zip(("q", "k"), TARGETS):
                valid = rows[f"{short}_status"].eq("valid")
                inclusive = valid | rows[f"{short}_status"].eq("suspect_zero")
                records.append({"cutoff": cutoff, "arc": arc, "role": rows.role.iloc[0], "variable": target,
                                "expected_hours": int(test.sum()),
                                "legacy_scored": int((test & rows.legacy_quality.eq("ok")).sum()),
                                "target_scored": int((test & valid).sum()),
                                "scored_including_suspect_zeros": int((test & inclusive).sum()),
                                "legacy_training": int((history & rows.legacy_quality.eq("ok")).sum()),
                                "target_training": int((history & valid).sum())})
    return pd.DataFrame(records, columns=ELIGIBILITY_COLUMNS)


def zero_context(grid: pd.DataFrame) -> pd.DataFrame:
    """Original flagged hours, consecutive run lengths and exact +/-1 hour neighbours."""
    flagged = grid.q_status.eq("suspect_zero")
    starts = flagged & ~flagged.groupby(grid.arc, sort=False).shift(1, fill_value=False)
    runs = starts.cumsum()
    result = grid.loc[flagged].copy()
    result["run_hours"] = result.groupby(["arc", runs.loc[flagged]]).datetime.transform("size")
    indexed = grid.set_index(KEYS)
    for label, delta in (("previous", -1), ("next", 1)):
        lookup = pd.MultiIndex.from_arrays([result.arc, result.datetime + pd.Timedelta(hours=delta)], names=KEYS)
        neighbour = indexed.reindex(lookup)
        for field in ("q_value", "k_value", "legacy_quality"):
            result[f"{label}_{field}"] = neighbour[field].to_numpy()
    return result


def inventory_report(snapshot: dict, arcs: list[traffic.Arc]) -> pd.DataFrame:
    records = []
    for arc in arcs:
        found = [entry for entry in snapshot["entries"] if entry["arc"] == arc.name and entry["direction"] == "configured"]
        if len(found) != 1:
            raise ValueError(f"Sensor snapshot must contain one configured-direction entry for {arc.name}")
        entry = found[0]
        for response in entry["responses"].values():
            if type(response.get("total_count")) is not int or len(response["results"]) != response["total_count"]:
                raise ValueError("Sensor snapshot contains an incomplete response")
        observed = entry["responses"]["measurements"]["results"]
        reference = entry["responses"]["reference"]["results"]
        ids = sorted({int(row["iu_ac"]) for row in observed})
        reference_ids = sorted({int(row["iu_ac"]) for row in reference})
        pairs_match = all(row["libelle_nd_amont"] == arc.upstream_node
                          and row["libelle_nd_aval"] == arc.downstream_node for row in [*observed, *reference])
        reverse = [item for item in snapshot["entries"] if item["arc"] == arc.name and item["direction"] == "reverse"]
        reverse_ids = sorted({int(row["iu_ac"]) for item in reverse
                              for row in item["responses"]["measurements"]["results"]})
        for item in reverse:
            for response in item["responses"].values():
                if type(response.get("total_count")) is not int or len(response["results"]) != response["total_count"]:
                    raise ValueError("Sensor snapshot contains an incomplete reverse-direction response")
        records.append({"arc": arc.name, "expected_id": arc.iu_ac, "upstream": arc.upstream_node,
                        "downstream": arc.downstream_node, "measurement_ids": ";".join(map(str, ids)),
                        "reference_ids": ";".join(map(str, reference_ids)),
                        "reverse_measurement_ids": ";".join(map(str, reverse_ids)),
                        "reverse_checked": len(reverse) == 1,
                        "snapshot_ids_match": (pairs_match and ids == [arc.iu_ac] and arc.iu_ac in reference_ids
                                               and len(reverse) == 1 and arc.iu_ac not in reverse_ids),
                        "checked_at_utc": snapshot["retrieved_at_utc"]})
    return pd.DataFrame(records)


def run_audit(inputs_path: Path, output_dir: Path, *, chunk_rows: int = traffic.CHUNK_ROWS,
              cache_dir: Path | None = None) -> dict:
    """Read the ordered source manifest, verify every hash, and publish auditable sidecars."""
    traffic._validate_chunk_rows(chunk_rows)
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    config = _resolve(inputs.get("config", "configs/baseline.yaml"))
    # Git checkouts may use CRLF on Windows; only normalize this tracked text config.
    # Raw-source checksums below still verify the original bytes without normalization.
    config_digest = hashlib.sha256(config.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if inputs.get("config_sha256", config_digest) != config_digest:
        raise ValueError("Arc configuration checksum mismatch")
    arcs = traffic.load_arcs(config)
    first, last = traffic.period_bounds(inputs["start"], inputs["end"])
    if first.tz is not None or last.tz is not None or pd.isna(first) or pd.isna(last):
        raise ValueError("Audit period must be timezone-free inclusive Paris dates")
    snapshot_path = _resolve(inputs["sensor_snapshot"]["path"])
    if traffic.file_sha256(snapshot_path) != inputs["sensor_snapshot"]["sha256"]:
        raise ValueError("Sensor snapshot checksum mismatch")
    sensors = inventory_report(json.loads(snapshot_path.read_text(encoding="utf-8")), arcs)
    cache_dir = cache_dir or traffic.REPO_ROOT / "data/interim/quality_audit_cache"
    parts, sources, seen, copies = [], [], set(), []
    for source in inputs["sources"]:
        if source["sha256"] in seen:
            # Still verify the claimed duplicate; never trust a manifest's declaration alone.
            if traffic.file_sha256(_resolve(source["path"])) != source["sha256"]:
                raise ValueError(f"Source checksum mismatch: {source['path']}")
            copies.append(source["path"])
            continue
        selected, metadata = extract_source(_resolve(source["path"]), source["sha256"], arcs, cache_dir, chunk_rows)
        seen.add(source["sha256"])
        selected = selected.assign(source_file=source["path"])
        # Parse per file: archive timestamps and API timestamps have different timezone conventions.
        parts.append(classify(selected))
        sources.append({**source, **metadata})
    if not parts or not any(len(part) for part in parts):
        raise ValueError("No configured arc observations in the supplied sources")
    grid, duplicates = hourly_grid(pd.concat(parts, ignore_index=True), arcs, first, last)
    monthly = monthly_report(grid)
    effects = eligibility_report(grid, load_cutoffs())
    zeros = zero_context(grid)
    summary = monthly.groupby(["arc", "role", "variable"], sort=False).agg(
        expected_hours=("expected_hours", "sum"), source_hours=("source_hours", "sum"),
        absent_source_hours=("absent_source_hours", "sum"), null_source_values=("null_source_values", "sum"),
        invalid_source_values=("invalid_source_values", "sum"), suspect_zero_flow=("suspect_zero_flow", "sum"),
        raw_zero_values=("raw_zero_values", "sum"), valid_target_values=("valid_target_values", "sum"),
        legacy_pair_ok=("legacy_pair_ok", "sum"), excluded_by_pair_requirement=("excluded_by_pair_requirement", "sum"),
        latest_valid_hour=("latest_valid_hour", "max"),
    ).reset_index()
    summary["coverage_percent"] = 100 * summary.valid_target_values / summary.expected_hours
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("monthly", monthly), ("summary", summary), ("eligibility", effects), ("sensors", sensors)):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    grid.to_parquet(output_dir / "observations.parquet", index=False, compression="zstd")
    zeros.to_parquet(output_dir / "zero_context.parquet", index=False, compression="zstd")
    report = {"schema_version": 1, "period": {"start": str(first), "end": str(last)},
              "sources": sources, "identical_files_skipped": copies, "duplicate_hour_rows_removed": duplicates,
              "source_order": "manifest order; last supplied row wins; no source row means absent",
              "rows": len(grid), "months": int(grid.month.nunique()), "suspect_zero_flow_hours": len(zeros),
              "unexpected_sensor_rows": int(grid.unexpected_sensor_id.eq(True).sum()),
              "sensor_inventory_matches": bool(sensors.snapshot_ids_match.all()),
              "configuration": [asdict(arc) for arc in arcs], "config_sha256": config_digest,
              "sensor_snapshot_sha256": inputs["sensor_snapshot"]["sha256"],
              "inputs_sha256": traffic.file_sha256(inputs_path),
              "code_sha256": traffic.file_sha256(Path(__file__)),
              "dependency_sha256": {name: traffic.file_sha256(traffic.REPO_ROOT / name) for name in (
                  "src/traffic_prediction/data/traffic.py", "src/traffic_prediction/data/historical.py",
                  "src/traffic_prediction/evaluate.py",
              )},
              "environment": {"python": platform.python_version(),
                              **{name: version(name) for name in ("numpy", "pandas", "pyarrow", "PyYAML", "stream-unzip")}}}
    _json(output_dir / "report.json", report)
    _json(output_dir / "inputs.json", {**inputs, "config_sha256": config_digest})
    print(f"Saved traffic quality audit: {output_dir} ({len(grid):,} arc/hour rows)", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="Ordered, fingerprinted raw-source JSON manifest")
    parser.add_argument("--output-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/quality_52")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=traffic.CHUNK_ROWS)
    parser.add_argument("--capture-sensors", type=Path, help="Explicit online inventory snapshot; does not fetch traffic history")
    args = parser.parse_args()
    if args.capture_sensors:
        capture_sensors(args.capture_sensors, traffic.load_arcs())
    if args.inputs:
        run_audit(args.inputs, args.output_dir, chunk_rows=args.chunk_rows, cache_dir=args.cache_dir)
    if not args.inputs and not args.capture_sensors:
        parser.error("Provide --inputs and/or --capture-sensors")


if __name__ == "__main__":
    main()
