"""Validate processed traffic datasets, saved integrity reports, and submission CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from traffic_prediction.data import build, query, traffic

MANIFEST_VERSION = 1
SUBMISSION_COLUMNS = ["arc", "datetime", "debit_horaire", "taux_occupation"]
SUBMISSION_ARCS = ["Champs-Elysées", "Convention", "Saint-Pères"]


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _configuration(arcs) -> dict:
    values = [asdict(arc) for arc in arcs]
    return {"arcs": values, "sha256": _digest(values)}


def _inventory(root: Path) -> list[dict]:
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No processed Parquet files in {root}; build or update first.")
    entries = []
    for path in files:
        if not pq.read_schema(path).equals(build.TRAFFIC_SCHEMA):
            raise ValueError(f"Unexpected processed traffic schema: {path}")
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": traffic.file_sha256(path),
            "rows": pq.read_metadata(path).num_rows,
        })
    return entries


def _timestamp(value):
    return pd.Timestamp(value).isoformat() if value is not None and pd.notna(value) else None


def _identity(report: dict) -> dict:
    return {"files": report["files"], "configuration_sha256": report["configuration"]["sha256"]}


def validate_dataset(
    data_dir: str | Path = build.DEFAULT_TRAFFIC_DIR,
    *,
    config_path: str | Path = traffic.DEFAULT_CONFIG,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Check the hourly grid one year at a time and return lightweight metadata.

    With no dates, validate the complete stored span. Paired dates restrict the
    row-level checks to those inclusive Paris days; file integrity still covers
    every Parquet file. Missing measurements are reported, not imputed or rejected.
    """
    if (start is None) != (end is None):
        raise ValueError("Provide both --start and --end, or neither.")
    root = Path(data_dir).expanduser().resolve()
    lock = root / ".traffic-update.lock"
    if lock.exists():
        raise ValueError("An update lock exists; validate after the updater has finished.")
    arcs = traffic.load_arcs(Path(config_path))
    configuration = _configuration(arcs)
    files = _inventory(root)
    quality = {arc.name: {
        "arc": arc.name, "role": arc.role, "iu_ac": arc.iu_ac,
        "rows": 0, "source_rows": 0, "measured_q": 0, "measured_k": 0,
        "missing_q": 0, "missing_k": 0, "last_source_hour": None,
        "quality_counts": dict.fromkeys(build.QUALITY_LABELS, 0),
    } for arc in arcs}
    with query.connect_traffic(root) as connection:
        earliest, latest, last_source, stored_rows, null_times = connection.execute(
            """SELECT min(datetime), max(datetime),
                      max(datetime) FILTER (WHERE qualite_mesure <> 'absente'),
                      count(*), count(*) FILTER (WHERE datetime IS NULL)
               FROM traffic"""
        ).fetchone()
        if null_times:
            raise ValueError(f"Missing timestamps: {null_times} rows.")
        if earliest is None:
            raise ValueError("The dataset has no rows; coverage cannot be validated.")
        if stored_rows != sum(entry["rows"] for entry in files):
            raise ValueError("Stored row count differs from Parquet metadata.")
        first, last = traffic.period_bounds(
            earliest if start is None else start, latest if end is None else end
        )
        if first.tzinfo is not None or last.tzinfo is not None or pd.isna(first) or pd.isna(last):
            raise ValueError("Validation dates must be timezone-free Paris calendar dates.")
        available = {row[0].lower() for row in connection.execute("DESCRIBE traffic").fetchall()}
        for partition in ["year", "month"]:
            if partition in available:
                mismatches = connection.execute(
                    f'''SELECT count(*) FROM traffic
                        WHERE try_cast("{partition}" AS INTEGER) IS NULL
                           OR try_cast("{partition}" AS INTEGER) <> date_part('{partition}', datetime)'''
                ).fetchone()[0]
                if mismatches:
                    raise ValueError(f"The {partition} partition disagrees with datetime: {mismatches} rows.")
        columns = ", ".join(f'"{column}"' for column in traffic.OUTPUT_COLUMNS)
        checked_rows = 0
        for year in range(first.year, last.year + 1):
            lower = max(first, pd.Timestamp(year=year, month=1, day=1))
            upper = min(last, pd.Timestamp(year=year, month=12, day=31, hour=23))
            table = connection.execute(
                f"SELECT {columns} FROM traffic WHERE datetime >= ? AND datetime <= ?",
                [lower, upper],
            ).fetchdf()
            build.validate_traffic(table, arcs, lower, upper)
            checked_rows += len(table)
            for arc in arcs:
                rows = table[table["arc"].eq(arc.name)]
                sources = rows[rows["qualite_mesure"].ne("absente")]
                summary = quality[arc.name]
                summary["rows"] += len(rows)
                summary["source_rows"] += len(sources)
                for short, column in [("q", "debit_horaire"), ("k", "taux_occupation")]:
                    summary[f"measured_{short}"] += int(rows[column].notna().sum())
                    summary[f"missing_{short}"] += int(rows[column].isna().sum())
                if not sources.empty:
                    timestamp = _timestamp(sources["datetime"].max())
                    summary["last_source_hour"] = max(summary["last_source_hour"] or timestamp, timestamp)
                for label, count in rows["qualite_mesure"].value_counts().items():
                    summary["quality_counts"][label] += int(count)

    if lock.exists() or _inventory(root) != files:
        raise ValueError("Dataset changed during validation; retry on stable files.")
    if _configuration(traffic.load_arcs(Path(config_path))) != configuration:
        raise ValueError("Arc configuration changed during validation; retry.")
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "duckdb", "PyYAML", "stream-unzip"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "environment": {"python": platform.python_version(), "system": platform.system(), "packages": packages},
        "coverage": {
            "stored_first_hour": _timestamp(earliest), "stored_last_hour": _timestamp(latest),
            "stored_last_source_hour": _timestamp(last_source), "stored_rows": stored_rows,
            "validated_first_hour": _timestamp(first), "validated_last_hour": _timestamp(last),
            "validated_rows": checked_rows,
            "expected_rows": len(arcs) * (int((last - first) / pd.Timedelta(hours=1)) + 1),
        },
        "quality": list(quality.values()),
        "files": files,
    }
    report["fingerprint"] = _digest(_identity(report))
    return report


def write_manifest(report: dict, path: str | Path) -> Path:
    """Atomically save a validation report, without replacing Parquet data."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("The validation manifest must have a .json extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def check_manifest(report: dict, path: str | Path) -> None:
    """Verify exact dataset artifacts and semantic arc configuration against a report."""
    with Path(path).open(encoding="utf-8") as source:
        previous = json.load(source)
    try:
        if type(previous["manifest_version"]) is not int or previous["manifest_version"] != MANIFEST_VERSION:
            raise ValueError("Unsupported validation manifest version.")
        identity = _identity(previous)
        if _digest(previous["configuration"]["arcs"]) != previous["configuration"]["sha256"]:
            raise ValueError("Saved manifest arc configuration is inconsistent.")
        if _digest(identity) != previous["fingerprint"]:
            raise ValueError("Saved manifest fingerprint is inconsistent with its contents.")
        if identity != _identity(report):
            raise ValueError("Dataset files or arc configuration differ from the saved manifest.")
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid validation manifest structure.") from error


def validate_submission(table: pd.DataFrame) -> None:
    """Check the official 360-row November 7–11, 2026 prediction contract.

    This validates predictions supplied by a future model; it does not generate
    forecasts or apply provisional cleaning rules for observed measurements.
    """
    if len(table.columns) != 4 or set(table.columns) != set(SUBMISSION_COLUMNS):
        raise ValueError(f"Submission requires exactly these columns: {SUBMISSION_COLUMNS}")
    if len(table) != 360:
        raise ValueError(f"Submission requires 360 rows, got {len(table)}.")
    if not table["arc"].isin(SUBMISSION_ARCS).all():
        raise ValueError("Submission contains an unknown arc or a proxy.")
    text = table["datetime"]
    if not text.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Submission datetime must contain strings in YYYY-MM-DD HH:MM format.")
    if not text.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}").all():
        raise ValueError("Submission datetime must match YYYY-MM-DD HH:MM exactly.")
    timestamps = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="raise")
    hours = pd.date_range("2026-11-07", "2026-11-11 23:00", freq="h")
    if not timestamps.isin(hours).all():
        raise ValueError("Submission timestamps must be hourly on November 7–11, 2026.")
    if table.duplicated(["arc", "datetime"]).any():
        raise ValueError("Submission contains duplicate arc/hour pairs.")
    for column in ["debit_horaire", "taux_occupation"]:
        values = table[column]
        if not pd.api.types.is_float_dtype(values.dtype):
            raise ValueError(f"Submission {column} must have a floating-point dtype.")
        if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"Submission {column} must be finite and nonnegative.")
    if table["taux_occupation"].gt(100).any():
        raise ValueError("Submission taux_occupation must be between 0 and 100.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=build.DEFAULT_TRAFFIC_DIR)
    parser.add_argument("--config", type=Path, default=traffic.DEFAULT_CONFIG)
    parser.add_argument("--start", metavar="YYYY-MM-DD")
    parser.add_argument("--end", metavar="YYYY-MM-DD")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--write-manifest", action="store_true")
    actions.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--submission", type=Path, help="Validate the official semicolon-separated forecast CSV instead.")
    args = parser.parse_args()
    if args.submission and (args.write_manifest or args.check_manifest or args.start or args.end or args.manifest_path):
        parser.error("Submission validation cannot be combined with dataset range or manifest options.")
    try:
        if args.submission:
            if not re.fullmatch(r"output_.+\.csv", args.submission.name):
                raise ValueError("Submission filename must be output_<group_name>.csv.")
            table = pd.read_csv(args.submission, sep=";", dtype={
                "arc": str, "datetime": str, "debit_horaire": float, "taux_occupation": float,
            })
            # The organizer's unchanged exporter writes its default RangeIndex.
            if list(table.columns) == ["Unnamed: 0", *SUBMISSION_COLUMNS]:
                if not table["Unnamed: 0"].equals(pd.Series(range(len(table)))):
                    raise ValueError("Submission CSV index must be consecutive from zero.")
                table = table.drop(columns="Unnamed: 0")
            validate_submission(table)
            print(f"Valid submission: {args.submission} (360 rows).")
            return
        report = validate_dataset(args.data_dir, config_path=args.config, start=args.start, end=args.end)
        path = args.manifest_path or args.data_dir / "_validation.json"
        if args.check_manifest:
            check_manifest(report, path)
        if args.write_manifest:
            write_manifest(report, path)
        coverage = report["coverage"]
        print(f"Valid dataset: {coverage['validated_rows']} rows, {coverage['validated_first_hour']} to {coverage['validated_last_hour']}")
        print(f"Fingerprint: {report['fingerprint']}")
        for arc in report["quality"]:
            print(f"{arc['arc']}: {arc['source_rows']} source rows, q={arc['measured_q']}, k={arc['measured_k']} measured hours")
        if args.write_manifest:
            print(f"Saved validation manifest: {path}")
        if args.check_manifest:
            print("Saved manifest verified.")
    except (OSError, ValueError, RuntimeError, duckdb.Error) as error:
        parser.exit(1, f"Traffic validation failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Traffic validation interrupted.\n")


if __name__ == "__main__":
    main()
