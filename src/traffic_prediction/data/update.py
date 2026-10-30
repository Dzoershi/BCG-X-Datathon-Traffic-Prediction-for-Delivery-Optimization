"""Refresh local traffic partitions without rebuilding historical archives."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_prediction.data import build, query, traffic


@dataclass(frozen=True)
class Coverage:
    first_hour: pd.Timestamp | None = None
    last_hour: pd.Timestamp | None = None
    last_source_hour: pd.Timestamp | None = None


@dataclass
class UpdateResult:
    start: pd.Timestamp
    end: pd.Timestamp
    previous_coverage: Coverage
    raw_paths: list[Path]
    written: list[Path]
    unchanged: list[Path]
    stats: traffic.ProcessingStats


def _partition_paths(output_dir: Path) -> list[Path]:
    paths = sorted(output_dir.rglob("*.parquet"))
    for path in paths:
        parts = path.relative_to(output_dir).parts
        if (
            len(parts) != 2 or not re.fullmatch(r"year=[0-9]{4}", parts[0])
            or parts[1] != "part-0.parquet"
        ):
            raise ValueError(
                "Updates require the build layout year=YYYY/part-0.parquet; "
                f"unexpected file: {path}"
            )
    return paths


def dataset_coverage(output_dir: str | Path = build.DEFAULT_TRAFFIC_DIR) -> Coverage:
    """Inspect stored hours without loading all historical rows into Pandas."""
    output_dir = Path(output_dir)
    if not _partition_paths(output_dir):
        return Coverage()
    with query.connect_traffic(output_dir) as connection:
        values = connection.execute(
            """SELECT min(datetime), max(datetime),
                      max(datetime) FILTER (WHERE qualite_mesure <> 'absente')
               FROM traffic"""
        ).fetchone()
    return Coverage(*(pd.Timestamp(value) if value is not None else None for value in values))


@contextmanager
def _dataset_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".traffic-update.lock"
    try:
        lock = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Another update holds {path}. After a hard stop, verify that no "
            "update is running before removing this stale lock."
        ) from error
    try:
        with lock:
            lock.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=traffic.PARIS_TZ)


def _update_period(coverage, start, end, refresh_days, now):
    if isinstance(refresh_days, bool) or not isinstance(refresh_days, int) or refresh_days < 1:
        raise ValueError("refresh_days must be a positive integer.")
    if start is None:
        if coverage.last_source_hour is None:
            raise ValueError("No source coverage exists; provide --start for the first update.")
        start = coverage.last_source_hour.normalize() - pd.Timedelta(days=refresh_days - 1)
    if end is None:
        end = now.tz_localize(None).normalize()
    first, last = traffic.period_bounds(start, end)
    if first.tzinfo is not None or last.tzinfo is not None:
        raise ValueError("Update dates must be timezone-free Paris calendar dates.")
    lower = first.tz_localize(traffic.PARIS_TZ)
    upper = (last.normalize() + pd.Timedelta(days=1)).tz_localize(traffic.PARIS_TZ)
    # Never request observations after the invocation time, even for a partial day.
    upper = min(upper, now)
    if lower >= upper:
        raise ValueError("The update period contains no past or current observations.")
    return first, last, lower.to_pydatetime(), upper.to_pydatetime()


def _merged_partitions(incoming, arcs, output_dir, unchanged):
    for year, fresh_grid in incoming.groupby(incoming["datetime"].dt.year):
        observed = fresh_grid[fresh_grid["qualite_mesure"].ne("absente")]
        if observed.empty:
            continue
        target = output_dir / f"year={year}" / "part-0.parquet"
        old = pd.DataFrame(columns=traffic.OUTPUT_COLUMNS)
        if target.exists():
            if not pq.read_schema(target).equals(build.TRAFFIC_SCHEMA):
                raise ValueError(f"Unexpected processed schema: {target}")
            old = pq.ParquetFile(target).read().to_pandas()
            if not old.empty:
                if not old["datetime"].dt.year.eq(year).all():
                    raise ValueError(f"Rows do not belong to their year partition: {target}")
                build.validate_traffic(
                    old, arcs, old["datetime"].min(), old["datetime"].max()
                )

        # Fillers have the lowest priority; only actual API rows replace history.
        pieces = [fresh_grid[fresh_grid["qualite_mesure"].eq("absente")]]
        if not old.empty:
            pieces.append(old)
        pieces.append(observed)
        rows = pd.concat([part for part in pieces if not part.empty], ignore_index=True)
        rows = rows.drop_duplicates(["arc", "datetime"], keep="last")
        first, last = rows["datetime"].min(), rows["datetime"].max()
        merged = traffic.build_hourly_table(rows, arcs, first, last)
        build.validate_traffic(merged, arcs, first, last)
        if old.equals(merged):
            unchanged.append(target)
        else:
            yield target, merged


def _publish_partitions(partitions, output_dir: Path) -> list[Path]:
    """Stage and validate everything, then atomically replace each partition.

    Raised failures roll back completed replacements. A hard process kill can
    leave completed partitions updated, but never exposes a partially written
    Parquet file; rerunning the requested period is safe.
    """
    staging = Path(mkdtemp(prefix=".traffic-update-", dir=output_dir)).resolve()
    pending, published = [], []
    keep_backups = False
    try:
        for number, (target, table) in enumerate(partitions):
            temporary = staging / f"{number}.new"
            pq.write_table(
                pa.Table.from_pandas(table, schema=build.TRAFFIC_SCHEMA, preserve_index=False),
                temporary, compression="zstd",
            )
            build.check_written([temporary], len(table))
            backup = staging / f"{number}.old" if target.exists() else None
            if backup is not None:
                shutil.copy2(target, backup)
            pending.append((target, temporary, backup))

        try:
            for target, temporary, backup in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                published.append((target, backup))
                temporary.replace(target)
        except BaseException as error:
            failures = []
            for target, backup in reversed(published):
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        backup.replace(target)
                except OSError:
                    failures.append(str(target))
            if failures:
                keep_backups = True
                raise RuntimeError(
                    f"Rollback failed for {failures}; recovery backups remain in {staging}"
                ) from error
            raise
        return [target for target, _, _ in pending]
    finally:
        if not keep_backups:
            # The generated cleanup path must remain directly under this dataset.
            assert staging.parent == output_dir.resolve()
            assert staging.name.startswith(".traffic-update-")
            shutil.rmtree(staging)


def update_traffic(
    start: str | None = None,
    end: str | None = None,
    *,
    refresh_days: int = 7,
    output_dir: str | Path = build.DEFAULT_TRAFFIC_DIR,
    raw_dir: str | Path = traffic.REPO_ROOT / "data" / "raw" / "recent",
    config_path: str | Path = traffic.DEFAULT_CONFIG,
) -> UpdateResult:
    """Refresh each configured arc, preserving history outside supplied rows.

    Dates are inclusive Paris calendar dates. The default starts with a recent
    overlap based on the latest source row, and ends at the invocation time.
    Missing API rows never erase stored observations. Supplied corrections
    replace complete rows, including nulls, using existing processing rules.
    """
    output_dir = Path(output_dir).resolve()
    arcs = traffic.load_arcs(Path(config_path))
    with _dataset_lock(output_dir):
        coverage = dataset_coverage(output_dir)
        first, last, lower, upper = _update_period(
            coverage, start, end, refresh_days, _now()
        )
        raw_paths = [
            traffic.download_observations(lower, upper, Path(raw_dir), road_id=arc.iu_ac)
            for arc in arcs
        ]
        incoming, stats = traffic.process_sources(
            [], arcs, first, last, recent_paths=raw_paths
        )
        build.validate_traffic(incoming, arcs, first, last)
        unchanged = []
        written = _publish_partitions(
            _merged_partitions(incoming, arcs, output_dir, unchanged), output_dir
        )
        return UpdateResult(first, last, coverage, raw_paths, written, unchanged, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", metavar="YYYY-MM-DD", help="First inclusive Paris day; required to bootstrap.")
    parser.add_argument("--end", metavar="YYYY-MM-DD", help="Last inclusive Paris day; default: today, capped at now.")
    parser.add_argument("--refresh-days", type=int, default=7, help="Overlap with latest source day (default: 7).")
    parser.add_argument("--output-dir", type=Path, default=build.DEFAULT_TRAFFIC_DIR)
    parser.add_argument("--raw-dir", type=Path, default=traffic.REPO_ROOT / "data" / "raw" / "recent")
    parser.add_argument("--config", type=Path, default=traffic.DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        result = update_traffic(
            args.start, args.end, refresh_days=args.refresh_days,
            output_dir=args.output_dir, raw_dir=args.raw_dir, config_path=args.config,
        )
    except (OSError, ValueError, RuntimeError, HTTPException, duckdb.Error) as error:
        parser.exit(1, f"Traffic update failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Traffic update interrupted; rerun the requested period to retry.\n")
    print(f"Updated period: {result.start:%Y-%m-%d} to {result.end:%Y-%m-%d}")
    print(f"Previous source coverage ended: {result.previous_coverage.last_source_hour}")
    print(f"Raw snapshots: {len(result.raw_paths)}; source rows: {result.stats.arc_rows}")
    print(f"Written partitions: {[str(path) for path in result.written]}")
    print(f"Unchanged partitions: {[str(path) for path in result.unchanged]}")


if __name__ == "__main__":
    main()
