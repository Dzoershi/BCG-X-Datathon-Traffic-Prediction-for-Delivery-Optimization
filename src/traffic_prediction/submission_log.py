"""Best-effort, offline provenance for an already published submission CSV (#48)."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from traffic_prediction.data.dataset import content_sha256
from traffic_prediction.data.traffic import REPO_ROOT, file_sha256
from traffic_prediction.models.chronos import MODEL_ID, MODEL_REVISION
from traffic_prediction.models.timestral import MODEL_REVISIONS, SOURCE_REVISION

TRAFFIC_COLUMNS = ["arc", "datetime", "role", "qualite_mesure", "debit_horaire", "taux_occupation"]


def git_value(*args: str) -> str | None:
    """Allow source archives and hosts without Git; never access the network."""
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def timestral_package() -> dict:
    """Record installed VCS provenance rather than presenting the required pin as installed."""
    try:
        distribution = importlib.metadata.distribution("timestral")
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "commit": None}
    direct = distribution.read_text("direct_url.json")
    commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    return {"version": distribution.version, "commit": commit}


def write_run_log(csv_path: Path, *, history: pd.DataFrame, table: pd.DataFrame,
                  model: object, model_name: str, cutoff: pd.Timestamp,
                  seed: int, started: float) -> None:
    """Write atomically after export; any metadata/serialization/I/O failure only warns.

    Traffic fingerprints cover the exact cutoff-limited evaluated traffic supplied
    to prediction, independent of Parquet encoding. Additional model inputs (e.g.
    weather merged into history) and the model calendar have separate hashes.
    Runtime measures wall time from run start through CSV export and metadata collection.
    """
    try:
        history = history.loc[history["datetime"].le(cutoff)].sort_values(
            ["arc", "datetime"], kind="stable").reset_index(drop=True)
        keys = table[["arc", "datetime"]].copy()
        keys["datetime"] = pd.to_datetime(keys["datetime"])
        index = pd.MultiIndex.from_frame(keys)
        sources = getattr(model, "sources_", None)
        sources = (sources.reindex(index) if sources is not None
                   else pd.Series(type(model).__name__, index=index))
        rows = keys.copy()
        rows["datetime"] = rows["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        rows["source"] = sources.to_numpy()
        member_sources = getattr(model, "member_sources_", pd.DataFrame())
        if not member_sources.empty:
            rows["member_sources"] = member_sources.reindex(index).to_dict(orient="records")
        # A row with both measured targets is an observation; missing/invalid rows
        # must not make an arc with a stale sensor appear current.
        values = history[["debit_horaire", "taux_occupation"]].to_numpy(dtype=float, na_value=np.nan)
        valid = (history["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool)
                 & np.isfinite(values).all(axis=1) & (values[:, 0] >= 0)
                 & (values[:, 1] >= 0) & (values[:, 1] <= 100))
        last = history.loc[valid].groupby("arc")["datetime"].max()
        status = git_value("status", "--porcelain", "--untracked-files=normal")
        calendar = getattr(model, "calendar", None)
        payload = {
            "schema_version": 1,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "model_choice": model_name,
            "model_class": type(model).__name__,
            "model_settings": model.describe() if hasattr(model, "describe") else type(model).__name__,
            "weights": getattr(model, "weights", None),
            "model_revisions": {
                "chronos_2": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
                "timestral": {size: revision for size, revision in MODEL_REVISIONS.items()},
                "timestral_package_required_commit": SOURCE_REVISION,
                "timestral_package_installed": timestral_package(),
            },
            "cutoff": cutoff.isoformat(),
            "seed": seed,
            "last_observed_hour_per_arc": {
                arc: last[arc].isoformat() if arc in last else None
                for arc in sorted(history["arc"].unique())
            },
            "traffic_fingerprint": content_sha256(history[TRAFFIC_COLUMNS]),
            "model_history_fingerprint": content_sha256(history),
            "calendar_fingerprint": content_sha256(calendar) if calendar is not None else None,
            "sources": rows.where(pd.notna(rows), None).to_dict(orient="records"),
            "closure_correction_applied": bool(sources.str.startswith("closure_copy_").fillna(False).any()),
            "csv_file": csv_path.name,
            "csv_sha256": file_sha256(csv_path),
            "runtime_seconds": time.perf_counter() - started,
        }
        destination = csv_path.with_name(f"{csv_path.stem}_run.json")
        with TemporaryDirectory(dir=csv_path.parent, prefix=".run-log-") as staging:
            temporary = Path(staging) / destination.name
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(destination)
    except Exception as error:
        print(f"Warning: submission run log could not be written: {error}", file=sys.stderr)
