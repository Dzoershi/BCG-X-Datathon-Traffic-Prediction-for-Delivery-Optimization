"""Issue #54 reproducible baseline comparison from the saved local backtests.

Run at repository root after all four run_local_backtests.py runs.
Does not download data, modify models, or use observations beyond the specified end date.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

from traffic_prediction.data import dataset, traffic
from traffic_prediction.data.query import DEFAULT_TRAFFIC_DIR
from traffic_prediction.evaluate import KEYS, TARGETS, load_cutoffs, target_window

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics" / "local"
NAMES = ["baseline_holiday_aware", "baseline_day_type", "baseline_previous_week", "baseline_four_week"]
END = "2026-11-04"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    data = dataset.read_traffic(DEFAULT_TRAFFIC_DIR, traffic.load_arcs(), end=END)
    expected_data_hash = dataset.content_sha256(data)
    truth = data.loc[data["role"].eq("evalue"), [*KEYS, "qualite_mesure", *TARGETS]].copy()
    prediction_tables = {}
    run_records = {}
    for name in NAMES:
        p = OUT / f"{name}_predictions.parquet"
        meta = OUT / f"{name}_run.json"
        if not p.exists() or not meta.exists():
            raise SystemExit(f"Missing output for {name}. Run the backtest first.")
        pred = pd.read_parquet(p)
        record = json.loads(meta.read_text(encoding="utf-8"))
        actual_hash = record["data"].get("content_sha256")
        if actual_hash != expected_data_hash:
            raise SystemExit(f"Data fingerprint mismatch in {name}: {actual_hash} != {expected_data_hash}")
        assert record.get("data", {}).get("end") == END, f"Different end date for {name}"
        if pred.duplicated(["cutoff", *KEYS]).any():
            raise SystemExit(f"Duplicate prediction keys: {name}")
        prediction_tables[name] = pred
        run_records[name] = record

    # Verify identical chronological prediction keys across candidates and full 3x120 grid per cutoff.
    expected = pd.MultiIndex.from_tuples(
        [(cutoff, arc, dt) for cutoff in load_cutoffs()
         for arc in sorted(truth["arc"].unique()) for dt in target_window(cutoff)],
        names=["cutoff", *KEYS],
    )
    for name, pred in prediction_tables.items():
        keys = pd.MultiIndex.from_frame(pred[["cutoff", *KEYS]])
        if len(keys) != len(expected) or not keys.is_unique or not keys.sort_values().equals(expected.sort_values()):
            raise SystemExit(f"Prediction key/grid mismatch for {name}")

    rows = []
    source_rows = []
    for name, pred in prediction_tables.items():
        joined = pred.merge(truth, on=KEYS, how="left", suffixes=("_pred", "_true"), validate="many_to_one")
        for arc, frame in joined.groupby("arc", sort=True):
            for target in TARGETS:
                y = pd.to_numeric(frame[f"{target}_true"], errors="coerce")
                p = pd.to_numeric(frame[f"{target}_pred"], errors="coerce")
                valid = frame["qualite_mesure"].eq("ok") & np.isfinite(y)
                valid &= y.ge(0)
                if target == "taux_occupation":
                    valid &= y.le(100)
                n = int(valid.sum())
                bad = valid & ~np.isfinite(p)
                if bad.any():
                    raise SystemExit(f"Nonfinite predictions with valid truth: {name}/{arc}/{target}")
                rmse = float(np.sqrt(np.mean(np.square((p[valid] - y[valid]).to_numpy())))) if n else None
                rows.append(dict(model=name, arc=arc, target=target, rmse=rmse,
                                 valid_truth=n, total_hours=len(frame),
                                 truth_coverage=round(n / len(frame), 6),
                                 forecast_coverage=round(float(np.isfinite(p).mean()), 6)))
            for source, count in frame["forecast_source"].fillna("<missing>").value_counts().items():
                source_rows.append(dict(model=name, arc=arc, source=str(source), predictions=int(count)))

    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    sources = pd.DataFrame(source_rows)
    scores.to_csv(OUT / "issue54_per_arc_scores.csv", index=False, float_format="%.8f")
    sources.to_csv(OUT / "issue54_sources.csv", index=False)

    files = [ROOT / "configs/baseline.yaml", ROOT / "data/processed/calendar/calendar.parquet"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}
    metadata = dict(
        input_content_sha256=expected_data_hash,
        evaluated_through=END,
        cutoffs=[str(d.date()) for d in load_cutoffs()],
        config_and_calendar_sha256=hashes,
        Python=platform.python_version(),
        model_settings={n: run_records[n]["settings"] for n in NAMES},
        versions={n: run_records[n]["versions"] for n in NAMES},
        local_run_json={n: sha256(OUT / f"{n}_run.json") for n in NAMES},
    )
    (OUT / "issue54_fingerprint.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (ROOT / "docs/issue54_baseline_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Issue #54 — Baseline comparison (pre-freeze evidence)\n\n")
        f.write("Generated from identical chronological prediction keys and cutoff-limited traffic. ")
        f.write("Missing/invalid truth excluded per target, matching the validity criteria of `evaluate.backtest`. ")
        f.write("Truth coverage is valid observed hours / all candidate hours; forecast coverage is finite predictions / all hours.\n\n")
        f.write("The pooled aggregate of `run_local_backtests.py` covers Convention and Saint-Pères; ")
        f.write("this file additionally reports all three arcs. The RMSE below is pooled by arc/target over valid hours, ")
        f.write("which is not the same as `evaluate.backtest`'s unweighted mean over cutoff-specific RMSE values.\n\n")
        f.write("## RMSE and coverage per arc/target\n\n")
        f.write("| Model | Arc | Target | RMSE | Valid truth | Truth coverage | Forecast coverage |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            rmse = "N/A (no truth)" if r["rmse"] is None else f'{r["rmse"]:.3f}'
            f.write(f'| {r["model"]} | {r["arc"]} | {r["target"]} | {rmse} | {r["valid_truth"]}/{r["total_hours"]} | {r["truth_coverage"]:.1%} | {r["forecast_coverage"]:.1%} |\n')
        f.write("\n## Forecast source distribution by model and arc\n\n")
        f.write("| Model | Arc | Source / fallback | Hours |\n|---|---|---|---:|\n")
        for r in source_rows:
            f.write(f'| {r["model"]} | {r["arc"]} | {r["source"]} | {r["predictions"]} |\n')
        f.write("\n## Reproduction\n\n")
        f.write("```powershell\n")
        f.write(".\\.venv\\Scripts\\python.exe scripts/run_local_backtests.py baseline_holiday_aware baseline_day_type baseline_previous_week baseline_four_week --end 2026-11-04\n")
        f.write(".\\.venv\\Scripts\\python.exe scripts/issue54_report.py\n")
        f.write(".\\.venv\\Scripts\\python.exe -m pytest tests/test_baselines.py -q\n")
        f.write("```\n\n")
        f.write(f"Processed traffic content SHA-256: `{expected_data_hash}`\n\n")
        f.write("Further checks required before closure: the simulated Champs-Elysées outages and proxy evidence; ")
        f.write("the final frozen 2026-11-05 23:00 cutoff; repeat-run identical submission CSV SHA-256; ")
        f.write("and documented baseline selection/rationale. Do not use November 7–11 observations to select the model.\n")
    print(f"Wrote {ROOT / 'docs/issue54_baseline_comparison.md'}")
    print(scores.to_string(index=False))
    print("Forecast sources saved to outputs/metrics/local/issue54_sources.csv")

if __name__ == "__main__":
    main()
