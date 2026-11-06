"""Measure what a planner gains from a forecast: busy hours flagged and clearest shift found.

From the repository root, after ``scripts/run_local_backtests.py baseline_holiday_aware
timestral_5m lightgbm_raw``::

    python scripts/slot_choice_eval.py

For each of the 8 backtest weeks, each scored road (Convention, Saint-Pères) and each
day, the planner picks the 4-hour block between 08:00 and 20:00 with the lowest
*forecast* traffic, then we look at the *measured* traffic. The E5 ensemble is rebuilt
from its members with the frozen ``E5_WEIGHTS``. Only valid measurements are used
(``qualite_mesure == "ok"``); a day needs at least 8 valid daytime hours and 3
complete blocks. With about 40 day-road cases, a gap of a few points is noise.

Writes ``docs/planner/slot_choice_results.csv`` and prints a summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from traffic_prediction.models.ensemble import E5_WEIGHTS, day_kinds  # noqa: E402

LOCAL = ROOT / "outputs/metrics/local"
MEMBERS = ["baseline_holiday_aware", "timestral_5m", "lightgbm_raw"]
ROADS = ["Convention", "Saint-Pères"]
TARGETS = ["taux_occupation", "debit_horaire"]


def load_truth() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted((ROOT / "data/processed/traffic").glob("year=*/part-0.parquet"))]
    truth = pd.concat(parts)
    truth = truth[(truth.role == "evalue") & truth.arc.isin(ROADS) & (truth.qualite_mesure == "ok")]
    return truth[["arc", "datetime", *TARGETS]]


def rebuild_e5(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    calendar = pd.read_parquet(ROOT / "data/processed/calendar/calendar.parquet")
    key = ["cutoff", "arc", "datetime"]
    merged = preds[MEMBERS[0]][key].copy()
    for name in MEMBERS:
        cols = preds[name][key + TARGETS].rename(columns={t: f"{t}__{name}" for t in TARGETS})
        merged = merged.merge(cols, on=key)
    kinds = day_kinds(pd.DatetimeIndex(merged.datetime), calendar)
    for target in TARGETS:
        values = np.zeros(len(merged))
        for kind, weights in E5_WEIGHTS[target].items():
            mask = kinds == kind
            values[mask] = sum(w * merged.loc[mask, f"{target}__{name}"].to_numpy() for name, w in weights.items())
        merged[target] = values
    merged["kind"] = kinds
    return merged[key + TARGETS + ["kind"]]


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, kinds: pd.DataFrame, target: str) -> pd.DataFrame:
    d = pred[pred.arc.isin(ROADS)][["cutoff", "arc", "datetime", target]].merge(
        truth[["arc", "datetime", target]], on=["arc", "datetime"], suffixes=("_p", "_t"))
    d = d.merge(kinds, on=["cutoff", "arc", "datetime"], how="left")
    d["start"] = d.datetime - pd.Timedelta(hours=1)  # labels mark the end of the hour
    d["day"], d["h"] = d.start.dt.date, d.start.dt.hour
    d = d[(d.h >= 8) & (d.h < 20)]
    rows = []
    for (cutoff, arc, day), g in d.groupby(["cutoff", "arc", "day"]):
        g = g.set_index("h")
        starts = [s for s in range(8, 17) if all(h in g.index for h in range(s, s + 4))]
        if len(g) < 8 or len(starts) < 3:
            continue
        forecast = {s: g.loc[s:s + 3, f"{target}_p"].mean() for s in starts}
        measured = {s: g.loc[s:s + 3, f"{target}_t"].mean() for s in starts}
        chosen, best = min(forecast, key=forecast.get), min(measured, key=measured.get)
        busy_t = g[f"{target}_t"] > g[f"{target}_t"].quantile(2 / 3)
        busy_p = g[f"{target}_p"] > g[f"{target}_p"].quantile(2 / 3)
        rows.append({"cutoff": cutoff, "arc": arc, "day": day, "kind": g["kind"].iloc[0],
                     "best_shift_found": chosen == best,
                     "extra_vs_best_shift": (measured[chosen] - measured[best]) / measured[best],
                     "busy_hours_flagged": float((busy_t == busy_p).mean()),
                     "hour_rank_corr": g[f"{target}_p"].rank().corr(g[f"{target}_t"].rank())})
    return pd.DataFrame(rows)


def main() -> None:
    missing = [n for n in MEMBERS if not (LOCAL / f"{n}_predictions.parquet").exists()]
    if missing:
        raise SystemExit(f"Run scripts/run_local_backtests.py {' '.join(missing)} first")
    preds = {n: pd.read_parquet(LOCAL / f"{n}_predictions.parquet") for n in MEMBERS}
    e5 = rebuild_e5(preds)
    preds["E5"] = e5[["cutoff", "arc", "datetime", *TARGETS]]
    kinds = e5[["cutoff", "arc", "datetime", "kind"]]
    truth = load_truth()
    results = []
    for target in TARGETS:
        for name in ["baseline_holiday_aware", "E5"]:
            r = evaluate(preds[name], truth, kinds, target)
            r.insert(0, "model", name)
            r.insert(1, "target", target)
            results.append(r)
    out = pd.concat(results)
    path = ROOT / "docs/planner/slot_choice_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    summary = out.groupby(["target", "model"]).agg(
        cases=("best_shift_found", "size"), best_shift_found=("best_shift_found", "mean"),
        busy_hours_flagged=("busy_hours_flagged", "mean"), hour_rank_corr=("hour_rank_corr", "mean"),
        extra_vs_best_shift=("extra_vs_best_shift", "mean"))
    print(summary.round(3).to_string())
    print(f"Per-case results: {path}")


if __name__ == "__main__":
    main()
