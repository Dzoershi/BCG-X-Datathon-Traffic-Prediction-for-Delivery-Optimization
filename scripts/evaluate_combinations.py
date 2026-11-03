"""Score single models, day-kind rules and weighted ensembles from saved hourly forecasts (#32).

From the repository root, after ``scripts/run_local_backtests.py``::

    .venv/bin/python scripts/evaluate_combinations.py             # print and write docs/ensemble_results.*
    .venv/bin/python scripts/evaluate_combinations.py --no-write  # print only

No model is run: everything is recombined from ``outputs/metrics/local/*_predictions.parquet``.
Every strategy tested is written to ``docs/ensemble_results.md`` (readable report) and
``docs/ensemble_results.csv`` (one row per strategy and variable), so that the results and what
was tried can be read without rerunning anything. Scores are on Convention + Saint-Pères (the
Champs-Elysées sensor has no truth since 2025-11-19, so every model uses the same fallback there):

- ``pooled``: one RMSE over all valid hours of the eight windows, the #44 decision metric;
- ``by_week``: one RMSE per window, then their mean, closest to BCG X scoring one week;
- per day kind, and the submission week estimate of docs/methodology.md (48 weekend, 48
  working and 24 holiday hours).

Flow and occupancy are always scored, and weights always chosen, separately. Ensemble weights
take one value per day kind and variable, chosen leave-one-window-out: for each window, the
weights minimising the error on the seven others are applied to it, so the score is measured
on unseen windows only. The nested strategy also chooses the ensemble itself per day kind on
the seven other windows. The rules C1-C5 of docs/methodology.md were picked on these same
windows: optimistic. Differences are compared with a bootstrap over the eight windows.
"""

from __future__ import annotations

import argparse
import getpass
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from traffic_prediction.data.traffic import REPO_ROOT
from traffic_prediction.evaluate import KEYS, TARGETS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_local_backtests import OUTPUT_DIR, load_data  # noqa: E402

RESULTS_MD = REPO_ROOT / "docs/ensemble_results.md"
RESULTS_CSV = REPO_ROOT / "docs/ensemble_results.csv"
POOLED_ARCS = ("Convention", "Saint-Pères")
DAY_KINDS = ("ouvre", "weekend", "ferie")
SUBMISSION_HOURS = {"ouvre": 48, "weekend": 48, "ferie": 24}  # 9-10/11, 7-8/11 and 11/11/2026.
REFERENCE = "baseline_holiday_aware"
WEIGHT_GRID = np.round(np.arange(0, 1.01, 0.1), 2)
BOOTSTRAP_DRAWS, SEED = 5000, 0
# Rules from docs/methodology.md ("Pistes pour la #32"): model per day kind, optionally per variable.
RULES = {
    "C1 chronos ouvre / baseline we+ferie": {"ouvre": "chronos2", "weekend": "baseline_holiday_aware",
                                             "ferie": "baseline_holiday_aware"},
    "C2 C1 + lightgbm flow ferie": {"ouvre": "chronos2", "weekend": "baseline_holiday_aware",
                                    "ferie": {"debit_horaire": "lightgbm_raw",
                                              "taux_occupation": "baseline_holiday_aware"}},
    "C3 ts5m ouvre / baseline we / ts209m ferie": {"ouvre": "timestral_5m", "weekend": "baseline_holiday_aware",
                                                   "ferie": "timestral_209m"},
    "C4 C3 + lightgbm flow ferie": {"ouvre": "timestral_5m", "weekend": "baseline_holiday_aware",
                                    "ferie": {"debit_horaire": "lightgbm_raw", "taux_occupation": "timestral_209m"}},
    "C5 chronos ouvre / baseline we / ts209m ferie": {"ouvre": "chronos2", "weekend": "baseline_holiday_aware",
                                                      "ferie": "timestral_209m"},
    "ts209m ouvre+ferie / baseline we": {"ouvre": "timestral_209m", "weekend": "baseline_holiday_aware",
                                         "ferie": "timestral_209m"},
}
# Members of each honest ensemble, justified by the single-model scores per day kind (run 1,
# outputs/metrics/local/combinations_v1.txt): the baseline wins weekends, Timestral-5M and
# Chronos-2 win working days, Timestral-209M wins holidays. No single model wins everywhere.
ENSEMBLES = {
    "E1 baseline + ts5m + ts209m": ["baseline_holiday_aware", "timestral_5m", "timestral_209m"],
    "E2 E1 + chronos2": ["baseline_holiday_aware", "timestral_5m", "timestral_209m", "chronos2"],
    "E3 E1 + baseline day type": ["baseline_holiday_aware", "baseline_day_type", "timestral_5m", "timestral_209m"],
    "E4 E1 + lightgbm": ["baseline_holiday_aware", "timestral_5m", "timestral_209m", "lightgbm_raw"],
    # Run 3 (combinations_v3.txt): LightGBM wins flow on holidays (123) and pairs best with the
    # baseline (87.7), while four members make holiday weights unstable; keep three members.
    "E5 baseline + ts5m + lightgbm": ["baseline_holiday_aware", "timestral_5m", "lightgbm_raw"],
    "E6 baseline + ts209m + lightgbm": ["baseline_holiday_aware", "timestral_209m", "lightgbm_raw"],
}
FAMILIES = {
    "single": ("Modèles seuls", "sans choix"),
    "rule": ("Règles de docs/methodology.md (choisies sur ces semaines)", "non"),
    "ensemble": ("Ensembles, poids par type de jour", "oui"),
    "nested": ("Ensemble choisi par type de jour (validation imbriquée)", "oui"),
    "pair": ("Paires w·A + (1−w)·B, poids par type de jour", "oui"),
}


def quantile_mean(name: str, saved: pd.DataFrame) -> pd.DataFrame | None:
    """`saved` with model hours replaced by the mean of the 19 evenly spaced quantiles 0.05-0.95.

    For RMSE the optimal point is the mean, not the median; fallback hours are unchanged.
    """
    path = OUTPUT_DIR / f"{name}_quantiles.parquet"
    if not path.exists():
        return None
    raw = pd.read_parquet(path).rename(columns={"id": "arc", "timestamp": "datetime", "target_name": "variable"})
    levels = [column for column in raw.columns if column in {f"{0.05 * k:.2f}".rstrip("0") for k in range(1, 20)}]
    if len(levels) != 19:
        return None
    raw["mean"] = raw[levels].mean(axis=1)
    means = raw.pivot_table(index=["cutoff", *KEYS], columns="variable", values="mean")
    result = saved.set_index(["cutoff", *KEYS])
    native = ~result["forecast_source"].astype(str).str.startswith("fallback")
    rows = means.index.intersection(result.index[native])
    result.loc[rows, list(TARGETS)] = means.loc[rows, list(TARGETS)].to_numpy()
    result["debit_horaire"] = result["debit_horaire"].clip(lower=0)
    result["taux_occupation"] = result["taux_occupation"].clip(0, 100)
    return result.reset_index()


def load_frame() -> tuple[pd.DataFrame, list[str], dict]:
    """Truth of the pooled arcs joined to every saved model's forecasts, with day kinds and data source."""
    data, _, source = load_data()
    # Saved ensemble runs use weights chosen on these windows: not a model to recombine.
    names = sorted(path.name.removesuffix("_predictions.parquet")
                   for path in OUTPUT_DIR.glob("*_predictions.parquet") if not path.name.startswith("ensemble"))
    if not names:
        raise SystemExit(f"No saved forecasts in {OUTPUT_DIR}; run scripts/run_local_backtests.py first")
    forecasts = {}
    for name in names:
        saved = pd.read_parquet(OUTPUT_DIR / f"{name}_predictions.parquet")
        forecasts[name] = saved
        if (averaged := quantile_mean(name, saved)) is not None:
            forecasts[f"{name}_qmean"] = averaged
    models = sorted(forecasts)
    frame = None
    for name, saved in forecasts.items():
        saved = saved.loc[saved["arc"].isin(POOLED_ARCS), ["cutoff", *KEYS, *TARGETS]]
        saved = saved.rename(columns={target: f"{target}__{name}" for target in TARGETS})
        frame = saved if frame is None else frame.merge(saved, on=["cutoff", *KEYS], how="inner")
    truth = data.set_index(KEYS)[[*TARGETS, "qualite_mesure"]]
    frame = frame.join(truth, on=KEYS).reset_index(drop=True)
    calendar = holidays.France(years=sorted(frame["datetime"].dt.year.unique()))
    holiday = frame["datetime"].dt.date.map(lambda day: day in calendar).to_numpy(dtype=bool)
    weekend = frame["datetime"].dt.dayofweek.ge(5).to_numpy()
    frame["kind"] = np.where(holiday, "ferie", np.where(weekend, "weekend", "ouvre"))
    return frame, models, source


def truth_mask(frame: pd.DataFrame, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Observed values and the backtest's validity rule (ok, finite, within bounds)."""
    y = frame[variable].to_numpy(dtype=float)
    valid = frame["qualite_mesure"].eq("ok").fillna(False).to_numpy(dtype=bool) & np.isfinite(y)
    valid &= (y >= 0) if variable == "debit_horaire" else ((y >= 0) & (y <= 100))
    return y, valid


def scores(prediction: np.ndarray, y: np.ndarray, valid: np.ndarray, frame: pd.DataFrame) -> dict:
    """Pooled RMSE overall and per day kind, the mean of per-window RMSEs and the submission week estimate."""
    squared = (prediction - y) ** 2
    result = {"pooled": np.sqrt(squared[valid].mean())}
    for kind in DAY_KINDS:
        selected = valid & (frame["kind"].to_numpy() == kind)
        result[kind] = np.sqrt(squared[selected].mean())
    cutoffs = frame["cutoff"].to_numpy()
    weekly = [np.sqrt(squared[valid & (cutoffs == cutoff)].mean()) for cutoff in np.unique(cutoffs[valid])]
    result["by_week"] = float(np.mean(weekly))
    result["submission_week"] = np.sqrt(sum(hours * result[kind] ** 2 for kind, hours in SUBMISSION_HOURS.items())
                                        / sum(SUBMISSION_HOURS.values()))
    return result


def rule_models(rule: dict) -> set[str]:
    return {model for choice in rule.values() for model in (choice.values() if isinstance(choice, dict) else [choice])}


def apply_rule(frame: pd.DataFrame, variable: str, rule: dict) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    for kind, model in rule.items():
        model = model[variable] if isinstance(model, dict) else model
        selected = frame["kind"].to_numpy() == kind
        prediction[selected] = frame.loc[selected, f"{variable}__{model}"].to_numpy(dtype=float)
    return prediction


def simplex(size: int, step: float = 0.1) -> np.ndarray:
    """Every weight vector of `size` nonnegative multiples of `step` summing to one."""
    units = round(1 / step)
    grid = [c for c in itertools.product(range(units + 1), repeat=size) if sum(c) == units]
    return np.array(grid, dtype=float) / units


def modes(size: int) -> tuple[tuple[str, np.ndarray], ...]:
    """One-hot rows give the honest "best model" rule; the simplex gives weighted blends."""
    return ("best model", np.eye(size)), ("blend", simplex(size))


def stack(frame: pd.DataFrame, variable: str, models: list[str]) -> np.ndarray:
    return np.column_stack([frame[f"{variable}__{model}"].to_numpy(dtype=float) for model in models])


def choose(stacked: np.ndarray, candidates: np.ndarray, train: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The candidate weight vector with the smallest squared error on the `train` hours."""
    errors = ((stacked[train] @ candidates.T - y[train, None]) ** 2).mean(axis=0)
    return candidates[int(np.argmin(errors))]


def select_leave_one_window_out(frame: pd.DataFrame, stacked: np.ndarray, candidates: np.ndarray,
                                y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Held-out forecasts of the best weight vector per day kind, chosen on the seven other windows."""
    cutoffs, kinds = frame["cutoff"].to_numpy(), frame["kind"].to_numpy()
    prediction = np.full(len(frame), np.nan)
    chosen = {kind: [] for kind in DAY_KINDS}
    for cutoff in np.unique(cutoffs):
        for kind in DAY_KINDS:
            weights = choose(stacked, candidates, valid & (cutoffs != cutoff) & (kinds == kind), y)
            chosen[kind].append(weights)
            test = (cutoffs == cutoff) & (kinds == kind)
            prediction[test] = stacked[test] @ weights
    return prediction, chosen


def nested_leave_one_window_out(frame: pd.DataFrame, variable: str, models: list[str],
                                y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Choose the ensemble and mode per day kind, then its weights, without the held-out window.

    For each held-out window, every configuration (ensemble x mode) is scored leave-one-window-out
    on the seven others; the best one per day kind is refitted on those seven and applied.
    """
    configs = {f"{label.split()[0]} [{mode}]": (stack(frame, variable, members), candidates)
               for label, members in ENSEMBLES.items() if set(members) <= set(models)
               for mode, candidates in modes(len(members))}
    cutoffs, kinds = frame["cutoff"].to_numpy(), frame["kind"].to_numpy()
    windows = np.unique(cutoffs)
    prediction = np.full(len(frame), np.nan)
    picked = {kind: [] for kind in DAY_KINDS}
    for outer in windows:
        for kind in DAY_KINDS:
            inner_errors = {}
            for label, (stacked, candidates) in configs.items():
                squared, hours = 0.0, 0
                for inner in windows[windows != outer]:
                    test = valid & (kinds == kind) & (cutoffs == inner)
                    if not test.any():
                        continue
                    train = valid & (kinds == kind) & (cutoffs != outer) & (cutoffs != inner)
                    weights = choose(stacked, candidates, train, y)
                    squared += ((stacked[test] @ weights - y[test]) ** 2).sum()
                    hours += test.sum()
                inner_errors[label] = squared / hours if hours else np.inf
            best = min(inner_errors, key=inner_errors.get)
            picked[kind].append(best)
            stacked, candidates = configs[best]
            weights = choose(stacked, candidates, valid & (kinds == kind) & (cutoffs != outer), y)
            test = (kinds == kind) & (cutoffs == outer)
            prediction[test] = stacked[test] @ weights
    return prediction, picked


def short(model: str) -> str:
    return f"{model.split('_')[0][:4]}{model.split('_')[-1][:4]}" if "_" in model else model[:8]


def describe_weights(models: list[str], chosen: dict) -> str:
    """Mean weight per model and day kind over the eight held-out choices, with its range."""
    parts = []
    for kind, vectors in chosen.items():
        vectors = np.array(vectors)
        terms = [f"{short(model)}={vectors[:, i].mean():.2f}[{vectors[:, i].min():.1f}-{vectors[:, i].max():.1f}]"
                 for i, model in enumerate(models) if vectors[:, i].max() > 0]
        parts.append(f"{kind}: " + " ".join(terms))
    return " | ".join(parts)


def describe_picks(picked: dict) -> str:
    """How often each configuration was chosen per day kind over the eight held-out windows."""
    parts = []
    for kind, labels in picked.items():
        values, counts = np.unique(labels, return_counts=True)
        parts.append(f"{kind}: " + ", ".join(f"{value} x{count}" for value, count in zip(values, counts)))
    return " | ".join(parts)


def strategies(frame: pd.DataFrame, variable: str, models: list[str], y: np.ndarray,
               valid: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Every strategy tested with its held-out forecast, and the final ensemble weights.

    Final weights are chosen on all eight windows: what a submission would use.
    """
    rows = [{"family": "single", "strategy": model, "prediction": frame[f"{variable}__{model}"].to_numpy(dtype=float),
             "detail": ""} for model in models]
    rows += [{"family": "rule", "strategy": label, "prediction": apply_rule(frame, variable, rule), "detail": ""}
             for label, rule in RULES.items() if rule_models(rule) <= set(models)]
    final, kinds = [], frame["kind"].to_numpy()
    for label, members in ENSEMBLES.items():
        if not set(members) <= set(models):
            continue
        stacked = stack(frame, variable, members)
        for mode, candidates in modes(len(members)):
            prediction, chosen = select_leave_one_window_out(frame, stacked, candidates, y, valid)
            rows.append({"family": "ensemble", "strategy": f"{label} [{mode}]", "prediction": prediction,
                         "detail": describe_weights(members, chosen)})
            for kind in DAY_KINDS:
                weights = choose(stacked, candidates, valid & (kinds == kind), y)
                final.append({"variable": variable, "strategy": f"{label} [{mode}]", "kind": kind,
                              **{model: float(weight) for model, weight in zip(members, weights)}})
    prediction, picked = nested_leave_one_window_out(frame, variable, models, y, valid)
    rows.append({"family": "nested", "strategy": "Best E1-E6 configuration per day kind", "prediction": prediction,
                 "detail": describe_picks(picked)})
    pair_grid = np.column_stack([WEIGHT_GRID, 1 - WEIGHT_GRID])
    for first, second in itertools.combinations(models, 2):
        if first.removesuffix("_qmean") == second.removesuffix("_qmean"):
            continue  # A model and its own quantile mean: nothing to combine.
        prediction, chosen = select_leave_one_window_out(frame, stack(frame, variable, [first, second]),
                                                         pair_grid, y, valid)
        spread = " ".join(f"{kind[:2]}={min(w[0] for w in v):.1f}-{max(w[0] for w in v):.1f}"
                          for kind, v in chosen.items())
        rows.append({"family": "pair", "strategy": f"{first} + {second}", "prediction": prediction,
                     "detail": f"w(A): {spread}"})
    return rows, final


def compare(rows: list[dict], frame: pd.DataFrame, y: np.ndarray, valid: np.ndarray) -> None:
    """Add scores, windows beaten against the baseline and bootstrap intervals to every row.

    The eight windows are resampled with replacement (same draws for every strategy, so the
    comparisons are paired); an interval excluding zero means the gap survives that resampling.
    """
    cutoffs = frame["cutoff"].to_numpy()
    windows = np.array([cutoff for cutoff in np.unique(cutoffs) if (valid & (cutoffs == cutoff)).any()])
    hours = np.array([(valid & (cutoffs == cutoff)).sum() for cutoff in windows])
    draws = np.random.default_rng(SEED).integers(0, len(windows), size=(BOOTSTRAP_DRAWS, len(windows)))
    for row in rows:
        squared = (row["prediction"] - y) ** 2
        row["sse"] = np.array([squared[valid & (cutoffs == cutoff)].sum() for cutoff in windows])
        row["resampled"] = np.sqrt(row["sse"][draws].sum(axis=1) / hours[draws].sum(axis=1))
        row.update(scores(row["prediction"], y, valid, frame))
    reference = next(row for row in rows if row["strategy"] == REFERENCE)
    best = min((row for row in rows if FAMILIES[row["family"]][1] == "oui"), key=lambda row: row["pooled"])
    for row in rows:
        row["windows_beating_reference"] = int((row["sse"] < reference["sse"]).sum())
        for name, other in (("reference", reference), ("best", best)):
            low, high = np.percentile(row["resampled"] - other["resampled"], [2.5, 97.5])
            row[f"gap_vs_{name}_low"], row[f"gap_vs_{name}_high"] = low, high
        row["best_honest"] = row is best


def print_rows(variable: str, rows: list[dict], hours: int) -> None:
    print(f"\n=== {variable} ({hours} valid hours) ===")
    print(f"  {'':60s} {'pooled':>7s} {'by_week':>8s} | {'ouvre':>7s} {'weekend':>7s} {'ferie':>7s} | "
          f"{'rendu':>7s} | vs {REFERENCE} IC95")
    for family, (title, honest) in FAMILIES.items():
        print(f"  -- {title} (honnête : {honest})")
        for row in (row for row in rows if row["family"] == family):
            print(f"  {row['strategy'][:60]:60s} {row['pooled']:7.2f} {row['by_week']:8.2f} | {row['ouvre']:7.2f} "
                  f"{row['weekend']:7.2f} {row['ferie']:7.2f} | {row['submission_week']:7.2f} | "
                  f"[{row['gap_vs_reference_low']:+.2f}, {row['gap_vs_reference_high']:+.2f}]")
            if row["detail"]:
                print(f"      {row['detail']}")


def fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def interval(row: dict, name: str, digits: int) -> str:
    low, high = row[f"gap_vs_{name}_low"], row[f"gap_vs_{name}_high"]
    text = f"[{fr(low, digits)} ; {fr(high, digits)}]"
    return f"**{text}**" if high < 0 or low > 0 else text


def runs_table() -> list[str]:
    """Provenance of every saved model run, from its _run.json, flagging runs on different data."""
    lines = ["| Run | Lancé (UTC) | Données (sha256) | Durée | Réglages |", "|---|---|---|---:|---|"]
    sources = set()
    for path in sorted(OUTPUT_DIR.glob("*_run.json")):
        if path.name.startswith("ensemble"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        sources.add(record["data"]["content_sha256"])
        lines.append(f"| `{record['run']}` | {record['started_utc'][:16]} | {record['data']['kind']} "
                     f"(`{record['data']['content_sha256'][:8]}`) | {record['seconds_total']:.0f} s | "
                     f"{record['settings'].replace('|', '/')} |")
    if len(sources) > 1:
        lines += ["", "**Attention** : ces runs n'ont pas tous lu les mêmes données. Relancer les runs concernés "
                      "(`scripts/run_local_backtests.py <run>`) ou vérifier que leurs prévisions sont identiques."]
    return lines


def write_results(results: dict, final: list[dict], source: dict, windows: int, author: str) -> None:
    """docs/ensemble_results.md (report) and docs/ensemble_results.csv (every strategy and variable)."""
    columns = ["family", "strategy", "pooled", "by_week", *DAY_KINDS, "submission_week", "windows_beating_reference",
               "gap_vs_reference_low", "gap_vs_reference_high", "gap_vs_best_low", "gap_vs_best_high", "detail"]
    table = pd.concat([pd.DataFrame([{key: row[key] for key in columns} for row in rows]).assign(variable=variable)
                       for variable, rows in results.items()], ignore_index=True)
    generated = datetime.now(timezone.utc)
    table.insert(0, "date", generated.isoformat())
    table.insert(1, "author", author)
    table.insert(4, "honest", table["family"].map(lambda family: FAMILIES[family][1]))
    table[["date", "author", "variable", *table.columns.drop(["date", "author", "variable"])]].round(4).to_csv(RESULTS_CSV, index=False)

    units = {"debit_horaire": ("Débit", "véh/h", 2), "taux_occupation": ("Occupation", "points de %", 3)}
    lines = [
        "# Résultats des combinaisons de modèles (#32)",
        "",
        f"Généré par `scripts/evaluate_combinations.py` le {generated:%Y-%m-%d %H:%M} UTC ; "
        "ne pas modifier à la main. Toutes les lignes, avec plus de décimales : `docs/ensemble_results.csv`.",
        "",
        "## Protocole",
        "",
        f"- Prévisions horaires des {windows} semaines de backtest de `configs/baseline.yaml`, rejouées par "
        "`scripts/run_local_backtests.py` ; aucun modèle n'est réentraîné ici.",
        f"- Données : {source['kind']}, empreinte de contenu `{source['content_sha256'][:8]}`"
        + (f", fingerprint `{source['fingerprint'][:8]}`" if source.get("fingerprint") else "") + ".",
        f"- Score : RMSE globale (*pooled*) sur les heures valides de {' + '.join(POOLED_ARCS)} ; débit et "
        "occupation sont toujours notés, et les poids toujours choisis, séparément.",
        "- `Par semaine` : RMSE de chaque semaine puis moyenne. `Semaine du rendu` : erreurs par type de jour "
        "repondérées sur 48 h de week-end, 48 h ouvrées et 24 h fériées (estimation, pas une mesure).",
        "- **Honnête** : les poids (et pour la validation imbriquée, l'ensemble lui-même) sont choisis sur les "
        "7 autres semaines puis appliqués à la semaine notée, à tour de rôle. Les règles de la doc ont été "
        "repérées sur ces mêmes semaines : leurs scores sont optimistes.",
        f"- IC 95 % : écart de RMSE globale, bootstrap de {BOOTSTRAP_DRAWS} tirages des semaines. En gras : "
        "l'intervalle exclut zéro, l'écart résiste au rééchantillonnage des semaines.",
        "",
        "## Runs de modèles utilisés",
        "",
        *runs_table(),
    ]
    for variable, rows in results.items():
        title, unit, digits = units[variable]
        best = next(row for row in rows if row["best_honest"])
        lines += ["", f"## {title} ({unit})", "",
                  f"Meilleure stratégie honnête : **{best['strategy']}** ({fr(best['pooled'], digits)}). "
                  f"Écarts : stratégie − `{REFERENCE}` et stratégie − meilleure honnête.", ""]
        for family, (family_title, honest) in FAMILIES.items():
            selected = sorted((row for row in rows if row["family"] == family), key=lambda row: row["pooled"])
            lines += [f"### {family_title} — honnête : {honest}", "",
                      "| Stratégie | Global | Par semaine | Ouvrés | Week-ends | Fériés | Semaine du rendu | "
                      "Semaines gagnées vs baseline | IC 95 % vs baseline | IC 95 % vs meilleure | Détail |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
            for row in selected:
                name = f"**{row['strategy']}**" if row["best_honest"] else row["strategy"]
                lines.append(f"| {name} | {fr(row['pooled'], digits)} | {fr(row['by_week'], digits)} | "
                             + " | ".join(fr(row[kind], digits) for kind in DAY_KINDS)
                             + f" | {fr(row['submission_week'], digits)} | {row['windows_beating_reference']}/{windows}"
                             f" | {interval(row, 'reference', digits)} | {interval(row, 'best', digits)}"
                             f" | {row['detail'].replace(' | ', '<br>')} |")
            lines.append("")
    lines += ["## Poids figés des ensembles", "",
              "Choisis sur les 8 semaines à la fois, par type de jour et par variable : les poids qu'utiliserait "
              "le rendu. Leur score honnête est celui des tableaux ci-dessus.", "",
              "| Variable | Ensemble | Type de jour | Poids |", "|---|---|---|---|"]
    for entry in final:
        weights = ", ".join(f"{model} {fr(weight, 1)}" for model, weight in entry.items()
                            if model not in {"variable", "strategy", "kind"} and weight > 0)
        lines.append(f"| {entry['variable']} | {entry['strategy']} | {entry['kind']} | {weights} |")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten {RESULTS_MD.relative_to(REPO_ROOT)} and {RESULTS_CSV.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author", default=getpass.getuser(), help="author recorded in docs/ensemble_results.csv")
    parser.add_argument("--no-write", action="store_true", help="print only, leave docs/ensemble_results.* unchanged")
    args = parser.parse_args()
    frame, models, source = load_frame()
    windows = frame["cutoff"].nunique()
    print(f"Models: {', '.join(models)}; {windows} windows; arcs {' + '.join(POOLED_ARCS)}")
    results, final = {}, []
    for variable in TARGETS:
        y, valid = truth_mask(frame, variable)
        rows, weights = strategies(frame, variable, models, y, valid)
        compare(rows, frame, y, valid)
        print_rows(variable, rows, int(valid.sum()))
        results[variable], final = rows, final + weights
    if not args.no_write:
        write_results(results, final, source, windows, args.author)


if __name__ == "__main__":
    main()
