"""Reproducible traffic patterns, coverage and forecast-error figures (#56)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_prediction import evaluate, validation
from traffic_prediction.constants import TARGET_START
from traffic_prediction.data import traffic, disruptions
from traffic_prediction.data.audit import _json
from traffic_prediction.data.build import DEFAULT_CALENDAR_DIR

MODELS = ("baseline_holiday_aware", "lightgbm_all", "timestral_209m_calendar")
LABELS = {"baseline_holiday_aware": "Holiday-aware baseline", "lightgbm_all": "LightGBM",
          "timestral_209m_calendar": "Timestral-209M"}
UNITS = {"debit_horaire": "Flow (vehicles/hour)", "taux_occupation": "Occupancy (%)"}


def observations(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or raw.datetime.max() >= TARGET_START:
        raise ValueError("Use a nonempty historical snapshot; final-test observations are reserved")
    if raw.duplicated(evaluate.KEYS).any():
        raise ValueError("Duplicate source arc/hour")
    parts = []
    for short, variable in zip(("q", "k"), evaluate.TARGETS):
        part = raw[[*evaluate.KEYS, "role"]].copy()
        part["variable"] = variable
        part["supplied"] = raw.source_file.notna()
        part["status"] = raw[f"{short}_status"].where(part.supplied, "absent")
        part["value"] = raw[f"{short}_value"].where(part.status.eq("valid"))
        part["partial_pair"] = raw.legacy_quality.eq("partielle")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True).merge(
        calendar[["datetime", "is_holiday", "is_school_holiday"]], on="datetime", how="left", validate="many_to_one")
    if result[["is_holiday", "is_school_holiday"]].isna().any().any():
        raise ValueError("Calendar does not cover every observation")
    result["hour"], result["weekday"] = result.datetime.dt.hour, result.datetime.dt.dayofweek
    result["month"] = result.datetime.dt.strftime("%Y-%m")
    result["day_kind"] = np.select([result.is_holiday, result.weekday.ge(5)], ["holiday", "weekend"], default="working")
    return result


def pattern_table(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = data.groupby(keys, sort=True, dropna=False).agg(
        n_expected=("value", "size"), n_valid=("value", "count"), source_rows=("supplied", "sum"),
        mean=("value", "mean"), median=("value", "median"), latest_valid=("datetime", lambda x: x[data.loc[x.index, "value"].notna()].max()),
    ).reset_index()
    result["coverage_percent"] = 100 * result.n_valid / result.n_expected
    return result


def errors(hourly: pd.DataFrame) -> pd.DataFrame:
    validation.check_alignment(hourly)
    selected = hourly.loc[hourly.scenario.eq("observed")].copy()
    selected["hour"] = selected.datetime.dt.hour
    selected["error"] = (selected.prediction - selected.truth).where(selected.truth_valid)
    selected["absolute_error"] = selected.error.abs()
    selected["squared_error"] = selected.error.pow(2)
    result = selected.groupby(["phase", "model", "arc", "variable", "hour"], sort=True).agg(
        n_expected=("error", "size"), n_scored=("error", "count"), bias=("error", "mean"),
        mae=("absolute_error", "mean"), mse=("squared_error", "mean"),
    ).reset_index()
    result["rmse"] = np.sqrt(result.pop("mse"))
    result["n_ignored"] = result.n_expected - result.n_scored
    return result


def examples(hourly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    chosen = hourly.loc[hourly.model.eq(MODELS[0]) & hourly.phase.eq("development") & hourly.scenario.eq("observed")]
    cells = validation.metrics(chosen, ["cutoff", "arc", "variable"])
    eligible = cells.loc[cells.variable.eq("debit_horaire") & cells.n_scored.ge(72)]
    if eligible.empty:
        raise ValueError("Need a baseline development cell with at least 72 scored flow hours")
    worst = eligible.sort_values(["rmse", "cutoff", "arc"], ascending=[False, True, True]).iloc[0]
    detail = chosen.loc[chosen.truth_valid].copy()
    detail["absolute_error"] = (detail.prediction - detail.truth).abs()
    # Separate units: never rank flow errors against occupancy errors.
    largest = detail.sort_values("absolute_error", ascending=False).groupby("variable", sort=False).head(10)
    return {"cutoff": str(worst.cutoff), "arc": worst.arc, "flow_rmse": float(worst.rmse),
            "selection": "largest baseline development arc/window flow RMSE with >=72 observed hours"}, largest


def plots(data: pd.DataFrame, hourly: pd.DataFrame, tables: dict, chosen: dict, output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "traffic56", "svg.fonttype": "none"})
    output.mkdir(parents=True, exist_ok=True)
    official = sorted(data.loc[data.role.eq("evalue"), "arc"].unique())
    files = []
    period = f"{data.datetime.min():%Y-%m-%d} to {data.datetime.max():%Y-%m-%d}"

    def publish(fig, name, title, caption):
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.text(0.02, 0.015, caption, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, .07, 1, .94))
        for extension in ("png", "svg", "pdf"):
            path = output / f"{name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(path, dpi=170, metadata=metadata)
            if extension == "svg":
                # Matplotlib path data has trailing spaces; normalize for reviewable Git diffs.
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
            files.append(path)
        plt.close(fig)

    monthly = tables["monthly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), squeeze=False)
    for axis, variable in zip(axes[:, 0], evaluate.TARGETS):
        table = monthly.loc[monthly.variable.eq(variable)].pivot(index="arc", columns="month", values="coverage_percent")
        image = axis.imshow(table.to_numpy(), vmin=0, vmax=100, aspect="auto", cmap="viridis")
        axis.set_yticks(range(len(table)), [str(arc).replace("Champs-Elysées proxy ", "Proxy ") for arc in table.index])
        axis.set_xticks(range(len(table.columns)), table.columns, rotation=60, fontsize=7)
        axis.set_title(UNITS[variable] + " — valid observed hours / expected hours")
        fig.colorbar(image, ax=axis, label="Coverage (%)", pad=.015)
    publish(fig, "01_coverage", "Measured coverage by arc, target and month",
            f"Source: #52 frozen raw audit | {period} | 24 local labels/day; last month is partial.\nZero coverage is missing usable truth, never an assumed zero measurement. Counts: monthly.csv.")

    fig, axes = plt.subplots(len(official), 2, figsize=(12, 3 * len(official)), squeeze=False)
    profiles = tables["hourly_patterns"]
    for row, arc in enumerate(official):
        for column, variable in enumerate(evaluate.TARGETS):
            axis = axes[row, column]
            for kind, color in [("working", "#17628b"), ("weekend", "#e58e26"), ("holiday", "#8a427a")]:
                part = profiles.loc[profiles.arc.eq(arc) & profiles.variable.eq(variable) & profiles.day_kind.eq(kind)]
                axis.plot(part.hour, part["mean"], label=f"{kind}, n={int(part.n_valid.sum())}", color=color)
            axis.set(title=f"{arc} — {UNITS[variable]}", xlabel="Paris hour", ylabel="Observed mean")
            axis.legend(fontsize=7)
    publish(fig, "02_hourly_patterns", "Hourly profiles use measured values only",
            f"Source: #52 + recorded calendar | {period}\nCounts differ across arcs and seasons. Holiday has priority over weekend; associations are descriptive.")

    fig, axes = plt.subplots(len(official), 2, figsize=(12, 3 * len(official)), squeeze=False)
    for row, arc in enumerate(official):
        for column, variable in enumerate(evaluate.TARGETS):
            axis = axes[row, column]
            part = monthly.loc[monthly.arc.eq(arc) & monthly.variable.eq(variable)]
            axis.plot(part.month, part["mean"], marker=".", color="#17628b")
            axis.set(title=f"{arc} — {UNITS[variable]}", ylabel="Observed monthly mean")
            ticks = list(range(0, len(part), 6))
            if len(part) - 1 not in ticks:
                ticks.append(len(part) - 1)
            axis.set_xlim(-.5, len(part) - .5)
            axis.set_xticks(ticks, part.month.iloc[ticks], rotation=35)
            axis.text(.02, .95, f"n={int(part.n_valid.sum())} valid hours", transform=axis.transAxes, va="top", fontsize=8)
    publish(fig, "03_monthly_patterns", "Monthly variation with observation gaps preserved",
            f"Source: #52 | {period} | Means are conditional on available observations.\nMissing months remain gaps; November 2026 contains only four days. Counts and coverage: monthly.csv.")

    fig, axes = plt.subplots(len(official), 2, figsize=(12, 3 * len(official)), squeeze=False)
    error_table = tables["hourly_errors"]
    for row, arc in enumerate(official):
        for column, variable in enumerate(evaluate.TARGETS):
            axis = axes[row, column]
            for model in MODELS:
                part = error_table.loc[error_table.arc.eq(arc) & error_table.variable.eq(variable)
                                       & error_table.model.eq(model) & error_table.phase.eq("development")]
                axis.plot(part.hour, part.rmse, label=f"{LABELS[model]}, n={int(part.n_scored.sum())}")
            axis.set(title=f"{arc} — {UNITS[variable]}", xlabel="Forecast hour", ylabel="RMSE")
            axis.legend(fontsize=7)
    publish(fig, "04_hourly_errors", "Forecast error by hour on identical observed truth",
            "Source: #55, eight development windows | All three arcs retained, independent targets.\nSaved attribution includes native/fallback forecasts; unavailable truths are ignored and counted, not estimated.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), squeeze=False)
    sample = hourly.loc[hourly.cutoff.eq(pd.Timestamp(chosen["cutoff"])) & hourly.arc.eq(chosen["arc"])
                        & hourly.scenario.eq("observed") & hourly.phase.eq("development")]
    for axis, variable in zip(axes[:, 0], evaluate.TARGETS):
        truth = sample.loc[sample.model.eq(MODELS[0]) & sample.variable.eq(variable)].sort_values("datetime")
        axis.plot(truth.datetime, truth.truth.where(truth.truth_valid), color="black", linewidth=2, label=f"Observed, n={int(truth.truth_valid.sum())}/120")
        for model, color in zip(MODELS, ["#17628b", "#e58e26", "#8a427a"]):
            part = sample.loc[sample.model.eq(model) & sample.variable.eq(variable)].sort_values("datetime")
            for mask, style, suffix in [(part.is_fallback.eq(False).fillna(False), "-", ""),
                                        (part.is_fallback.eq(True).fillna(False), "--", " fallback"),
                                        (part.is_fallback.isna(), ":", " source unknown")]:
                if mask.any():
                    axis.plot(part.datetime, part.prediction.where(mask), label=LABELS[model] + suffix,
                              color=color, linestyle=style, alpha=.85)
        axis.set(ylabel=UNITS[variable]); axis.legend(fontsize=8); axis.tick_params(axis="x", rotation=20)
    publish(fig, "05_error_example", f"Historical error example: {chosen['arc']} | cutoff {chosen['cutoff'][:10]}",
            "Selection: largest baseline development flow RMSE with >=72 observed hours.\nBlack = observed; solid = model, dashed = fallback, dotted = unknown. Attribution: example_predictions.csv.")
    fig, axes = plt.subplots(len(official), 2, figsize=(12, 3 * len(official)), squeeze=False)
    schools = tables["school_patterns"]
    for row, arc in enumerate(official):
        for column, variable in enumerate(evaluate.TARGETS):
            axis = axes[row, column]
            for holiday, color in [(False, "#17628b"), (True, "#e58e26")]:
                part = schools.loc[schools.arc.eq(arc) & schools.variable.eq(variable) & schools.day_kind.eq("working")
                                   & schools.is_school_holiday.eq(holiday)]
                axis.plot(part.hour, part["mean"], color=color, label=f"School holiday={holiday}, n={int(part.n_valid.sum())}")
            axis.set(title=f"{arc} — {UNITS[variable]}", xlabel="Paris hour", ylabel="Observed mean")
            axis.legend(fontsize=7)
    publish(fig, "06_school_holidays", "School-holiday associations within working days",
            f"Source: #52 + recorded Zone C calendar | {period}\nSame hour/day-kind strata; seasonal and coverage differences remain. Descriptive association, not a causal holiday effect.")
    return files


def run(args) -> None:
    raw = pd.read_parquet(args.audit_dir / "observations.parquet")
    calendar = pd.read_parquet(args.calendar)
    data = observations(raw, calendar)
    hourly = pd.read_parquet(args.comparison_dir / "hourly_predictions.parquet")
    validation.check_alignment(hourly)
    metadata = json.loads((args.comparison_dir / "run.json").read_text(encoding="utf-8"))
    if metadata["audit_observations_sha256"] != traffic.file_sha256(args.audit_dir / "observations.parquet"):
        raise ValueError("Analysis/comparison observation snapshots differ")
    if metadata["calendar_sha256"] != traffic.file_sha256(args.calendar):
        raise ValueError("Analysis/comparison calendar snapshots differ")
    if set(MODELS) - set(hourly.model):
        raise ValueError("Run the baseline, LightGBM and Timestral-209M comparison first")
    tables = {
        "monthly": pattern_table(data, ["arc", "role", "variable", "month"]),
        "hourly_patterns": pattern_table(data, ["arc", "role", "variable", "day_kind", "hour"]),
        "weekday_patterns": pattern_table(data, ["arc", "role", "variable", "weekday", "hour"]),
        "school_patterns": pattern_table(data, ["arc", "role", "variable", "day_kind", "hour", "is_school_holiday"]),
        "hourly_errors": errors(hourly),
    }
    chosen, largest = examples(hourly)
    tables["largest_errors"] = largest
    tables["example_predictions"] = hourly.loc[hourly.cutoff.eq(pd.Timestamp(chosen["cutoff"]))
        & hourly.arc.eq(chosen["arc"]) & hourly.scenario.eq("observed") & hourly.model.isin(MODELS)]
    events = pd.read_csv(args.events_file, parse_dates=["date_debut", "date_fin"])
    times = evaluate.target_window(chosen["cutoff"])
    tables["example_event_context"] = events.loc[events.arc.isin([chosen["arc"], disruptions.ALL_ARCS])
        & events.date_debut.le(times.max()) & events.date_fin.ge(times.min())].assign(
            interpretation="Manual contextual source; association is not proof of causation or verified impact")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False, lineterminator="\n")
    files = plots(data, hourly, tables, chosen, args.figures_dir)
    _json(args.output_dir / "run.json", {
        "period": {"first": str(raw.datetime.min()), "last": str(raw.datetime.max())},
        "observation_rows": len(raw), "long_rows": len(data), "chosen_example": chosen,
        "inputs": {"audit": traffic.file_sha256(args.audit_dir / "observations.parquet"),
                   "predictions": traffic.file_sha256(args.comparison_dir / "hourly_predictions.parquet"),
                   "calendar": traffic.file_sha256(args.calendar)}, "comparison_data_sha256": metadata["data_sha256"],
        "manual_event_context_sha256": traffic.file_sha256(args.events_file),
        "code_sha256": traffic.file_sha256(Path(__file__)),
        "figures": {path.name: traffic.file_sha256(path) for path in files},
        "interpretation": "Descriptive associations; counts/coverage differ. No causal school-holiday or event effect is estimated.",
    })
    print(f"Saved {len(tables)} tables and {len(files)} figure exports. Example: {chosen}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/quality_52")
    parser.add_argument("--comparison-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/complexity_55")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_DIR / "calendar.parquet")
    parser.add_argument("--events-file", type=Path, default=disruptions.DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=traffic.REPO_ROOT / "outputs/metrics/traffic_56")
    parser.add_argument("--figures-dir", type=Path, default=traffic.REPO_ROOT / "outputs/figures/traffic_56")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
