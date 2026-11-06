"""Draw the two charts used in the deliverable-2 slides.

From the repository root::

    python scripts/presentation_figures.py

Writes ``docs/presentation/accuracy_by_day_type.png`` and
``docs/presentation/driver_cost_waterfall.png``.

Sources of the numbers:
- accuracy: honest leave-one-week-out scores of the baseline and E5 (``docs/ensemble_results.md``,
  pooled RMSE on Convention + Saint-Pères over the 8 backtest weeks);
- cost: business-case assumptions of the deck (appendices H and I): 100 drivers, 2,500 h a year,
  €10.6/h average (brief rates €10 day / €14 night, 85/15 mix), lever 1 +3% parcels/h
  (TomTom Traffic Index 2025, Paris), lever 2 -5% paid hours (assumption, tested -3% to -8%).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs/presentation"
DG, G, GR, TX = "#03522D", "#29BA74", "#B5B5B5", "#575757"

CATS = ["All days", "Weekdays", "Weekends", "Holidays"]
FLOW = {"baseline": [97.5, 94.1, 84.7, 142.3], "E5": [88.0, 74.6, 83.7, 135.9]}
OCC = {"baseline": [2.04, 2.71, 1.28, 1.22], "E5": [1.71, 2.10, 1.31, 1.31]}

BASE_COST = 100 * 2_500 * 10.6 / 1e6            # €M a year
LEVER1 = BASE_COST * (1 - 1 / 1.03)             # +3% parcels per hour
LEVER2 = (BASE_COST - LEVER1) * 0.05            # -5% paid hours, after lever 1


def accuracy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), dpi=200)
    for ax, values, title, fmt, gain in [(axes[0], FLOW, "Flow error (vehicles/hour)", "{:.1f}", "−10%"),
                                         (axes[1], OCC, "Occupancy error (points)", "{:.2f}", "−16%")]:
        x, w = range(len(CATS)), 0.38
        bars_b = ax.bar([i - w / 2 for i in x], values["baseline"], w, color=GR, label="Same as last week (baseline)")
        bars_e = ax.bar([i + w / 2 for i in x], values["E5"], w, color=[DG] + [G] * 3, label="Our forecast")
        for bars, vals, color, bold in [(bars_b, values["baseline"], TX, "normal"), (bars_e, values["E5"], DG, "bold")]:
            for r, v in zip(bars, vals):
                ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=color, fontweight=bold)
        ax.text(0.19, values["E5"][0] * 0.45, gain, ha="center", color="white", fontsize=8.5, fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(CATS, fontsize=9, color=TX)
        ax.set_title(title, fontsize=11, color=TX, loc="left")
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GR); ax.set_yticks([]); ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "accuracy_by_day_type.png", transparent=True)


def waterfall() -> None:
    end = BASE_COST - LEVER1 - LEVER2
    fig, ax = plt.subplots(figsize=(5.9, 4.95), dpi=200)
    ax.bar(0, BASE_COST, color=GR, width=0.6)
    ax.bar(1, LEVER1, bottom=BASE_COST - LEVER1, color=G, width=0.6)
    ax.bar(2, LEVER2, bottom=end, color=G, width=0.6)
    ax.bar(3, end, color=DG, width=0.6)
    for x, y, text, color in [(0, BASE_COST, f"€{BASE_COST:.2f}M", TX), (1, BASE_COST, f"−€{LEVER1 * 1000:.0f}k", G),
                              (2, BASE_COST - LEVER1, f"−€{LEVER2 * 1000:.0f}k", G), (3, end, f"€{end:.2f}M", DG)]:
        ax.text(x, y + 0.03, text, ha="center", color=color, fontsize=11, fontweight="bold" if x else "normal")
    for x0, y in [(0.3, BASE_COST), (1.3, BASE_COST - LEVER1), (2.3, end)]:
        ax.plot([x0, x0 + 0.4], [y, y], color=GR, lw=0.8)
    ax.set_ylim(2.0, 2.78)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Today\n(no traffic\nforecast)", "Lever 1\nclear slots", "Lever 2\nright-size\nstaffing", "Traffic-\ninformed\nplan"], fontsize=9.5, color=TX)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GR); ax.set_yticks([]); ax.tick_params(length=0)
    ax.set_title("Annual driver cost (€M), axis starts at €2.0M", fontsize=10, color=TX, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "driver_cost_waterfall.png", transparent=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy()
    waterfall()
    print(f"Figures written to {OUT}")
