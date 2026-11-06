# Frozen baseline reference — issue #54

## Decision

Freeze **`DayTypeProfileBaseline`** as the common simple reference and standalone fallback candidate.
On identical measured truth across all three official arcs it has the lowest development pooled RMSE
of the four existing baselines for **both** targets. No weighted sum combines incompatible units.
The recent diagnostic supports the choice but was not used to fit parameters. This replaces the
pre-freeze PR recommendation based on a two-arc pool; neither that older score nor the older CSV hash
is the final freeze. This decision does **not** replace production ensemble E5 or change its internal
holiday-aware fallback: any final-model change requires an aligned comparison and explicit decision.

Development is the eight Thursday cutoffs in `configs/baseline.yaml`; history ends at Thursday 23:00,
Friday is skipped, then 120 Saturday–Wednesday hours are forecast per official arc. The separate
recent cutoff is October 29 (October 31–November 4 forecasts). Settings were not tuned on it.
All dates are Paris local end-of-hour labels, retaining the project's unresolved label/DST convention.
Historical publication/revision latency cannot be reconstructed. November 7–11 observations are never
read as truth or used for model selection. Final-horizon RMSE is not yet available.

## Comparable results

Flow RMSE is vehicles/hour; occupancy RMSE is percentage points. Pooled RMSE is
`sqrt(sum(squared errors)/scored hours)`. Macro RMSE, unweighted across nonempty arc/window cells,
is also exported, but does not drive selection. Independent per-target eligibility follows #52/#53:
only valid observed values, excluding suspect zero flow and out-of-domain measurements; proxy estimates
never become truth. All models retain the same full grid, truth and validity mask.

| phase | model | variable | rmse | n_scored | n_expected |
|---|---|---|---|---|---|
| development | day_type | debit_horaire | 145.133 | 1691 | 2880 |
| development | day_type | taux_occupation | 4.240 | 1684 | 2880 |
| development | holiday_aware | debit_horaire | 166.667 | 1691 | 2880 |
| development | holiday_aware | taux_occupation | 5.809 | 1684 | 2880 |
| development | previous_week | debit_horaire | 182.543 | 1691 | 2880 |
| development | previous_week | taux_occupation | 5.674 | 1684 | 2880 |
| development | four_week | debit_horaire | 171.998 | 1691 | 2880 |
| development | four_week | taux_occupation | 4.529 | 1684 | 2880 |
| recent_check | day_type | debit_horaire | 70.312 | 239 | 360 |
| recent_check | day_type | taux_occupation | 1.854 | 239 | 360 |
| recent_check | holiday_aware | debit_horaire | 77.381 | 239 | 360 |
| recent_check | holiday_aware | taux_occupation | 2.314 | 239 | 360 |
| recent_check | previous_week | debit_horaire | 77.381 | 239 | 360 |
| recent_check | previous_week | taux_occupation | 2.314 | 239 | 360 |
| recent_check | four_week | debit_horaire | 76.242 | 239 | 360 |
| recent_check | four_week | taux_occupation | 2.237 | 239 | 360 |

### Selected baseline by arc

| phase | arc | variable | rmse | n_scored | n_expected |
|---|---|---|---|---|---|
| development | Champs-Elysées | debit_horaire | 188.629 | 697 | 960 |
| development | Champs-Elysées | taux_occupation | 6.055 | 690 | 960 |
| development | Convention | debit_horaire | 108.432 | 659 | 960 |
| development | Convention | taux_occupation | 2.319 | 659 | 960 |
| development | Saint-Pères | debit_horaire | 95.741 | 335 | 960 |
| development | Saint-Pères | taux_occupation | 2.071 | 335 | 960 |
| recent_check | Champs-Elysées | debit_horaire | N/A | 0 | 120 |
| recent_check | Champs-Elysées | taux_occupation | N/A | 0 | 120 |
| recent_check | Convention | debit_horaire | 57.355 | 119 | 120 |
| recent_check | Convention | taux_occupation | 1.618 | 119 | 120 |
| recent_check | Saint-Pères | debit_horaire | 81.143 | 120 | 120 |
| recent_check | Saint-Pères | taux_occupation | 2.062 | 120 | 120 |

All candidates have 100% finite forecast coverage, including hours without measured truth. Development
coverage differs by arc and target, so the global winner need not win every arc. The holiday-aware
baseline remains better for Convention; that tradeoff is visible in `per_arc.csv`. We select a single
simple policy rather than tune a patchwork of rules on these reused windows. Saint-Pères has only
335 development truth hours per target. Champs has zero recent measured truth: its 120 recent forecasts
are retained and attributed but no recent accuracy is claimed. Pooled gains are conditional on the
observed mix of hours, not proof of superiority on the unobserved final week.

## Rules, missing values and public holidays

For each arc and target hour, average valid historical measurements for the same **weekday and hour**;
a metropolitan French public holiday uses Sunday's group. This is seven weekday groups, not merely
working-day/weekend groups. If unavailable, use the same arc/hour averaged across days, then the same
day-type/hour pooled across evaluated arcs. If every level is empty, fail explicitly. No proxy or
weather dependency is used. Clip flow at zero and occupancy to [0,100]; submission rounds to two decimals.

The existing baseline training contract requires complete pairs with `qualite_mesure == ok`, although
the evaluator independently scores either valid target. Partial pairs therefore reduce training coverage;
they are not silently imputed. The same source rule supplies both targets. Calendar flags are known in
advance. No event correction is part of this reference. Profiles average all eligible history and cannot
adapt to recent changes without new measurements. Holidays being Sunday-like is an assumption, not a
measured causal effect. A generic profile cannot anticipate a ceremony closure; #57 documents provisional
restrictions and why unapproved schedules must not automatically change the predictions.

`submission_sources.csv` records the rule for each of the 360 official arc/hour predictions.
`per_forecast_source.csv` reports all forecast counts separately from scored fallback/native counts;
`per_day_kind.csv` separates working days, weekends and holidays, with sample sizes. Per-window scores
and common-truth coverage are retained, including undefined scores on empty groups.

## Simulated outages and proxy decision (#51/#53)

Mask Champs history for the last 28, 90, 180 or 365 days before each eligible cutoff, retain untouched
observed forecast-window truth, and replay **all four baselines**. Also replay the #53 proxy candidates:
direct copy, historical target-minus-proxy offset, and linear calibration for 4272 and 4265. Calibration
precedes the outage; inputs use the known previous week's proxy, never contemporaneous future observations.
Missing/ineligible proxies fall back to the pre-outage target profile. At least 56 pre-outage calendar
days are required; skipped windows are recorded in `run.json`. The real 2026 outage has no observed target
truth and cannot be scored. No hidden target is used for calibration or reconstruction.

The following table includes the selected baseline, profile control and the Washington–Balzac linear
candidate (the best contributing proxy in #53); all other proxy methods remain in the CSVs.

| scenario | model | variable | rmse | n_scored | n_expected |
|---|---|---|---|---|---|
| outage_28d | profile | debit_horaire | 189.879 | 697 | 840 |
| outage_28d | profile | taux_occupation | 6.184 | 690 | 840 |
| outage_28d | linear:Champs-Elysées proxy Washington-Balzac | debit_horaire | 268.779 | 697 | 840 |
| outage_28d | linear:Champs-Elysées proxy Washington-Balzac | taux_occupation | 7.916 | 690 | 840 |
| outage_90d | profile | debit_horaire | 193.220 | 697 | 840 |
| outage_90d | profile | taux_occupation | 6.314 | 690 | 840 |
| outage_90d | linear:Champs-Elysées proxy Washington-Balzac | debit_horaire | 275.416 | 697 | 840 |
| outage_90d | linear:Champs-Elysées proxy Washington-Balzac | taux_occupation | 7.852 | 690 | 840 |
| outage_180d | profile | debit_horaire | 236.377 | 697 | 840 |
| outage_180d | profile | taux_occupation | 6.289 | 690 | 840 |
| outage_180d | linear:Champs-Elysées proxy Washington-Balzac | debit_horaire | 269.545 | 697 | 840 |
| outage_180d | linear:Champs-Elysées proxy Washington-Balzac | taux_occupation | 7.836 | 690 | 840 |
| outage_365d | profile | debit_horaire | 152.833 | 368 | 480 |
| outage_365d | profile | taux_occupation | 6.045 | 368 | 480 |
| outage_365d | linear:Champs-Elysées proxy Washington-Balzac | debit_horaire | 287.504 | 368 | 480 |
| outage_365d | linear:Champs-Elysées proxy Washington-Balzac | taux_occupation | 6.651 | 368 | 480 |
| outage_28d | day_type | debit_horaire | 189.879 | 697 | 840 |
| outage_28d | day_type | taux_occupation | 6.184 | 690 | 840 |
| outage_90d | day_type | debit_horaire | 193.220 | 697 | 840 |
| outage_90d | day_type | taux_occupation | 6.314 | 690 | 840 |
| outage_180d | day_type | debit_horaire | 236.377 | 697 | 840 |
| outage_180d | day_type | taux_occupation | 6.289 | 690 | 840 |
| outage_365d | day_type | debit_horaire | 152.833 | 368 | 480 |
| outage_365d | day_type | taux_occupation | 6.045 | 368 | 480 |

For these long outages all four baseline lag rules fall through to the same profile; their forecasts
agree. Tested proxy policies do not improve the control's pooled RMSE in these comparisons. Retain
historical-profile fallback rather than enable a proxy in the frozen baseline. This is a downstream
forecast comparison, not a claim about the benefit of every possible proxy reconstruction method.
The #51 reconstruction study and #53's independently implemented forecasting experiment have different
scopes; the versioned, cutoff-safe #53 engine is the reproducible evidence used here.

## Frozen inputs, execution and byte checks

A compact **versioned snapshot** under `inputs/` makes reproduction possible from a clean clone,
without API access, large archives or model weights. Traffic spans January 1, 2024 through the official
cutoff **November 5, 2026 at 23:00**, including absent hours; the calendar extends to November 11.
The November 6 retrieval is strictly bounded before November 6 local time and cannot include test truth.
Original raw-source URLs/hashes, capture times, inventory checks and extraction policy are preserved in
`audit_inputs.json`, `audit_report.json` and `sensor_snapshot.json`; all five directed sensor mappings
passed inventory checks. Raw archives contain some surrounding-period rows, but audit filtering occurs
before the snapshot and evaluation; none after the cutoff enter training or selection. The unavailable
January 1 00:00 boundary source is left missing rather than reconstructed or filled.

These are freshly audited frozen inputs, not the older #53 snapshot. Historical development counts
remain identical; minor recent scores can change with later source revisions. Byte hashes and content
hashes identify the difference. Refreshing live data defines a new experiment, never an implicit rerun.

Create Python **3.11.8**, install `requirements-baseline54.txt` and install the project with `pip install
--no-deps -e .`. Then a single offline command reproduces scores, predictions, provenance, independently
verifies every aggregate, and exports the submission twice:

```sh
python -m traffic_prediction.baseline_freeze
```

Outputs are written to `outputs/metrics/baseline_54/`. `--output-dir` permits a fresh directory. The
legacy `scripts/issue54_report.py` delegates to this command and no longer overwrites this documentation.
The command verifies input/code/environment fingerprints, both in-process CSV exports, and every saved
CSV/Parquet byte hash against `inputs/expected.json`. Missing expected evidence or changed inputs/code
fail explicitly. `--record-expected` is reserved for intentionally creating a NEW reviewed freeze and
refuses to overwrite an existing expected manifest. It is not part of ordinary reproduction.

Traffic content SHA-256: `75b41bd5e9c48c3b57009ef969c0ae3d483bd9f90720e374eddf65442b3f570b`.

Submission CSV SHA-256: `4e4413d3d50b713e4e27c090e2279f3c81b3f3492ba71bf9dc6d968ac13f82ee`.

The frozen CSV contains the three official arcs × 120 hours = **360 valid rows** for November 7–11,
with nonnegative flow and occupancy in [0,100]. Saved CSVs here are baseline evidence, not a replacement
for the team's final ensemble submission. Repeating the entire offline command in a separate output
directory must match all expected artifact hashes, not merely the two exports from one process.

Focused tests cover baseline rules, missing data, holidays, future-data isolation, common truth,
independent aggregate verification, submission validation, freeze tampering and masked-history isolation.
Test results and full-run repetition are recorded in `validation.json`.
