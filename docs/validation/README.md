# Comparable model validation — issue #53

```powershell
# Requires the frozen #52 snapshots and the recorded local calendar.
python -m traffic_prediction.validation

# Recompute all aggregates from saved hourly predictions, without training.
python -m traffic_prediction.validation --verify-only
```

Outputs go to `outputs/metrics/validation_53/`. Use `--audit-dir`, `--calendar` and `--output-dir` for other local paths. `--models baseline_day_type baseline_holiday_aware` omits LightGBM; `--outage-days` with no following values skips the outage experiment. Changing inputs/settings produces a different experiment.

## Scoring rules and what is still unconfirmed

The supplied LivraisonCo brief, PDF pages 6 and 12, specifies **RMSE for flow and occupancy on the three official arcs over November 7–11**. It does not specify the weighting between arcs/targets, any cross-target normalization, rounding before scoring, or the treatment of unavailable truth. Those organiser rules remain unconfirmed. We do not invent a combined score mixing vehicles/hour and occupancy percentage points.

For each target, this report provides:

- **Pooled RMSE:** `sqrt(sum((prediction - observed_truth)^2) / n_scored)` across the specified rows. This weights observed hours equally.
- **Macro RMSE:** the unweighted mean of nonempty arc/window RMSEs. Unequal coverage can reverse model rankings; it is not a pooled metric.
- **Coverage:** `n_expected = n_scored + n_ignored`, retaining all 120 hourly labels per arc/window even when truth is missing. No usable truth means undefined RMSE, never zero.
- **Sources:** `n_fallback`, `n_native`, `n_source_unknown` concern scored hours. `n_forecast_fallback`, `n_forecast_native`, `n_forecast_unknown` concern **all** expected forecasts, including those without truth. `n_estimated_truth` counts excluded proxy-estimate rows; such estimates never become observed truth.

Working days, weekends and metropolitan French public holidays are exclusive groups, with public holidays taking priority over weekends. Per-source breakdowns are conditional diagnostics, not rankings on equal samples. An ensemble counts as a total fallback only if all positive-weight members for that target fall back; a mixture with a native member is not a total fallback.

The reusable `backtest()` now scores available targets independently. Its historical aggregate `rmse` column remains a macro mean for compatibility, also labelled `macro_rmse`; use **`pooled_rmse`** for pooled comparisons. Per-window/arc `scores.rmse` is an ordinary RMSE on that cell. `truth_policy="paired"` reproduces the previous mask. The previous-week reference and normalized ratios retain their explicitly limited coverage; they are not a combined organiser score.

## Frozen data and chronological protocol

The input is the #52 audit sidecar over **2024-01-01 through 2026-11-04 23:00 Paris**, not an implicit refresh of the production dataset. Raw values, sensor mapping and source fingerprints are documented in [the quality audit](../data_quality/README.md). This validation's measured-input content fingerprint is `848d155025972573d77aa22de7195a84a39f7a30624b7a3ecba62cff822f0094`.

Development uses the eight existing Thursday cutoffs: **2024-11-07, 2024-12-19, 2024-12-26, 2025-04-17, 2025-07-10, 2025-10-16, 2025-11-06, 2026-10-22**. Each model receives only history through Thursday 23:00, skips Friday, and forecasts Saturday–Wednesday, 120 local hourly labels. Inputs may have arrived later in the archive; historical publication/revision latency is not reconstructed. Existing Paris end-of-hour and DST normalization conventions are retained.

The limited recent check fixes the same model settings at cutoff **2026-10-29** and predicts **October 31–November 4**. It is a separate, single-settings diagnostic, not claimed to be untouched by every teammate's previous exploration. We do not tune settings on this check. **November 7–11 remains the final test**: this protocol rejects its use before calling models, and refuses windows extending beyond the frozen data availability.

Every model must provide the same full grid and is scored on identical eligible measured truth. Missing/nonfinite or out-of-domain predictions on eligible truth fail the comparison; models cannot improve their score by dropping difficult hours. Proxy estimates are explicitly excluded as truth. Partial measurements and valid partners of masked values remain usable independently. The conservative `q=0, k>0` flow exclusion remains, while valid occupancy can be scored. Training eligibility inside existing models is unchanged and still generally requires complete pairs.

## Executed candidates and results

The run compares existing `DayTypeProfileBaseline`, `HolidayAwarePreviousWeekBaseline`, and a **calendar/traffic-only LightGBM diagnostic**, with the #31 raw-flow defaults (200 trees, depth 6, learning rate 0.05, seed 42, one CPU thread). Weather inputs are explicitly all missing in history and covariates; the existing lagged-weather feature columns therefore carry no observations. This avoids assuming historical weather forecasts were issued by the cutoff. It is **not** a rerun or replacement of production LightGBM or ensemble E5. Settings and code/data/environment fingerprints are saved in [run.json](run.json).

| Candidate | Development pooled flow / occupancy RMSE | Recent pooled flow / occupancy RMSE |
|---|---:|---:|
| Day-type profile | 145.13 / 4.24 | 70.32 / 1.85 |
| Holiday-aware previous week | 166.67 / 5.81 | 77.38 / 2.31 |
| LightGBM, weather unavailable | 140.13 / 3.92 | 62.52 / 1.75 |

Each development comparison expects **2,880 hours per target** and scores **1,691 flow / 1,684 occupancy**, ignoring 1,189 / 1,196. Each recent comparison expects **360** and scores **239 for each target**. Champs-Élysées has no recent measured truth (120 ignored hours); Convention has one unusable hour and Saint-Pères has 120 usable hours. Recent predictions for Champs are retained with their source attribution but cannot produce a measured RMSE. Its historical-profile fallback must not be mistaken for a newly validated sensor forecast.

These numbers are for all three official arcs under the new mask and snapshot. Earlier reports, including the two-arc #32 pool, have different scopes and cannot be compared directly. This issue establishes comparable evaluation; it does not freeze a baseline, retune the ensemble, or change the final submission.

## Simulated Champs-Élysées outages

The implemented experiment follows the candidate families discussed in #51: a target-profile control, direct proxy copy, a mean target-minus-proxy offset, and a linear target-on-proxy fit, separately for proxies **4272 and 4265** and for each target. It does not claim to reproduce the unversioned code/results behind that document-only study.

For each eligible development cutoff, hide every Champs measurement during the last **28, 90, 180 or 365 days** from both the profile and calibration. Original observations remain separate held-out truth. Calibration uses only measured pairs **strictly before the outage**; offset/linear require at least 20 pairs, and linear requires nonzero proxy variance. At forecast time, use the corresponding proxy **seven days before the forecast hour**, always at or before Thursday's cutoff; when unavailable or ineligible, use the pre-outage target profile for the forecast hour. Offset and linear forecasts are clipped to the physical output domain. No contemporaneous future proxy or hidden target measurement enters the forecasts.

At least 56 pre-outage calendar days and some measured forecast-window truth are required. Shorter outages admit seven configured windows (840 expected hours); 365 days admits four (480 expected). The recent 2026 window has no measured Champs truth and is skipped, not replaced with proxy truth. Other skips and exact dates are in `run.json`; calibration counts, last fitting hour and each proxy input hour are saved per variable in hourly `forecast_details`.

| Outage | Scored flow / occupancy | Control pooled flow / occupancy RMSE | Best proxy actually contributing flow / occupancy RMSE |
|---:|---:|---:|---:|
| 28 days | 697 / 690 | 189.88 / 6.18 | 268.80 / 7.92 |
| 90 days | 697 / 690 | 193.22 / 6.31 | 275.43 / 7.85 |
| 180 days | 697 / 690 | 236.38 / 6.29 | 269.58 / 7.84 |
| 365 days | 368 / 368 | 152.83 / 6.04 | 287.52 / 6.65 |

The best active proxy in these pooled cells is Washington–Balzac linear. La Boétie's flow candidates exactly match the control because they always fall back, not because proxy flow works. Even Washington's 365-day case uses fallback on 310 of 368 scored flow hours and 250 of 368 occupancy hours. Complete per-method/window/source counts remain in the CSVs. These limited windows support retaining the existing fallback for these methods; they do not establish that no proxy method can ever help. The real 2026 Champs outage has no observed truth to score.

## Saved evidence and reproduction

| File | Contents |
|---|---|
| [summary.csv](summary.csv) | Pooled and macro RMSE, separate development/recent/scenario/target coverage and forecast-source counts. |
| [per_arc.csv](per_arc.csv), [per_window.csv](per_window.csv) | Arc-level and window-level pooled errors and counts, including unscored groups. |
| [scores.csv](scores.csv) | Each window × arc × target cell, retaining zero-coverage groups. |
| [per_day_kind.csv](per_day_kind.csv) | Working-day, weekend and holiday errors and counts. |
| [per_forecast_source.csv](per_forecast_source.csv) | Native rules, proxies, fallback rules and unknown attribution; coverage of all forecast rows. |
| [run.json](run.json) | Fixed settings, cutoffs, skipped experiments, input/code/package fingerprints and full audit provenance. |
| [verification.json](verification.json) | Independent pooled/macro error sums, coverage/source counts and common-truth alignment checks. |
| `outputs/metrics/validation_53/hourly_predictions.parquet` | Every prediction, measured truth, eligibility, source, fallback flag, day kind and outage calibration details. |

The frozen run saves **61,440 hourly records** and independently checks **1,402 aggregate rows**. The verifier uses Python loops and `math.fsum`, independently of Pandas aggregation, and detects changed pooled/macro RMSE, counts or squared-error sums. Tests cover ranking reversal under unequal coverage, partial/null/invalid truth, duplicate/misaligned/missing predictions, unknown attribution, full grids, final-test rejection, native LightGBM integration, and poisoning hidden target/future proxy measurements. They use small fixtures and require no model downloads or live API.

For exact local handoff, share `outputs/metrics/validation_53_bundle.zip` separately from Git. It contains the hourly outputs, calendar and small frozen audit inputs; preserve their relative paths when extracting into the repository root. The large 2024/2025 ZIPs remain separate (source URLs and hashes are in #52). Rebuild the audit if needed, then run the commands above. Model code is versioned with this issue; saved scores from another snapshot are not silently imported into this comparison. Production traffic, model settings and final submission files are not overwritten.
