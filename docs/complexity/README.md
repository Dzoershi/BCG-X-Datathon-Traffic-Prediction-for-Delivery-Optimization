# Model complexity and conditional feature value — #55

```powershell
python -m traffic_prediction.comparison
python -m traffic_prediction.validation --verify-only --output-dir outputs/metrics/complexity_55
```

Requires the frozen #52 audit outputs, the recorded calendar, pinned project dependencies,
and local Chronos/Timestral weights. Add `--allow-download` to fetch pinned checkpoints.
`--runs NAME ...` selects cases; use a separate `--output-dir` for a smaller comparison.
Matching per-run caches are reused only when input/code/settings/version fingerprints
and saved prediction checksums agree. Network access is disabled by default.

## Experimental contract

- Same measured data and independent-target truth policy as [#53](../validation/README.md):
  2024-01-01 through 2026-11-04, eight development windows and the separate October 29
  recent diagnostic. No November 7–11 observations enter selection.
- All three official arcs appear. Each development case expects 2,880 hours per target,
  scoring 1,691 flow and 1,684 occupancy values. Recent cases score 239 of 360 for each.
  Complete grids, truth values and masks must agree across candidates.
- The reference is the existing `HolidayAwarePreviousWeekBaseline`, frozen here by code,
  settings and identical inputs. This does not complete or alter Alice's separate #54
  baseline-freezing PR. The day-type profile is retained as an additional reference.
- Every case is refitted or supplied only cutoff-limited history. Training eligibility,
  model seeds, tree hyperparameters and fallback thresholds are held fixed.
- The 28- and 365-day simulated Champs outages hide target measurements before the cutoff;
  truth remains untouched. At least 56 pre-outage calendar days and observed forecast
  truth are required. Other arcs are auxiliary during this target-only experiment.
- Pooled RMSE, macro RMSE, per-arc/window/day-kind scores, missing truth and fallback counts
  use #53's common engine. Historical publication/revision latency is not reconstructed.
- Scores evaluate the raw model policies. The submission's ceremony-copy post-processing
  is not replayed here; #57 assesses its evidence and assumptions separately.

## Controlled removals and unavailable features

LightGBM retains arc identity and the same ordered rows/labels. One group is removed
from its fitted and predicted feature matrices at a time:

| Group | Columns removed | Interpretation |
|---|---|---|
| Calendar | Explicit cyclical time and calendar flags | Profiles and weekly lags still encode calendar patterns; this is a conditional contribution, not removal of all calendar information. |
| Traffic lags | Flow/occupancy at 168, 336, 504 and 672 hours | Profile and calendar inputs remain. |
| Profiles | Historical target profiles and profile count | Lag and calendar inputs remain. |
| Weather | Weather feature columns | All weather inputs are deliberately missing: a no-information control, not a test of actual forecast-weather value. |

The full feature builder still prepares all groups. These experiments isolate prediction
effects; they do not claim an optimized implementation or a preparation-time saving from
removing a group. Chronos and Timestral-5M also compare explicit calendar covariates versus
history only; the fallback retains its own calendar rules in both cases.

Target-hour weather is excluded because archived issuance/lead-time availability has not
been verified. No hindsight weather gain is claimed. Recent-trend features are not
implemented in the existing builder, so no marginal effect is invented. Proxy information
is not a production feature: the separate copy/offset/linear evaluation in #53 found no
pooled improvement in its tested outages. This experiment measures existing model fallbacks
under outages; it does not relabel proxy estimates as observed truth.

## Measured results

| Case | Development flow / occupancy RMSE | Recent flow / occupancy RMSE | Observed windows, seconds |
|---|---:|---:|---:|
| `baseline_holiday_aware` | 166.67 / 5.81 | 77.38 / 2.31 | 2.2 |
| `baseline_day_type` | 145.13 / 4.24 | 70.32 / 1.85 | 1.8 |
| `lightgbm_all` | 140.13 / 3.92 | 62.52 / 1.75 | 17.7 |
| `lightgbm_without_calendar` | 141.68 / 4.24 | 60.44 / 1.88 | 12.7 |
| `lightgbm_without_lags` | 138.08 / 3.96 | 77.33 / 1.92 | 11.4 |
| `lightgbm_without_profiles` | 150.45 / 4.47 | 62.79 / 1.91 | 15.2 |
| `lightgbm_without_weather` | 140.13 / 3.92 | 62.52 / 1.75 | 16.0 |
| `chronos_calendar` | 160.79 / 3.88 | 58.03 / 2.13 | 82.4 |
| `chronos_history` | 170.09 / 3.93 | 64.47 / 2.21 | 21.6 |
| `timestral_5m_calendar` | 151.72 / 3.88 | 62.72 / 1.92 | 29.7 |
| `timestral_5m_history` | 170.65 / 4.14 | 64.15 / 1.97 | 6.0 |
| `timestral_28m_calendar` | 148.62 / 3.99 | 57.11 / 1.93 | 107.6 |
| `timestral_209m_calendar` | 129.49 / 3.59 | 52.09 / 1.70 | 549.9 |

- **Profiles:** removing them increases development RMSE from 140.13 / 3.92 to 150.45 / 4.47; retain them in this conditional comparison.
- **Weekly lags:** removal improves development flow slightly (140.13 to 138.08), but recent flow worsens (62.52 to 77.33), and occupancy worsens in both phases. Keep the existing lags; do not select on the development improvement alone.
- **Explicit calendar:** removal worsens development flow/occupancy and recent occupancy; recent flow improves slightly. Retain the full group as the fixed reference, while reporting this mixed result. Calendar-enabled Chronos and Timestral-5M improve both targets versus their history-only variants in both phases.
- **Size and cost:** Timestral-209M gives the lowest flow and occupancy RMSE in both phases among these fixed cases (129.49 / 3.59 development; 52.09 / 1.70 recent). Its observed-window time is 549.9 s versus 29.7 s for Timestral-5M and 17.7 s for full LightGBM. The 209M logical weights are about 837 MB. Retain it as an accuracy candidate for team review; LightGBM and 5M remain substantially cheaper options. This is not an automatic replacement of production E5.
- **Other candidates:** the day-type baseline is a useful alternate reference on the three-arc pool. Chronos-calendar improves on its history-only version but has mixed tradeoffs versus LightGBM. Timestral-28M improves flow versus 5M but slightly worsens occupancy and takes longer; it is not the preferred size tradeoff in this experiment. History-only foundation variants remain diagnostic controls.
- **Outages:** every tested model takes the same historical-profile policy on the simulated target outages, yielding 189.88 / 6.18 at 28 days and 152.83 / 6.04 at 365 days. This equality measures fallback behaviour, not a foundation-model improvement. Original observed truth and forecast attribution stay separate.
- **Weather/proxies/trends:** removing all-missing weather columns yields identical predictions, a sanity check with no evidence about actual forecast-weather value. Recent trends and production proxy features were not implemented or assigned invented gains.

The run stores **118,560 hourly rows** and independently verifies **2,294 aggregate rows**. Per-window and day-kind tables show variation and coverage; a single recent diagnostic does not establish universal or statistically significant superiority.

Validation: **755 tests and 27 subtests pass**, including feature-removal contracts,
unchanged training rows, matching inference columns, checkpoint selection and outage isolation.


All findings are conditional on this snapshot and protocol. They do not compare directly
with earlier two-arc ensemble tables, retune E5 weights, or change the production submission.
The fixed current checkpoints are evaluated retrospectively; this does not claim they
were available at the older historical cutoffs. Their pretraining corpora are not audited.

## Runtime, resources and reproduction

`runtime.csv` separates cold checkpoint loading, observed-window wall time and total time
including outage experiments. LightGBM additionally records preparation, fitting and
inference sub-times per observed window. Preparation includes feature construction, and
native foundation-window times include their preparation and inference. Runs use one CPU
process, six Torch threads and one LightGBM thread. Timings are single-pass descriptive
measurements, not isolated repeated benchmarks; import/allocator warm-up and model order
can affect them.

RSS is sampled before/after each case; Windows process-lifetime peaks, where available,
remain in run metadata and are not claimed as isolated per-model peaks. Package installed
file sizes and logical snapshot weight sizes are recorded separately from download sizes
and physical cache duplication. Model revisions and exact settings are pinned by the adapters.

| Artifact | Contents |
|---|---|
| `summary.csv`, `per_arc.csv`, `per_window.csv`, `scores.csv` | RMSE, squared-error sums and coverage by target/scenario/phase. |
| `per_day_kind.csv`, `per_forecast_source.csv` | Day-kind and native/fallback diagnostics; these conditional groups are not automatically equal-sample comparisons. |
| `ablations.csv` | Removal effect: RMSE without group minus full-model RMSE. Positive favours retaining the group. |
| `runtime.csv` | Measured loading/window/total time and RSS samples. |
| `run.json`, individual `*_run.json` | Input/code/settings/environment fingerprints, per-window timings, package/weight sizes and skipped outage windows. |
| `hourly_predictions.parquet`, individual `*_hourly.parquet` | Saved values, truth eligibility, native/fallback source and scenarios, outside Git. |
| `verification.json` | Independent Python-loop checks of every aggregate and common-truth alignment. |

Aggregate tables and provenance are versioned here; larger hourly files stay in
`outputs/metrics/complexity_55/`. Re-running the final command against unchanged frozen
inputs reuses verified individual cases and rebuilds all comparison tables.
