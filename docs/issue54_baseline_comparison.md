# Issue #54 — Baseline comparison (pre-freeze evidence)

Generated from identical chronological prediction keys and cutoff-limited traffic. Missing/invalid truth excluded per target, matching the validity criteria of `evaluate.backtest`. Truth coverage is valid observed hours / all candidate hours; forecast coverage is finite predictions / all hours.

The pooled aggregate of `run_local_backtests.py` covers Convention and Saint-Pères; this file additionally reports all three arcs. The RMSE below is pooled by arc/target over valid hours, which is not the same as `evaluate.backtest`'s unweighted mean over cutoff-specific RMSE values.

## RMSE and coverage per arc/target


| Model | Arc | Target | RMSE | Valid truth | Truth coverage | Forecast coverage |
|---|---|---|---:|---:|---:|---:|
| baseline_holiday_aware | Champs-Elysées | debit_horaire | 216.806 | 690/960 | 71.9% | 100.0% |
| baseline_holiday_aware | Champs-Elysées | taux_occupation | 8.738 | 690/960 | 71.9% | 100.0% |
| baseline_holiday_aware | Convention | debit_horaire | 92.959 | 659/960 | 68.6% | 100.0% |
| baseline_holiday_aware | Convention | taux_occupation | 1.926 | 659/960 | 68.6% | 100.0% |
| baseline_holiday_aware | Saint-Pères | debit_horaire | 105.747 | 335/960 | 34.9% | 100.0% |
| baseline_holiday_aware | Saint-Pères | taux_occupation | 2.254 | 335/960 | 34.9% | 100.0% |
| baseline_day_type | Champs-Elysées | debit_horaire | 169.781 | 690/960 | 71.9% | 100.0% |
| baseline_day_type | Champs-Elysées | taux_occupation | 6.055 | 690/960 | 71.9% | 100.0% |
| baseline_day_type | Convention | debit_horaire | 108.432 | 659/960 | 68.6% | 100.0% |
| baseline_day_type | Convention | taux_occupation | 2.319 | 659/960 | 68.6% | 100.0% |
| baseline_day_type | Saint-Pères | debit_horaire | 95.741 | 335/960 | 34.9% | 100.0% |
| baseline_day_type | Saint-Pères | taux_occupation | 2.071 | 335/960 | 34.9% | 100.0% |
| baseline_previous_week | Champs-Elysées | debit_horaire | 238.031 | 690/960 | 71.9% | 100.0% |
| baseline_previous_week | Champs-Elysées | taux_occupation | 8.434 | 690/960 | 71.9% | 100.0% |
| baseline_previous_week | Convention | debit_horaire | 113.908 | 659/960 | 68.6% | 100.0% |
| baseline_previous_week | Convention | taux_occupation | 2.263 | 659/960 | 68.6% | 100.0% |
| baseline_previous_week | Saint-Pères | debit_horaire | 110.648 | 335/960 | 34.9% | 100.0% |
| baseline_previous_week | Saint-Pères | taux_occupation | 2.289 | 335/960 | 34.9% | 100.0% |
| baseline_four_week | Champs-Elysées | debit_horaire | 215.759 | 690/960 | 71.9% | 100.0% |
| baseline_four_week | Champs-Elysées | taux_occupation | 6.453 | 690/960 | 71.9% | 100.0% |
| baseline_four_week | Convention | debit_horaire | 113.363 | 659/960 | 68.6% | 100.0% |
| baseline_four_week | Convention | taux_occupation | 2.295 | 659/960 | 68.6% | 100.0% |
| baseline_four_week | Saint-Pères | debit_horaire | 128.787 | 335/960 | 34.9% | 100.0% |
| baseline_four_week | Saint-Pères | taux_occupation | 2.645 | 335/960 | 34.9% | 100.0% |

## Forecast source distribution by model and arc

| Model | Arc | Source / fallback | Hours |
|---|---|---|---:|
| baseline_holiday_aware | Champs-Elysées | previous_week | 549 |
| baseline_holiday_aware | Champs-Elysées | two_weeks_before | 145 |
| baseline_holiday_aware | Champs-Elysées | day_type_profile | 122 |
| baseline_holiday_aware | Champs-Elysées | last_sunday | 98 |
| baseline_holiday_aware | Champs-Elysées | sunday_before | 46 |
| baseline_holiday_aware | Convention | previous_week | 598 |
| baseline_holiday_aware | Convention | day_type_profile | 240 |
| baseline_holiday_aware | Convention | last_sunday | 96 |
| baseline_holiday_aware | Convention | two_weeks_before | 26 |
| baseline_holiday_aware | Saint-Pères | day_type_profile | 600 |
| baseline_holiday_aware | Saint-Pères | previous_week | 310 |
| baseline_holiday_aware | Saint-Pères | last_sunday | 47 |
| baseline_holiday_aware | Saint-Pères | two_weeks_before | 2 |
| baseline_holiday_aware | Saint-Pères | sunday_before | 1 |
| baseline_day_type | Champs-Elysées | day_type_profile | 960 |
| baseline_day_type | Convention | day_type_profile | 960 |
| baseline_day_type | Saint-Pères | day_type_profile | 960 |
| baseline_previous_week | Champs-Elysées | previous_week | 650 |
| baseline_previous_week | Champs-Elysées | two_weeks_before | 181 |
| baseline_previous_week | Champs-Elysées | day_type_profile | 129 |
| baseline_previous_week | Convention | previous_week | 670 |
| baseline_previous_week | Convention | day_type_profile | 240 |
| baseline_previous_week | Convention | two_weeks_before | 50 |
| baseline_previous_week | Saint-Pères | day_type_profile | 600 |
| baseline_previous_week | Saint-Pères | previous_week | 358 |
| baseline_previous_week | Saint-Pères | two_weeks_before | 2 |
| baseline_four_week | Champs-Elysées | four_week_mean | 840 |
| baseline_four_week | Champs-Elysées | day_type_profile | 120 |
| baseline_four_week | Convention | four_week_mean | 720 |
| baseline_four_week | Convention | day_type_profile | 240 |
| baseline_four_week | Saint-Pères | day_type_profile | 600 |
| baseline_four_week | Saint-Pères | four_week_mean | 360 |

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/run_local_backtests.py baseline_holiday_aware baseline_day_type baseline_previous_week baseline_four_week --end 2026-11-04
.\.venv\Scripts\python.exe scripts/issue54_report.py
.\.venv\Scripts\python.exe -m pytest tests/test_baselines.py -q
```

Processed traffic content SHA-256: `9d8d2e9e29cd3fe400c8e40c7dce76483d6a3179f3958efe971fcc50b2334049`

Further checks required before closure: the simulated Champs-Elysées outages and proxy evidence; the final frozen 2026-11-05 23:00 cutoff; repeat-run identical submission CSV SHA-256; and documented baseline selection/rationale. Do not use November 7–11 observations to select the model.



## Submission reproducibility

Selected candidate: `HolidayAwarePreviousWeekBaseline`

Command:

```powershell
.\.venv\Scripts\python.exe -m traffic_prediction.predict --team-name baseline54 --model holiday-aware --skip-update
```

Validation command:

```powershell
.\.venv\Scripts\python.exe -m traffic_prediction.data.validate --submission outputs/submissions/output_baseline54.csv
```

Results:
- 360 valid submission rows.
- Three official arcs.
- Seed: 42.
- Two consecutive runs produced identical CSV bytes.
- SHA-256: `A2DDF62F61B6B6BFB037CA102390D725B01B4B81A3BFBCAE54F4941A29A82284`
- Baseline tests: 30 passed.

The byte-level reproducibility check was performed on the locally available inputs. Final verification against the frozen official cutoff (2026-11-05 23:00) remains required.
