# Champs-Elysées proxy study

The production forecast retains its existing historical-profile fallback. Across
all eligible weekly cutoffs in the project's 2024–2025 historical period, none
of the tested proxy methods had a lower mean downstream forecast RMSE for either
target at any of the four outage lengths. Individual windows do improve, mostly
in 2024. No proxy estimate is enabled in the submission pipeline.

## Arcs and observations

The historical processor matches node pairs and verifies the sensor IDs. The
recent API records also identify these directed arcs:

| Arc | ID | Upstream → downstream |
|---|---:|---|
| Champs-Elysées (target) | 4264 | Washington → Berri |
| La Boétie-Berri | 4272 | La Boétie → Berri |
| Washington-Balzac | 4265 | Washington → Balzac |

A `partielle` row is usable for its valid variable. For
2024–2025, overlap was 506/0 for 4272 flow (2024/2025), 7,752/7,143 for 4272
occupancy, 7,801/2,498 for 4265 flow, and 7,723/4,009 for 4265 occupancy.
The bounded 2026 snapshots run through November 2 and contain no valid target
measurements. Of 7,344 grid hours, 4272 has zero valid flow and 6,762 valid
occupancy; 4265 has 2,616 valid flow and 3,623 valid occupancy. The last valid
4265 occupancy was July 29. This is availability through November 2, not a
claim about later hours or the final cutoff.

## Method and leakage controls

`traffic_prediction.models.proxy.reconstruct` returns a separate estimate
and source per variable; it never mutates or relabels observations. It compares
copy, a global mean target-minus-proxy correction, and an ordinary least-squares
linear relationship. Correction and regression require at least 20 paired
observations; regression also requires nonzero proxy variance. Fits use only
valid measured pairs strictly before the first masked hour. Missing proxy or
insufficient calibration falls back to a pre-window target day-type/hour,
hourly, then overall mean; missing target history can leave an estimate absent.

For each Thursday, the evaluation masks all target values during the simulated
outage and fits the existing `DayTypeProfileBaseline` using only pre-outage
history. Reconstruction is scored against the original untouched observations.
For the forecast, the candidate uses a reconstructed value from the same hour
one week earlier, which is checked to be at or before Thursday 23:00. Where
that proxy estimate is unavailable, the forecast uses the baseline for the
forecast hour directly. Neither future proxy observations nor target truth in
the outage or forecast horizon enter the predictions. The outcome grid retains
`Champs-Elysées` as the official target. Existing production and submission
behavior remains unchanged.

We used every Thursday whose forecast ends no later than the target's last
observed calendar day, with at least 56 calendar days of data before the outage.
The last day's later hours may be missing and are not scored. The outage lengths
and cutoff ranges are 28 days (86 cutoffs, 2024-03-28–2025-11-13), 90 days
(77, 2024-05-30–2025-11-13), 180 days (64, 2024-08-29–2025-11-13), and 365
days (38, 2025-02-27–2025-11-13). A full forecast window has 120 hours. One
window has no flow truth in each of the first three sets; scored counts for
all windows and both variables are in the results. The 365-day experiment is
closest to the length of the real outage but uses a shorter training history.
Overlapping windows are not independent.

The local experiment outputs include per-window reconstruction and forecast
RMSE, truth coverage, fit sample sizes and source counts. The compressed hourly
outputs preserve individual values and sources. These outputs are not included
in this document-only commit. A tie caused by missing proxy measurements is not
evidence of a successful proxy.

Mean forecast RMSE below is the **unweighted mean of scored window RMSEs**, not
a pooled score. The best 4265 method is shown for flow; 4272 always fell back
for forecast flow in these windows. The best proxy method is shown for occupancy.

| Outage | Scored flow windows | Flow fallback | Best proxy flow | Occupancy fallback | Best proxy occupancy |
|---:|---:|---:|---:|---:|---:|
| 28 days | 85 | 189.37 | 241.27 (4265 linear) | 5.70 | 7.10 (4265 linear) |
| 90 days | 76 | 200.87 | 244.64 (4265 linear) | 5.79 | 7.06 (4265 linear) |
| 180 days | 63 | 181.85 | 252.57 (4265 linear) | 5.53 | 7.14 (4265 linear) |
| 365 days | 38 | 175.69 | 229.20 (4265 copy) | 5.36 | 6.42 (4265 linear) |

For 28-day reconstruction, baseline mean RMSE was 191.52 flow and 6.13
occupancy, compared with 206.07 and 7.21 for the best 4265 linear candidate.
For 365 days, baseline means were 240.65 and 6.76 versus 249.65 and 7.73.
The 28-day 4265 linear forecast improves 15 of 85 scored flow windows and 12
of 85 scored occupancy windows, but its mean is worse; the 365-day equivalent
improves zero of 38 flow and one of 38 occupancy windows. In the 28-day 2025
windows, fallback mean forecast RMSE is 135.72 flow and 4.88 occupancy, while
4265 linear gives 208.58 and 6.48. These comparisons support retaining the
fallback, not a claim that proxy forecasting can never improve a different model.

The half-year difference analysis separates hour and holiday-adjusted day type. For groups with at least five pairs in each
half-year, the median absolute change in group mean from 2024 H2 to 2025 H1 was
249.14 flow for 4265 and 2.01 occupancy for 4265; for 4272 occupancy it was
1.55. For 4265 flow, the 90th percentile change was 369.05. Such changes and
sparse later groups do not support assuming a stable global correction. These
full-period summaries are descriptive only and are never used to fit or select a
window's model.

This downstream experiment is a previous-week forecast based on reconstructed
history, not a rerun of the full production ensemble or its closure correction.
It is a conservative decision to retain the current production fallback given
these candidates and the evidence, not proof about all possible proxy methods.
There is no valid 2026 target truth with which to score the real outage.

## Reproduction

The commands below describe the local experiment. The experimental reconstruction
module, evaluation script, and result tables are not part of this document-only
commit; they must be obtained separately to rerun the evaluation.

The raw 2024 archive SHA-256 is
`f74a620bd8ccb96e63a1c1d12368aa979a6048cc0473bf490528882a75c6ae7e`.
Both 2025 attachments have SHA-256
`9549fea12ac05d7f46d6d3c951a094604e47019e9b95a6aacb6a56eb01cdcd09`;
the processor deduplicates them. The first 2024 local hour may be missing because
we did not download the 2023 archive. The snapshots were downloaded November 3,
2026, using the recent API for IDs 4264, 4265 and 4272 with inclusive start
`2026-01-01T00:00:00+01:00` and exclusive end `2026-11-03T00:00:00+01:00`.
Their hashes are `65c1bf1b0afb3ce05a1c1c87f4f3562fb7739cc86c1845338171afcbc04b21da`
(4264), `8784a913c8bf2761c1b9eeb39cecd0d685eef358ed10986dc0f42cc74b459b71`
(4265), and `59cd4358b73117680b4291511a8d4c6f6a2934a9368855af8c45c500e472bd4b`
(4272). Recent data can change or expire, so retain these snapshots for exact
reproduction. Calendar parquet SHA-256 is
`ccf6d22f27c005c4993c0c1fef9103f11f9b4d9811a5d1d731788342325b0225`.

```sh
PYTHONPATH=src .env/bin/python -m traffic_prediction.data.historical --download --years 2024-2025
# Download each bounded arc snapshot as described above using data.traffic --download --road-id.
PYTHONPATH=src .env/bin/python -m traffic_prediction.data.build --start 2024-01-01 --end 2024-12-31 --output-dir data/processed/traffic_51
PYTHONPATH=src .env/bin/python -m traffic_prediction.data.build --start 2025-01-01 --end 2026-11-02 --output-dir data/processed/traffic_51 \
  --recent-file data/raw/recent/issue51/traffic_20251231T230000000000Z_20261102T230000000000Z_hd55dxrr.jsonl \
  --recent-file data/raw/recent/issue51/traffic_20251231T230000000000Z_20261102T230000000000Z_l75mqbx7.jsonl \
  --recent-file data/raw/recent/issue51/traffic_20251231T230000000000Z_20261102T230000000000Z_msvn1gjp.jsonl
for days in 28 90 180 365; do
  PYTHONPATH=src .env/bin/python scripts/evaluate_champs_proxies.py \
    --traffic data/processed/traffic_51 --calendar data/processed/calendar/calendar.parquet \
    --all-cutoffs --outage-days "$days" --output "outputs/metrics/proxy_51_$days"
done
PYTHONPATH=src .env/bin/python -m pytest -q
```

The processed partition hashes in year order are
`4e2cc3d470292bc852bc03aad330c66b86b4835316bfca4446f070fd8d4abcb0`,
`7783f0ed67d76e054ad6dceea100dc75b63923e64db3ba646183ad2a7043a97f`, and
`2beb126ce2ed978ac02a225e9e64e260358261a14535dc8cad89d62b43b5e1a9`.
