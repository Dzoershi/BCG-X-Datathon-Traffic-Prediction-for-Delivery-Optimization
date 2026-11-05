# Submission-period disruption review — #57

```powershell
python -m traffic_prediction.data.disruption_review --refresh --assess-history
python -m traffic_prediction.data.disruption_review --snapshot data/raw/disruption_57/20261105T223608663128Z --assess-history
python docs/disruption_review/verify.py
```

Run from the repository root with pinned dependencies. The historical assessment requires
the frozen #52 observation audit and #55 hourly predictions. Omit `--assess-history` to
refresh/replay sources without those experiment artifacts. Raw JSON and document bytes
stay outside Git; the small review tables here are the recorded results. Each refresh
creates a new directory. Replay checks source checksums, counts and the official nodes;
it does not access the network. An incomplete feed capture is rejected. Unavailable
document endpoints are recorded explicitly, never treated as successful captures.

## Decision for the current submission

**No numeric disruption adjustment is currently approved.** The November 11, 2026
Champs-Elysées restriction remains provisional. The existing monument-hours source
establishes neither road hours nor the evaluated direction. This review did not verify
a 2026 Police Prefecture traffic order; that does not establish that none exists.

Issue #57 therefore changes the automatic #47 behavior: adding a `ceremonie` row to
`events_manual.csv` alone no longer copies last year's measurements. The submission
preserves the model forecast, prints the unresolved decision, and records it in its
run JSON. This leaves a known risk of overpredicting during an actual ceremony;
an explicit, reviewed correction can be enabled before delivery. No zeros, capacity
reductions or detour effects are invented.

The recorded source capture is **November 5, 2026, 23:36 Paris time**. Source retrieval
times are UTC in [run.json](run.json) and [evidence.csv](evidence.csv). These are evidence
retrieval times, not new traffic observations or claims of historical availability.
The gazette, 2025 order and monument page were archived successfully. The City notice
returned 503 and the decree/operator endpoints 403 to the direct downloader; those
primary pages were inspected through the browser, with archival failures retained.

## Sources and directed targets

[Verified arc identities](verified_arcs.csv) match the live reference by unique ID and
ordered upstream/downstream labels. The reverse direction is a separate target:

| Arc | ID | Upstream → downstream |
|---|---:|---|
| Champs-Elysées | 4264 | Av_Champs_Elysees-Washington → Av_Champs_Elysees-Berri |
| Convention | 5671 | Lecourbe-Convention → Convention-Blomet |
| Saint-Pères | 191 | Sts_Peres-Voltaire → Sts_Peres-Universite |

The capture includes current/planned works near each line, plus complete
`circulation_evenement` and `fermetures-voirie` feeds. Works are filtered to 200 m;
the two additional feeds to 500 m. Exact geometry/dates, not street-name similarity,
select candidates. Whole-day works dates are not confirmed clock hours. Geometric
proximity and an operator's `ONE_DIRECTION` flag do not identify the official direction.
Absence from these feeds does not prove absence of a restriction.

| Recorded evidence | Scope and operational interpretation |
|---|---|
| [84 works source rows](works.csv): Champs 47, Convention 20, Saint-Pères 17 | Declared works overlapping possible submitted hours. Most are building/parking works; direction, actual daily activity and traffic effect remain unknown. |
| Champs `EC509350`, 37.1 m, July 1–December 18 | Roadway work inside the existing 50 m screening threshold. No verified lane/direction/hours or quantitative effect; no numeric correction. |
| [3 additional restriction candidates](other_restrictions.csv), Saint-Pères, 188.5–307.8 m | Two Louvre-work records and one Bac record. Nearby network context, not proof that arc 191 is closed. |
| [Official monument information](https://www.paris-arc-de-triomphe.fr/visiter/informations-pratiques) | Annual November 11 morning closure of the monument, not a road traffic schedule. |
| [City flag-display notice](https://bovp.apps.paris.fr/index.php?id=55395&lvl=notice_display), published October 15 | Confirms commemorative context only. Browser notice inspected; direct archival requests returned HTTP 503, recorded in evidence.csv. |
| [2025 order 2025-01460](https://cdn.paris.fr/paris/2025/11/07/arrete_2025-01460_03112025-F8oV.pdf), article 3 and annex 1 | Historical 07:00–14:00 motor-vehicle prohibition within a perimeter containing Washington–Berri, both directions. Priority-vehicle exemptions and operational changes are possible. It does not authorize a 2026 assumption. |
| [November 4, 2025 official gazette](https://www.prefectures-regions.gouv.fr/ile-de-france/irecontenu/telechargement/132389/971747/file/recueil-75-2025-678-recueil-des-actes-administratifs-special%20du%2004.11.2025.pdf), p44 onward | Establishes that the historical order was published before the November 6 backtest cutoff. The City-hosted PDF path is dated November 7. |
| [Paris Peace Forum decree](https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000055023178) | November 10–11 major-event context; no target-road closure/hours established. |
| [Operator's November 7–13 bulletin](https://www.bonjour-ratp.fr/actualites/articles/bulletin-travaux-7-novembre/) | Regional public-transport works; no quantified effect on the evaluated road arcs. |

The complete sourced hour table has **9,878 candidate rows**, not 9,878 disrupted hours:
several works can overlap the same hour. [candidate_hour_counts.csv](candidate_hour_counts.csv)
is its compact version by target/hour/kind/status. The detailed table, with per-record
URLs, direction and check times, is `outputs/metrics/disruption_57/candidate_hours.csv`.

## Hours and the historical copy

Processed timestamps remain Paris local **end-hour labels**, pending organizer
confirmation for the final CSV. A clock restriction `[07:00, 14:00)` intersects
measurement hours labelled **08:00–14:00**, seven rows. Label 07:00 describes the hour
before the restriction. The old #47 implementation included it as a transition
heuristic; that extra hour is now excluded. Partial-hour restrictions select every
positively overlapping hour, without pretending the whole hour has zero traffic.

[Historical observations](historical_observations.csv) preserve both November 11 days,
source member/row and independent target status. [Historical scores](historical_scores.csv)
reassess the #55 forecasts at cutoff November 6, 2025. Only seven measured hours from
**November 11, 2024**, available before that cutoff, replace predictions. Truth and
eligibility masks stay unchanged; missing/invalid reference pairs stop the copy.

| Model, Champs November 11, 2025 | Flow RMSE before → copy | Occupancy RMSE before → copy |
|---|---:|---:|
| Holiday-aware baseline | 436.14 → 161.79 | 11.38 → 7.14 |
| LightGBM | 299.03 → 95.92 | 10.12 → 8.17 |
| Timestral-209M | 256.84 → 103.08 | 9.42 → 7.64 |

Each daily score uses 24 observed values per target; each complete Champs forecast
window scores 116/120 per target. On the seven replaced hours, the copied profile's
RMSE is 107.17 veh/h and 1.66 occupancy points for every model. This is **one annual
transfer**, already investigated in #47, not an independent holdout or a guaranteed
2026 improvement. Nonzero historical measurements, exemptions and boundaries are
why a legal restriction is not converted into imposed zero measurements.

No November 7–11, 2026 observations enter the assessment. Raw model predictions and
truth remain unchanged on disk. All 48 score cells and 2,160 historical prediction
records were independently checked; see [verification.json](verification.json).

## Delivery checklist — handoff to #32, #33 and #35

1. Refresh again before the **November 6, 18:00 Paris** submission deadline. Inspect
   source retrieval failures and the actual Police Prefecture order/annex, not only
   a search snippet, ceremonial announcement or monument access page.
2. Verify the exact year, legal interval, interior/boundary status, exceptions and
   **Washington → Berri** direction. Record unresolved information explicitly.
   Check EC509350 and the Saint-Pères neighbours for a real lane/direction effect.
3. If the road restriction is confirmed, update `events_manual.csv` and the exact
   matching entry in `configs/disruptions_review.json`. Record `iu_ac`, ordered nodes,
   `source_url`, UTC `checked_at`, reviewer and rationale. Set `status=confirmed`,
   `evidence_kind=traffic_order`, `direction_verified=true`, `hours_verified=true`.
   Only then consider `action=copy_prior_year`; confirmation alone does not require it.
   Keep the reviewed source bytes. Different hours need a new matching review.
4. Document why the previous-year profile is transferable and inspect every required
   reference pair. A missing/invalid hour fails the export rather than filling a zero.
   Leave `action=none` if the evidence remains insufficient. Do not tune on final truth.
5. Inspect the October 31–November 1 reference weekend and November 5 traffic before
   using them as normal-week analogues. The manual protest/strike descriptions do
   not establish effects on these arcs; no numerical adjustment is inferred here.
6. Run the chosen submission command (E5 by default, `--model holiday-aware` for plan B),
   then validate the 360-row CSV. Inspect the adjacent `_run.json`: `closure_review`,
   `closure_adjustments` with original/copied values and reference dates, source markers,
   cutoff, fingerprints and CSV hash. No approved correction means an empty adjustment list.
7. Repeat offline with identical inputs and compare CSV hashes. Retain the source capture,
   review configuration, CSV and run log. Explain the unresolved Champs sensor/closure
   uncertainty in #35; #33 must not claim measured delivery savings from these scenarios.

Tests cover provisional/default decisions, wrong directions/IDs, incomplete approvals,
interval mismatch, hour boundaries, missing reference observations, immutable replay,
checksum failures, historical-only scoring, 360-row export and adjustment provenance.
The raw snapshot, full hourly detail and reports are bundled locally in
`outputs/metrics/disruption_57_bundle.zip`; model weights and final submission are excluded.
