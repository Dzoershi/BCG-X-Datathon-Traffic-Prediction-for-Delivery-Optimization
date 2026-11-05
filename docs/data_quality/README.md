# Traffic quality audit — issue #52

```powershell
python -m traffic_prediction.data.audit --inputs docs/data_quality/inputs.json --output-dir outputs/metrics/quality_52
```

Run from the repository root with the recorded raw snapshots present. The command is offline: it verifies source checksums, streams ZIPs on the first run, then reuses a checked, configuration-specific cache under `data/interim/quality_audit_cache/`. It produces the CSVs below and two local Parquet sidecars. It does not rewrite production traffic, train a model, or change the final submission.

## Scope and provenance

**Audited labels: January 1, 2024 00:00 through November 4, 2026 23:00, Paris local time without a timezone.** These are end-of-hour labels. November 2026 therefore contains four audited days, not a full month. No November 7–11, 2026 observations are used. Latest-valid timestamps below refer to this bounded audit period, not every label the live API may publish afterward.

The inputs are the 2024 archive, both 2025 attachments (identical bytes; one skipped), the previously extracted January 1, 2024 boundary from the CRC-checked 2023 archive members, and five newly downloaded 2026 API snapshots. The latter use inclusive January 1 and exclusive November 5 Paris bounds. [inputs.json](inputs.json) records their order, paths, hashes, request URLs, retrieval times and boundary provenance. The independent node-pair inventory was retrieved on **November 5, 2026 at 08:28 UTC**.

The scanner read **56,282,519 rows** across unique sources and retained the five configured directed arcs. After ten duplicate arc/hour rows were resolved, the full grid has **124,680 arc/hour rows**, 24,936 per arc. Later supplied source rows win, including explicit null corrections. Missing source rows receive their own absent label. The original files remain intact; no estimates are written over them.

| File | Contents |
|---|---|
| [monthly.csv](monthly.csv) | 350 rows: arc × target × month, with expected/source hours, absent rows, explicit nulls, invalid numbers, suspect zeros, usable values, pair exclusions and latest valid hour. |
| [summary.csv](summary.csv) | The ten arc/target totals over the audited period. |
| [eligibility.csv](eligibility.csv) | Training and truth eligibility under paired versus independent-target rules for the eight complete configured backtest windows, including proxies separately. |
| [sensors.csv](sensors.csv) | Configured direction, expected ID, IDs found by independent node-pair queries, reverse-direction controls and verification time. |
| [verification.json](verification.json) | Independent count checks, original source-row spot checks and replay against the existing production partitions. |
| `outputs/metrics/quality_52/observations.parquet` | Canonical raw observations with `q`, `k`, numeric values, per-target status, existing pair-quality label and source coordinates. |
| `outputs/metrics/quality_52/zero_context.parquet` | All 209 flagged zero-flow hours, their original values/states, consecutive run lengths, and exact ±1-hour neighbours. |

The Parquet sidecars and raw snapshots stay outside Git. Source coordinates are `source_file`, `source_member` (ZIP), and `source_row` (one-based JSONL line or CSV record position including the header). CSV literals are preserved, including non-finite text rather than silently interpreting it as an empty field. JSON scalar values use a common string representation; the fingerprinted original file is authoritative for its original bytes. Source row count, selected row count, skipped duplicate files, parser/dependency hashes and environment are also recorded in the local `report.json`.

## Availability and exclusions

“Usable” means finite, in the target's numerical domain and supplied by a source row. Flow additionally excludes the separately flagged ambiguous zero-flow cases under the conservative policy below. It does not certify that every sensor value is physically accurate.

| Arc / ID | Usable flow | Usable occupancy | Complete pairs | Extra flow / occupancy excluded by the pair requirement | Latest flow / occupancy |
|---|---:|---:|---:|---:|---|
| Champs-Elysées / 4264 | 15,088 | 15,023 | 15,022 | 66 / 1 | 2025-11-19 11:00 / 2025-11-19 12:00 |
| Convention / 5671 | 15,279 | 15,532 | 15,181 | 98 / 351 | Both 2026-11-04 23:00 |
| Saint-Pères / 191 | 13,665 | 13,665 | 13,665 | 0 / 0 | Both 2026-11-04 23:00 |
| La Boétie–Berri proxy / 4272 | 507 | 22,782 | 507 | 0 / 22,275 | 2024-01-29 12:00 / 2026-11-04 23:00 |
| Washington–Balzac proxy / 4265 | 13,426 | 16,408 | 12,108 | 1,318 / 4,300 | 2026-11-04 23:00 / 2026-07-29 14:00 |

Each arc has **640 absent source hours**, distinct from explicit null measurements in supplied rows: 511 in 2024 and 129 in 2026. Some gaps span multiple days; one is the nonexistent spring-DST label. Their common occurrence across the five arcs is evidence about availability of these extracts, not proof that every sensor failed simultaneously. None is converted to zero.

Across all five arcs, independent-target eligibility retains **57,965 flow values and 83,410 occupancy values**, compared with **56,483 complete pairs**: +1,482 flow and +26,927 occupancy observations. These are availability gains, not measured forecast improvements. The La Boétie proxy is particularly unsuitable as a general flow substitute even though its occupancy is well represented. The Washington proxy's recent occupancy is stale despite recent flow. Both remain auxiliary candidates; the official targets are unchanged.

On the three official arcs in the eight existing backtest windows, independent-target truth eligibility changes flow **1,684 → 1,691** and leaves occupancy **1,684 → 1,684**. The seven additional flow truths are all on Champs-Elysées in the **2024-12-26 cutoff** window (113 → 120). Training eligibility effects by arc and cutoff are in `eligibility.csv`. No model has been retrained or rescored under the proposed independent-target rule in #52; #53 must apply a common evaluation mask before comparing model scores.

## Rules retained and interpretation corrected

| Rule | Decision and rationale | Measured effect |
|---|---|---|
| Blank/null value versus absent source row | Keep them distinct. `missing` means a supplied row lacks the target; an absent grid row has no source coordinates. | 640 absent hours per arc; target-specific null counts are reported separately each month. |
| Non-numeric, non-finite, negative flow; occupancy outside [0,100] | Preserve the supplied literal and exclude that target from usability. Never replace it with zero. | No such supplied numeric-domain violations were found in the canonical five-arc observations; focused fixtures cover these cases. |
| Only one target available | Count and expose its valid value independently; a bad/missing partner does not invalidate that value. Preserve the existing production pair-quality label for compatibility. | +1,482 flow and +26,927 occupancy observations versus paired eligibility. Production model/backtest adoption is a separate #53 change. |
| `q=0, k>0` | Retain the existing conservative flow exclusion, identify it as `suspect_zero`, keep its raw zero and valid occupancy. The pattern alone does **not** prove a sensor fault. | 209 hours, all on proxies. Including these zeros would add 209 proxy flow values; no official-arc flow truth changes. |
| Traffic/open-state codes | Keep as context; do not use them as an additional measurement-quality filter. | Archive closure-state code 3 does not, by itself, establish that the numeric measurements are invalid. |
| Duplicates and timestamps | Keep documented last-source precedence and existing Paris end-of-hour convention, with 24 local labels/day. | Ten duplicate hours removed; the established normalized values and labels replay exactly on all 124,080 existing production rows. |

The suspect-zero breakdown is 98 hours on La Boétie–Berri (occupancy 0.27889–10.26722%, longest run 7 hours) and 111 on Washington–Balzac (0.09556–48.79389%, longest run 38 hours). Most carry a fluid traffic state; none carries the blocked state. These observations do not establish a cause or calibrate a correction. Therefore neither automatic reinstatement as known-good flow nor a claim of confirmed sensor faults is justified. The source values and contextual alternatives remain available for sensitivity analysis.

The existing paired production policy is now an explicit limitation: it should not be used to describe independent target availability. Keeping the policy during this audit preserves frozen comparisons while #53 establishes consistent evaluation. This audit does not reopen the closed proxy-model decision or silently alter the submitted ensemble.

## Node pairs, IDs and extraction risk

The three target directions match the brief's specified node pairs. The geographic reference and rolling recent measurements resolve them to **4264, 5671 and 191**; proxy IDs are **4272 and 4265**. Reverse-direction queries return different IDs where records exist (4262, 5672, 4274, 4267); no reverse record was returned for Saint-Pères. That absence is not evidence that the reverse physical street is absent. Raw selection excludes reverse directions, and no unexpected ID occurred in the audited forward records.

The ID-filtered recent downloader could miss a future ID replacement before `filter_arcs` ever sees it. The audit's inventory queries therefore filter by **directed node labels without an ID condition** and compare all returned IDs with configuration. Today's snapshot supports the current mapping; it cannot guarantee future mappings. Capture a new inventory under a **new dated filename** before the final refresh. Existing raw inventory files cannot be overwritten by this command. Reference `date_fin` values are not used as measurement end dates: the traffic observations and reference metadata must be distinguished.

```powershell
python -m traffic_prediction.data.audit --capture-sensors data/raw/quality_52/sensors_NEW_TIMESTAMP.json
```

## Reproduction and handoff

1. Obtain the fingerprinted raw ZIPs using the existing historical downloader (`--years 2024-2025`). The two 2025 attachments are identical for this snapshot, and the audit verifies and skips the duplicate.
2. Use the recorded JSONL/inventory snapshots and extracted boundary from `outputs/metrics/quality_52_bundle.zip` (share separately; ignored by Git). Extract into the repository root to restore the recorded relative paths. The bundle includes audit results and the small raw snapshots; obtain the large historical ZIPs separately in step 1. Live API responses may change or expire; downloading them again is a new snapshot, not guaranteed exact reproduction.
3. Run the command at the top. `--chunk-rows` adjusts streaming chunk size; `--cache-dir` isolates a cache. Source/configuration checksum mismatches fail explicitly. The tracked configuration's hash uses UTF-8 text with LF line endings so Windows and Linux Git checkouts agree; raw-source hashes always verify original bytes. A cache with changed bytes is rebuilt from its verified source.
4. Compare the four aggregate CSVs with the versioned files here. The same frozen inputs reproduce identical CSV bytes. Keep the generated `report.json`, `inputs.json` and `verification.json` with results.

Independent verification recomputed counts for all **124,680 rows / 350 monthly rows**, checked four original source coordinates directly (including ZIP members), verified production file hashes and found **zero differences** in existing flow, occupancy and quality labels over 124,080 rows. Production traffic and the #22 modelling dataset remain on their previous snapshots; this new bounded audit must not be mistaken for a production data refresh or mixed silently with the newer #32 score snapshot.

The local test suite passes **705 tests and 27 subtests**, including 25 audit tests and the audit import check. Tests use small synthetic ZIP/JSONL data and mocked inventory responses, including missing/invalid values, partial targets, genuine and suspect zeros, directions, ID changes, source order, cache integrity, raw-file preservation, incomplete API responses, complete-window eligibility and deterministic report regeneration.

Source references: [Paris traffic measurements and definitions](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/), [historical archives](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/), [geographic reference](https://opendata.paris.fr/explore/dataset/referentiel-comptages-routiers/). The supplied **7 June 2019 traffic notice**, pages 1–3, defines occupancy-only stations, end-of-hour timestamps and state codes; it does not establish an automatic fault rule for `q=0, k>0`. The supplied LivraisonCo brief fixes the three official directed target arcs and the November 7–11 forecast requirement.
