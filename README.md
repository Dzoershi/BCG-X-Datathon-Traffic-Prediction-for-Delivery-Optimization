# Traffic Prediction for Delivery Optimization

BCG X Datathon: forecast hourly traffic flow and occupancy on three Paris road segments.

**Status:** data download, processing, updates, SQL queries, dataset/submission validation, and chronological backtesting are implemented. Forecasting models and final prediction export are still to do.

Run commands from the repository root. Terminal examples use PowerShell.

## 1. Set up your environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . --no-deps
python -m pip install numpy==2.2.6 pandas==2.3.3 pyarrow==25.0.1 `
  PyYAML==6.0.3 duckdb==1.4.5 holidays==0.107 pytest==9.2.0 ipykernel==7.3.0
python -m pip install stream-unzip==0.0.100 stream-inflate==0.0.43 pycryptodome==4.2.0
```

Use Python 3.10–3.13; development is tested on 3.12. Select `.venv` in VS Code and your notebook kernel. These commands install the current data workflow; all project pins are in [requirements.txt](requirements.txt).

## 2. Download data

### Historical data: yearly ZIP files

```powershell
python -m traffic_prediction.data.historical --list
python -m traffic_prediction.data.historical --download --years 2024-2025
```

**Output:** `data/raw/historical/<year>/`. Change `--years` to one year or a range. Valid ZIPs are reused; interrupted downloads restart safely. ZIP integrity checks can take time.

### Recent data: one road and time range

```powershell
python -m traffic_prediction.data.traffic --download --road-id 5671 `
  --start "2026-10-29T00:00:00Z" --end "2026-10-30T00:00:00Z"
```

**Output:** `data/raw/recent/*.jsonl`. Start is inclusive, end exclusive; timestamps require a timezone. Replace `--download` with `--sample --limit 1` to preview.

Sources: [historical archives](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/) · [recent observations](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/).

### Weather: history and forecast

```powershell
python -m traffic_prediction.data.build --weather

# Before the final forecast (November 6): keep history, add missing past days, replace the forecast.
python -m traffic_prediction.data.build --weather --refresh-forecast
```

**Output:** `data/processed/weather/weather.parquet`, one row per hour from the first local traffic day to November 11, 2026 23:00. It joins traffic directly on `datetime`. Columns are `precipitation_mm`, `temperature_c`, `wind_speed_kmh`, and `source`: Open-Meteo Historical Forecast until yesterday, the live forecast from today. Raw responses are kept in `data/raw/weather/`, named with their UTC download time. [Weather sources, time convention, and limits](docs/methodology.md#météo-issue-21).

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0); credit it in the presentation.

### Road works and events on the evaluated arcs

```powershell
# Rerun on November 6: the sources only list current and planned works.
python -m traffic_prediction.data.build --disruptions
```

**Output:** `data/processed/disruptions/disruptions.parquet`, one row per arc and hour from January 1, 2024 to November 11, 2026 23:00, with `chantier_actif`, `evenement_actif` and a `description`; and `chantiers.parquet`, the works within 200 m of each arc. It combines Open Data Paris road works with the hand-checked list [data/raw/events_manual.csv](data/raw/events_manual.csv), one source per row. The command ends with a check of November 7–11. Use the table to check the test period and explain past errors, not as a training feature. [Sources, rules, and the November 6 checklist](docs/methodology.md#chantiers-et-événements-issue-34).

## 3. Process data into Parquet

```powershell
python -m traffic_prediction.data.build --start 2024-01-01 --end 2025-12-31
```

**Output:** `data/processed/traffic/year=YYYY/part-0.parquet`. Reads downloaded ZIPs, keeps the arcs in [configs/baseline.yaml](configs/baseline.yaml), standardizes measurements, and completes the hourly grid. Dates are inclusive Paris days.

**A build replaces each annual file it covers.** Use the update command below to preserve existing history. Use `--output-dir PATH` for experiments.

To include a recent download, add `--recent-file "data/raw/recent/YOUR_SNAPSHOT.jsonl"`; repeat the option for multiple files. See [processing rules](docs/methodology.md#contrat-de-données) for timestamps, missing values, and overlap handling.

**Modelling dataset:** `python -m traffic_prediction.data.build --dataset --end 2026-10-30` joins traffic, calendar, weather, and disruption flags into `data/processed/dataset/dataset.parquet` (one row per traffic row) and `covariates.parquet` (covariates of every hour until November 11, for target hours), and prints the fingerprint to compare with the team. [Columns, rules, and frozen version](docs/methodology.md#table-de-modélisation-issue-22).

## 4. Refresh data without rebuilding history

```powershell
python -m traffic_prediction.data.update

# First update on an empty dataset, or refresh a chosen period:
python -m traffic_prediction.data.update --start 2026-10-23 --end 2026-10-29
```

By default, refreshes seven calendar days ending on the latest stored source day, then fetches new observations through the invocation time. Use `--refresh-days N` to change the overlap.

Downloads each configured arc separately, applies corrections, and rewrites only changed annual partitions. Missing API rows preserve history; explicit null corrections replace old values. Repeating an update adds no duplicates.

Use the same arc configuration as the initial build. [Update details and recovery](docs/methodology.md#actualisation-des-observations-issue-8).

## 5. Explore data with SQL

Run in a notebook or Python session after processing:

```python
from traffic_prediction.data.query import query_traffic

df = query_traffic(
    """SELECT datetime, debit_horaire, taux_occupation
       FROM traffic WHERE iu_ac = ? AND year = ?
       ORDER BY datetime""",
    [5671, 2025],
)
```

Returns a Pandas DataFrame. Pass `path=...` for another Parquet file or folder. Only the query result is loaded into Pandas; large results still need memory.

For date filters, aggregations, and joins, see [SQL examples](docs/methodology.md#exemples-sql).

## 6. Run tests

```powershell
python -m pytest -q
```

Tests use small fixtures and mocked network responses.

## 7. Validate data and the final CSV

```powershell
python -m traffic_prediction.data.validate --write-manifest
python -m traffic_prediction.data.validate --check-manifest

# Once your forecasting code has produced the submission:
python -m traffic_prediction.data.validate --submission "outputs/submissions/output_team.csv"
```

Checks the hourly grid, schema, duplicates, partition labels, and data constraints. The manifest at `data/processed/traffic/_validation.json` records coverage, missing measurements, configuration, versions, and file checksums. Rewrite it after an intentional dataset update.

Use `--data-dir`, `--config`, or paired `--start` / `--end` dates to choose what to validate. Missing measurements are reported separately from missing rows. The CSV check enforces the brief's 360-row format; it does not generate forecasts. PR tests are configured for Windows/Linux and Python 3.10/3.13. [Validation details](docs/methodology.md#vérification-et-tests).

## 8. Backtest a forecasting model

```python
from traffic_prediction.data.query import query_traffic
from traffic_prediction.evaluate import backtest, PreviousWeekModel
data = query_traffic("SELECT * FROM traffic")
result = backtest(PreviousWeekModel(), data, author="your-name", description="Weekly reference")
print(result.summary)
```

Replace `PreviousWeekModel()` with any model implementing `predict(history, target_times)`.
The eight configured Thursday cutoffs produce Saturday–Wednesday forecasts after a Friday gap.
Results contain per-window/arc scores (`scores`), means by arc (`per_arc`), and overall means
(`summary`), including atypical days, coverage counts, and ratios to the previous-week baseline.
Each successful call appends a run to `docs/experiments.csv`; use `experiments_path=None` to
skip logging. Data is read only from existing local Parquet files; no download is triggered.
[Model contract, metric definitions, and limitations](docs/methodology.md#backtest-issue-15).

## Project deliverables

The supplied LivraisonCo brief remains the source of truth.

- **Forecast:** flow and occupancy for `Champs-Elysées`, `Convention`, and `Saint-Pères`, November 7–11, 2026. Validate with RMSE.
- **Submit by November 6, 2026, 18:00 Paris time:** reproducible code, a 360-row prediction CSV, and seven presentation slides plus appendices.
- **CSV:** `output_<group_name>.csv`, with `arc`, `datetime`, `debit_horaire`, `taux_occupation`. This is separate from the internal Parquet dataset.
- **Presentation:** approach, results, and a delivery-optimization roadmap. Refresh observations before the final forecasts; proxies remain auxiliary features.

[Full requirements and business assumptions](docs/methodology.md#périmètre-du-brief).

## Where things live

```text
src/traffic_prediction/data/   Download, processing, update, and query modules
configs/baseline.yaml          Evaluated arcs and experimental proxies
data/raw/                     Original downloads — outside Git
data/processed/               Parquet datasets — outside Git
notebooks/                    Exploration and feature engineering
tests/                        Automated checks
docs/methodology.md            Data contract, decisions, and detailed examples
```

One issue, one branch, one PR. Keep raw data unchanged and use a personal copy of notebook templates.
