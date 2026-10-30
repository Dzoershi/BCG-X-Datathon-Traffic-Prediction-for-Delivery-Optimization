# Traffic Prediction for Delivery Optimization

BCG X Datathon project to forecast hourly traffic flow and occupancy on three Paris road segments and support delivery planning.

Historical archive ingestion and traffic processing are implemented. The remaining data and modeling modules are scaffolds.

## Repository structure

```text
data/raw/                  Original downloads, excluded from Git
data/interim/              Cleaned intermediate data
data/processed/            Model-ready datasets
src/traffic_prediction/    Reusable code; see data/historical.py, data/traffic.py, data/build.py
tests/                     Automated tests
notebooks/                 Numbered notebook templates
configs/                   Configuration
docs/                      Methodology and experiment log
models/                    Saved models, excluded from Git
outputs/                   Figures, metrics, and submissions, excluded from Git
```

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment if needed (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . --no-deps
```

Both ingestion modules and their tests use Python's standard library. Other project dependencies are listed in `requirements.txt`.

## Historical traffic ingestion

Source: [Paris Open Data historical traffic archives](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/).

Run commands from the repository root.

List available archives:

```powershell
python src/traffic_prediction/data/historical.py --list
```

Download one year or an inclusive range:

```powershell
python src/traffic_prediction/data/historical.py --download --years 2024
python src/traffic_prediction/data/historical.py --download --years 2023-2025
```

- Downloads require `--years`. An unavailable year causes the command to fail before downloading.
- Archives are stored unchanged at `data/raw/historical/<year>/<attachment_id>.zip`. Use `--output-dir PATH` to override the destination.
- Downloads stream in chunks and use temporary `.part` files until ZIP integrity checks pass. Validation reads decompressed contents without extracting them and can take time.
- Valid existing archives are checked and reused. Remote revisions are not automatically detected.
- Temporary failures are retried up to three attempts. Interrupted downloads restart from the beginning.
- All matching attachments are preserved separately, including the two listed for 2025.

Road filtering, cleaning, and Parquet conversion are done by the traffic processing step below.

## Traffic data processing

Turns the downloaded archives into one hourly table for the arcs listed in `configs/baseline.yaml` (the three evaluated arcs and Champs-Elysées proxies):

```powershell
python -m traffic_prediction.data.build --start 2024-01-01 --end 2025-12-31
```

- Output: `data/processed/traffic/year=YYYY/part-0.parquet`, compressed Parquet with one file per year. Read it with `pd.read_parquet("data/processed/traffic")`.
- Columns: `arc, role, iu_ac, datetime, debit_horaire, taux_occupation, etat_trafic, etat_barre, qualite_mesure`. Columns, time convention, quality labels, and duplicate rules are described in the data contract in `docs/methodology.md`.
- `--start` and `--end` are inclusive days (00:00 to 23:00); they default to 2024-01-01 and 2025-12-31. Archives are read from `data/raw/historical/<year>/`; use `--raw-dir`, `--output-dir`, or `--config` to change paths.
- Archives are streamed without extraction (about 2 minutes and under 200 MB of memory for 2024–2025). Identical archives are read once. Years covered by the period are replaced; other years already on disk are kept.
- The final summary lists rows read, dropped, or added, and missing values per arc, variable, and month.
- The 2023 archive uses Deflate64 compression, which Python's `zipfile` cannot read; it is not processed yet.

## Recent traffic ingestion

Source: [Paris Open Data recent traffic observations](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/).

Preview observations for a timestamp range:

```powershell
python src/traffic_prediction/data/traffic.py --sample --limit 1 --start "2026-10-29T22:00:00Z" --end "2026-10-29T23:00:00Z"
```

Download all matching observations:

```powershell
python src/traffic_prediction/data/traffic.py --download --start "2026-10-29T22:00:00Z" --end "2026-10-29T23:00:00Z"
```

- Downloads require both boundaries. Start is inclusive; end is exclusive, applied to the observation timestamp `t_1h`.
- Timestamps require `Z` for UTC or an explicit timezone offset.
- Each run saves a separate raw `.jsonl` snapshot under `data/raw/recent/`. Use `--output-dir PATH` to change the destination.
- Exports stream one observation at a time and preserve raw values, including nulls.
- Row counts are checked against the API before and after extraction. Empty periods produce empty snapshots.
- Temporary failures are retried up to three attempts, restarting extraction. Temporary `.part` files receive their final name after validation succeeds.
- `--limit` controls previews only; downloads retrieve all matching observations available from the source.

## Tests

Run the ingestion tests without a live connection or large downloads:

```powershell
python -m unittest discover -s tests -p test_historical.py -v -b
python -m unittest discover -s tests -p test_traffic.py -v -b
```

The complete test suite, including traffic processing tests on small synthetic archives, runs with `python -m pytest`.

## Collaboration

- One issue, one branch, one pull request reviewed by another teammate. Never push directly to `main`.
- Keep raw downloads unchanged and generated datasets out of Git.
- Work in a named copy of a notebook template, such as `01_data_exploration_cécil.ipynb`.
- Put reusable code in `src/traffic_prediction/` and document decisions in `docs/methodology.md`.
