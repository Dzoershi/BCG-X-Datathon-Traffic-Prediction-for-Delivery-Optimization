# Traffic Prediction for Delivery Optimization

BCG X Datathon project to forecast hourly traffic flow and occupancy on three Paris road segments and support delivery planning.

Historical and recent traffic ingestion are implemented. Processing and modeling modules remain scaffolds.

## Repository structure

```text
data/raw/                  Original downloads, excluded from Git
data/interim/              Cleaned intermediate data
data/processed/            Model-ready datasets
src/traffic_prediction/    Reusable code; ingestion in data/historical.py and data/traffic.py
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

Road filtering, cleaning, and Parquet conversion belong to the processing pipeline.

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

The complete test suite can also run with `python -m pytest` when pytest is installed.

## Collaboration

- One issue, one branch, one pull request reviewed by another teammate. Never push directly to `main`.
- Keep raw downloads unchanged and generated datasets out of Git.
- Work in a named copy of a notebook template, such as `01_data_exploration_cécil.ipynb`.
- Put reusable code in `src/traffic_prediction/` and document decisions in `docs/methodology.md`.
