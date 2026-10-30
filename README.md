# Traffic Prediction for Delivery Optimization

BCG X Datathon project to forecast hourly traffic flow and occupancy on three Paris road segments and support delivery planning.

Historical archive ingestion, recent traffic ingestion, traffic processing from both sources, and DuckDB queries over processed Parquet are implemented. The remaining data and modeling modules are scaffolds.

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

Recent downloads and ordinary historical ZIP downloads use Python's standard library. The shared `traffic.py` module loads processing dependencies only when processing functions are called.

For Deflate64 archives, including 2023, install the optional streaming decoder and its pinned dependencies. These are also required by the historical ingestion tests, which include genuine Deflate64 fixtures:

```powershell
python -m pip install stream-unzip==0.0.100 stream-inflate==0.0.43 pycryptodome==4.2.0
```

For traffic processing, install the data dependencies pinned in `requirements.txt`:

```powershell
python -m pip install numpy==2.2.6 pandas==2.3.3 pyarrow==25.0.1 PyYAML==6.0.3
```

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
- The published 2023 archive uses Deflate64. With the optional decoder installed, validation and processing stream its contents without extraction and check member sizes and CRCs. Missing decoder errors include the installation command.

Road filtering, cleaning, and Parquet conversion are done by the traffic processing step below.

## Traffic data processing

Turns downloaded archives and explicitly selected recent snapshots into one hourly table for the arcs listed in `configs/baseline.yaml` (the three evaluated arcs and Champs-Elysées proxies):

```powershell
python -m traffic_prediction.data.build --start 2024-01-01 --end 2025-12-31
```

- Output: `data/processed/traffic/year=YYYY/part-0.parquet`, compressed Parquet with one file per year. Read it with `pd.read_parquet("data/processed/traffic")`.
- Columns: `arc, role, iu_ac, datetime, debit_horaire, taux_occupation, etat_trafic, etat_barre, qualite_mesure`. Columns, time convention, quality labels, and duplicate rules are described in the data contract in `docs/methodology.md`.
- `--start` and `--end` are inclusive days (00:00 to 23:00); they default to 2024-01-01 and 2025-12-31. Archives are read from `data/raw/historical/<year>/`; use `--raw-dir`, `--output-dir`, or `--config` to change paths.
- When the period starts on January 1, available readable archives from the preceding year are also included to recover midnight. Missing or unsupported optional boundary archives produce a warning.
- Archives are streamed without extraction; JSONL snapshots are read in batches. Only configured arcs inside the requested period are accumulated for the hourly table. The selected rows and final hourly grid still occupy RAM; processing is not fully out of core. Identical archives are read once.
- Each run replaces the entire output file for each year in its period, even for a one-day build. Use a separate `--output-dir` for experiments. Updating existing partitions without replacement belongs to the incremental-update issue.
- The final summary lists rows read, dropped, or added, and missing values per arc, variable, and month.
- Deflate64 archives, including 2023, require the optional decoder above. Their members are read in physical archive order; ordinary ZIP members retain filename order.

Include a completed recent download by passing its path (replace the example path with your file):

```powershell
python -m traffic_prediction.data.build --start 2026-10-29 --end 2026-10-29 --recent-file "data/raw/recent/YOUR_SNAPSHOT.jsonl" --output-dir "data/processed/recent_check"
```

- Repeat `--recent-file PATH` to include multiple snapshots. Files are read in argument order, after historical archives. Recent files are not discovered automatically.
- For overlapping arc/hour pairs, the last row wins in full, including nulls. This does not fill missing measurements from an older row or from a proxy. Identical and conflicting duplicates are counted in the summary.
- Recent-only builds work when no historical archive is available for the requested period. API timestamps are converted to Paris local time before filtering by the inclusive build dates.

## Recent traffic ingestion

Source: [Paris Open Data recent traffic observations](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/).

`traffic.py` contains both the recent downloader and processing helpers. Its command line previews or downloads raw observations; `python -m traffic_prediction.data.build` processes historical archives and snapshots selected with `--recent-file` into Parquet.

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

## SQL queries from notebooks

Install the pinned query dependency in the same Python environment as your notebook kernel:

```powershell
python -m pip install duckdb==1.4.5
```

After building the processed traffic dataset, use this in the existing exploration or feature-engineering notebooks. The default path is anchored to the repository, so it also works when the notebook's working directory is `notebooks/`:

```python
from datetime import datetime
from traffic_prediction.data.query import query_traffic

observations = query_traffic(
    """SELECT arc, datetime, debit_horaire, taux_occupation, qualite_mesure
       FROM traffic
       WHERE datetime >= ? AND datetime < ? AND iu_ac IN (?, ?)
         AND role = ?
       ORDER BY arc, datetime""",
    [datetime(2025, 10, 1), datetime(2025, 11, 1), 5671, 191, "evalue"],
)
```

`observations` is a Pandas DataFrame. SQL boundaries above include the start and exclude the end; timestamps are the stored timezone-free Paris hour labels. Add SQL predicates on `qualite_mesure`, `debit_horaire`, `taux_occupation`, or other available attributes as needed. Missing measurements stay null, and proxies retain their own identifiers and `role='proxy'`.

For aggregations, several queries, or joins to a notebook DataFrame, reuse one connection:

```python
import pandas as pd
from traffic_prediction.data.query import connect_traffic

with connect_traffic() as connection:
    monthly = connection.execute(
        """SELECT arc, date_trunc('month', datetime) AS month,
                  avg(debit_horaire) AS mean_flow,
                  count(debit_horaire) AS measured_hours
           FROM traffic WHERE role = 'evalue' AND year = ?
           GROUP BY arc, month ORDER BY arc, month""",
        [2025],
    ).fetchdf()

    roads = pd.DataFrame({"iu_ac": [5671, 191], "delivery_zone": ["south", "central"]})
    connection.register("roads", roads)
    by_zone = connection.execute(
        """SELECT delivery_zone, avg(debit_horaire) AS mean_flow
           FROM traffic JOIN roads USING (iu_ac)
           WHERE year = 2025 GROUP BY delivery_zone"""
    ).fetchdf()
```

- Both functions accept `path=...` for a single `.parquet` file or a directory scanned recursively. The default is `data/processed/traffic/`.
- `traffic` is a view over the files in an in-memory DuckDB connection. No database server, database file, or full Pandas import of the dataset is needed. Only the SQL result is converted to a DataFrame; a large result can still use substantial RAM.
- Hive partition folders such as `year=2025/month=01` become SQL columns. Types are inferred; zero-padded month values may be strings. Use `CAST(month AS INTEGER)` when numeric month comparisons are needed. Filters on partition columns let DuckDB skip unrelated files.
- Files must follow the processed traffic schema. A missing dataset or missing required columns produces an explicit error.
- `query_traffic` closes its connection automatically. With `connect_traffic`, use `with` as above or call `.close()`. Reconnect after adding or removing files to refresh the registered file list.
- See [DuckDB's Parquet documentation](https://duckdb.org/docs/1.4/data/parquet/overview) for projection and filter pushdown.

## Tests

Run the ingestion tests without a live connection or large downloads:

```powershell
python -m unittest discover -s tests -p test_historical.py -v -b
python -m unittest discover -s tests -p test_traffic.py -v -b
```

The complete test suite, including mixed-source builds, DuckDB queries over partitioned Parquet, CLI checks, and a mocked recent-download-to-Parquet integration test, runs with `python -m pytest`. It requires the processing and SQL dependencies above and `pytest==9.2.0`.

## Collaboration

- One issue, one branch, one pull request reviewed by another teammate. Never push directly to `main`.
- Keep raw downloads unchanged and generated datasets out of Git.
- Work in a named copy of a notebook template, such as `01_data_exploration_cécil.ipynb`.
- Put reusable code in `src/traffic_prediction/` and document decisions in `docs/methodology.md`.
