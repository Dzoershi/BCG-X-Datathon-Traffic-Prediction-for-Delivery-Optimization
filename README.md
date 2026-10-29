# Traffic Prediction for Delivery Optimization

A BCG X Datathon project for predicting traffic and assessing its impact on delivery operations.

## Repository structure

```text
.
|-- data/
|   |-- raw/                 # Downloaded by scripts; never add or edit by hand
|   |-- interim/             # Cleaned intermediate data
|   `-- processed/           # Model-ready features
|-- notebooks/               # Numbered notebook templates; work in your own copy
|-- src/traffic_prediction/  # Reusable Python code
|   |-- data/                # One module per data source
|   |   |-- traffic.py         # Open Data Paris traffic counts
|   |   |-- calendar_events.py # Public and school holidays
|   |   |-- weather.py         # Weather
|   |   |-- disruptions.py     # Roadworks and events
|   |   `-- build.py           # Assembly of all sources
|   |-- models/              # One module per model family
|   |-- features.py          # Time, location, and historical features
|   |-- train.py             # Model training
|   |-- predict.py           # Inference
|   `-- evaluate.py          # Metrics and error analysis
|-- configs/baseline.yaml    # Paths and experiment settings
|-- models/                  # Saved models (ignored by Git)
|-- outputs/
|   |-- figures/
|   |-- metrics/
|   `-- submissions/
|-- tests/                   # Tests for reusable code
`-- docs/
    |-- methodology.md       # Data contract, decisions, findings, model choice
    `-- experiments.csv      # One row per experiment, shared by the team
```

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
nbstripout --install
jupyter lab
```

Select the virtual environment's Python kernel when opening notebooks.

Run `nbstripout --install` once on each computer, and again if you recreate `.venv`. It removes notebook outputs and execution counts when you commit, so diffs stay readable and notebooks do not conflict. Check it with `nbstripout --status`.

`chronos-forecasting` installs PyTorch. On Windows and macOS, the default build runs on CPU only and weighs about 125 MB. On Linux, pip also downloads several GB of CUDA libraries, so install the CPU build first, using the version pinned in `requirements.txt`:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

## Workflow

1. Download the datasets with the code in `src/traffic_prediction/data/`. Never place or edit files in `data/raw/` by hand: we will rerun the whole pipeline from scratch on 6 November. Dataset files are excluded from Git.
2. Work in your own copy of a numbered notebook, suffixed with your first name (for example, `01_data_exploration_hassan.ipynb`). Leave the numbered templates unchanged.
3. Move code that works into `src/traffic_prediction/` and import it in notebooks.
4. Define the prediction target, chronological validation periods, and metric before training.
5. Save model artifacts in `models/`, metrics in `outputs/metrics/`, and prediction files in `outputs/submissions/`.
6. Record findings and delivery impact in `docs/methodology.md`, and add one row per experiment to `docs/experiments.csv`.

The notebooks and Python modules are starting templates, and there is no model pipeline yet. Update `configs/baseline.yaml` once we know the dataset schema and the challenge requirements. Add the training, evaluation, and submission commands here once they exist.

## Collaboration

- One issue = one branch = one reviewed pull request.
- Never push directly to `main`. Every change goes through a pull request that another team member reviews.

## Data and validation conventions

- Keep raw datasets unchanged; write derived data to `interim/` or `processed/`.
- For forecasting, split chronologically and fit preprocessing on training data only.
- Build historical features using only information available at prediction time.
- Compare experiments against a simple baseline on the same validation split.
- Keep large datasets, saved models, and generated outputs out of Git.

## Checks

Run `python -m pytest` from the repository root once there are tests. For now, the test files only contain guidance.

## Project brief

Place the project brief at `docs/project_brief.pdf` if you have it. This scaffold does not include it.
