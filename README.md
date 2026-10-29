# Traffic Prediction for Delivery Optimization

A BCG X Datathon project for predicting traffic and assessing its impact on delivery operations.

## Repository structure

```text
.
|-- data/
|   |-- raw/                 # Original data; do not modify
|   |-- interim/             # Cleaned intermediate data
|   `-- processed/           # Model-ready features
|-- notebooks/               # Numbered exploration and experiment notebooks
|-- src/traffic_prediction/  # Reusable Python code
|   |-- data.py              # Loading, cleaning, validation
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
`-- docs/methodology.md      # Decisions and experiment notes
```

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
jupyter lab
```

Select the virtual environment's Python kernel when opening notebooks.

## Workflow

1. Place the original datasets in `data/raw/`. Dataset files are excluded from Git.
2. Work through the numbered notebooks, starting with data exploration.
3. Move reusable functions into `src/traffic_prediction/` and import them in notebooks.
4. Define the prediction target, chronological validation periods, and metric before training.
5. Save model artifacts in `models/`, metrics in `outputs/metrics/`, and prediction files in `outputs/submissions/`.
6. Record findings and delivery impact in `docs/methodology.md`.

The notebooks and Python modules are starting templates; no model pipeline is implemented yet. Update `configs/baseline.yaml` once the dataset schema and challenge requirements are known. Training, evaluation, and submission commands can be documented here when implemented.

## Data and validation conventions

- Keep raw datasets unchanged; write derived data to `interim/` or `processed/`.
- For forecasting, split chronologically and fit preprocessing on training data only.
- Build historical features using only information available at prediction time.
- Compare experiments against a simple baseline on the same validation split.
- Keep large datasets, saved models, and generated outputs out of Git.

## Checks

Once tests have been added, run `python -m pytest` from the repository root. The current test files contain guidance only.

## Project brief

Place the project brief at `docs/project_brief.pdf` if available. It has not been included in this scaffold.
