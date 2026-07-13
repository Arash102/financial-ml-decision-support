# Financial ML Decision Support

A reproducible financial machine-learning pipeline for stock-level decision-support
signals in the Iranian capital market.

## Current implemented stage

- Notebook 01: Data preparation

Notebook 02 through Notebook 11 are present as placeholders and will be implemented
sequentially after each previous stage is executed, audited, and frozen.

## Current local structure

The project folder is expected to be:

```text
E:/Iran_stock_trade/financial-ml-decision-support
```

The raw stock files are expected inside:

```text
E:/Iran_stock_trade/financial-ml-decision-support/raw_data
```

Because `configs/paths.yaml` uses:

```yaml
data_root_mode: "repository_root"
```

no absolute path needs to be edited. The notebook automatically uses the repository
root as the data root.

## Running in VS Code Jupyter without Conda

### Option A: Use the Python interpreter already selected in VS Code

Open the VS Code terminal in the project folder and run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then open:

```text
notebooks/01_data_preparation.ipynb
```

Select the same Python interpreter as the notebook kernel and choose `Run All`.

### Option B: Create a standard Python virtual environment

```powershell
cd E:\Iran_stock_trade\financial-ml-decision-support
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code:

1. Press `Ctrl+Shift+P`.
2. Select `Python: Select Interpreter`.
3. Choose `.venv`.
4. Open the notebook.
5. Select the `.venv` kernel.
6. Choose `Run All`.

## Core research design

- Training/model-selection data end: 2021-03-20
- Final unseen-test signal dates: 2021-03-21 to 2024-09-22
- Primary models: Random Forest and XGBoost
- Hyperparameter optimization: Optuna with 30 trials
- Primary validation: purged anchored walk-forward
- Robustness validation: CPCV
- Evaluation: machine-learning, signal-level, and portfolio-level

## Notebook 01 scope

Notebook 01:

1. inventories raw CSV files;
2. validates the required schema;
3. parses and sorts market dates;
4. removes duplicate dates deterministically;
5. coerces configured numeric fields;
6. creates transparent data-quality flags;
7. separates candidate model inputs from legacy future-derived audit fields;
8. writes canonical prepared files;
9. generates audit tables and a reproducibility manifest.

## Safeguards

- `insCode` and `dEven` are metadata, not model features.
- The legacy `class` column is not treated as the new target.
- Future-derived legacy fields are isolated from prepared model inputs.
- Missing values are not imputed in Notebook 01.
- Candidate features are preserved but are not assumed leakage-free.
  Notebook 04 will perform the formal feature and leakage audit.
- `raw_data/` and `data_ready/` are excluded from Git by `.gitignore`.
