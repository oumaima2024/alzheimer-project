# MLOps additions

This project includes lightweight, local-first MLOps without changing the scientific split or metrics.

## 1. MLflow experiment tracking

Install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the experiments as usual:

```powershell
python scripts\train_baseline.py --config config.yaml --no-shap
python scripts\train_longitudinal.py --config config.yaml --model-type GRU
python scripts\train_longitudinal.py --config config.yaml --model-type LSTM
python scripts\compare_models.py --results-dir results
```

Each training script records parameters, test metrics, and selected output artifacts under the local `mlruns/` directory. The tracking metadata is stored in the local `mlflow.db` SQLite database. Raw ADNI data is not copied into MLflow.

Start the dashboard in a second PowerShell window:

```powershell
mlflow ui --backend-store-uri .\mlflow.db --port 5000
```

Open <http://127.0.0.1:5000>. The dashboard is local; do not commit `mlflow.db`, `mlartifacts/`, or `mlruns/` to Git.

## 2. Data validation

`src/data/validation.py` checks required columns, duplicate participant/visit rows, valid CDR ranges, binary labels, and disjoint participant splits. The training scripts run these checks immediately after loading and cohort/split creation.

## 3. Continuous integration

`.github/workflows/tests.yml` installs the declared dependencies and runs `pytest -q` on pushes and pull requests. This is a quality gate; it does not train on private ADNI data.

## Reproducibility

MLflow improves traceability but does not improve metrics by itself. Results remain genuine outputs of the declared data, split, code, and configuration. Local MLflow files are intentionally ignored by Git because they contain local artifacts and paths.
