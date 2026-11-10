# Final package manifest

This archive contains the corrected ADNI RQ3 project with source code, tests, documentation, real result artifacts, experiment logs, and local MLflow tracking artifacts.

Included MLOps components:

- `src/ops/tracking.py`: SQLite-backed local MLflow helper.
- `src/data/validation.py`: input, labels, and split validation.
- `.github/workflows/tests.yml`: CI test workflow.
- `MLOPS.md`: setup and dashboard instructions.
- `MLOPS_RUN_SUMMARY.md`: verified runs and genuine metrics.
- `mlflow.db` and `mlruns/`: recorded local experiment metadata/artifacts from the final runs.

Verification performed before packaging:

- 59 automated tests passed.
- Baseline, GRU, and LSTM ran on the real supplied ADNI-derived data.
- Five-model comparison regenerated.
- Required model rows are present in `results/rq3_summary.csv`.

The Python virtual environment and Python caches are intentionally excluded. Install dependencies with `python -m pip install -r requirements.txt`.
