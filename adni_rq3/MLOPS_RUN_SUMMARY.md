# Verified MLOps run summary

Date: 2026-10-28

## Verification

- `python -m pytest -q --disable-warnings`: **59 passed**.
- Baseline real-data run: completed and tracked in MLflow.
- GRU real-data run: completed and tracked in MLflow.
- LSTM real-data run: completed and tracked in MLflow.
- `scripts/compare_models.py --results-dir results`: completed.
- MLflow experiment `ADNI_RQ3`: 3 finished successful runs are present for the final experiments. A first baseline attempt is also retained as a failed run because it correctly exposed an overly strict validator; the validator was then calibrated to the observed ADNI `CDGLOBAL=-1` missing sentinel and the baseline was rerun successfully.

## Final held-out test metrics

These are genuine results from the same 270-participant held-out test set used by the corrected project:

| Model | ROC-AUC | Average Precision | Balanced Accuracy | Sensitivity | Specificity | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7161 | 0.6099 | 0.6906 | 0.6429 | 0.7384 | 0.6861 |
| Random Forest | 0.7021 | 0.6127 | 0.6768 | 0.6327 | 0.7209 | 0.6716 |
| MLP | 0.7084 | 0.6009 | 0.6884 | 0.5918 | 0.7849 | 0.6896 |
| GRU longitudinal | 0.8465 | 0.7620 | 0.7564 | 0.9082 | 0.6047 | 0.7139 |
| LSTM longitudinal | 0.8475 | 0.7688 | 0.7658 | 0.8980 | 0.6337 | 0.7280 |

The results are not a claim of clinical readiness or external generalization. DXSUM, ADNIMERGE, and an independent validation cohort were not supplied.
