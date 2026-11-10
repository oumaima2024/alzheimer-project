# ADNI RQ3 – MCI-to-Alzheimer Prediction

> **Research Question 3**: Does longitudinal modelling of successive ADNI visits improve MCI-to-AD conversion prediction compared to a cross-sectional (T0-only) baseline?

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Scientific Background](#scientific-background)
4. [Architecture](#architecture)
5. [Installation (Windows)](#installation-windows)
6. [Data Preparation](#data-preparation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Running Tests](#running-tests)
9. [Results](#results)
10. [Configuration Reference](#configuration-reference)
11. [Methodology Notes](#methodology-notes)
12. [References](#references)

---

## Overview

This project implements and compares:

| Approach | Models | Input |
|----------|--------|-------|
| **Baseline (T0-only)** | Logistic Regression, Random Forest, MLP | Single baseline visit features |
| **Longitudinal** | GRU / LSTM | Sequence of all available visits |

The cohort consists of **MCI participants from ADNI** who either:
- **Convert** to AD at a follow-up visit (label = 1)
- **Remain stable** MCI throughout follow-up (label = 0)

---

## Project Structure

```
adni_rq3/
├── config.yaml                  # All hyperparameters and paths
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                     ← Place ADNI CSV files here
│   └── processed/               ← Auto-generated intermediate files
│
├── results/
│   ├── figures/                 ← All plots (ROC, PR, confusion matrices…)
│   ├── models/                  ← Saved model checkpoints
│   ├── baseline_results.json
│   ├── longitudinal_results.json
│   └── rq3_summary.csv
│
├── scripts/
│   ├── train_baseline.py        ← Step 1: train LR / RF / MLP
│   ├── train_longitudinal.py    ← Step 2: train GRU / LSTM
│   └── compare_models.py        ← Step 3: generate comparison report
│
├── src/
│   ├── data/
│   │   ├── loader.py            ← Load CDR, ITEM, DXSUM CSV files
│   │   ├── cohort.py            ← Define MCI cohort + labels
│   │   ├── preprocessing.py     ← Impute / scale / build feature matrices
│   │   └── splitting.py         ← Participant-level train/val/test split
│   ├── models/
│   │   ├── baseline_models.py   ← LR, RF, MLP implementations
│   │   └── longitudinal_model.py ← GRU / LSTM implementation
│   └── evaluation/
│       ├── metrics.py           ← AUC, AP, sensitivity, specificity, CI
│       └── visualizations.py    ← All publication-quality plots
│
└── tests/
    ├── conftest.py              ← Shared synthetic data fixtures
    ├── test_data.py             ← Cohort / preprocessing / splitting tests
    ├── test_models.py           ← Model fit / predict / save / load tests
    ├── test_metrics.py          ← Metrics computation tests
    └── test_integration.py      ← End-to-end pipeline tests
```

---

## Scientific Background

### Cohort Definition

Following Petersen et al. (2010) and Jack et al. (2018):
- **MCI at baseline**: CDR global score = 0.5 (or DXCHANGE codes 2/4/5 in DXSUM)
- **Converter**: any follow-up visit with CDR ≥ 1.0 (or DXCHANGE = 5/6)
- **Stable MCI**: CDR remains 0.5 throughout all follow-ups
- Minimum 2 visits required to assign a label

### Feature Groups

| Group | Features | Source |
|-------|----------|--------|
| CDR | CDR global, CDR-SB, 6 box scores | `CDR_*.csv` |
| Neuropsychological | MMSE total, ADAS-Cog 11, AVLT T5, AVLT Delayed, Boston Naming, TMT-B | `ITEM.csv` |
| Demographics | Age, Education, Sex, APOE ε4 | `ADNIMERGE.csv` (if available) |

### Data Leakage Prevention

- **Train/val/test splits** are at the **participant level** — no participant appears in more than one set
- **Imputer and scaler** are fit **only on the training set** and applied to val/test
- **Labels** are derived from **follow-up visits only** — baseline data does not influence labeling

---

## Architecture

### Baseline Models (T0 only)

```
Baseline features (1 row per participant)
  ↓
Imputer (median) + StandardScaler  [fit on train only]
  ↓
┌──────────────────────────────────────────┐
│  Logistic Regression (L2, balanced)       │
│  Random Forest (300 trees, balanced)      │
│  MLP: FC(128) → BN → ReLU → Dropout     │
│       FC(64)  → BN → ReLU → Dropout     │
│       FC(32)  → BN → ReLU → Dropout     │
│       FC(1)   → sigmoid                  │
└──────────────────────────────────────────┘
  ↓
Binary prediction (Converter / Stable)
```

### Longitudinal Model (GRU/LSTM)

```
Sequence of visits  (N, T, F)  [zero-padded]
  ↓
LayerNorm(F)
  ↓
Pack padded sequence
  ↓
Stacked GRU / LSTM  (hidden=64, layers=2, dropout=0.3)
  ↓
Unpack → gather last valid hidden state
  ↓
FC(32) → ReLU → Dropout → FC(1) → sigmoid
  ↓
Binary prediction
```

---

## Installation (Windows)

### Requirements

- Python 3.10+ (download from python.org)
- pip

### Steps

```powershell
# 1. Open PowerShell and navigate to the project
cd C:\path\to\adni_rq3

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch, sklearn, pandas; print('OK')"
```

> **GPU support** (optional): If you have an NVIDIA GPU, install the CUDA-enabled
> PyTorch build from https://pytorch.org/get-started/locally/

---

## Data Preparation

### Required files (place in `data/raw/`)

| File | Source | Required |
|------|--------|----------|
| `CDR_27Oct2026.csv` | ADNI data portal → CDR | ✅ Required |
| `ITEM.csv` | ADNI data portal → Neuropsych Battery | ✅ Required |
| `DXSUM_PDXCONV_ADNIALL.csv` | ADNI → Diagnosis Summary | Optional (falls back to CDR) |
| `ADNIMERGE.csv` | ADNI → Merge spreadsheet | Optional (adds demographics) |

### How to download ADNI data

1. Register at [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. After approval, go to **Download → Study Data → Test Results**
3. Download: `CDR`, `ADASSCORES` or `ITEM`, `DXSUM`
4. Export as CSV and place in `data/raw/`

> **Note**: If you already have the files from another location (e.g. the parent
> folder), the scripts will find them automatically if you set `data.raw_dir` in
> `config.yaml`.

---

## Running the Pipeline

### Step 0: Activate environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 1: Train baseline T0 models

```powershell
python scripts/train_baseline.py --config config.yaml
```

Outputs:
- `results/baseline_results.json` – test metrics with 95% CI
- `results/baseline_metrics.csv`
- `results/figures/roc_baseline.png`, `pr_baseline.png`, `metric_comparison_baseline.png`
- `results/figures/cm_*.png` – confusion matrices
- `results/figures/feature_importance_rf.png`
- `results/figures/cohort_summary.png`
- `results/models/LogisticRegression.pkl`, `RandomForest.pkl`, `MLP.pt`

### Step 2: Train longitudinal GRU model

```powershell
python scripts/train_longitudinal.py --config config.yaml
```

Outputs:
- `results/longitudinal_results.json`
- `results/longitudinal_metrics.csv`
- `results/figures/roc_GRU_Longitudinal.png`
- `results/figures/loss_GRU_Longitudinal.png`
- `results/models/GRU_Longitudinal.pt`

To use LSTM instead of GRU:

```powershell
python scripts/train_longitudinal.py --config config.yaml --model-type LSTM
```

### Step 3: Compare all models

```powershell
python scripts/compare_models.py --results-dir results
```

Outputs:
- `results/rq3_summary.csv` – full comparison table
- `results/figures/rq3_full_comparison.png`
- `results/figures/rq3_summary_table.png`

### Run everything at once

```powershell
python scripts/train_baseline.py --config config.yaml
python scripts/train_longitudinal.py --config config.yaml
python scripts/compare_models.py --results-dir results
```

---

## Running Tests

```powershell
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_metrics.py -v
pytest tests/test_integration.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

Tests do **not** require real ADNI data — they use synthetic data generated by `conftest.py`.

---

## Results

After running all scripts, the main comparison table (`results/rq3_summary.csv`) will contain:

| Model | ROC-AUC | Avg Precision | Bal. Accuracy | Sensitivity | Specificity | F1 |
|-------|---------|---------------|---------------|-------------|-------------|-----|
| LogisticRegression | – | – | – | – | – | – |
| RandomForest | – | – | – | – | – | – |
| MLP | – | – | – | – | – | – |
| GRU_Longitudinal | – | – | – | – | – | – |

> Values will be populated after running the pipeline with real ADNI data.

---

## Configuration Reference

All parameters are in `config.yaml`:

```yaml
data:
  raw_dir: "data/raw"      # Path to ADNI CSV files
  results_dir: "results"

cohort:
  min_visits: 2            # Minimum visits to include participant
  cdr_mci_threshold: 0.5   # CDR global for MCI at baseline
  cdr_ad_threshold: 1.0    # CDR ≥ this at follow-up → Converter

split:
  test_size: 0.20          # Fraction for held-out test
  val_size: 0.10           # Fraction for validation
  stratify: true           # Stratify by Converter/Stable label

baseline:
  mlp:
    hidden_layer_sizes: [128, 64, 32]
    epochs: 100
    patience: 15           # Early stopping

longitudinal:
  model_type: "GRU"        # "GRU" or "LSTM"
  hidden_size: 64
  num_layers: 2
  max_seq_len: 10          # Pad/truncate visit sequences to this length
  patience: 20

evaluation:
  bootstrap_iterations: 1000  # For 95% CI computation
```

---

## Methodology Notes

### Why participant-level splitting?

If a participant's visits were split across train and test sets, the model would
see some of their clinical trajectory at training time and then be evaluated on
other visits of the same person — this constitutes **data leakage** and inflates
test performance.

### Why GRU over LSTM?

GRU has fewer parameters than LSTM (no cell state), making it less prone to
overfitting with the small sample sizes typical in ADNI studies (typically
200–800 MCI participants). Both are available via the `--model-type` flag.

### Handling missing data

- Missing feature values are imputed with the **training-set median**
- Padded time steps are handled with **attention masks** in the RNN
- If ITEM.csv is unavailable, the pipeline falls back to CDR features only

### Class imbalance

Converters are typically 30–40% of MCI participants. We address this with:
- `class_weight='balanced'` in LR and RF
- Inverse class frequency weighting in BCE loss for MLP and GRU

---

## References

1. **Petersen et al. (2010)**. Alzheimer's disease neuroimaging initiative (ADNI): clinical characterization. *Neurology*, 74(3), 201–209.
2. **Jack et al. (2018)**. NIA-AA Research Framework: toward a biological definition of Alzheimer's disease. *Alzheimer's & Dementia*, 14(4), 535–562.
3. **Moradi et al. (2015)**. Machine learning framework for early MRI-based Alzheimer's conversion prediction in MCI subjects. *NeuroImage*, 104, 398–412.
4. **Nguyen et al. (2020)**. Predicting Alzheimer's disease progression using deep recurrent neural networks. *NeuroImage*, 222, 117203.
5. **Bhagwat et al. (2018)**. Modeling and prediction of clinical symptom trajectories in Alzheimer's disease using longitudinal data. *PLOS Computational Biology*, 14(9).
6. **Ghazi et al. (2019)**. Training recurrent neural networks robust to incomplete data. *IEEE Transactions on Medical Imaging*, 38(7).
7. **Ding et al. (2019)**. A deep learning model to predict a diagnosis of Alzheimer disease by using 18F-FDG PET of the brain. *Radiology*, 290(2).
8. **Saito & Rehmsmeier (2015)**. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3).
