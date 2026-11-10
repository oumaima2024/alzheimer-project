"""
conftest.py
-----------
Shared pytest fixtures for the ADNI RQ3 test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Synthetic ADNI-like data fixtures ────────────────────────────────────────

N_PARTICIPANTS = 80
N_VISITS_PER = 5
SEED = 42
rng = np.random.default_rng(SEED)

CDR_FEATURES = ['CDGLOBAL', 'CDRSB', 'CDMEMORY', 'CDORIENT',
                 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']
NEURO_FEATURES = ['MMSE_TOTAL', 'ADAS_TOTAL', 'AVLT_T5', 'AVLT_DELAY']
ALL_FEATURES = CDR_FEATURES + NEURO_FEATURES


@pytest.fixture(scope='session')
def fake_merged_df():
    """Synthetic longitudinal DataFrame mimicking merged CDR + ITEM data."""
    rows = []
    for rid in range(1, N_PARTICIPANTS + 1):
        for visit_idx in range(N_VISITS_PER):
            visit_month = visit_idx * 12
            row = {'RID': rid, 'VISIT_MONTH': visit_month,
                   'VISCODE_NORM': f'm{visit_month:02d}' if visit_month > 0 else 'bl'}
            # CDR baseline 0.5 for all (MCI cohort)
            row['CDGLOBAL'] = 0.5 if visit_idx == 0 else rng.choice(
                [0.5, 1.0], p=[0.6, 0.4])
            row['CDRSB'] = rng.uniform(0.5, 4.5)
            row['CDMEMORY'] = rng.choice([0.5, 1.0])
            row['CDORIENT'] = rng.choice([0.0, 0.5])
            row['CDJUDGE'] = rng.choice([0.0, 0.5])
            row['CDCOMMUN'] = rng.choice([0.0, 0.5])
            row['CDHOME'] = rng.choice([0.0, 0.5])
            row['CDCARE'] = rng.choice([0.0, 0.5])
            row['MMSE_TOTAL'] = rng.integers(18, 30)
            row['ADAS_TOTAL'] = rng.uniform(5, 35)
            row['AVLT_T5'] = rng.integers(0, 14)
            row['AVLT_DELAY'] = rng.integers(0, 15)
            rows.append(row)

    df = pd.DataFrame(rows)
    df['RID'] = df['RID'].astype('Int64')
    return df


@pytest.fixture(scope='session')
def fake_labels_df(fake_merged_df):
    """Synthetic labels: ~40% converters."""
    rids = fake_merged_df['RID'].unique()
    labels = rng.choice([0, 1], size=len(rids), p=[0.60, 0.40])
    df = pd.DataFrame({'RID': rids, 'LABEL': labels, 'N_VISITS': N_VISITS_PER})
    df['RID'] = df['RID'].astype('Int64')
    return df


@pytest.fixture(scope='session')
def fake_long_df(fake_merged_df, fake_labels_df):
    """Merged longitudinal df with LABEL column."""
    df = fake_merged_df.merge(
        fake_labels_df[['RID', 'LABEL']], on='RID', how='left'
    )
    return df


@pytest.fixture(scope='session')
def feature_cols():
    return ALL_FEATURES
