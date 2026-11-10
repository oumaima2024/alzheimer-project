import numpy as np
import pandas as pd

from src.data.cohort import define_cohort
from src.data.preprocessing import build_baseline_features, build_sequence_features


def _rows():
    return pd.DataFrame([
        {'RID': 1, 'VISIT_MONTH': 0, 'CDGLOBAL': 0.5, 'CDRSB': 1.0,
         'MMSE_TOTAL': np.nan},
        {'RID': 1, 'VISIT_MONTH': 6, 'CDGLOBAL': 1.0, 'CDRSB': 4.0,
         'MMSE_TOTAL': 20.0},
        {'RID': 1, 'VISIT_MONTH': 12, 'CDGLOBAL': 2.0, 'CDRSB': 6.0,
         'MMSE_TOTAL': 18.0},
        {'RID': 2, 'VISIT_MONTH': 0, 'CDGLOBAL': 0.5, 'CDRSB': 1.5,
         'MMSE_TOTAL': 26.0},
        {'RID': 2, 'VISIT_MONTH': 3, 'CDGLOBAL': 0.5, 'CDRSB': 1.5,
         'MMSE_TOTAL': 25.0},
    ])


def test_baseline_does_not_backfill_from_future_visit():
    df = _rows()
    labels, long_df = define_cohort(df, require_followup_months=3)
    X, _ = build_baseline_features(long_df, labels, ['CDGLOBAL', 'MMSE_TOTAL'])
    assert np.isnan(X.loc[1, 'MMSE_TOTAL'])


def test_followup_requirement_is_enforced():
    labels, _ = define_cohort(_rows(), require_followup_months=6)
    assert set(labels['RID']) == {1}


def test_sequence_excludes_conversion_and_post_conversion_visits():
    df = _rows()
    labels, long_df = define_cohort(df, require_followup_months=3)
    X, masks, _, rids = build_sequence_features(
        long_df, labels, ['CDGLOBAL'], max_seq_len=5
    )
    assert rids == [1, 2]
    assert masks.sum(axis=1).tolist() == [1.0, 2.0]
    assert X[0, 0, 0] == 0.5
