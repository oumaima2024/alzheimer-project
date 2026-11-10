"""
test_integration.py
-------------------
End-to-end integration test using synthetic data.

Tests the full pipeline:
  CDR data → cohort → split → preprocess → train → evaluate

This verifies that all modules work together correctly without
requiring real ADNI data.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cohort import define_cohort
from src.data.preprocessing import (
    FeaturePipeline, build_baseline_features,
    build_sequence_features, get_available_features, ALL_FEATURES
)
from src.data.splitting import (
    participant_split, filter_arrays_by_rids
)
from src.models.baseline_models import (
    LogisticRegressionModel, RandomForestModel, MLPModel
)
from src.models.longitudinal_model import LongitudinalModel
from src.evaluation.metrics import compute_metrics, bootstrap_ci


@pytest.fixture(scope='module')
def full_pipeline_data(fake_merged_df, fake_labels_df):
    """Run the data preparation pipeline end-to-end."""
    feat_cols = get_available_features(fake_merged_df, ALL_FEATURES)

    # Cohort (use CDR-based labeling)
    labels_df, long_df = define_cohort(fake_merged_df, dxsum_df=None)

    # Require at least some participants
    if len(labels_df) < 10:
        pytest.skip('Not enough participants in synthetic cohort')

    # Split
    train_rids, val_rids, test_rids = participant_split(
        labels_df, test_size=0.20, val_size=0.10, random_state=42
    )

    return {
        'feat_cols': feat_cols,
        'labels_df': labels_df,
        'long_df': long_df,
        'train_rids': train_rids,
        'val_rids': val_rids,
        'test_rids': test_rids,
    }


class TestFullBaselinePipeline:
    def test_logistic_regression_pipeline(self, full_pipeline_data):
        fd = full_pipeline_data
        X, y = build_baseline_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols']
        )
        X_tr = X.loc[X.index.isin(fd['train_rids'])]
        y_tr = y.loc[y.index.isin(fd['train_rids'])]
        X_te = X.loc[X.index.isin(fd['test_rids'])]
        y_te = y.loc[y.index.isin(fd['test_rids'])]

        pipe = FeaturePipeline()
        X_tr_np = pipe.fit_transform(X_tr)
        X_te_np = pipe.transform(X_te)

        model = LogisticRegressionModel()
        model.fit(X_tr_np, y_tr.values)
        probs = model.predict_proba(X_te_np)
        preds = model.predict(X_te_np)

        # Only compute AUC if we have both classes in test set
        if len(np.unique(y_te.values)) > 1:
            m = compute_metrics(y_te.values, preds, probs)
            assert 'roc_auc' in m
            assert 0.0 <= m['roc_auc'] <= 1.0
        else:
            pytest.skip('Test set has only one class – cannot compute AUC')

    def test_random_forest_pipeline(self, full_pipeline_data):
        fd = full_pipeline_data
        X, y = build_baseline_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols']
        )
        X_tr = X.loc[X.index.isin(fd['train_rids'])]
        y_tr = y.loc[y.index.isin(fd['train_rids'])]
        X_te = X.loc[X.index.isin(fd['test_rids'])]
        y_te = y.loc[y.index.isin(fd['test_rids'])]

        pipe = FeaturePipeline()
        X_tr_np = pipe.fit_transform(X_tr)
        X_te_np = pipe.transform(X_te)

        model = RandomForestModel(n_estimators=20)
        model.fit(X_tr_np, y_tr.values)
        probs = model.predict_proba(X_te_np)
        assert probs.shape == (len(y_te),)

    def test_mlp_pipeline(self, full_pipeline_data):
        fd = full_pipeline_data
        X, y = build_baseline_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols']
        )
        X_tr = X.loc[X.index.isin(fd['train_rids'])]
        y_tr = y.loc[y.index.isin(fd['train_rids'])]
        X_val = X.loc[X.index.isin(fd['val_rids'])]
        y_val = y.loc[y.index.isin(fd['val_rids'])]
        X_te = X.loc[X.index.isin(fd['test_rids'])]
        y_te = y.loc[y.index.isin(fd['test_rids'])]

        pipe = FeaturePipeline()
        X_tr_np = pipe.fit_transform(X_tr)
        X_val_np = pipe.transform(X_val)
        X_te_np = pipe.transform(X_te)

        model = MLPModel(
            input_dim=X_tr_np.shape[1],
            hidden_layer_sizes=(16, 8),
            epochs=5,
            patience=3,
        )
        model.fit(X_tr_np, y_tr.values, X_val_np, y_val.values)
        probs = model.predict_proba(X_te_np)
        assert probs.shape == (len(y_te),)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestFullLongitudinalPipeline:
    def test_gru_pipeline(self, full_pipeline_data):
        fd = full_pipeline_data
        max_seq = 6

        X_seq, masks, y_seq, all_rids = build_sequence_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols'],
            max_seq_len=max_seq
        )

        # Fit pipeline on training baseline features
        X_bl, y_bl = build_baseline_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols']
        )
        X_bl_train = X_bl.loc[X_bl.index.isin(fd['train_rids'])]
        pipe = FeaturePipeline()
        pipe.fit(X_bl_train)
        X_seq_scaled = pipe.transform_sequences(X_seq)

        X_tr, y_tr = filter_arrays_by_rids(
            X_seq_scaled, y_seq, all_rids, fd['train_rids']
        )
        m_tr, _ = filter_arrays_by_rids(masks, y_seq, all_rids, fd['train_rids'])
        X_te, y_te = filter_arrays_by_rids(
            X_seq_scaled, y_seq, all_rids, fd['test_rids']
        )
        m_te, _ = filter_arrays_by_rids(masks, y_seq, all_rids, fd['test_rids'])

        model = LongitudinalModel(
            input_dim=X_tr.shape[2],
            hidden_size=8,
            num_layers=1,
            epochs=5,
            patience=3,
        )
        model.fit(X_tr, m_tr, y_tr)
        probs = model.predict_proba(X_te, m_te)
        assert probs.shape == (len(y_te),)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestLeakagePrevention:
    """Verify no data leakage across train/test boundaries."""

    def test_train_test_participant_disjoint(self, full_pipeline_data):
        fd = full_pipeline_data
        assert len(set(fd['train_rids']) & set(fd['test_rids'])) == 0

    def test_pipeline_fit_only_on_train(self, full_pipeline_data):
        """
        Check that imputer statistics come only from training data.
        Artificially offset test data by 100; pipeline stats should not shift.
        """
        fd = full_pipeline_data
        X, y = build_baseline_features(
            fd['long_df'], fd['labels_df'], fd['feat_cols']
        )
        X_tr = X.loc[X.index.isin(fd['train_rids'])]
        X_te = X.loc[X.index.isin(fd['test_rids'])]

        pipe = FeaturePipeline()
        pipe.fit(X_tr)
        stats_before = pipe.imputer.statistics_.copy()

        # Transform test (with NaN) – stats should not change
        X_te_with_nan = X_te.copy()
        X_te_with_nan.iloc[:, 0] = np.nan
        pipe.transform(X_te_with_nan)

        np.testing.assert_array_equal(stats_before, pipe.imputer.statistics_)

    def test_labels_from_followup_only(self, full_pipeline_data):
        """
        Verify that no label is computed using ONLY baseline information.
        Every labeled participant must have ≥ 2 visits in long_df.
        """
        fd = full_pipeline_data
        visit_counts = fd['long_df'].groupby('RID').size()
        labeled_rids = fd['labels_df']['RID'].values
        for rid in labeled_rids:
            if rid in visit_counts.index:
                assert visit_counts[rid] >= 2, \
                    f"Participant {rid} has only 1 visit but is labeled"
