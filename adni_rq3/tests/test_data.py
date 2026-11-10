"""
test_data.py
------------
Tests for data loading, cohort definition, and preprocessing modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cohort import define_cohort, _label_by_cdr
from src.data.preprocessing import (
    FeaturePipeline, build_baseline_features,
    build_sequence_features, get_available_features
)
from src.data.splitting import participant_split, filter_arrays_by_rids


# ─── Cohort Tests ─────────────────────────────────────────────────────────────

class TestCohortDefinition:
    def test_cdr_labeling_returns_dataframe(self, fake_merged_df):
        labels = _label_by_cdr(fake_merged_df)
        assert isinstance(labels, pd.DataFrame)
        assert 'RID' in labels.columns
        assert 'LABEL' in labels.columns

    def test_labels_are_binary(self, fake_merged_df):
        labels = _label_by_cdr(fake_merged_df)
        assert set(labels['LABEL'].unique()).issubset({0, 1})

    def test_excludes_single_visit_participants(self, fake_merged_df):
        # Add a single-visit participant
        single_visit = pd.DataFrame([{
            'RID': 9999, 'VISIT_MONTH': 0, 'VISCODE_NORM': 'bl',
            'CDGLOBAL': 0.5, 'CDRSB': 2.0,
            'CDMEMORY': 0.5, 'CDORIENT': 0.0, 'CDJUDGE': 0.0,
            'CDCOMMUN': 0.0, 'CDHOME': 0.0, 'CDCARE': 0.0,
            'MMSE_TOTAL': 25, 'ADAS_TOTAL': 15.0,
            'AVLT_T5': 5, 'AVLT_DELAY': 4
        }])
        single_visit['RID'] = single_visit['RID'].astype('Int64')
        df = pd.concat([fake_merged_df, single_visit], ignore_index=True)
        labels = _label_by_cdr(df)
        assert 9999 not in labels['RID'].values

    def test_define_cohort_returns_tuple(self, fake_merged_df):
        labels_df, long_df = define_cohort(fake_merged_df, dxsum_df=None)
        assert isinstance(labels_df, pd.DataFrame)
        assert isinstance(long_df, pd.DataFrame)

    def test_define_cohort_long_df_has_label_col(self, fake_merged_df):
        _, long_df = define_cohort(fake_merged_df, dxsum_df=None)
        assert 'LABEL' in long_df.columns

    def test_no_non_mci_at_baseline(self, fake_merged_df):
        """All participants in cohort should have CDR = 0.5 at baseline."""
        labels_df, long_df = define_cohort(fake_merged_df, dxsum_df=None)
        bl_cdr = (
            long_df.sort_values('VISIT_MONTH')
                   .groupby('RID')['CDGLOBAL']
                   .first()
        )
        assert (bl_cdr == 0.5).all(), "Non-MCI participants found at baseline"


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_feature_pipeline_fit_transform(self, fake_merged_df, fake_labels_df,
                                            feature_cols):
        X, y = build_baseline_features(fake_merged_df, fake_labels_df,
                                        feature_cols)
        rids = list(X.index[:50])
        X_train = X.loc[X.index.isin(rids)]

        pipeline = FeaturePipeline()
        X_arr = pipeline.fit_transform(X_train)
        assert X_arr.shape == (len(rids), X_train.shape[1])
        assert not np.isnan(X_arr).any()

    def test_pipeline_no_nan_after_impute(self, fake_merged_df,
                                          fake_labels_df, feature_cols):
        # Introduce some NaN values
        df = fake_merged_df.copy()
        df.loc[df.index[:20], 'MMSE_TOTAL'] = np.nan

        X, y = build_baseline_features(df, fake_labels_df, feature_cols)
        pipeline = FeaturePipeline()
        X_arr = pipeline.fit_transform(X)
        assert not np.isnan(X_arr).any()

    def test_pipeline_transform_unseen_data(self, fake_merged_df,
                                             fake_labels_df, feature_cols):
        X, y = build_baseline_features(fake_merged_df, fake_labels_df,
                                        feature_cols)
        n = len(X)
        X_train = X.iloc[:int(n * 0.8)]
        X_test = X.iloc[int(n * 0.8):]

        pipeline = FeaturePipeline()
        pipeline.fit(X_train)
        X_test_arr = pipeline.transform(X_test)
        assert X_test_arr.shape[0] == len(X_test)
        assert not np.isnan(X_test_arr).any()

    def test_transform_before_fit_raises(self, fake_merged_df,
                                          fake_labels_df, feature_cols):
        X, y = build_baseline_features(fake_merged_df, fake_labels_df,
                                        feature_cols)
        pipeline = FeaturePipeline()
        with pytest.raises(RuntimeError):
            pipeline.transform(X)

    def test_build_baseline_features_shape(self, fake_merged_df,
                                            fake_labels_df, feature_cols):
        X, y = build_baseline_features(fake_merged_df, fake_labels_df,
                                        feature_cols)
        n_labeled = len(fake_labels_df)
        assert X.shape[0] == n_labeled
        assert len(y) == n_labeled

    def test_build_sequence_features_shape(self, fake_merged_df,
                                            fake_labels_df, feature_cols):
        max_seq = 6
        X_seq, masks, y, rids = build_sequence_features(
            fake_merged_df, fake_labels_df, feature_cols, max_seq_len=max_seq
        )
        n = len(fake_labels_df)
        f = len([c for c in feature_cols if c in fake_merged_df.columns])
        assert X_seq.shape == (n, max_seq, f)
        assert masks.shape == (n, max_seq)
        assert y.shape == (n,)

    def test_sequence_masks_sum_correctly(self, fake_merged_df,
                                           fake_labels_df, feature_cols):
        max_seq = 10
        _, masks, _, _ = build_sequence_features(
            fake_merged_df, fake_labels_df, feature_cols, max_seq_len=max_seq
        )
        # Each participant has 5 visits, so masks should sum to 5
        assert (masks.sum(axis=1) == 5).all()

    def test_get_available_features_filters(self, fake_merged_df):
        feats = get_available_features(fake_merged_df, ['CDGLOBAL', 'NONEXISTENT'])
        assert 'CDGLOBAL' in feats
        assert 'NONEXISTENT' not in feats


# ─── Splitting Tests ──────────────────────────────────────────────────────────

class TestSplitting:
    def test_split_no_overlap(self, fake_labels_df):
        train, val, test = participant_split(fake_labels_df, random_state=42)
        train_set, val_set, test_set = set(train), set(val), set(test)
        assert len(train_set & val_set) == 0, "Train/val overlap!"
        assert len(train_set & test_set) == 0, "Train/test overlap!"
        assert len(val_set & test_set) == 0, "Val/test overlap!"

    def test_split_covers_all_participants(self, fake_labels_df):
        train, val, test = participant_split(fake_labels_df, random_state=42)
        all_in_split = set(train) | set(val) | set(test)
        all_rids = set(fake_labels_df['RID'].astype(int).tolist())
        assert all_in_split == all_rids

    def test_test_size_respected(self, fake_labels_df):
        n = len(fake_labels_df)
        train, val, test = participant_split(
            fake_labels_df, test_size=0.20, val_size=0.10, random_state=42
        )
        assert abs(len(test) / n - 0.20) < 0.05

    def test_stratification_maintains_ratio(self, fake_labels_df):
        """Converter fraction in train/test should be similar to overall."""
        overall_rate = fake_labels_df['LABEL'].mean()
        train, val, test = participant_split(fake_labels_df, stratify=True,
                                             random_state=42)
        label_map = dict(zip(fake_labels_df['RID'].astype(int),
                             fake_labels_df['LABEL']))

        test_rate = np.mean([label_map[r] for r in test])
        assert abs(test_rate - overall_rate) < 0.15, (
            f'Test converter rate {test_rate:.2f} far from overall {overall_rate:.2f}'
        )

    def test_filter_arrays_by_rids(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        rids = list(range(10))
        target = [0, 2, 4]
        X_sub, y_sub = filter_arrays_by_rids(X, y, rids, target)
        assert X_sub.shape == (3, 2)
        assert list(y_sub) == [0, 2, 4]
