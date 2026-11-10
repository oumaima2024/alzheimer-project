"""
test_metrics.py
---------------
Tests for the evaluation metrics module.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics, bootstrap_ci, get_roc_curve,
    get_pr_curve, find_optimal_threshold
)


@pytest.fixture
def binary_example():
    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, 100)
    y_prob = np.clip(y_true + rng.normal(0, 0.3, 100), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    def test_returns_expected_keys(self, binary_example):
        y_true, y_pred, y_prob = binary_example
        m = compute_metrics(y_true, y_pred, y_prob)
        for key in ['roc_auc', 'average_precision', 'balanced_accuracy',
                    'f1', 'sensitivity', 'specificity']:
            assert key in m, f'Missing key: {key}'

    def test_roc_auc_above_chance(self, binary_example):
        y_true, y_pred, y_prob = binary_example
        m = compute_metrics(y_true, y_pred, y_prob)
        assert m['roc_auc'] > 0.5

    def test_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        preds = np.array([0, 0, 1, 1])
        m = compute_metrics(y, preds, probs)
        assert m['roc_auc'] == 1.0
        assert m['sensitivity'] == 1.0
        assert m['specificity'] == 1.0

    def test_sensitivity_specificity_in_range(self, binary_example):
        y_true, y_pred, y_prob = binary_example
        m = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m['sensitivity'] <= 1.0
        assert 0.0 <= m['specificity'] <= 1.0

    def test_no_proba_gives_nan_auc(self, binary_example):
        y_true, y_pred, _ = binary_example
        m = compute_metrics(y_true, y_pred, y_prob=None)
        assert np.isnan(m['roc_auc'])


class TestBootstrapCI:
    def test_returns_ci_structure(self, binary_example):
        y_true, _, y_prob = binary_example
        ci = bootstrap_ci(y_true, y_prob, n_iterations=50)
        assert 'roc_auc' in ci
        assert 'mean' in ci['roc_auc']
        assert 'ci_lower' in ci['roc_auc']
        assert 'ci_upper' in ci['roc_auc']

    def test_ci_lower_le_mean_le_upper(self, binary_example):
        y_true, _, y_prob = binary_example
        ci = bootstrap_ci(y_true, y_prob, n_iterations=50)
        for metric, vals in ci.items():
            assert vals['ci_lower'] <= vals['mean'] <= vals['ci_upper'], \
                f"CI violated for {metric}"

    def test_ci_width_positive(self, binary_example):
        y_true, _, y_prob = binary_example
        ci = bootstrap_ci(y_true, y_prob, n_iterations=100)
        # CI should have some width (not zero)
        assert ci['roc_auc']['ci_upper'] > ci['roc_auc']['ci_lower']


class TestCurves:
    def test_roc_curve_shape(self, binary_example):
        y_true, _, y_prob = binary_example
        fpr, tpr, thresh = get_roc_curve(y_true, y_prob)
        assert len(fpr) == len(tpr)
        assert fpr[0] <= fpr[-1]  # non-decreasing

    def test_pr_curve_shape(self, binary_example):
        y_true, _, y_prob = binary_example
        prec, recall, thresh = get_pr_curve(y_true, y_prob)
        assert len(prec) == len(recall)

    def test_find_optimal_threshold_youden(self, binary_example):
        y_true, _, y_prob = binary_example
        t = find_optimal_threshold(y_true, y_prob, metric='youden')
        assert 0.0 <= t <= 1.0

    def test_find_optimal_threshold_f1(self, binary_example):
        y_true, _, y_prob = binary_example
        t = find_optimal_threshold(y_true, y_prob, metric='f1')
        assert 0.0 <= t <= 1.0
