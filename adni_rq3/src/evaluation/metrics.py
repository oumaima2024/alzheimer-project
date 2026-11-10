"""
metrics.py
----------
Evaluation metrics for binary classification of MCI-to-AD conversion.

Computes:
  - ROC-AUC
  - Average Precision (PR-AUC)
  - Balanced Accuracy
  - F1 Score (macro)
  - Sensitivity (Recall / True Positive Rate)
  - Specificity (True Negative Rate)
  - Bootstrap 95% CI for all metrics

References:
  - Hanley & McNeil (1982) – AUC interpretation
  - Saito & Rehmsmeier (2015) – PR-AUC for imbalanced data
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true : (N,) int array – true labels (0/1)
    y_pred : (N,) int array – predicted labels (0/1)
    y_prob : (N,) float array – predicted probabilities of class 1
    threshold : float – decision threshold (used only if y_pred is None)

    Returns
    -------
    dict with keys: roc_auc, average_precision, balanced_accuracy,
                    f1, sensitivity, specificity
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    results = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        results['roc_auc'] = roc_auc_score(y_true, y_prob)
        results['average_precision'] = average_precision_score(y_true, y_prob)
    else:
        results['roc_auc'] = float('nan')
        results['average_precision'] = float('nan')

    return results


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.95,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for all metrics.

    Parameters
    ----------
    y_true : (N,) int
    y_prob : (N,) float
    n_iterations : int
    alpha : float – CI level (e.g. 0.95)
    random_state : int
    threshold : float

    Returns
    -------
    dict { metric_name: {'mean': ..., 'ci_lower': ..., 'ci_upper': ...} }
    """
    rng = np.random.default_rng(random_state)
    N = len(y_true)
    records: Dict[str, list] = {}

    for _ in range(n_iterations):
        idx = rng.integers(0, N, size=N)
        yt = y_true[idx]
        yp_prob = y_prob[idx]
        yp_pred = (yp_prob >= threshold).astype(int)

        # Need at least one of each class
        if len(np.unique(yt)) < 2:
            continue

        m = compute_metrics(yt, yp_pred, yp_prob, threshold)
        for k, v in m.items():
            if k in ('tp', 'fp', 'tn', 'fn'):
                continue
            records.setdefault(k, []).append(v)

    lo = (1 - alpha) / 2
    hi = 1 - lo

    ci_results = {}
    for k, vals in records.items():
        arr = np.array(vals)
        ci_results[k] = {
            'mean': float(np.nanmean(arr)),
            'ci_lower': float(np.nanquantile(arr, lo)),
            'ci_upper': float(np.nanquantile(arr, hi)),
        }

    return ci_results


def get_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)


def get_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_prob)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'youden',
) -> float:
    """
    Find the decision threshold that maximises a given metric.

    Parameters
    ----------
    metric : str
        'youden'      → Youden's J = sensitivity + specificity - 1
        'f1'          → F1 score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if metric == 'youden':
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    elif metric == 'f1':
        f1s = []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_true, yp, zero_division=0))
        best_idx = int(np.argmax(f1s))
    else:
        raise ValueError(f'Unknown metric: {metric}')

    return float(thresholds[best_idx])
