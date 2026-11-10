"""
visualizations.py
-----------------
Publication-ready plots for RQ3 evaluation.

Figures produced:
  1. ROC curves (all models, single plot)
  2. PR curves (all models, single plot)
  3. Metric comparison bar chart with 95% CI error bars
  4. Feature importance (Random Forest)
  5. SHAP summary plot (if shap is available)
  6. Training loss curves (MLP + GRU)
  7. Confusion matrices (one per model)
  8. Cohort flow diagram (text-based)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
STYLE_KWARGS = dict(dpi=150, bbox_inches='tight')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **STYLE_KWARGS)
    plt.close(fig)
    logger.info(f'Saved figure: {path}')


# ─── 1. ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: str,
    title: str = 'ROC Curves – RQ3',
):
    """
    Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    results : dict  { model_name: {'fpr': ..., 'tpr': ..., 'auc': ...} }
    save_path : str
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, res) in enumerate(results.items()):
        fpr = res['fpr']
        tpr = res['tpr']
        auc = res.get('auc', np.nan)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate (1 – Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _save(fig, save_path)


# ─── 2. PR Curves ─────────────────────────────────────────────────────────────

def plot_pr_curves(
    results: Dict[str, Dict],
    save_path: str,
    baseline_rate: float = 0.5,
    title: str = 'Precision-Recall Curves – RQ3',
):
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, res) in enumerate(results.items()):
        prec = res['precision']
        recall = res['recall']
        ap = res.get('ap', np.nan)
        ax.plot(recall, prec, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f'{name} (AP = {ap:.3f})')

    ax.axhline(baseline_rate, color='k', linestyle='--', lw=1,
               alpha=0.5, label=f'Baseline (prev={baseline_rate:.2f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    _save(fig, save_path)


# ─── 3. Metric Comparison Bar Chart ───────────────────────────────────────────

def plot_metric_comparison(
    ci_results: Dict[str, Dict[str, Dict]],
    metrics: List[str],
    save_path: str,
    title: str = 'Model Comparison – RQ3 (95% CI)',
):
    """
    Parameters
    ----------
    ci_results : { model_name: { metric: {'mean': ..., 'ci_lower': ..., 'ci_upper': ...} } }
    metrics : list of metric names to plot
    """
    n_metrics = len(metrics)
    n_models = len(ci_results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    model_names = list(ci_results.keys())
    x = np.arange(n_models)
    width = 0.6

    for ax, metric in zip(axes, metrics):
        means, lo_errs, hi_errs = [], [], []
        for name in model_names:
            m = ci_results[name].get(metric, {})
            mean = m.get('mean', 0)
            ci_lo = m.get('ci_lower', mean)
            ci_hi = m.get('ci_upper', mean)
            means.append(mean)
            lo_errs.append(mean - ci_lo)
            hi_errs.append(ci_hi - mean)

        bars = ax.bar(x, means, width,
                      color=PALETTE[:n_models], alpha=0.85, zorder=3)
        ax.errorbar(x, means, yerr=[lo_errs, hi_errs],
                    fmt='none', color='black', capsize=5, lw=2, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + 0.02, f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)


# ─── 4. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: str,
    top_n: int = 15,
    title: str = 'Random Forest Feature Importances',
):
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals[::-1], color='#2196F3', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Mean Decrease in Impurity')
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)


# ─── 5. Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: str,
    labels: List[str] = ['Stable MCI', 'Converter'],
):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False, linewidths=0.5,
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix – {model_name}')
    plt.tight_layout()
    _save(fig, save_path)


# ─── 6. Training Loss Curves ──────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    model_name: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color='#2196F3', lw=2, label='Train Loss')
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses, color='#FF5722',
                lw=2, linestyle='--', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title(f'Training Curves – {model_name}')
    ax.legend()
    _save(fig, save_path)


# ─── 7. Cohort Sankey / Flow ──────────────────────────────────────────────────

def plot_cohort_summary(
    total: int,
    mci_baseline: int,
    n_converters: int,
    n_stable: int,
    n_train: int,
    n_val: int,
    n_test: int,
    save_path: str,
):
    """Simple text-based cohort flow figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    lines = [
        f"ADNI Full Database: N = {total}",
        f"  ↓  Filter: baseline CDR = 0.5 (MCI)",
        f"  ↓  Require ≥ 2 visits",
        f"  ─────────────────────────────────────",
        f"  MCI Cohort: N = {mci_baseline}",
        f"    ├── Converters (MCI→AD): N = {n_converters}  ({100*n_converters/max(mci_baseline,1):.1f}%)",
        f"    └── Stable MCI:          N = {n_stable}  ({100*n_stable/max(mci_baseline,1):.1f}%)",
        f"  ─────────────────────────────────────",
        f"  Split (participant-level):",
        f"    Train:      N = {n_train}",
        f"    Validation: N = {n_val}",
        f"    Test:       N = {n_test}",
    ]

    for i, line in enumerate(lines):
        ax.text(0.05, 0.95 - i * 0.085, line,
                transform=ax.transAxes,
                fontsize=11, family='monospace',
                verticalalignment='top')

    ax.set_title('Cohort Definition – ADNI RQ3', fontsize=13, pad=10)
    _save(fig, save_path)


# ─── 8. Visit Distribution ────────────────────────────────────────────────────

def plot_visit_distribution(
    long_df,
    labels_df,
    save_path: str,
):
    """Histogram of number of visits per participant by group."""
    label_map = dict(zip(labels_df['RID'], labels_df['LABEL']))
    visit_counts = long_df.groupby('RID').size().reset_index(name='N_VISITS')
    visit_counts['GROUP'] = visit_counts['RID'].map(label_map).map(
        {0: 'Stable MCI', 1: 'Converter'}
    )
    visit_counts = visit_counts.dropna(subset=['GROUP'])

    fig, ax = plt.subplots(figsize=(8, 4))
    for group, color in [('Stable MCI', '#2196F3'), ('Converter', '#FF5722')]:
        subset = visit_counts[visit_counts['GROUP'] == group]['N_VISITS']
        ax.hist(subset, bins=range(1, 15), alpha=0.6, color=color,
                label=group, edgecolor='white')

    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Number of Participants')
    ax.set_title('Visit Distribution by Group')
    ax.legend()
    _save(fig, save_path)
