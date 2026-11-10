"""
compare_models.py
-----------------
Load saved results from baseline and longitudinal experiments
and produce a combined comparison report and figures.

Usage:
  python scripts/compare_models.py --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.visualizations import plot_metric_comparison, PALETTE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('compare_models')


def parse_args():
    p = argparse.ArgumentParser(description='Compare all RQ3 model results')
    p.add_argument('--results-dir', default='results')
    return p.parse_args()


METRICS_TO_SHOW = ['roc_auc', 'average_precision', 'balanced_accuracy',
                   'sensitivity', 'specificity', 'f1']


def load_results(results_dir: str) -> dict:
    combined = {}
    for fname in ('baseline_results.json', 'longitudinal_results.json'):
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                combined.update(json.load(f))
        else:
            logger.warning(f'{fpath} not found – skipping.')
    return combined


def build_summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for model, data in results.items():
        row = {'model': model}
        ci = data.get('ci', {})
        for m in METRICS_TO_SHOW:
            val = data.get(m, float('nan'))
            ci_m = ci.get(m, {})
            row[m] = round(val, 4)
            row[f'{m}_ci_lo'] = round(ci_m.get('ci_lower', float('nan')), 4)
            row[f'{m}_ci_hi'] = round(ci_m.get('ci_upper', float('nan')), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_combined_roc(results_dir: str, figures_dir: str):
    """Re-draw ROC curves from saved probability arrays if available."""
    # This would load saved y_prob arrays; for now we plot AUC bars
    pass


def plot_summary_table_figure(df: pd.DataFrame, save_path: str):
    """Plot a coloured table of metrics."""
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.8 + 1.5))
    ax.axis('off')

    display_cols = ['model'] + METRICS_TO_SHOW
    display_df = df[display_cols].copy()
    for m in METRICS_TO_SHOW:
        display_df[m] = display_df[m].apply(lambda x: f'{x:.3f}')

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Colour header
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colour rows alternately
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            table[i, j].set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')

    ax.set_title('RQ3 – All Model Metrics (Test Set)', fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved summary table to {save_path}')


def main():
    args = parse_args()
    results_dir = args.results_dir
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    results = load_results(results_dir)

    if not results:
        logger.error('No results found. Run train_baseline.py and '
                     'train_longitudinal.py first.')
        sys.exit(1)

    # CI results for comparison plot
    ci_results = {}
    for model, data in results.items():
        ci_results[model] = data.get('ci', {
            m: {'mean': data.get(m, 0), 'ci_lower': 0, 'ci_upper': 1}
            for m in METRICS_TO_SHOW
        })

    # Metric comparison plot
    plot_metric_comparison(
        ci_results,
        metrics=['roc_auc', 'average_precision', 'balanced_accuracy', 'f1'],
        save_path=os.path.join(figures_dir, 'rq3_full_comparison.png'),
        title='RQ3: Baseline vs Longitudinal – Test Set (95% CI)',
    )

    # Summary table
    df = build_summary_table(results)
    df.to_csv(os.path.join(results_dir, 'rq3_summary.csv'), index=False)
    plot_summary_table_figure(
        df, os.path.join(figures_dir, 'rq3_summary_table.png')
    )

    # Print to console
    print('\n' + '=' * 70)
    print('RQ3 RESULTS SUMMARY')
    print('=' * 70)
    print(df[['model'] + METRICS_TO_SHOW].to_string(index=False))
    print('=' * 70 + '\n')

    logger.info('Comparison complete.')


if __name__ == '__main__':
    main()
