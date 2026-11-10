"""
train_baseline.py
-----------------
Training pipeline for T0-only (cross-sectional) baseline models.

Workflow:
  1. Load and merge ADNI data (CDR + ITEM)
  2. Define MCI cohort and assign labels (Converter / Stable)
  3. Participant-level train / val / test split (no leakage)
  4. Feature imputation + standardisation (fit on train only)
  5. Train: Logistic Regression, Random Forest, MLP
  6. Evaluate on held-out test set
  7. Save models + metrics + figures

Usage (Windows):
  python scripts/train_baseline.py --config config.yaml
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
import yaml
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_all, load_dxsum
from src.data.validation import validate_merged_data, validate_labels, validate_split_disjoint
from src.data.cohort import define_cohort
from src.data.preprocessing import (
    FeaturePipeline, build_baseline_features, get_available_features, ALL_FEATURES
)
from src.data.splitting import participant_split, filter_by_rids
from src.models.baseline_models import (
    LogisticRegressionModel, RandomForestModel, MLPModel
)
from src.evaluation.metrics import (
    compute_metrics, bootstrap_ci, get_roc_curve, get_pr_curve
)
from src.ops.tracking import start_run, log_config, log_metrics, log_artifact, end_run
from src.evaluation.visualizations import (
    plot_roc_curves, plot_pr_curves, plot_metric_comparison,
    plot_feature_importance, plot_confusion_matrix, plot_training_curves,
    plot_cohort_summary, plot_visit_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s – %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('train_baseline')


def parse_args():
    p = argparse.ArgumentParser(description='Train baseline T0 models for RQ3')
    p.add_argument('--config', default='config.yaml',
                   help='Path to config.yaml')
    p.add_argument('--data-dir', default=None,
                   help='Override data.raw_dir from config')
    p.add_argument('--results-dir', default=None,
                   help='Override data.results_dir from config')
    p.add_argument('--no-shap', action='store_true',
                   help='Skip SHAP computation (faster)')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    start_run('ADNI_RQ3', 'baseline_models')
    log_config(cfg)
    try:
        import mlflow
        mlflow.log_param('pipeline_type', 'baseline_T0')
        mlflow.log_param('data_source', 'real_ADNI')
    except Exception:
        pass

    data_dir = args.data_dir or cfg['data']['raw_dir']
    results_dir = args.results_dir or cfg['data']['results_dir']
    figures_dir = os.path.join(results_dir, 'figures')
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    seed = cfg.get('seed', 42)
    np.random.seed(seed)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info('=== Step 1: Loading ADNI data ===')
    merged_df = load_all(data_dir, cfg)
    validate_merged_data(merged_df)

    dxsum_path = str(Path(data_dir) / 'DXSUM_PDXCONV_ADNIALL.csv')
    dxsum_df = load_dxsum(dxsum_path)

    # ── 2. Cohort definition ──────────────────────────────────────────────────
    logger.info('=== Step 2: Defining MCI cohort ===')
    cohort_cfg = cfg.get('cohort', {})
    labels_df, long_df = define_cohort(
        merged_df,
        dxsum_df=dxsum_df,
        min_visits=cohort_cfg.get('min_visits', 2),
        require_followup_months=cohort_cfg.get('require_followup_months', 6),
    )
    validate_labels(labels_df)

    if len(labels_df) < 20:
        logger.error(
            f'Only {len(labels_df)} participants found in cohort. '
            'Check that CDR data is in data/raw/ directory.'
        )
        sys.exit(1)

    # ── 3. Train / val / test split ───────────────────────────────────────────
    logger.info('=== Step 3: Participant-level split ===')
    split_cfg = cfg.get('split', {})
    train_rids, val_rids, test_rids = participant_split(
        labels_df,
        test_size=split_cfg.get('test_size', 0.20),
        val_size=split_cfg.get('val_size', 0.10),
        random_state=split_cfg.get('random_state', seed),
        stratify=split_cfg.get('stratify', True),
    )
    validate_split_disjoint(train_rids, val_rids, test_rids)

    # ── 4. Feature extraction ──────────────────────────────────────────────────
    logger.info('=== Step 4: Building baseline feature matrices ===')
    feat_cols = (
        get_available_features(long_df, ALL_FEATURES)
    )

    X_all, y_all = build_baseline_features(long_df, labels_df, feat_cols)

    # Filter by split
    X_train = X_all.loc[X_all.index.isin(train_rids)]
    y_train = y_all.loc[y_all.index.isin(train_rids)]
    X_val   = X_all.loc[X_all.index.isin(val_rids)]
    y_val   = y_all.loc[y_all.index.isin(val_rids)]
    X_test  = X_all.loc[X_all.index.isin(test_rids)]
    y_test  = y_all.loc[y_all.index.isin(test_rids)]

    # Pipeline (fit on train only)
    pipeline = FeaturePipeline()
    X_train_np = pipeline.fit_transform(X_train)
    X_val_np   = pipeline.transform(X_val)
    X_test_np  = pipeline.transform(X_test)

    y_train_np = y_train.values
    y_val_np   = y_val.values
    y_test_np  = y_test.values

    logger.info(
        f'Train: {X_train_np.shape[0]} | Val: {X_val_np.shape[0]} | '
        f'Test: {X_test_np.shape[0]} | Features: {X_train_np.shape[1]}'
    )

    # ── 5. Train models ───────────────────────────────────────────────────────
    logger.info('=== Step 5: Training models ===')
    bl_cfg = cfg.get('baseline', {})
    mlp_cfg = bl_cfg.get('mlp', {})

    models = {
        'LogisticRegression': LogisticRegressionModel(
            **{k: v for k, v in bl_cfg.get('logistic_regression', {}).items()
               if k != 'random_state'},
            random_state=seed,
        ),
        'RandomForest': RandomForestModel(
            **{k: v for k, v in bl_cfg.get('random_forest', {}).items()
               if k not in ('random_state', 'n_jobs')},
            random_state=seed,
        ),
        'MLP': MLPModel(
            input_dim=X_train_np.shape[1],
            hidden_layer_sizes=mlp_cfg.get('hidden_layer_sizes', [128, 64, 32]),
            dropout=mlp_cfg.get('dropout', 0.3),
            lr=mlp_cfg.get('lr', 0.001),
            batch_size=mlp_cfg.get('batch_size', 32),
            epochs=mlp_cfg.get('epochs', 100),
            patience=mlp_cfg.get('patience', 15),
            random_state=seed,
        ),
    }

    for name, model in models.items():
        model.fit(X_train_np, y_train_np, X_val_np, y_val_np)
        model.save(os.path.join(models_dir, f'{name}.pkl'))

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info('=== Step 6: Evaluating on test set ===')
    all_test_results = {}
    all_ci_results = {}
    roc_data = {}
    pr_data = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test_np)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        ci = bootstrap_ci(y_test_np, y_prob,
                          n_iterations=cfg['evaluation']['bootstrap_iterations'],
                          random_state=seed)

        all_test_results[name] = metrics
        all_ci_results[name] = ci

        fpr, tpr, _ = get_roc_curve(y_test_np, y_prob)
        prec, recall, _ = get_pr_curve(y_test_np, y_prob)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['roc_auc']}
        pr_data[name] = {'precision': prec, 'recall': recall,
                          'ap': metrics['average_precision']}

        # Per-model confusion matrix
        cm = confusion_matrix(y_test_np, y_pred, labels=[0, 1])
        plot_confusion_matrix(
            cm, name,
            os.path.join(figures_dir, f'cm_{name}.png')
        )

        logger.info(
            f'{name}: AUC={metrics["roc_auc"]:.3f} | '
            f'AP={metrics["average_precision"]:.3f} | '
            f'BalAcc={metrics["balanced_accuracy"]:.3f} | '
            f'Sens={metrics["sensitivity"]:.3f} | '
            f'Spec={metrics["specificity"]:.3f}'
        )

        # MLP training curves
        if hasattr(model, 'train_losses_') and model.train_losses_:
            plot_training_curves(
                model.train_losses_, model.val_losses_,
                name, os.path.join(figures_dir, f'loss_{name}.png')
            )

    # ── 7. Figures ────────────────────────────────────────────────────────────
    logger.info('=== Step 7: Generating figures ===')
    plot_roc_curves(roc_data, os.path.join(figures_dir, 'roc_baseline.png'),
                    title='ROC – Baseline T0 Models (RQ3)')
    plot_pr_curves(pr_data,
                   os.path.join(figures_dir, 'pr_baseline.png'),
                   baseline_rate=y_test_np.mean(),
                   title='PR – Baseline T0 Models (RQ3)')

    metric_keys = ['roc_auc', 'average_precision', 'balanced_accuracy',
                   'f1', 'sensitivity', 'specificity']
    plot_metric_comparison(
        all_ci_results, metric_keys,
        os.path.join(figures_dir, 'metric_comparison_baseline.png'),
        title='Baseline Model Comparison – Test Set (95% CI)',
    )

    # RF feature importance
    rf_model = models['RandomForest']
    imp = rf_model.feature_importances()
    plot_feature_importance(
        pipeline.feature_names_, imp,
        os.path.join(figures_dir, 'feature_importance_rf.png'),
    )

    # Cohort summary
    n_total = merged_df['RID'].nunique()
    n_conv = int(labels_df['LABEL'].sum())
    n_stable = int((labels_df['LABEL'] == 0).sum())
    plot_cohort_summary(
        total=n_total,
        mci_baseline=len(labels_df),
        n_converters=n_conv,
        n_stable=n_stable,
        n_train=len(train_rids),
        n_val=len(val_rids),
        n_test=len(test_rids),
        save_path=os.path.join(figures_dir, 'cohort_summary.png'),
    )

    # Visit distribution
    plot_visit_distribution(
        long_df, labels_df,
        os.path.join(figures_dir, 'visit_distribution.png'),
    )

    # ── 8. Save results ───────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, 'baseline_results.json')
    serializable = {}
    for model_name, m in all_test_results.items():
        serializable[model_name] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
            for k, v in m.items()
        }
        serializable[model_name]['ci'] = all_ci_results[model_name]

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f'Results saved to {out_path}')

    # Also save as CSV for easy inspection
    rows = []
    for model_name, m in all_test_results.items():
        row = {'model': model_name}
        row.update({k: v for k, v in m.items() if k not in ('tp', 'fp', 'tn', 'fn')})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, 'baseline_metrics.csv'), index=False
    )

    for model_name, model_metrics in all_test_results.items():
        log_metrics(model_metrics, prefix=model_name)
    log_artifact(out_path)
    log_artifact(os.path.join(results_dir, 'baseline_metrics.csv'))
    for model_name in models:
        log_artifact(os.path.join(models_dir, f'{model_name}.pkl'))
    end_run()
    logger.info('=== Baseline training complete ===')
    return serializable


if __name__ == '__main__':
    main()
