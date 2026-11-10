"""
train_longitudinal.py
---------------------
Training pipeline for the longitudinal LSTM/GRU model.

Uses the SAME cohort + split as train_baseline.py but builds 3-D sequence
tensors (N, T, F) from all available visits per participant.

Usage (Windows):
  python scripts/train_longitudinal.py --config config.yaml
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_all, load_dxsum
from src.data.validation import validate_merged_data, validate_labels, validate_split_disjoint
from src.data.cohort import define_cohort
from src.data.preprocessing import (
    FeaturePipeline, build_baseline_features,
    build_sequence_features, get_available_features, ALL_FEATURES
)
from src.data.splitting import (
    participant_split, filter_arrays_by_rids
)
from src.models.longitudinal_model import LongitudinalModel
from src.evaluation.metrics import (
    compute_metrics, bootstrap_ci, get_roc_curve, get_pr_curve
)
from src.ops.tracking import start_run, log_config, log_metrics, log_artifact, end_run
from src.evaluation.visualizations import (
    plot_roc_curves, plot_pr_curves, plot_metric_comparison,
    plot_confusion_matrix, plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s – %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('train_longitudinal')


def parse_args():
    p = argparse.ArgumentParser(
        description='Train longitudinal GRU/LSTM model for RQ3'
    )
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--results-dir', default=None)
    p.add_argument('--model-type', default=None,
                   choices=['LSTM', 'GRU'],
                   help='Override model type from config')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    long_cfg = cfg.get('longitudinal', {})
    model_type = args.model_type or long_cfg.get('model_type', 'GRU')
    start_run('ADNI_RQ3', f'{model_type}_longitudinal')
    log_config(cfg)
    try:
        import mlflow
        mlflow.log_param('model_type', model_type)
        mlflow.log_param('pipeline_type', 'longitudinal')
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

    max_seq_len = long_cfg.get('max_seq_len', 10)

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
        logger.error('Cohort too small. Check data in data/raw/.')
        sys.exit(1)

    # ── 3. Split ──────────────────────────────────────────────────────────────
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

    # ── 4. Feature sequences ──────────────────────────────────────────────────
    logger.info('=== Step 4: Building sequence feature arrays ===')
    feat_cols = get_available_features(long_df, ALL_FEATURES)

    # Build baseline features first so the pipeline can identify columns that
    # are entirely absent in the training split.
    X_bl, y_bl = build_baseline_features(long_df, labels_df, feat_cols)
    X_train_bl = X_bl.loc[X_bl.index.isin(train_rids)]

    pipeline = FeaturePipeline()
    pipeline.fit(X_train_bl)

    # Use the same observed feature schema for sequences; otherwise an
    # all-missing training column causes a shape mismatch during scaling.
    X_seq, masks, y_seq, all_rids = build_sequence_features(
        long_df, labels_df, pipeline.feature_names_, max_seq_len=max_seq_len
    )

    # Scale sequences
    X_seq_scaled = pipeline.transform_sequences(X_seq)

    # Split sequences by participant
    X_train, y_train = filter_arrays_by_rids(X_seq_scaled, y_seq,
                                              all_rids, train_rids)
    m_train, _ = filter_arrays_by_rids(masks, y_seq, all_rids, train_rids)

    X_val_arr, y_val = filter_arrays_by_rids(X_seq_scaled, y_seq,
                                              all_rids, val_rids)
    m_val, _ = filter_arrays_by_rids(masks, y_seq, all_rids, val_rids)

    X_test_arr, y_test = filter_arrays_by_rids(X_seq_scaled, y_seq,
                                               all_rids, test_rids)
    m_test, _ = filter_arrays_by_rids(masks, y_seq, all_rids, test_rids)

    logger.info(
        f'Train: {X_train.shape[0]} | Val: {X_val_arr.shape[0]} | '
        f'Test: {X_test_arr.shape[0]} | T={max_seq_len}, F={X_train.shape[2]}'
    )

    # ── 5. Train model ────────────────────────────────────────────────────────
    logger.info(f'=== Step 5: Training {model_type} model ===')
    model = LongitudinalModel(
        input_dim=X_train.shape[2],
        model_type=model_type,
        hidden_size=long_cfg.get('hidden_size', 64),
        num_layers=long_cfg.get('num_layers', 2),
        dropout=long_cfg.get('dropout', 0.3),
        bidirectional=long_cfg.get('bidirectional', False),
        lr=long_cfg.get('lr', 0.001),
        batch_size=long_cfg.get('batch_size', 32),
        epochs=long_cfg.get('epochs', 150),
        patience=long_cfg.get('patience', 20),
        random_state=seed,
    )
    model.fit(X_train, m_train, y_train,
              X_val_arr, m_val, y_val)
    model.save(os.path.join(models_dir, f'{model.name}.pt'))

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info('=== Step 6: Evaluating on test set ===')
    y_prob = model.predict_proba(X_test_arr, m_test)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    ci = bootstrap_ci(y_test, y_prob,
                      n_iterations=cfg['evaluation']['bootstrap_iterations'],
                      random_state=seed)

    logger.info(
        f'{model.name}: AUC={metrics["roc_auc"]:.3f} | '
        f'AP={metrics["average_precision"]:.3f} | '
        f'BalAcc={metrics["balanced_accuracy"]:.3f} | '
        f'Sens={metrics["sensitivity"]:.3f} | '
        f'Spec={metrics["specificity"]:.3f}'
    )

    # ── 7. Figures ────────────────────────────────────────────────────────────
    logger.info('=== Step 7: Generating figures ===')
    fpr, tpr, _ = get_roc_curve(y_test, y_prob)
    prec, recall, _ = get_pr_curve(y_test, y_prob)

    roc_data = {model.name: {'fpr': fpr, 'tpr': tpr, 'auc': metrics['roc_auc']}}
    pr_data  = {model.name: {'precision': prec, 'recall': recall,
                              'ap': metrics['average_precision']}}

    plot_roc_curves(roc_data,
                    os.path.join(figures_dir, f'roc_{model.name}.png'),
                    title=f'ROC – {model.name} (RQ3)')
    plot_pr_curves(pr_data,
                   os.path.join(figures_dir, f'pr_{model.name}.png'),
                   baseline_rate=y_test.mean(),
                   title=f'PR – {model.name} (RQ3)')

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion_matrix(
        cm, model.name,
        os.path.join(figures_dir, f'cm_{model.name}.png')
    )

    if model.train_losses_:
        plot_training_curves(
            model.train_losses_, model.val_losses_,
            model.name,
            os.path.join(figures_dir, f'loss_{model.name}.png'),
        )

    # ── 8. Save results ───────────────────────────────────────────────────────
    out = {
        model.name: {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
            for k, v in metrics.items()
        }
    }
    out[model.name]['ci'] = ci

    longitudinal_path = os.path.join(results_dir, 'longitudinal_results.json')
    prior = {}
    if os.path.exists(longitudinal_path):
        with open(longitudinal_path) as f:
            prior = json.load(f)
    prior.update(out)
    with open(longitudinal_path, 'w') as f:
        json.dump(prior, f, indent=2)

    pd.DataFrame([
        {'model': name,
         **{k: v for k, v in values.items()
            if k not in ('ci', 'tp', 'fp', 'tn', 'fn')}}
        for name, values in prior.items()
    ]).to_csv(os.path.join(results_dir, 'longitudinal_metrics.csv'), index=False)

    log_metrics(metrics, prefix=model.name)
    log_artifact(os.path.join(results_dir, 'longitudinal_results.json'))
    log_artifact(os.path.join(results_dir, 'longitudinal_metrics.csv'))
    log_artifact(os.path.join(models_dir, f'{model.name}.pt'))
    end_run()
    logger.info('=== Longitudinal training complete ===')
    return out


if __name__ == '__main__':
    main()
