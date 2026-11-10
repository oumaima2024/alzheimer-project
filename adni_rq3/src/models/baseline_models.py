"""
baseline_models.py
------------------
Scikit-learn baseline classifiers for T0-only (cross-sectional) prediction
of MCI-to-AD conversion.

Models:
  - LogisticRegressionClassifier
  - RandomForestClassifier
  - MLPClassifier (PyTorch-based for consistency with longitudinal model)

All models expose a common interface:
  fit(X_train, y_train, X_val=None, y_val=None)
  predict_proba(X) -> np.ndarray
  predict(X) -> np.ndarray
  save(path) / load(path)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─── Logistic Regression ──────────────────────────────────────────────────────

class LogisticRegressionModel:
    """
    Balanced logistic regression with L2 regularisation.

    References:
      - Moradi et al. (2015) – logistic regression on ADNI for MCI conversion
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 class_weight: str = 'balanced', solver: str = 'lbfgs',
                 random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            random_state=random_state,
        )
        self.name = 'LogisticRegression'

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val=None, y_val=None) -> 'LogisticRegressionModel':
        logger.info(f'Fitting {self.name} on {X_train.shape[0]} samples...')
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f'Saved {self.name} to {path}')

    @classmethod
    def load(cls, path: str) -> 'LogisticRegressionModel':
        obj = cls()
        with open(path, 'rb') as f:
            obj.model = pickle.load(f)
        return obj


# ─── Random Forest ────────────────────────────────────────────────────────────

class RandomForestModel:
    """
    Random Forest with class-balancing and probability calibration.

    References:
      - Grassi et al. (2019) – ensemble methods for ADNI MCI conversion
    """

    def __init__(self, n_estimators: int = 300, max_depth=None,
                 min_samples_leaf: int = 2, class_weight: str = 'balanced',
                 random_state: int = 42, n_jobs: int = -1):
        self.model = _RFC(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.name = 'RandomForest'
        self._calibrated = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val=None, y_val=None) -> 'RandomForestModel':
        logger.info(f'Fitting {self.name} on {X_train.shape[0]} samples...')
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f'Saved {self.name} to {path}')

    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        obj = cls()
        with open(path, 'rb') as f:
            obj.model = pickle.load(f)
        return obj


# ─── MLP (PyTorch) ───────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel:
    """
    Multi-Layer Perceptron (feed-forward) for T0 classification.

    Uses BCE loss with class-weight correction for imbalanced data.

    References:
      - Ding et al. (2019) – deep learning for ADNI MCI conversion prediction
    """

    def __init__(self, input_dim: int = 10,
                 hidden_layer_sizes=(128, 64, 32),
                 dropout: float = 0.3,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 15,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.name = 'MLP'
        self.net_: Optional[_MLPNet] = None
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses_: list = []
        self.val_losses_: list = []

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool) -> DataLoader:
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'MLPModel':

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.input_dim = X_train.shape[1]
        self.net_ = _MLPNet(self.input_dim,
                            self.hidden_layer_sizes,
                            self.dropout).to(self.device_)

        # Class-weighted loss
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32
        ).to(self.device_)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        no_improve = 0

        logger.info(f'Training {self.name} for up to {self.epochs} epochs...')

        for epoch in range(self.epochs):
            self.net_.train()
            total_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()
                logits = self.net_(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train = total_loss / len(train_loader)
            self.train_losses_.append(avg_train)

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion)
                self.val_losses_.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
            logger.info('Restored best model weights.')

        return self

    def _eval_loss(self, X: np.ndarray, y: np.ndarray,
                   criterion) -> float:
        self.net_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device_)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device_)
        with torch.no_grad():
            logits = self.net_(X_t)
            return criterion(logits, y_t).item()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device_)
        with torch.no_grad():
            logits = self.net_(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.net_.state_dict(),
            'input_dim': self.input_dim,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'dropout': self.dropout,
        }, path)
        logger.info(f'Saved {self.name} to {path}')

    @classmethod
    def load(cls, path: str) -> 'MLPModel':
        ck = torch.load(path, map_location='cpu')
        obj = cls(input_dim=ck['input_dim'],
                  hidden_layer_sizes=ck['hidden_layer_sizes'],
                  dropout=ck['dropout'])
        obj.net_ = _MLPNet(ck['input_dim'],
                           ck['hidden_layer_sizes'],
                           ck['dropout'])
        obj.net_.load_state_dict(ck['state_dict'])
        obj.net_.eval()
        return obj
