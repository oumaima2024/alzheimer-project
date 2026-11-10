"""
longitudinal_model.py
---------------------
LSTM / GRU-based model for longitudinal MCI-to-AD conversion prediction.

Architecture:
  Input: (batch, T, F)  – padded sequence of clinical features
  → Masking to ignore padded time steps
  → Bidirectional or unidirectional GRU / LSTM
  → Last valid hidden state
  → FC head → logit

References:
  - Nguyen et al. (2020) – LSTM for longitudinal ADNI prediction
  - Bhagwat et al. (2018) – RNN on ADNI longitudinal data
  - Ghazi et al. (2019) – deep ensemble for AD progression
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─── Dataset helper ────────────────────────────────────────────────────────────

class SequenceDataset(TensorDataset):
    """TensorDataset for (X_seq, masks, y) triples."""

    def __init__(self, X: np.ndarray, masks: np.ndarray, y: np.ndarray):
        X_t = torch.from_numpy(X.astype(np.float32))
        m_t = torch.from_numpy(masks.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        super().__init__(X_t, m_t, y_t)


# ─── Network ──────────────────────────────────────────────────────────────────

class _LSTMGRUNet(nn.Module):
    """
    Stacked LSTM or GRU with attention-free last-valid-step pooling.

    For each sample in the batch we use the hidden state at the LAST
    non-padded time step, then pass it through a classification head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        model_type: str = 'GRU',
    ):
        super().__init__()
        self.model_type = model_type.upper()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.GRU if self.model_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Input layer norm for stability
        self.input_norm = nn.LayerNorm(input_dim)

        fc_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)
        masks : (B, T) – 1 for real, 0 for padded

        Returns
        -------
        logits : (B,)
        """
        x = self.input_norm(x)

        # Compute sequence lengths from masks
        lengths = masks.sum(dim=1).long().clamp(min=1)

        # Pack padded sequence for efficient RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        # Gather the last valid hidden state per sample
        # lengths - 1 gives the 0-indexed position of last real step
        idx = (lengths - 1).clamp(min=0)
        # idx: (B,) → expand to (B, 1, hidden*dirs)
        idx_exp = idx.unsqueeze(1).unsqueeze(2).expand(
            -1, 1, output.size(2)
        )
        last_h = output.gather(1, idx_exp).squeeze(1)  # (B, H)

        logits = self.head(last_h).squeeze(-1)          # (B,)
        return logits


# ─── Training wrapper ─────────────────────────────────────────────────────────

class LongitudinalModel:
    """
    Wrapper around _LSTMGRUNet with fit / predict_proba interface
    matching the baseline models.

    References
    ----------
    Nguyen et al. (2020), Ding et al. (2019), Ghazi et al. (2019).
    """

    def __init__(
        self,
        input_dim: int = 10,
        model_type: str = 'GRU',
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 150,
        patience: int = 20,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        self.name = f'{model_type}_Longitudinal'
        self.net_: Optional[_LSTMGRUNet] = None
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses_: list = []
        self.val_losses_: list = []
        self.best_epoch_: int = 0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_net(self) -> _LSTMGRUNet:
        return _LSTMGRUNet(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            model_type=self.model_type,
        ).to(self.device_)

    def _make_loader(self, X, masks, y, shuffle=True) -> DataLoader:
        ds = SequenceDataset(X, masks, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _compute_loss(self, X, masks, y, criterion) -> float:
        self.net_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device_)
        m_t = torch.from_numpy(masks.astype(np.float32)).to(self.device_)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device_)
        with torch.no_grad():
            logits = self.net_(X_t, m_t)
            return criterion(logits, y_t).item()

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        masks_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        masks_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'LongitudinalModel':
        """
        Train the RNN.

        Parameters
        ----------
        X_train : (N, T, F)
        masks_train : (N, T)
        y_train : (N,)
        X_val, masks_val, y_val : optional validation set
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.input_dim = X_train.shape[2]
        self.net_ = self._make_net()

        # Class weight
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(self.device_)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, factor=0.5)


        loader = self._make_loader(X_train, masks_train, y_train, shuffle=True)

        best_val = float('inf')
        best_state = None
        no_improve = 0

        logger.info(f'Training {self.name} for up to {self.epochs} epochs '
                    f'on {self.device_}...')

        for epoch in range(self.epochs):
            self.net_.train()
            total_loss = 0.0

            for Xb, mb, yb in loader:
                Xb = Xb.to(self.device_)
                mb = mb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = self.net_(Xb, mb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train = total_loss / len(loader)
            self.train_losses_.append(avg_train)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, masks_val, y_val, criterion)
                self.val_losses_.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net_.state_dict().items()}
                    self.best_epoch_ = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch + 1} '
                                f'(best epoch: {self.best_epoch_})')
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f'Epoch {epoch+1:4d} | train={avg_train:.4f} '
                        f'val={val_loss:.4f}'
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch {epoch+1:4d} | train={avg_train:.4f}')

        if best_state is not None:
            self.net_.load_state_dict(best_state)
            logger.info(f'Restored best weights from epoch {self.best_epoch_}')

        return self

    def predict_proba(self, X: np.ndarray,
                      masks: np.ndarray) -> np.ndarray:
        """Return predicted probabilities of class 1 (Converter)."""
        self.net_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device_)
        m_t = torch.from_numpy(masks.astype(np.float32)).to(self.device_)
        with torch.no_grad():
            logits = self.net_(X_t, m_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, masks: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X, masks) >= threshold).astype(int)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.net_.state_dict(),
            'hparams': {
                'input_dim': self.input_dim,
                'model_type': self.model_type,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
            },
        }, path)
        logger.info(f'Saved {self.name} to {path}')

    @classmethod
    def load(cls, path: str) -> 'LongitudinalModel':
        ck = torch.load(path, map_location='cpu')
        hp = ck['hparams']
        obj = cls(**hp)
        obj.net_ = obj._make_net()
        obj.net_.load_state_dict(ck['state_dict'])
        obj.net_.eval()
        return obj
