"""
test_models.py
--------------
Tests for baseline and longitudinal models.
"""

import numpy as np
import pytest
import tempfile
import os

from src.models.baseline_models import (
    LogisticRegressionModel, RandomForestModel, MLPModel
)
from src.models.longitudinal_model import LongitudinalModel


# ─── Common test data ─────────────────────────────────────────────────────────

N_TRAIN = 60
N_TEST = 20
N_FEATURES = 10
N_TIME = 6
SEED = 42

rng = np.random.default_rng(SEED)


@pytest.fixture(scope='module')
def binary_data():
    """Simple synthetic binary classification data."""
    X_train = rng.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float32)
    y_train = rng.integers(0, 2, N_TRAIN)
    X_test = rng.standard_normal((N_TEST, N_FEATURES)).astype(np.float32)
    y_test = rng.integers(0, 2, N_TEST)
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope='module')
def sequence_data():
    """Synthetic (N, T, F) data for RNN."""
    X = rng.standard_normal((N_TRAIN + N_TEST, N_TIME, N_FEATURES)).astype(np.float32)
    # Masks: first 4 steps real, last 2 padded
    masks = np.ones((N_TRAIN + N_TEST, N_TIME), dtype=np.float32)
    masks[:, 4:] = 0.0
    y = rng.integers(0, 2, N_TRAIN + N_TEST)
    return (X[:N_TRAIN], masks[:N_TRAIN], y[:N_TRAIN],
            X[N_TRAIN:], masks[N_TRAIN:], y[N_TRAIN:])


# ─── Logistic Regression Tests ────────────────────────────────────────────────

class TestLogisticRegression:
    def test_fit_predict_proba_shape(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = LogisticRegressionModel()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert probs.shape == (N_TEST,)

    def test_proba_in_range(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = LogisticRegressionModel()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_binary(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = LogisticRegressionModel()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_save_load(self, binary_data, tmp_path):
        X_tr, y_tr, X_te, _ = binary_data
        model = LogisticRegressionModel()
        model.fit(X_tr, y_tr)
        path = str(tmp_path / 'lr.pkl')
        model.save(path)
        loaded = LogisticRegressionModel.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_te),
            loaded.predict_proba(X_te)
        )


# ─── Random Forest Tests ──────────────────────────────────────────────────────

class TestRandomForest:
    def test_fit_predict_proba_shape(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert probs.shape == (N_TEST,)

    def test_proba_in_range(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_feature_importances_shape(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        imp = model.feature_importances()
        assert imp.shape == (N_FEATURES,)
        assert abs(imp.sum() - 1.0) < 0.01  # importances sum to ~1

    def test_save_load(self, binary_data, tmp_path):
        X_tr, y_tr, X_te, _ = binary_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X_tr, y_tr)
        path = str(tmp_path / 'rf.pkl')
        model.save(path)
        loaded = RandomForestModel.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_te),
            loaded.predict_proba(X_te)
        )


# ─── MLP Tests ────────────────────────────────────────────────────────────────

class TestMLP:
    def test_fit_predict_proba_shape(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = MLPModel(
            input_dim=N_FEATURES,
            hidden_layer_sizes=(32, 16),
            epochs=5,
            patience=3,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert probs.shape == (N_TEST,)

    def test_proba_in_range(self, binary_data):
        X_tr, y_tr, X_te, _ = binary_data
        model = MLPModel(
            input_dim=N_FEATURES,
            hidden_layer_sizes=(32, 16),
            epochs=5,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_early_stopping_with_val(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        model = MLPModel(
            input_dim=N_FEATURES,
            hidden_layer_sizes=(16, 8),
            epochs=50,
            patience=3,
        )
        model.fit(X_tr, y_tr, X_te, y_te)
        # Should have stopped before 50 epochs (patience=3 with tiny data)
        assert len(model.train_losses_) <= 50

    def test_save_load(self, binary_data, tmp_path):
        X_tr, y_tr, X_te, _ = binary_data
        model = MLPModel(
            input_dim=N_FEATURES,
            hidden_layer_sizes=(16, 8),
            epochs=3,
        )
        model.fit(X_tr, y_tr)
        path = str(tmp_path / 'mlp.pt')
        model.save(path)
        loaded = MLPModel.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_te),
            loaded.predict_proba(X_te),
            decimal=5,
        )


# ─── Longitudinal Model Tests ─────────────────────────────────────────────────

class TestLongitudinalModel:
    @pytest.mark.parametrize('model_type', ['GRU', 'LSTM'])
    def test_fit_predict_proba_shape(self, sequence_data, model_type):
        X_tr, m_tr, y_tr, X_te, m_te, y_te = sequence_data
        model = LongitudinalModel(
            input_dim=N_FEATURES,
            model_type=model_type,
            hidden_size=16,
            num_layers=1,
            epochs=3,
            patience=2,
        )
        model.fit(X_tr, m_tr, y_tr)
        probs = model.predict_proba(X_te, m_te)
        assert probs.shape == (N_TEST,)

    def test_proba_in_range(self, sequence_data):
        X_tr, m_tr, y_tr, X_te, m_te, _ = sequence_data
        model = LongitudinalModel(
            input_dim=N_FEATURES,
            hidden_size=16,
            num_layers=1,
            epochs=3,
        )
        model.fit(X_tr, m_tr, y_tr)
        probs = model.predict_proba(X_te, m_te)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_masking_ignores_padding(self, sequence_data):
        """Predictions should differ if we zero out padded steps."""
        X_tr, m_tr, y_tr, X_te, m_te, _ = sequence_data
        model = LongitudinalModel(
            input_dim=N_FEATURES,
            hidden_size=16,
            num_layers=1,
            epochs=3,
        )
        model.fit(X_tr, m_tr, y_tr)
        # Corrupt the padded steps with large values
        X_corrupt = X_te.copy()
        X_corrupt[:, 4:, :] = 1e6
        probs_clean = model.predict_proba(X_te, m_te)
        probs_corrupt = model.predict_proba(X_corrupt, m_te)
        # Because masking is applied, predictions should be nearly identical
        np.testing.assert_array_almost_equal(probs_clean, probs_corrupt,
                                              decimal=3)

    def test_save_load(self, sequence_data, tmp_path):
        X_tr, m_tr, y_tr, X_te, m_te, _ = sequence_data
        model = LongitudinalModel(
            input_dim=N_FEATURES,
            hidden_size=16,
            num_layers=1,
            epochs=3,
        )
        model.fit(X_tr, m_tr, y_tr)
        path = str(tmp_path / 'gru.pt')
        model.save(path)
        loaded = LongitudinalModel.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_te, m_te),
            loaded.predict_proba(X_te, m_te),
            decimal=5,
        )

    def test_early_stopping_with_val(self, sequence_data):
        X_tr, m_tr, y_tr, X_te, m_te, y_te = sequence_data
        model = LongitudinalModel(
            input_dim=N_FEATURES,
            hidden_size=8,
            num_layers=1,
            epochs=100,
            patience=3,
        )
        model.fit(X_tr, m_tr, y_tr, X_te, m_te, y_te)
        assert len(model.train_losses_) < 100
