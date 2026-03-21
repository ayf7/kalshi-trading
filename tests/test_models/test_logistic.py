"""Tests for the logistic regression model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kalshi_trader.models.logistic import LogisticBaseline


class TestLogisticBaseline:
    def _make_data(self, n=200):
        """Generate synthetic binary classification data."""
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        feature_names = [f"f{i}" for i in range(5)]
        return X, y, feature_names

    def test_fit_returns_metrics(self):
        X, y, names = self._make_data()
        model = LogisticBaseline()
        metrics = model.fit(X, y, names)

        assert "train_brier" in metrics
        assert "val_brier" in metrics
        assert "train_log_loss" in metrics
        assert "val_log_loss" in metrics
        assert 0 <= metrics["train_brier"] <= 1
        assert 0 <= metrics["val_brier"] <= 1

    def test_predict_proba_shape(self):
        X, y, names = self._make_data()
        model = LogisticBaseline()
        model.fit(X, y, names)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_before_fit_raises(self):
        model = LogisticBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros((5, 3)))

    def test_save_and_load(self):
        X, y, names = self._make_data()
        model = LogisticBaseline()
        model.fit(X, y, names)
        original_probs = model.predict_proba(X[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(path)

            loaded = LogisticBaseline()
            loaded.load(path)
            loaded_probs = loaded.predict_proba(X[:10])

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_feature_importance(self):
        X, y, names = self._make_data()
        model = LogisticBaseline()
        model.fit(X, y, names)

        importance = model.feature_importance()
        assert importance is not None
        assert set(importance.keys()) == set(names)

    def test_name(self):
        model = LogisticBaseline()
        assert model.name == "logistic_baseline"
