"""Naive baseline models for comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kalshi_trader.models.base import ProbabilityModel


class RandomModel(ProbabilityModel):
    """Predicts 0.5 for everything (coin flip)."""

    @property
    def name(self) -> str:
        return "random"

    def fit(self, X, y, feature_names, validation_split=0.2):
        self._feature_names = feature_names
        return {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], 0.5)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class MostLikelyModel(ProbabilityModel):
    """Predicts 1.0 if mid_price > 0.5, else 0.0 (always bet the favorite)."""

    @property
    def name(self) -> str:
        return "most_likely"

    def fit(self, X, y, feature_names, validation_split=0.2):
        self._feature_names = feature_names
        self._mid_idx = feature_names.index("mid_price") if "mid_price" in feature_names else 0
        return {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mid = X[:, self._mid_idx]
        return np.where(mid >= 0.5, 0.99, 0.01)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class ContrarianModel(ProbabilityModel):
    """Predicts opposite of market (always bet the underdog)."""

    @property
    def name(self) -> str:
        return "contrarian"

    def fit(self, X, y, feature_names, validation_split=0.2):
        self._feature_names = feature_names
        self._mid_idx = feature_names.index("mid_price") if "mid_price" in feature_names else 0
        return {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mid = X[:, self._mid_idx]
        return np.where(mid < 0.5, 0.99, 0.01)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class MarketImpliedModel(ProbabilityModel):
    """Uses market mid-price directly as the probability (no ML)."""

    @property
    def name(self) -> str:
        return "market_implied"

    def fit(self, X, y, feature_names, validation_split=0.2):
        self._feature_names = feature_names
        self._mid_idx = feature_names.index("mid_price") if "mid_price" in feature_names else 0
        return {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X[:, self._mid_idx], 0.01, 0.99)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
