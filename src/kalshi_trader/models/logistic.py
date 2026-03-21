from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from kalshi_trader.models.base import ProbabilityModel


class LogisticBaseline(ProbabilityModel):
    """Logistic regression baseline model with standardized features."""

    def __init__(self, C: float = 1.0):
        self._C = C
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, penalty="l2", max_iter=1000, random_state=42)
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "logistic_baseline"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        validation_split: float = 0.2,
    ) -> dict:
        self._feature_names = feature_names
        n = len(X)

        has_val = validation_split > 0 and n > 1
        if has_val:
            split_idx = int(n * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Fit scaler on training data only
        X_train_scaled = self._scaler.fit_transform(X_train)

        self._model.fit(X_train_scaled, y_train)
        self._fitted = True

        # Compute metrics
        train_probs = self._model.predict_proba(X_train_scaled)[:, 1]

        metrics = {
            "train_brier": brier_score_loss(y_train, train_probs),
            "train_log_loss": log_loss(y_train, train_probs),
            "train_accuracy": self._model.score(X_train_scaled, y_train),
        }

        if has_val:
            X_val_scaled = self._scaler.transform(X_val)
            val_probs = self._model.predict_proba(X_val_scaled)[:, 1]
            metrics["val_brier"] = brier_score_loss(y_val, val_probs)
            metrics["val_log_loss"] = log_loss(y_val, val_probs)
            metrics["val_accuracy"] = self._model.score(X_val_scaled, y_val)
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self._scaler,
                    "model": self._model,
                    "feature_names": self._feature_names,
                    "C": self._C,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._scaler = data["scaler"]
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._C = data["C"]
        self._fitted = True

    def feature_importance(self) -> dict[str, float] | None:
        if not self._fitted:
            return None
        coefs = self._model.coef_[0]
        return dict(zip(self._feature_names, coefs.tolist()))
