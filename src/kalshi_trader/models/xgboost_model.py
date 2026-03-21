from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from kalshi_trader.models.base import ProbabilityModel


class XGBoostModel(ProbabilityModel):
    """XGBoost gradient-boosted tree model for probability estimation."""

    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_child_weight: int = 3,
    ):
        self._params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
        }
        self._model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            **self._params,
        )
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "xgboost"

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

        fit_kwargs = {"verbose": False}
        if has_val:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._fitted = True

        train_probs = self._model.predict_proba(X_train)[:, 1]

        metrics = {
            "train_brier": brier_score_loss(y_train, train_probs),
            "train_log_loss": log_loss(y_train, train_probs),
            "train_accuracy": self._model.score(X_train, y_train),
        }

        if has_val:
            val_probs = self._model.predict_proba(X_val)[:, 1]
            metrics["val_brier"] = brier_score_loss(y_val, val_probs)
            metrics["val_log_loss"] = log_loss(y_val, val_probs)
            metrics["val_accuracy"] = self._model.score(X_val, y_val)
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "feature_names": self._feature_names,
                    "params": self._params,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._params = data["params"]
        self._fitted = True

    def feature_importance(self) -> dict[str, float] | None:
        if not self._fitted:
            return None
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))
