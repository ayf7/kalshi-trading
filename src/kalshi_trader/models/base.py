from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ProbabilityModel(ABC):
    """Estimates the true probability of a YES outcome for a binary market."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        validation_split: float = 0.2,
    ) -> dict:
        """
        Train the model.

        Returns a dict of training metrics:
            train_brier, val_brier, train_log_loss, val_log_loss,
            train_accuracy, val_accuracy
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(YES) for each row. Shape (n_samples,), values in [0, 1]."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...

    def feature_importance(self) -> dict[str, float] | None:
        """Return feature name -> importance mapping, if supported."""
        return None
