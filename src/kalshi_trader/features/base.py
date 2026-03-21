from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class FeatureExtractor(ABC):
    """Extracts features from raw data for a single market at a single point in time."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        ...

    @abstractmethod
    def extract(
        self,
        ticker: str,
        as_of_ts: int,
        snapshots: pd.DataFrame,
        news: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Return a 1-D float array of len(self.feature_names)."""
        ...


class FeaturePipeline:
    """Composes multiple FeatureExtractors into a single feature vector."""

    def __init__(self, extractors: list[FeatureExtractor]):
        self.extractors = extractors

    @property
    def feature_names(self) -> list[str]:
        return [name for ext in self.extractors for name in ext.feature_names]

    def extract(self, ticker: str, as_of_ts: int, **kwargs) -> np.ndarray:
        parts = [ext.extract(ticker, as_of_ts, **kwargs) for ext in self.extractors]
        return np.concatenate(parts)
