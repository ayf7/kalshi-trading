from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kalshi_trader.features.base import FeatureExtractor


class NewsVolumeFeatures(FeatureExtractor):
    """Features derived from news article volume and coverage."""

    @property
    def name(self) -> str:
        return "news_volume"

    @property
    def feature_names(self) -> list[str]:
        return [
            "article_count_1h",
            "article_count_24h",
            "volume_change",
        ]

    def extract(
        self,
        ticker: str,
        as_of_ts: int,
        snapshots: pd.DataFrame,
        news: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> np.ndarray:
        features = np.zeros(len(self.feature_names), dtype=np.float64)

        if news is None or news.empty:
            return features

        if "fetched_ts" not in news.columns:
            return features

        ts_col = news["fetched_ts"]

        # Article counts in windows
        features[0] = float((ts_col >= as_of_ts - 3600).sum())
        features[1] = float((ts_col >= as_of_ts - 86400).sum())

        # Volume change: articles in last 6h vs previous 6h
        count_6h = float(((ts_col >= as_of_ts - 21600) & (ts_col <= as_of_ts)).sum())
        count_prev_6h = float(
            ((ts_col >= as_of_ts - 43200) & (ts_col < as_of_ts - 21600)).sum()
        )
        if count_prev_6h > 0:
            features[2] = count_6h / count_prev_6h
        elif count_6h > 0:
            features[2] = 2.0  # signal spike from zero

        return features


class NewsToneFeatures(FeatureExtractor):
    """Features derived from news article tone/sentiment."""

    @property
    def name(self) -> str:
        return "news_tone"

    @property
    def feature_names(self) -> list[str]:
        return [
            "avg_tone_1h",
            "avg_tone_24h",
            "tone_change",
            "positive_ratio",
        ]

    def extract(
        self,
        ticker: str,
        as_of_ts: int,
        snapshots: pd.DataFrame,
        news: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> np.ndarray:
        features = np.zeros(len(self.feature_names), dtype=np.float64)

        if news is None or news.empty:
            return features

        if "fetched_ts" not in news.columns or "tone" not in news.columns:
            return features

        ts_col = news["fetched_ts"]
        tone_col = news["tone"]

        # Average tone in last 1 hour
        mask_1h = (ts_col >= as_of_ts - 3600) & tone_col.notna()
        if mask_1h.any():
            features[0] = tone_col[mask_1h].mean()

        # Average tone in last 24 hours
        mask_24h = (ts_col >= as_of_ts - 86400) & tone_col.notna()
        if mask_24h.any():
            features[1] = tone_col[mask_24h].mean()

        # Tone change: avg last 6h vs previous 6h
        mask_6h = (ts_col >= as_of_ts - 21600) & (ts_col <= as_of_ts) & tone_col.notna()
        mask_prev_6h = (
            (ts_col >= as_of_ts - 43200) & (ts_col < as_of_ts - 21600) & tone_col.notna()
        )
        if mask_6h.any() and mask_prev_6h.any():
            features[2] = tone_col[mask_6h].mean() - tone_col[mask_prev_6h].mean()

        # Positive ratio in last 24h
        if mask_24h.any():
            features[3] = (tone_col[mask_24h] > 0).mean()

        return features
