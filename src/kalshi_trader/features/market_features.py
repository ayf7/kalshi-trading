from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kalshi_trader.features.base import FeatureExtractor


class MarketPriceFeatures(FeatureExtractor):
    """Features derived from the latest market snapshot."""

    @property
    def name(self) -> str:
        return "market_price"

    @property
    def feature_names(self) -> list[str]:
        return [
            "mid_price",
            "spread",
            "spread_pct",
            "log_volume",
            "bid_size_ratio",
            "time_to_close",
        ]

    def extract(
        self,
        ticker: str,
        as_of_ts: int,
        snapshots: pd.DataFrame,
        news: Optional[pd.DataFrame] = None,
        close_ts: Optional[int] = None,
    ) -> np.ndarray:
        features = np.zeros(len(self.feature_names), dtype=np.float64)

        # Get the latest snapshot at or before as_of_ts
        mask = snapshots["ts"] <= as_of_ts
        if not mask.any():
            return features

        latest = snapshots.loc[mask].iloc[-1]

        yes_bid = latest.get("yes_bid")
        yes_ask = latest.get("yes_ask")

        if yes_bid is not None and yes_ask is not None and pd.notna(yes_bid) and pd.notna(yes_ask):
            mid = (yes_bid + yes_ask) / 2.0
            spread = yes_ask - yes_bid

            features[0] = mid / 100.0  # scaled to [0, 1]
            features[1] = spread
            features[2] = spread / mid if mid > 0 else 0.0
        else:
            # Use last_price as fallback
            lp = latest.get("last_price")
            if lp is not None and pd.notna(lp):
                features[0] = float(lp) / 100.0

        volume = latest.get("volume")
        if volume is not None and pd.notna(volume):
            features[3] = np.log1p(float(volume))

        bid_size = latest.get("yes_bid_size")
        ask_size = latest.get("yes_ask_size")
        if (
            bid_size is not None
            and ask_size is not None
            and pd.notna(bid_size)
            and pd.notna(ask_size)
        ):
            total = float(bid_size) + float(ask_size)
            features[4] = float(bid_size) / total if total > 0 else 0.5

        if close_ts is not None:
            features[5] = max(0.0, (close_ts - as_of_ts) / 86400.0)

        return features


class MarketMomentumFeatures(FeatureExtractor):
    """Features derived from recent price history (momentum, volatility)."""

    @property
    def name(self) -> str:
        return "market_momentum"

    @property
    def feature_names(self) -> list[str]:
        return [
            "return_5m",
            "return_30m",
            "return_1h",
            "volatility_1h",
            "volume_spike",
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

        if snapshots.empty or "ts" not in snapshots.columns:
            return features

        mask = snapshots["ts"] <= as_of_ts
        if not mask.any():
            return features

        recent = snapshots.loc[mask].copy()

        # Compute mid prices
        bid = recent["yes_bid"]
        ask = recent["yes_ask"]
        valid = bid.notna() & ask.notna()
        recent = recent.loc[valid].copy()

        if recent.empty:
            return features

        recent["mid"] = (recent["yes_bid"] + recent["yes_ask"]) / 2.0
        current_mid = recent["mid"].iloc[-1]

        # Returns over different lookback windows
        for i, seconds in enumerate([300, 1800, 3600]):  # 5m, 30m, 1h
            cutoff = as_of_ts - seconds
            past = recent.loc[recent["ts"] <= cutoff]
            if not past.empty:
                features[i] = current_mid - past["mid"].iloc[-1]

        # Volatility (std of mid prices over last hour)
        hour_ago = as_of_ts - 3600
        last_hour = recent.loc[recent["ts"] >= hour_ago, "mid"]
        if len(last_hour) >= 2:
            features[3] = last_hour.std()

        # Volume spike (current volume relative to 24h average)
        day_ago = as_of_ts - 86400
        last_day = recent.loc[recent["ts"] >= day_ago]
        if len(last_day) >= 2 and "volume" in recent.columns:
            vol = last_day["volume"].dropna()
            if len(vol) >= 2:
                avg_vol = vol.mean()
                if avg_vol > 0:
                    features[4] = vol.iloc[-1] / avg_vol

        return features
