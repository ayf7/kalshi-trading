"""Tests for market feature extraction."""

import numpy as np
import pandas as pd
import pytest

from kalshi_trader.features.market_features import (
    MarketMomentumFeatures,
    MarketPriceFeatures,
)


def _make_snapshots(data: list[dict]) -> pd.DataFrame:
    """Create a snapshot DataFrame from a list of dicts."""
    return pd.DataFrame(data)


class TestMarketPriceFeatures:
    def test_basic_extraction(self):
        ext = MarketPriceFeatures()
        snapshots = _make_snapshots([
            {"ts": 1000, "yes_bid": 40, "yes_ask": 60, "last_price": 50,
             "volume": 100, "yes_bid_size": 30, "yes_ask_size": 70,
             "open_interest": 200},
        ])
        features = ext.extract("TEST", as_of_ts=1000, snapshots=snapshots, close_ts=87400)

        assert len(features) == len(ext.feature_names)
        assert features[0] == pytest.approx(0.5)  # mid_price: (40+60)/200 = 0.5
        assert features[1] == pytest.approx(20)  # spread: 60-40
        assert features[2] == pytest.approx(0.4)  # spread_pct: 20/50
        assert features[3] == pytest.approx(np.log1p(100))  # log_volume
        assert features[4] == pytest.approx(0.3)  # bid_size_ratio: 30/100
        assert features[5] == pytest.approx(1.0, abs=0.01)  # ~1 day to close

    def test_missing_bid_ask(self):
        ext = MarketPriceFeatures()
        snapshots = _make_snapshots([
            {"ts": 1000, "yes_bid": None, "yes_ask": None, "last_price": 50,
             "volume": 0, "yes_bid_size": None, "yes_ask_size": None,
             "open_interest": None},
        ])
        features = ext.extract("TEST", as_of_ts=1000, snapshots=snapshots)

        assert features[0] == pytest.approx(0.5)  # uses last_price fallback
        assert features[1] == 0.0  # no spread

    def test_empty_snapshots(self):
        ext = MarketPriceFeatures()
        snapshots = _make_snapshots([
            {"ts": 2000, "yes_bid": 40, "yes_ask": 60, "last_price": 50,
             "volume": 100, "yes_bid_size": 30, "yes_ask_size": 70,
             "open_interest": 200},
        ])
        # as_of_ts before any snapshot
        features = ext.extract("TEST", as_of_ts=500, snapshots=snapshots)
        assert np.all(features == 0.0)


class TestMarketMomentumFeatures:
    def test_returns(self):
        ext = MarketMomentumFeatures()
        # Create snapshots spanning >1 hour with a price trend
        data = []
        for i in range(150):  # 150 snapshots at 30s = 75 minutes
            ts = 10000 + i * 30  # 30-second intervals
            mid = 50 + i * 0.1  # slowly increasing
            data.append({
                "ts": ts,
                "yes_bid": int(mid - 5),
                "yes_ask": int(mid + 5),
                "last_price": int(mid),
                "volume": 100 + i,
                "yes_bid_size": 50,
                "yes_ask_size": 50,
                "open_interest": 200,
            })

        snapshots = _make_snapshots(data)
        as_of_ts = data[-1]["ts"]
        features = ext.extract("TEST", as_of_ts=as_of_ts, snapshots=snapshots)

        assert len(features) == len(ext.feature_names)
        # Price went up, so returns should be positive
        assert features[0] > 0  # return_5m
        assert features[1] > 0  # return_30m
        assert features[2] > 0  # return_1h
        assert features[3] > 0  # volatility_1h (there is variation)

    def test_empty_snapshots(self):
        ext = MarketMomentumFeatures()
        snapshots = _make_snapshots([])
        # Should not fail, just return zeros
        features = ext.extract("TEST", as_of_ts=1000, snapshots=snapshots)
        assert np.all(features == 0.0)
