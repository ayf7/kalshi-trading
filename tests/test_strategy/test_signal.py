"""Tests for the signal trading strategy."""

import pytest

from kalshi_trader.data.models import (
    Action,
    Market,
    MarketSnapshot,
    Position,
    Side,
)
from kalshi_trader.strategy.signal import SignalStrategy


@pytest.fixture
def market():
    return Market(
        ticker="TEST-MKT",
        event_ticker="TEST-EVT",
        title="Test Market",
        status="open",
    )


@pytest.fixture
def strategy():
    return SignalStrategy(min_edge=0.05, base_size=5, max_position=50)


class TestSignalStrategy:
    def test_no_signal_when_edge_too_small(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000, yes_bid=48, yes_ask=52)
        position = Position(ticker="TEST-MKT")
        # model_prob = 0.52, implied = 0.50, edge = 0.02 < 0.05
        signals = strategy.on_snapshot(market, snapshot, None, 0.52, position)
        assert signals == []

    def test_buy_yes_when_positive_edge(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000, yes_bid=45, yes_ask=55)
        position = Position(ticker="TEST-MKT")
        # model_prob = 0.60, implied = 0.50, edge = 0.10 > 0.05
        signals = strategy.on_snapshot(market, snapshot, None, 0.60, position)
        assert len(signals) == 1
        assert signals[0].side == Side.YES
        assert signals[0].action == Action.BUY
        assert signals[0].quantity == 5

    def test_buy_no_when_negative_edge(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000, yes_bid=45, yes_ask=55)
        position = Position(ticker="TEST-MKT")
        # model_prob = 0.40, implied = 0.50, edge = -0.10 < -0.05
        signals = strategy.on_snapshot(market, snapshot, None, 0.40, position)
        assert len(signals) == 1
        assert signals[0].side == Side.NO
        assert signals[0].action == Action.BUY

    def test_no_signal_when_position_full(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000, yes_bid=45, yes_ask=55)
        position = Position(ticker="TEST-MKT", net_contracts=50)  # at max
        signals = strategy.on_snapshot(market, snapshot, None, 0.60, position)
        assert signals == []

    def test_reduced_size_near_limit(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000, yes_bid=45, yes_ask=55)
        position = Position(ticker="TEST-MKT", net_contracts=48)  # 2 remaining
        signals = strategy.on_snapshot(market, snapshot, None, 0.60, position)
        assert len(signals) == 1
        assert signals[0].quantity == 2  # min(5, 50-48)

    def test_no_signal_when_no_bid_ask(self, strategy, market):
        snapshot = MarketSnapshot(ticker="TEST-MKT", ts=1000)  # no prices
        position = Position(ticker="TEST-MKT")
        signals = strategy.on_snapshot(market, snapshot, None, 0.60, position)
        assert signals == []

    def test_on_fill_returns_empty(self, strategy, market):
        position = Position(ticker="TEST-MKT")
        signals = strategy.on_fill(market, 50, 5, "yes", position)
        assert signals == []
