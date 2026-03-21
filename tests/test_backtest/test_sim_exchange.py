"""Tests for the simulated exchange."""

import pytest

from kalshi_trader.backtest.sim_exchange import SimulatedExchange
from kalshi_trader.data.models import (
    Action,
    MarketSnapshot,
    Order,
    OrderStatus,
    OrderType,
    Side,
)


_order_counter = 0


def _make_order(
    side=Side.YES,
    action=Action.BUY,
    price=50,
    qty=10,
    order_type=OrderType.GTC,
    ticker="TEST",
) -> Order:
    global _order_counter
    _order_counter += 1
    return Order(
        order_id=f"order-{_order_counter}",
        ticker=ticker,
        ts=1000,
        side=side,
        action=action,
        price_cents=price,
        quantity=qty,
        order_type=order_type,
        status=OrderStatus.SUBMITTED,
    )


def _make_snapshot(ticker="TEST", yes_bid=45, yes_ask=55) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        ts=1001,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
    )


class TestSimulatedExchange:
    def test_gtc_order_rests(self):
        exchange = SimulatedExchange()
        order = _make_order(price=40)  # bid at 40, ask at 55 -> no fill
        result = exchange.submit_order(order, _make_snapshot())
        assert result.status == OrderStatus.RESTING
        assert len(exchange.resting_orders) == 1

    def test_yes_buy_fills_when_ask_lte_price(self):
        exchange = SimulatedExchange()
        order = _make_order(side=Side.YES, action=Action.BUY, price=55)
        exchange.submit_order(order)

        snapshot = _make_snapshot(yes_ask=55)  # ask == our price -> fill
        fills = exchange.process_snapshot(snapshot)

        assert len(fills) == 1
        assert fills[0].status == OrderStatus.FILLED
        assert fills[0].filled_quantity == 10
        assert len(exchange.resting_orders) == 0

    def test_yes_buy_no_fill_when_ask_gt_price(self):
        exchange = SimulatedExchange()
        order = _make_order(side=Side.YES, action=Action.BUY, price=50)
        exchange.submit_order(order)

        snapshot = _make_snapshot(yes_ask=55)  # ask > our price -> no fill
        fills = exchange.process_snapshot(snapshot)

        assert len(fills) == 0
        assert len(exchange.resting_orders) == 1

    def test_no_buy_fills_when_no_ask_lte_price(self):
        exchange = SimulatedExchange()
        # Buying NO at 55 cents. NO ask = 100 - yes_bid.
        order = _make_order(side=Side.NO, action=Action.BUY, price=55)
        exchange.submit_order(order)

        # yes_bid = 45 -> NO ask = 100 - 45 = 55 -> fills (55 <= 55)
        snapshot = _make_snapshot(yes_bid=45)
        fills = exchange.process_snapshot(snapshot)

        assert len(fills) == 1
        assert fills[0].status == OrderStatus.FILLED

    def test_fok_fills_immediately(self):
        exchange = SimulatedExchange()
        order = _make_order(side=Side.YES, action=Action.BUY, price=55, order_type=OrderType.FOK)
        snapshot = _make_snapshot(yes_ask=55)
        result = exchange.submit_order(order, snapshot)

        assert result.status == OrderStatus.FILLED
        assert len(exchange.resting_orders) == 0

    def test_fok_cancels_when_no_fill(self):
        exchange = SimulatedExchange()
        order = _make_order(side=Side.YES, action=Action.BUY, price=50, order_type=OrderType.FOK)
        snapshot = _make_snapshot(yes_ask=55)
        result = exchange.submit_order(order, snapshot)

        assert result.status == OrderStatus.CANCELED
        assert len(exchange.resting_orders) == 0

    def test_fees_applied_on_fill(self):
        exchange = SimulatedExchange(fee_per_contract_cents=3)
        order = _make_order(side=Side.YES, action=Action.BUY, price=55)
        exchange.submit_order(order)

        snapshot = _make_snapshot(yes_ask=55)
        fills = exchange.process_snapshot(snapshot)

        assert fills[0].fees_cents == 30  # 10 contracts * 3 cents

    def test_settle_market_cancels_resting(self):
        exchange = SimulatedExchange()
        order = _make_order(price=40)
        exchange.submit_order(order)

        canceled = exchange.settle_market("TEST", "yes")
        assert len(canceled) == 1
        assert canceled[0].status == OrderStatus.CANCELED
        assert len(exchange.resting_orders) == 0

    def test_cancel_all(self):
        exchange = SimulatedExchange()
        exchange.submit_order(_make_order(price=40, ticker="A"))
        exchange.submit_order(_make_order(price=40, ticker="B"))
        assert len(exchange.resting_orders) == 2

        canceled = exchange.cancel_all(ticker="A")
        assert len(canceled) == 1
        assert len(exchange.resting_orders) == 1

    def test_different_tickers_independent(self):
        exchange = SimulatedExchange()
        order_a = _make_order(side=Side.YES, action=Action.BUY, price=55, ticker="A")
        order_b = _make_order(side=Side.YES, action=Action.BUY, price=55, ticker="B")
        exchange.submit_order(order_a)
        exchange.submit_order(order_b)

        # Snapshot only for ticker A
        snapshot = MarketSnapshot(ticker="A", ts=1001, yes_bid=45, yes_ask=55)
        fills = exchange.process_snapshot(snapshot)

        assert len(fills) == 1
        assert fills[0].ticker == "A"
        assert len(exchange.resting_orders) == 1  # B still resting
