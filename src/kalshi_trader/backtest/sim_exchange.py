from __future__ import annotations

import uuid
from typing import Optional

from kalshi_trader.data.models import (
    Action,
    MarketSnapshot,
    Order,
    OrderStatus,
    OrderType,
    Side,
)


class SimulatedExchange:
    """
    Simulates order matching against historical BBO snapshots.

    Matching rules:
    - A resting YES buy at price P fills if snapshot's yes_ask <= P.
    - A resting NO buy at price P fills if snapshot's (100 - yes_bid) <= P,
      i.e., yes_bid >= (100 - P).
    - Fill at the order's limit price (no price improvement in backtest).
    - FOK: fill entirely at current snapshot or cancel.
    - IOC: fill if possible at current snapshot, cancel remainder.
    - GTC: remain resting until filled or market closes.

    Limitations:
    - No partial fills (fills entirely or not at all).
    - No market impact (our orders don't move the book).
    - No queue priority.
    These make backtest results optimistic vs. live.
    """

    def __init__(self, fee_per_contract_cents: int = 2):
        self.fee = fee_per_contract_cents
        self.resting_orders: dict[str, Order] = {}  # order_id -> Order
        self.fill_log: list[Order] = []

    def submit_order(
        self, order: Order, snapshot: Optional[MarketSnapshot] = None
    ) -> Order:
        """
        Submit an order. For FOK/IOC, check immediate fill against current snapshot.
        For GTC, add to resting book.
        """
        if order.order_type == OrderType.FOK:
            if snapshot and self._can_fill(order, snapshot):
                return self._fill_order(order, snapshot)
            order.status = OrderStatus.CANCELED
            return order
        elif order.order_type == OrderType.IOC:
            if snapshot and self._can_fill(order, snapshot):
                return self._fill_order(order, snapshot)
            order.status = OrderStatus.CANCELED
            return order
        else:
            # GTC: add to resting
            order.status = OrderStatus.RESTING
            self.resting_orders[order.order_id] = order
            return order

    def process_snapshot(self, snapshot: MarketSnapshot) -> list[Order]:
        """
        Check all resting orders against a new snapshot.
        Returns list of newly filled orders.
        """
        filled = []
        to_remove = []

        for order_id, order in self.resting_orders.items():
            if order.ticker != snapshot.ticker:
                continue
            if self._can_fill(order, snapshot):
                self._fill_order(order, snapshot)
                filled.append(order)
                to_remove.append(order_id)

        for oid in to_remove:
            del self.resting_orders[oid]

        return filled

    def settle_market(self, ticker: str, result: str) -> list[Order]:
        """
        Cancel all resting orders for a settled market.
        Returns list of canceled orders.
        """
        canceled = []
        to_remove = []

        for order_id, order in self.resting_orders.items():
            if order.ticker == ticker:
                order.status = OrderStatus.CANCELED
                canceled.append(order)
                to_remove.append(order_id)

        for oid in to_remove:
            del self.resting_orders[oid]

        return canceled

    def _can_fill(self, order: Order, snapshot: MarketSnapshot) -> bool:
        """Check if an order would fill against the given snapshot."""
        if order.action == Action.BUY:
            if order.side == Side.YES:
                # Buy YES: fills if ask <= our bid price
                return (
                    snapshot.yes_ask is not None
                    and snapshot.yes_ask <= order.price_cents
                )
            else:
                # Buy NO: fills if NO ask <= our bid price
                # NO ask = 100 - YES bid
                return (
                    snapshot.yes_bid is not None
                    and (100 - snapshot.yes_bid) <= order.price_cents
                )
        else:
            # Sell
            if order.side == Side.YES:
                # Sell YES: fills if bid >= our ask price
                return (
                    snapshot.yes_bid is not None
                    and snapshot.yes_bid >= order.price_cents
                )
            else:
                # Sell NO: fills if NO bid >= our ask price
                # NO bid = 100 - YES ask
                return (
                    snapshot.yes_ask is not None
                    and (100 - snapshot.yes_ask) >= order.price_cents
                )

    def _fill_order(self, order: Order, snapshot: MarketSnapshot) -> Order:
        """Mark an order as filled."""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.fill_price_avg = order.price_cents
        order.fees_cents = order.quantity * self.fee
        self.fill_log.append(order)
        return order

    def cancel_all(self, ticker: Optional[str] = None) -> list[Order]:
        """Cancel all resting orders, optionally filtered by ticker."""
        canceled = []
        to_remove = []

        for order_id, order in self.resting_orders.items():
            if ticker is None or order.ticker == ticker:
                order.status = OrderStatus.CANCELED
                canceled.append(order)
                to_remove.append(order_id)

        for oid in to_remove:
            del self.resting_orders[oid]

        return canceled
