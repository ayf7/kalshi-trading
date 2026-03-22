from __future__ import annotations

from typing import Optional

from kalshi_trader.data.models import (
    Action,
    Market,
    MarketSnapshot,
    OrderBook,
    Position,
    Side,
    Signal,
)
from kalshi_trader.strategy.base import TradingStrategy


class SignalStrategy(TradingStrategy):
    """
    Signal-based trading strategy.

    Trades when the model's estimated probability differs from the market's
    implied probability by more than min_edge.
    """

    def __init__(self, min_edge: float = 0.05, base_size: int = 5, max_position: int = 50):
        self._min_edge = min_edge
        self._base_size = base_size
        self._max_position = max_position

    @property
    def name(self) -> str:
        return "signal"

    def on_snapshot(
        self,
        market: Market,
        snapshot: MarketSnapshot,
        orderbook: Optional[OrderBook],
        model_prob: float,
        current_position: Position,
    ) -> list[Signal]:
        implied = snapshot.implied_prob
        if implied is None:
            return []

        edge = model_prob - implied

        if abs(edge) < self._min_edge:
            return []

        if edge > 0:
            # Model thinks YES is underpriced -> buy YES at the ask (cross spread)
            side = Side.YES
            action = Action.BUY
            if snapshot.yes_ask is not None:
                price = snapshot.yes_ask
            else:
                price = int(model_prob * 100)
        else:
            # Model thinks NO is underpriced -> buy NO at the ask (cross spread)
            side = Side.NO
            action = Action.BUY
            # NO ask = 100 - YES bid
            if snapshot.yes_bid is not None:
                price = 100 - snapshot.yes_bid
            else:
                price = int((1 - model_prob) * 100)

        # Size: base_size, but don't exceed position limit
        remaining_capacity = self._max_position - abs(current_position.net_contracts)
        qty = min(self._base_size, remaining_capacity)
        if qty <= 0:
            return []

        mid = snapshot.mid_price or 50.0
        return [
            Signal(
                ticker=market.ticker,
                ts=snapshot.ts,
                model_prob=model_prob,
                market_mid=mid,
                edge=edge,
                side=side,
                action=action,
                price_cents=price,
                quantity=qty,
                confidence=abs(edge),
                strategy_name=self.name,
            )
        ]

    def on_fill(
        self,
        market: Market,
        fill_price: int,
        fill_quantity: int,
        side: str,
        current_position: Position,
    ) -> list[Signal]:
        # No adjustment signals for the simple signal strategy
        return []
