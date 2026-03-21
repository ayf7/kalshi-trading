from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from kalshi_trader.data.models import Market, MarketSnapshot, OrderBook, Position, Signal


class TradingStrategy(ABC):
    """Decides when and what to trade, given model output and market state."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def on_snapshot(
        self,
        market: Market,
        snapshot: MarketSnapshot,
        orderbook: Optional[OrderBook],
        model_prob: float,
        current_position: Position,
    ) -> list[Signal]:
        """Called on each market update. Return 0+ Signals to submit."""
        ...

    @abstractmethod
    def on_fill(
        self,
        market: Market,
        fill_price: int,
        fill_quantity: int,
        side: str,
        current_position: Position,
    ) -> list[Signal]:
        """Called when one of our orders is filled. Return adjustment signals if needed."""
        ...
