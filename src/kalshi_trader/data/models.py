from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    GTC = "gtc"
    FOK = "fok"
    IOC = "ioc"


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    RESTING = "resting"
    FILLED = "filled"
    CANCELED = "canceled"


@dataclass(frozen=True)
class Market:
    ticker: str
    event_ticker: str
    title: str
    status: str  # "open", "closed", "settled"
    series_ticker: Optional[str] = None
    category: Optional[str] = None
    result: Optional[str] = None  # "yes" or "no" once settled
    open_ts: Optional[int] = None
    close_ts: Optional[int] = None
    settled_ts: Optional[int] = None
    rules_primary: Optional[str] = None


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    ts: int  # unix epoch
    yes_bid: Optional[int] = None  # cents (1-99)
    yes_ask: Optional[int] = None
    last_price: Optional[int] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    yes_bid_size: Optional[int] = None
    yes_ask_size: Optional[int] = None

    @property
    def mid_price(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[int]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    @property
    def implied_prob(self) -> Optional[float]:
        mid = self.mid_price
        return mid / 100.0 if mid is not None else None


@dataclass(frozen=True)
class OrderBookLevel:
    price_cents: int
    quantity: int


@dataclass(frozen=True)
class OrderBook:
    ticker: str
    ts: int
    yes_levels: list[OrderBookLevel] = field(default_factory=list)
    no_levels: list[OrderBookLevel] = field(default_factory=list)

    @property
    def best_yes_bid(self) -> Optional[int]:
        return max((lvl.price_cents for lvl in self.yes_levels), default=None)

    @property
    def best_yes_ask(self) -> Optional[int]:
        best_no = max((lvl.price_cents for lvl in self.no_levels), default=None)
        return (100 - best_no) if best_no is not None else None


@dataclass(frozen=True)
class NewsArticle:
    url: str
    title: str
    seen_date: str  # ISO 8601 from GDELT
    domain: Optional[str] = None
    language: Optional[str] = None
    source_country: Optional[str] = None
    tone: Optional[float] = None


@dataclass
class Order:
    order_id: str
    ticker: str
    ts: int
    side: Side
    action: Action
    price_cents: int
    quantity: int
    order_type: OrderType
    status: OrderStatus
    filled_quantity: int = 0
    fill_price_avg: Optional[int] = None
    fees_cents: int = 0
    source: str = "backtest"
    strategy_name: Optional[str] = None
    model_prob: Optional[float] = None


@dataclass
class Position:
    ticker: str
    net_contracts: int = 0  # >0 = long YES, <0 = long NO
    avg_entry_price: int = 0  # cents
    cost_basis: int = 0  # total cents
    unrealized_pnl: int = 0
    realized_pnl: int = 0
    fees_paid: int = 0


@dataclass(frozen=True)
class Signal:
    ticker: str
    ts: int
    model_prob: float
    market_mid: float  # current midpoint in cents
    edge: float  # model_prob - implied_prob
    side: Side
    action: Action
    price_cents: int
    quantity: int
    confidence: float  # 0-1
    strategy_name: str
