from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from kalshi_trader.config import AppConfig
from kalshi_trader.data.models import Market, MarketSnapshot, OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and self._timestamps[0] < now - self.period:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            sleep_time = self._timestamps[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._timestamps.append(time.monotonic())


class KalshiClient:
    """Wrapper over the kalshi-python SDK with rate limiting."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._read_limiter = RateLimiter(max_calls=20, period_seconds=1.0)
        self._write_limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        self._markets_api, self._api_client = self._init_client()

    def _init_client(self):
        """Initialize the kalshi-python SDK client."""
        from kalshi_python import Configuration, KalshiClient as KalshiApiClient, MarketsApi

        key_id = self.config.kalshi_api_key_id
        private_key_path = self.config.kalshi_private_key_path

        if not key_id or not private_key_path:
            logger.warning(
                "Kalshi API credentials not configured. "
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
            )
            return None, None

        base_url = self.config.kalshi_base_url
        if self.config.kalshi_demo:
            base_url = base_url.replace("api.elections.kalshi.com", "demo-api.kalshi.co")

        try:
            config = Configuration(host=base_url)
            api_client = KalshiApiClient(configuration=config)
            api_client.set_kalshi_auth(key_id, private_key_path)
            markets_api = MarketsApi(api_client)
            logger.info("Kalshi SDK initialized (base_url=%s)", base_url)
            return markets_api, api_client
        except Exception as e:
            logger.error("Could not initialize Kalshi SDK client: %s", e)
            return None, None

    def _ensure_client(self):
        if self._markets_api is None:
            raise RuntimeError(
                "Kalshi client not initialized. Check API credentials."
            )

    def get_markets(
        self,
        series: list[str] | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[Market]:
        """Fetch markets, optionally filtered by series and status."""
        self._ensure_client()
        self._read_limiter.wait()

        try:
            kwargs = {"limit": limit}
            if status:
                kwargs["status"] = status
            response = self._markets_api.get_markets(**kwargs)
        except Exception as e:
            logger.error("Error fetching markets: %s", e)
            return []

        # Response is a Pydantic model -- access .markets attribute
        markets_data = _get_attr_or_key(response, "markets", [])

        markets = []
        for m in markets_data:
            ticker = _get_attr_or_key(m, "ticker", "")
            series_ticker = _get_attr_or_key(m, "series_ticker", "")

            if series and not any(ticker.startswith(s) or series_ticker == s for s in series):
                continue

            markets.append(
                Market(
                    ticker=ticker,
                    event_ticker=_get_attr_or_key(m, "event_ticker", ""),
                    title=_get_attr_or_key(m, "title", ""),
                    status=_get_attr_or_key(m, "status", ""),
                    series_ticker=series_ticker,
                    category=_get_attr_or_key(m, "category", ""),
                    result=_get_attr_or_key(m, "result"),
                    open_ts=_to_epoch(_get_attr_or_key(m, "open_time")),
                    close_ts=_to_epoch(_get_attr_or_key(m, "close_time")),
                    settled_ts=_to_epoch(_get_attr_or_key(m, "settlement_time")),
                    rules_primary=_get_attr_or_key(m, "rules_primary"),
                )
            )
        return markets

    def get_snapshot(self, ticker: str) -> MarketSnapshot:
        """Fetch a BBO snapshot for a single market."""
        self._ensure_client()
        self._read_limiter.wait()

        try:
            response = self._markets_api.get_market(ticker)
        except Exception as e:
            logger.error("Error fetching market %s: %s", ticker, e)
            return MarketSnapshot(ticker=ticker, ts=int(time.time()))

        m = _get_attr_or_key(response, "market", response)

        return MarketSnapshot(
            ticker=ticker,
            ts=int(time.time()),
            yes_bid=_cents(_get_attr_or_key(m, "yes_bid")),
            yes_ask=_cents(_get_attr_or_key(m, "yes_ask")),
            last_price=_cents(_get_attr_or_key(m, "last_price")),
            volume=_get_attr_or_key(m, "volume"),
            open_interest=_get_attr_or_key(m, "open_interest"),
            yes_bid_size=_get_attr_or_key(m, "yes_bid_size"),
            yes_ask_size=_get_attr_or_key(m, "yes_ask_size"),
        )

    def get_orderbook(self, ticker: str) -> OrderBook:
        """Fetch the full orderbook for a market."""
        self._ensure_client()
        self._read_limiter.wait()

        try:
            response = self._markets_api.get_market_orderbook(ticker)
        except Exception as e:
            logger.error("Error fetching orderbook for %s: %s", ticker, e)
            return OrderBook(ticker=ticker, ts=int(time.time()))

        ob = _get_attr_or_key(response, "orderbook", response)

        yes_levels = []
        no_levels = []

        for level in _get_attr_or_key(ob, "yes", []) or []:
            price = _get_attr_or_key(level, "price", None)
            qty = _get_attr_or_key(level, "quantity", None)
            if price is not None and qty is not None:
                yes_levels.append(OrderBookLevel(price_cents=int(price), quantity=int(qty)))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                yes_levels.append(OrderBookLevel(price_cents=int(level[0]), quantity=int(level[1])))

        for level in _get_attr_or_key(ob, "no", []) or []:
            price = _get_attr_or_key(level, "price", None)
            qty = _get_attr_or_key(level, "quantity", None)
            if price is not None and qty is not None:
                no_levels.append(OrderBookLevel(price_cents=int(price), quantity=int(qty)))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                no_levels.append(OrderBookLevel(price_cents=int(level[0]), quantity=int(level[1])))

        return OrderBook(
            ticker=ticker,
            ts=int(time.time()),
            yes_levels=yes_levels,
            no_levels=no_levels,
        )

    def get_candlesticks(
        self, ticker: str, period_seconds: int = 60
    ) -> list[dict]:
        """Fetch historical candlestick data."""
        self._ensure_client()
        self._read_limiter.wait()

        period_map = {60: 1, 3600: 60, 86400: 1440}
        interval = period_map.get(period_seconds, 1)

        try:
            response = self._markets_api.get_market_candlesticks(
                ticker, period_interval=interval
            )
            return _get_attr_or_key(response, "candlesticks", []) or []
        except Exception as e:
            logger.error("Error fetching candlesticks for %s: %s", ticker, e)
            return []


def _get_attr_or_key(obj, name, default=None):
    """Get a value from an object by attribute or dict key."""
    if obj is None:
        return default
    # Try attribute access first (Pydantic models)
    if hasattr(obj, name):
        return getattr(obj, name)
    # Try dict access
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _to_epoch(ts_val) -> Optional[int]:
    """Convert a timestamp value to unix epoch."""
    if ts_val is None:
        return None
    # Already an int/float (epoch)
    if isinstance(ts_val, (int, float)):
        return int(ts_val)
    # ISO string
    if isinstance(ts_val, str):
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return None
    return None


def _cents(value) -> Optional[int]:
    """Convert a price value to cents. Handles both cent ints and dollar floats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value < 1.0:
            return int(round(value * 100))
        return int(value)
    return None
