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

        all_markets_data = []
        series_list = series or [None]

        for series_ticker in series_list:
            cursor = None
            while True:
                self._read_limiter.wait()
                try:
                    kwargs = {"limit": limit}
                    if status:
                        kwargs["status"] = status
                    if series_ticker:
                        kwargs["series_ticker"] = series_ticker
                    if cursor:
                        kwargs["cursor"] = cursor
                    response = self._markets_api.get_markets(**kwargs)
                except Exception as e:
                    logger.error("Error fetching markets: %s", e)
                    break

                markets_data = _get_attr_or_key(response, "markets", [])
                if not markets_data:
                    break
                all_markets_data.extend(markets_data)

                cursor = _get_attr_or_key(response, "cursor", None)
                if not cursor:
                    break

        markets = []
        for m in all_markets_data:
            ticker = _get_attr_or_key(m, "ticker", "")

            # Derive series_ticker from the ticker prefix if the API
            # doesn't return it (common for game-level markets)
            st = _get_attr_or_key(m, "series_ticker")
            if not st and ticker:
                st = ticker.split("-")[0]

            markets.append(
                Market(
                    ticker=ticker,
                    event_ticker=_get_attr_or_key(m, "event_ticker", ""),
                    title=_get_attr_or_key(m, "title", ""),
                    status=_get_attr_or_key(m, "status", ""),
                    series_ticker=st,
                    category=_get_attr_or_key(m, "category", ""),
                    result=_get_attr_or_key(m, "result"),
                    open_ts=_to_epoch(_get_attr_or_key(m, "open_time")),
                    close_ts=_to_epoch(_get_attr_or_key(m, "close_time")),
                    settled_ts=_to_epoch(
                        _get_attr_or_key(m, "expiration_time")
                        or _get_attr_or_key(m, "settlement_time")
                    ),
                    rules_primary=_get_attr_or_key(m, "rules_primary"),
                )
            )
        return markets

    def get_snapshot(self, ticker: str) -> MarketSnapshot:
        """Fetch a BBO snapshot for a single market.

        Uses a raw HTTP request instead of the SDK because the API
        returns price fields as *_dollars strings and quantity fields
        as *_fp floats, which the SDK does not deserialize.
        """
        self._ensure_client()
        self._read_limiter.wait()

        try:
            url = f"{self.config.kalshi_base_url}/markets/{ticker}"
            if self.config.kalshi_demo:
                url = url.replace("api.elections.kalshi.com", "demo-api.kalshi.co")
            headers = {}
            self._api_client.update_params_for_auth(
                headers, {}, ["bearer"], None, None, None
            )
            import requests
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            m = resp.json().get("market", {})
        except Exception as e:
            logger.error("Error fetching market %s: %s", ticker, e)
            return MarketSnapshot(ticker=ticker, ts=int(time.time()))

        return MarketSnapshot(
            ticker=ticker,
            ts=int(time.time()),
            yes_bid=_dollars_or_cents(m, "yes_bid"),
            yes_ask=_dollars_or_cents(m, "yes_ask"),
            last_price=_dollars_or_cents(m, "last_price"),
            volume=_get_fp_or_int(m, "volume"),
            open_interest=_get_fp_or_int(m, "open_interest"),
            yes_bid_size=_get_fp_or_int(m, "yes_bid_size"),
            yes_ask_size=_get_fp_or_int(m, "yes_ask_size"),
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
    # datetime object (kalshi-python SDK returns these)
    from datetime import datetime
    if isinstance(ts_val, datetime):
        return int(ts_val.timestamp())
    # ISO string
    if isinstance(ts_val, str):
        try:
            from datetime import timezone
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
    if isinstance(value, str):
        try:
            f = float(value)
            return int(round(f * 100)) if f <= 1.0 else int(f)
        except ValueError:
            return None
    return None


def _dollars_or_cents(obj, field_name: str) -> Optional[int]:
    """Read a price field, trying the _dollars variant first (returns dollar
    strings like '0.7200'), then falling back to the plain field name."""
    val = _get_attr_or_key(obj, f"{field_name}_dollars")
    if val is not None:
        return _cents(val)
    return _cents(_get_attr_or_key(obj, field_name))


def _get_fp_or_int(obj, field_name: str) -> Optional[int]:
    """Read a quantity field, trying the _fp variant first (returns float
    strings like '3331473.00'), then falling back to the plain field name."""
    val = _get_attr_or_key(obj, f"{field_name}_fp")
    if val is not None:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            pass
    val = _get_attr_or_key(obj, field_name)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    return None
