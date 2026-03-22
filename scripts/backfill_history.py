#!/usr/bin/env python3
"""Backfill historical market data from Kalshi candlestick API.

Fetches settled/finalized NBA game markets and converts their hourly
candlestick data into market_snapshots rows for model training.

Usage:
    python scripts/backfill_history.py -v
    python scripts/backfill_history.py --series KXNBAGAME --days-back 30 -v
    python scripts/backfill_history.py --series KXNBAGAME KXNBASPREAD -v
"""

import argparse
import logging
import time

import requests

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.data.models import Market, MarketSnapshot

logger = logging.getLogger(__name__)


def _get_auth_headers(config: AppConfig) -> dict:
    """Initialize the Kalshi SDK client and return auth headers."""
    from kalshi_python import Configuration, KalshiClient as KalshiApiClient

    base_url = config.kalshi_base_url
    api_config = Configuration(host=base_url)
    api_client = KalshiApiClient(configuration=api_config)
    api_client.set_kalshi_auth(
        config.kalshi_api_key_id, config.kalshi_private_key_path
    )

    headers = {}
    api_client.update_params_for_auth(headers, {}, ["bearer"], None, None, None)
    return headers


def _cents(dollar_str) -> int | None:
    """Convert a dollar string like '0.7300' to cents (73)."""
    if dollar_str is None:
        return None
    try:
        return int(round(float(dollar_str) * 100))
    except (ValueError, TypeError):
        return None


def fetch_finalized_markets(
    base_url: str,
    headers: dict,
    series_list: list[str],
) -> list[dict]:
    """Fetch all finalized markets for the given series via raw API."""
    all_markets = []

    for series in series_list:
        cursor = None
        while True:
            params = {"limit": 200, "series_ticker": series}
            if cursor:
                params["cursor"] = cursor

            try:
                resp = requests.get(
                    f"{base_url}/markets", params=params, headers=headers, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("Error fetching markets for %s: %s", series, e)
                break

            markets = data.get("markets", [])
            if not markets:
                break

            for m in markets:
                if m.get("status") == "finalized":
                    all_markets.append(m)

            cursor = data.get("cursor")
            if not cursor:
                break

            time.sleep(0.3)  # rate limit (~3 req/s)

    return all_markets


def _rate_limited_get(url: str, params: dict, headers: dict, max_retries: int = 3):
    """GET with retry on 429 rate limit errors."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.debug("Rate limited, waiting %ds...", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # raise on final failure
    return resp


def fetch_candlesticks(
    base_url: str,
    headers: dict,
    series_ticker: str,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_minutes: int = 60,
) -> list[dict]:
    """Fetch candlestick data for a market."""
    url = f"{base_url}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    try:
        resp = _rate_limited_get(
            url,
            params={
                "period_interval": period_minutes,
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
            headers=headers,
        )
        return resp.json().get("candlesticks", [])
    except Exception as e:
        logger.error("Error fetching candlesticks for %s: %s", market_ticker, e)
        return []


def candle_to_snapshot(ticker: str, candle: dict) -> MarketSnapshot:
    """Convert a candlestick dict to a MarketSnapshot using close prices."""
    price = candle.get("price", {})
    yes_bid = candle.get("yes_bid", {})
    yes_ask = candle.get("yes_ask", {})

    return MarketSnapshot(
        ticker=ticker,
        ts=candle.get("end_period_ts", 0),
        yes_bid=_cents(yes_bid.get("close_dollars")),
        yes_ask=_cents(yes_ask.get("close_dollars")),
        last_price=_cents(price.get("close_dollars")),
        volume=int(float(candle.get("volume_fp", "0"))),
        open_interest=int(float(candle.get("open_interest_fp", "0"))),
    )


def raw_market_to_market(m: dict) -> Market:
    """Convert a raw API market dict to a Market dataclass."""
    from kalshi_trader.data.kalshi_client import _to_epoch

    ticker = m.get("ticker", "")
    return Market(
        ticker=ticker,
        event_ticker=m.get("event_ticker", ""),
        title=m.get("title", ""),
        status=m.get("status", ""),
        series_ticker=ticker.split("-")[0] if ticker else None,
        result=m.get("result"),
        open_ts=_to_epoch(m.get("open_time")),
        close_ts=_to_epoch(m.get("close_time")),
        settled_ts=_to_epoch(m.get("expiration_time") or m.get("settlement_time")),
        rules_primary=m.get("rules_primary"),
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill historical Kalshi data")
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Series to backfill (default: uses config tracked_series)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="How far back to fetch candlesticks (default: 90 days)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=60,
        help="Candlestick period in minutes (default: 60)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    conn = db.init_db(config.db_path)
    headers = _get_auth_headers(config)
    base_url = config.kalshi_base_url

    series_list = args.series or config.tracked_series

    logger.info("Fetching finalized markets for series: %s", series_list)
    raw_markets = fetch_finalized_markets(base_url, headers, series_list)
    logger.info("Found %d finalized markets", len(raw_markets))

    # Check which markets already have snapshots (for resume)
    existing = set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM market_snapshots"
        ).fetchall()
        existing = {r[0] for r in rows}
    except Exception:
        pass

    now = int(time.time())

    total_snapshots = 0
    skipped = 0
    for i, raw_m in enumerate(raw_markets):
        market = raw_market_to_market(raw_m)
        db.upsert_market(conn, market)

        if market.ticker in existing:
            skipped += 1
            continue

        # Use the market's own open_time as start, with a 1hr buffer.
        # Falls back to days-back window if open_time is missing.
        if market.open_ts:
            start_ts = market.open_ts - 3600
        else:
            start_ts = now - (args.days_back * 86400)

        end_ts = (market.settled_ts or market.close_ts or now) + 3600

        series = market.series_ticker or market.ticker.split("-")[0]
        candles = fetch_candlesticks(
            base_url, headers, series, market.ticker,
            start_ts=start_ts, end_ts=end_ts,
            period_minutes=args.period,
        )

        for candle in candles:
            snap = candle_to_snapshot(market.ticker, candle)
            db.insert_snapshot(conn, snap)
            total_snapshots += 1

        if (i + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d markets (%d skipped), %d snapshots so far",
                i + 1, len(raw_markets), skipped, total_snapshots,
            )

        time.sleep(0.3)  # rate limit (~3 req/s)

    conn.close()
    print(f"Backfilled {total_snapshots} snapshots from {len(raw_markets)} markets")


if __name__ == "__main__":
    main()
