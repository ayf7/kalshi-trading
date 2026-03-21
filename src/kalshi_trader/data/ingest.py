from __future__ import annotations

import logging
import sqlite3
import time

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


def run_ingestion_cycle(
    client: KalshiClient,
    conn: sqlite3.Connection,
    config: AppConfig,
) -> int:
    """
    Run one ingestion cycle: fetch all open markets and snapshot them.

    Returns the number of snapshots inserted.
    """
    # Fetch open markets in tracked series
    markets = client.get_markets(series=config.tracked_series, status="open")
    logger.info("Found %d open markets in tracked series", len(markets))

    snapshot_count = 0
    for market in markets:
        # Upsert market metadata
        db.upsert_market(conn, market)

        # Fetch and store BBO snapshot
        snapshot = client.get_snapshot(market.ticker)
        db.insert_snapshot(conn, snapshot)
        snapshot_count += 1

        # Fetch and store orderbook
        orderbook = client.get_orderbook(market.ticker)
        db.insert_orderbook(conn, orderbook)

    return snapshot_count


def refresh_market_metadata(
    client: KalshiClient,
    conn: sqlite3.Connection,
    config: AppConfig,
) -> int:
    """
    Refresh all market metadata (catches new markets, settlements, etc.).

    Returns number of markets updated.
    """
    count = 0
    for status in ["open", "closed", "settled"]:
        markets = client.get_markets(series=config.tracked_series, status=status)
        for market in markets:
            db.upsert_market(conn, market)
            count += 1
    logger.info("Refreshed %d market records", count)
    return count


def run_ingestion_loop(config: AppConfig) -> None:
    """
    Main ingestion loop. Runs continuously, snapshotting every N seconds.
    Refreshes market metadata every 10 minutes.
    """
    client = KalshiClient(config)
    conn = db.init_db(config.db_path)

    last_metadata_refresh = 0.0
    metadata_refresh_interval = 600.0  # 10 minutes

    logger.info(
        "Starting ingestion loop (interval=%ds, series=%s)",
        config.snapshot_interval_seconds,
        config.tracked_series,
    )

    try:
        while True:
            cycle_start = time.time()

            # Refresh metadata periodically
            if cycle_start - last_metadata_refresh > metadata_refresh_interval:
                refresh_market_metadata(client, conn, config)
                last_metadata_refresh = cycle_start

            # Run snapshot cycle
            count = run_ingestion_cycle(client, conn, config)
            logger.info("Ingested %d snapshots", count)

            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, config.snapshot_interval_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Ingestion loop stopped by user")
    finally:
        conn.close()
