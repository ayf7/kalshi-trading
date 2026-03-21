#!/usr/bin/env python3
"""CLI script to run Kalshi market data ingestion."""

import argparse
import logging
import sys

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.data.ingest import run_ingestion_cycle, run_ingestion_loop
from kalshi_trader.data.kalshi_client import KalshiClient


def main():
    parser = argparse.ArgumentParser(description="Ingest Kalshi market snapshots")
    parser.add_argument(
        "--once", action="store_true", help="Run a single ingestion cycle and exit"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override snapshot interval in seconds",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Override tracked series (e.g., KXNBA KXNFL)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    if args.interval:
        config.snapshot_interval_seconds = args.interval
    if args.series:
        config.tracked_series = args.series

    if args.once:
        client = KalshiClient(config)
        conn = db.init_db(config.db_path)
        try:
            count = run_ingestion_cycle(client, conn, config)
            print(f"Ingested {count} snapshots")
        finally:
            conn.close()
    else:
        run_ingestion_loop(config)


if __name__ == "__main__":
    main()
