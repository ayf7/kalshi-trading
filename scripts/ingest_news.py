#!/usr/bin/env python3
"""CLI script to run GDELT news ingestion."""

import argparse
import logging
from datetime import datetime, timedelta

import yaml

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.data.gdelt_client import GDELTClient


def main():
    parser = argparse.ArgumentParser(description="Ingest GDELT news articles")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Override keywords to search (e.g., 'Celtics' 'Lakers')",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Number of days to look back (default: 1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = AppConfig()
    conn = db.init_db(config.db_path)
    gdelt = GDELTClient()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days_back)).strftime("%Y-%m-%d")

    if args.keywords:
        # Direct keyword search
        articles = gdelt.search_articles(args.keywords, start_date, end_date)
        logger.info("Found %d articles for keywords: %s", len(articles), args.keywords)
        for article in articles:
            db.insert_news_article(conn, article)
    else:
        # Load keyword map and search for each tracked market
        try:
            with open("config/keyword_map.yaml") as f:
                keyword_map = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("config/keyword_map.yaml not found. Use --keywords instead.")
            conn.close()
            return

        # Get open markets
        markets = db.get_markets(conn, status="open")
        logger.info("Found %d open markets", len(markets))

        total_articles = 0
        for market in markets:
            keywords = _get_keywords_for_market(market.ticker, keyword_map)
            if not keywords:
                continue

            articles = gdelt.search_articles(keywords, start_date, end_date)
            for article in articles:
                article_id = db.insert_news_article(conn, article)
                if article_id is not None:
                    db.link_news_to_market(conn, article_id, market.ticker)
                    total_articles += 1

        logger.info("Ingested %d new articles total", total_articles)

    conn.close()


def _get_keywords_for_market(ticker: str, keyword_map: dict) -> list[str]:
    """Extract GDELT search keywords for a market ticker."""
    keywords = []

    for category, series_map in keyword_map.items():
        if not isinstance(series_map, dict):
            continue
        for series_prefix, series_config in series_map.items():
            if not ticker.startswith(series_prefix):
                continue
            if isinstance(series_config, dict):
                # Check for direct keywords
                if "keywords" in series_config:
                    keywords.extend(series_config["keywords"])
                # Check team map
                if "team_map" in series_config:
                    # Parse team abbreviations from ticker
                    parts = ticker.replace(series_prefix + "-", "").split("-")
                    for part in parts:
                        if part in series_config["team_map"]:
                            keywords.extend(series_config["team_map"][part])

    return keywords


if __name__ == "__main__":
    main()
