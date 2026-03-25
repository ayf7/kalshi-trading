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
    parser.add_argument(
        "--source",
        choices=["bigquery", "gdelt"],
        default="bigquery",
        help="News data source: 'bigquery' (default, no rate limits) or 'gdelt' (DOC 2.0 API)",
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

    if args.source == "bigquery":
        from kalshi_trader.data.bigquery_client import BigQueryGDELTClient

        gdelt = BigQueryGDELTClient(
            project=config.gcp_project or None,
            credentials_path=config.gcp_credentials_path or None,
        )
        logger.info("Using BigQuery GDELT source")
    else:
        gdelt = GDELTClient()
        logger.info("Using GDELT DOC 2.0 API source")

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

        # Get active and finalized markets
        markets = db.get_markets(conn, status="active")
        markets += db.get_markets(conn, status="finalized")
        logger.info("Found %d markets", len(markets))

        # Build mapping: keyword_key (frozenset) -> list of tickers to link
        # This deduplicates queries — "Boston Celtics" only gets queried once
        # even if the Celtics played 80 games.
        keyword_to_tickers: dict[frozenset[str], list[str]] = {}
        for market in markets:
            keywords = _get_keywords_for_market(market.ticker, keyword_map)
            if not keywords:
                continue
            key = frozenset(keywords)
            keyword_to_tickers.setdefault(key, []).append(market.ticker)

        unique_keyword_sets = list(keyword_to_tickers.items())
        logger.info(
            "Deduplicated %d markets into %d unique keyword sets",
            len(markets),
            len(unique_keyword_sets),
        )

        total_articles = 0
        for i, (kw_key, tickers) in enumerate(unique_keyword_sets):
            articles = gdelt.search_articles(list(kw_key), start_date, end_date)
            for article in articles:
                article_id = db.insert_news_article(conn, article)
                if article_id is not None:
                    for ticker in tickers:
                        db.link_news_to_market(conn, article_id, ticker)
                    total_articles += 1

            if (i + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d keyword sets, %d articles",
                    i + 1, len(unique_keyword_sets), total_articles,
                )

        logger.info("Ingested %d new articles total", total_articles)

    conn.close()


def _get_keywords_for_market(ticker: str, keyword_map: dict) -> list[str]:
    """Extract GDELT search keywords for a market ticker.

    Matches ticker against keyword_map entries using prefix matching.
    For example, KXNBAGAME-26MAR22BOSCLE-BOS will match the KXNBA entry
    because the ticker starts with "KXNBA".

    Team abbreviations are found by checking every known abbreviation
    against the ticker string, since game-level tickers embed team codes
    in various positions (e.g. KXNBAGAME-26MAR22BOSCLE-BOS).
    """
    keywords = []

    for category, series_map in keyword_map.items():
        if not isinstance(series_map, dict):
            continue

        for series_prefix, series_config in series_map.items():
            if not ticker.startswith(series_prefix):
                continue
            if not isinstance(series_config, dict):
                continue

            if "keywords" in series_config:
                keywords.extend(series_config["keywords"])

            if "team_map" in series_config:
                # Extract team abbreviations from the ticker.
                # Tickers like KXNBAGAME-26MAR24SACCHA-SAC have team
                # codes concatenated in the middle segment and as the
                # last segment. Match longer abbreviations first to
                # avoid "SA" matching inside "SAC".
                remainder = ticker[len(series_prefix):]
                sorted_abbrs = sorted(
                    series_config["team_map"].keys(),
                    key=lambda a: len(str(a)),
                    reverse=True,
                )
                for abbr in sorted_abbrs:
                    abbr_str = str(abbr)
                    if abbr_str in remainder:
                        keywords.extend(series_config["team_map"][abbr])
                        # Remove matched abbreviation to prevent
                        # shorter codes from matching its substring
                        remainder = remainder.replace(abbr_str, "")

            if keywords:
                return keywords

    return keywords


if __name__ == "__main__":
    main()
