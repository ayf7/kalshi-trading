"""Tests for the database layer."""

import sqlite3

import pandas as pd
import pytest

from kalshi_trader.data import db
from kalshi_trader.data.models import Market, MarketSnapshot, NewsArticle, OrderBook, OrderBookLevel


class TestInitDb:
    def test_creates_tables(self, in_memory_db):
        cursor = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            "markets",
            "market_snapshots",
            "orderbook_snapshots",
            "trades",
            "news_articles",
            "news_market_links",
            "news_timelines",
            "our_orders",
            "backtest_runs",
        }
        assert expected.issubset(tables)


class TestUpsertMarket:
    def test_insert_new_market(self, in_memory_db):
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test Market",
            status="open",
            category="sports",
        )
        db.upsert_market(in_memory_db, market)

        cursor = in_memory_db.execute(
            "SELECT ticker, status, category FROM markets WHERE ticker = ?",
            ("TEST-MKT-001",),
        )
        row = cursor.fetchone()
        assert row == ("TEST-MKT-001", "open", "sports")

    def test_update_existing_market(self, in_memory_db):
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test Market",
            status="open",
        )
        db.upsert_market(in_memory_db, market)

        # Update status to settled
        settled = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test Market",
            status="settled",
            result="yes",
        )
        db.upsert_market(in_memory_db, settled)

        cursor = in_memory_db.execute(
            "SELECT status, result FROM markets WHERE ticker = ?",
            ("TEST-MKT-001",),
        )
        row = cursor.fetchone()
        assert row == ("settled", "yes")


class TestInsertSnapshot:
    def test_insert_snapshot(self, in_memory_db):
        # Need a market first
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test",
            status="open",
        )
        db.upsert_market(in_memory_db, market)

        snap = MarketSnapshot(
            ticker="TEST-MKT-001",
            ts=1000000,
            yes_bid=45,
            yes_ask=55,
            last_price=50,
            volume=100,
        )
        db.insert_snapshot(in_memory_db, snap)

        cursor = in_memory_db.execute(
            "SELECT ticker, ts, yes_bid, yes_ask FROM market_snapshots"
        )
        row = cursor.fetchone()
        assert row == ("TEST-MKT-001", 1000000, 45, 55)

    def test_dedup_snapshot(self, in_memory_db):
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test",
            status="open",
        )
        db.upsert_market(in_memory_db, market)

        snap = MarketSnapshot(ticker="TEST-MKT-001", ts=1000000, yes_bid=45, yes_ask=55)
        db.insert_snapshot(in_memory_db, snap)
        db.insert_snapshot(in_memory_db, snap)  # duplicate

        cursor = in_memory_db.execute("SELECT COUNT(*) FROM market_snapshots")
        assert cursor.fetchone()[0] == 1


class TestInsertOrderbook:
    def test_insert_orderbook(self, in_memory_db):
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test",
            status="open",
        )
        db.upsert_market(in_memory_db, market)

        ob = OrderBook(
            ticker="TEST-MKT-001",
            ts=1000000,
            yes_levels=[
                OrderBookLevel(price_cents=45, quantity=100),
                OrderBookLevel(price_cents=44, quantity=200),
            ],
            no_levels=[
                OrderBookLevel(price_cents=55, quantity=150),
            ],
        )
        db.insert_orderbook(in_memory_db, ob)

        cursor = in_memory_db.execute("SELECT COUNT(*) FROM orderbook_snapshots")
        assert cursor.fetchone()[0] == 3


class TestInsertNewsArticle:
    def test_insert_article(self, in_memory_db):
        article = NewsArticle(
            url="https://example.com/article1",
            title="Test Article",
            seen_date="2025-03-14",
            domain="example.com",
            tone=5.2,
        )
        article_id = db.insert_news_article(in_memory_db, article)
        assert article_id is not None

    def test_dedup_article(self, in_memory_db):
        article = NewsArticle(
            url="https://example.com/article1",
            title="Test Article",
            seen_date="2025-03-14",
        )
        id1 = db.insert_news_article(in_memory_db, article)
        id2 = db.insert_news_article(in_memory_db, article)
        assert id1 is not None
        assert id2 is None  # duplicate returns None


class TestGetSnapshots:
    def test_get_snapshots_in_range(self, in_memory_db):
        market = Market(
            ticker="TEST-MKT-001",
            event_ticker="TEST-EVENT",
            title="Test",
            status="open",
        )
        db.upsert_market(in_memory_db, market)

        for ts in [1000, 2000, 3000, 4000, 5000]:
            snap = MarketSnapshot(ticker="TEST-MKT-001", ts=ts, yes_bid=45, yes_ask=55)
            db.insert_snapshot(in_memory_db, snap)

        df = db.get_snapshots(in_memory_db, "TEST-MKT-001", 2000, 4000)
        assert len(df) == 3  # ts=2000, 3000, 4000
        assert df["ts"].tolist() == [2000, 3000, 4000]


class TestGetSettledMarkets:
    def test_get_settled(self, in_memory_db):
        for i, (status, result) in enumerate(
            [("open", None), ("settled", "yes"), ("settled", "no")]
        ):
            m = Market(
                ticker=f"MKT-{i}",
                event_ticker="EVT",
                title=f"Market {i}",
                status=status,
                result=result,
            )
            db.upsert_market(in_memory_db, m)

        settled = db.get_settled_markets(in_memory_db)
        assert len(settled) == 2
        tickers = {m.ticker for m in settled}
        assert tickers == {"MKT-1", "MKT-2"}
