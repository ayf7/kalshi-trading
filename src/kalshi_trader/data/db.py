from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from kalshi_trader.data.models import (
    Market,
    MarketSnapshot,
    NewsArticle,
    OrderBook,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
    ticker          TEXT PRIMARY KEY,
    event_ticker    TEXT NOT NULL,
    series_ticker   TEXT,
    title           TEXT NOT NULL,
    category        TEXT,
    status          TEXT NOT NULL,
    result          TEXT,
    open_ts         INTEGER,
    close_ts        INTEGER,
    settled_ts      INTEGER,
    rules_primary   TEXT,
    last_updated    INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_ticker);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL REFERENCES markets(ticker),
    ts              INTEGER NOT NULL,
    yes_bid         INTEGER,
    yes_ask         INTEGER,
    last_price      INTEGER,
    volume          INTEGER,
    open_interest   INTEGER,
    yes_bid_size    INTEGER,
    yes_ask_size    INTEGER,
    UNIQUE(ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON market_snapshots(ticker, ts);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL REFERENCES markets(ticker),
    ts              INTEGER NOT NULL,
    side            TEXT NOT NULL CHECK(side IN ('yes', 'no')),
    price_cents     INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    UNIQUE(ticker, ts, side, price_cents)
);
CREATE INDEX IF NOT EXISTS idx_ob_ticker_ts ON orderbook_snapshots(ticker, ts);

CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL REFERENCES markets(ticker),
    ts              INTEGER NOT NULL,
    yes_price       INTEGER NOT NULL,
    count           INTEGER NOT NULL,
    taker_side      TEXT NOT NULL CHECK(taker_side IN ('yes', 'no'))
);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON trades(ticker, ts);

CREATE TABLE IF NOT EXISTS news_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    seen_date       TEXT NOT NULL,
    domain          TEXT,
    language        TEXT,
    source_country  TEXT,
    tone            REAL,
    fetched_ts      INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_news_seen ON news_articles(seen_date);

CREATE TABLE IF NOT EXISTS news_market_links (
    article_id      INTEGER NOT NULL REFERENCES news_articles(id),
    ticker          TEXT NOT NULL REFERENCES markets(ticker),
    relevance_score REAL DEFAULT 1.0,
    PRIMARY KEY (article_id, ticker)
);

CREATE TABLE IF NOT EXISTS news_timelines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT NOT NULL,
    mode            TEXT NOT NULL,
    ts              TEXT NOT NULL,
    value           REAL NOT NULL,
    UNIQUE(keyword, mode, ts)
);

CREATE TABLE IF NOT EXISTS our_orders (
    order_id        TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    side            TEXT NOT NULL CHECK(side IN ('yes', 'no')),
    action          TEXT NOT NULL CHECK(action IN ('buy', 'sell')),
    price_cents     INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    order_type      TEXT NOT NULL,
    status          TEXT NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    fill_price_avg  INTEGER,
    fees_cents      INTEGER DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'backtest',
    strategy_name   TEXT,
    model_prob      REAL,
    created_ts      INTEGER NOT NULL,
    updated_ts      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON our_orders(ticker, ts);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT PRIMARY KEY,
    run_ts          INTEGER NOT NULL,
    strategy_name   TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    total_pnl       INTEGER,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    num_trades      INTEGER,
    brier_score     REAL,
    notes           TEXT
);
"""


def init_db(path: str) -> sqlite3.Connection:
    """Create all tables and return a connection."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def upsert_market(conn: sqlite3.Connection, market: Market) -> None:
    """Insert or update a market record."""
    conn.execute(
        """INSERT INTO markets (ticker, event_ticker, series_ticker, title,
           category, status, result, open_ts, close_ts, settled_ts, rules_primary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
             status=excluded.status, result=excluded.result,
             settled_ts=excluded.settled_ts,
             last_updated=strftime('%s','now')""",
        (
            market.ticker,
            market.event_ticker,
            market.series_ticker,
            market.title,
            market.category,
            market.status,
            market.result,
            market.open_ts,
            market.close_ts,
            market.settled_ts,
            market.rules_primary,
        ),
    )
    conn.commit()


def insert_snapshot(conn: sqlite3.Connection, snap: MarketSnapshot) -> None:
    """Insert a market snapshot, ignoring duplicates."""
    conn.execute(
        """INSERT OR IGNORE INTO market_snapshots
           (ticker, ts, yes_bid, yes_ask, last_price, volume,
            open_interest, yes_bid_size, yes_ask_size)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snap.ticker,
            snap.ts,
            snap.yes_bid,
            snap.yes_ask,
            snap.last_price,
            snap.volume,
            snap.open_interest,
            snap.yes_bid_size,
            snap.yes_ask_size,
        ),
    )
    conn.commit()


def insert_orderbook(conn: sqlite3.Connection, ob: OrderBook) -> None:
    """Insert all orderbook levels for a snapshot."""
    rows = []
    for lvl in ob.yes_levels:
        rows.append((ob.ticker, ob.ts, "yes", lvl.price_cents, lvl.quantity))
    for lvl in ob.no_levels:
        rows.append((ob.ticker, ob.ts, "no", lvl.price_cents, lvl.quantity))
    conn.executemany(
        """INSERT OR IGNORE INTO orderbook_snapshots
           (ticker, ts, side, price_cents, quantity)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def insert_news_article(conn: sqlite3.Connection, article: NewsArticle) -> Optional[int]:
    """Insert a news article, returning its ID. Returns None if duplicate URL."""
    try:
        cursor = conn.execute(
            """INSERT INTO news_articles (url, title, seen_date, domain, language,
               source_country, tone)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                article.url,
                article.title,
                article.seen_date,
                article.domain,
                article.language,
                article.source_country,
                article.tone,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def link_news_to_market(
    conn: sqlite3.Connection, article_id: int, ticker: str, score: float = 1.0
) -> None:
    """Create a link between a news article and a market."""
    conn.execute(
        """INSERT OR IGNORE INTO news_market_links (article_id, ticker, relevance_score)
           VALUES (?, ?, ?)""",
        (article_id, ticker, score),
    )
    conn.commit()


def get_snapshots(
    conn: sqlite3.Connection, ticker: str, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """Get market snapshots for a ticker in a time range."""
    return pd.read_sql_query(
        """SELECT * FROM market_snapshots
           WHERE ticker = ? AND ts >= ? AND ts <= ?
           ORDER BY ts""",
        conn,
        params=(ticker, start_ts, end_ts),
    )


def get_all_snapshots_in_range(
    conn: sqlite3.Connection, start_ts: int, end_ts: int, tickers: list[str] | None = None
) -> pd.DataFrame:
    """Get all snapshots in a time range, optionally filtered by tickers."""
    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        return pd.read_sql_query(
            f"""SELECT * FROM market_snapshots
                WHERE ts >= ? AND ts <= ? AND ticker IN ({placeholders})
                ORDER BY ts""",
            conn,
            params=(start_ts, end_ts, *tickers),
        )
    return pd.read_sql_query(
        """SELECT * FROM market_snapshots
           WHERE ts >= ? AND ts <= ?
           ORDER BY ts""",
        conn,
        params=(start_ts, end_ts),
    )


def get_markets(
    conn: sqlite3.Connection,
    status: str | None = None,
    category: str | None = None,
) -> list[Market]:
    """Get markets, optionally filtered by status and category."""
    query = "SELECT * FROM markets WHERE 1=1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if category:
        query += " AND category = ?"
        params.append(category)
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    return [
        Market(
            ticker=row[cols.index("ticker")],
            event_ticker=row[cols.index("event_ticker")],
            title=row[cols.index("title")],
            status=row[cols.index("status")],
            series_ticker=row[cols.index("series_ticker")],
            category=row[cols.index("category")],
            result=row[cols.index("result")],
            open_ts=row[cols.index("open_ts")],
            close_ts=row[cols.index("close_ts")],
            settled_ts=row[cols.index("settled_ts")],
            rules_primary=row[cols.index("rules_primary")],
        )
        for row in rows
    ]


def get_settled_markets(
    conn: sqlite3.Connection, category: str | None = None
) -> list[Market]:
    """Get all settled markets."""
    return get_markets(conn, status="settled", category=category)


def get_news_for_ticker(
    conn: sqlite3.Connection, ticker: str, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """Get news articles linked to a specific market ticker in a time range."""
    return pd.read_sql_query(
        """SELECT na.*, nml.relevance_score
           FROM news_articles na
           JOIN news_market_links nml ON na.id = nml.article_id
           WHERE nml.ticker = ? AND na.fetched_ts >= ? AND na.fetched_ts <= ?
           ORDER BY na.seen_date""",
        conn,
        params=(ticker, start_ts, end_ts),
    )
