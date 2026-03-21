"""Shared test fixtures."""

import sqlite3

import pytest

from kalshi_trader.data.db import SCHEMA_SQL


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with the full schema."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn
    conn.close()
