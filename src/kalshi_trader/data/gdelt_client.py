from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from kalshi_trader.data.models import NewsArticle

logger = logging.getLogger(__name__)


class GDELTClient:
    """Fetches news articles and timelines from the GDELT DOC 2.0 API."""

    def __init__(self, request_delay: float = 5.0):
        self._request_delay = request_delay
        try:
            from gdeltdoc import GdeltDoc

            self._client = GdeltDoc()
        except ImportError:
            logger.warning("gdeltdoc not installed. Run: pip install gdeltdoc")
            self._client = None

    def search_articles(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        max_records: int = 250,
    ) -> list[NewsArticle]:
        """
        Search GDELT DOC 2.0 for articles matching keywords.

        Args:
            keywords: List of search terms (OR'd together).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            max_records: Max articles to return (GDELT caps at 250).

        Returns:
            List of NewsArticle dataclasses.
        """
        if self._client is None:
            return []

        from gdeltdoc import Filters

        # Filter to multi-word keywords only — GDELT rejects short/common
        # single words like "Nets", "Heat", "Jazz"
        multi_word = [kw for kw in keywords if len(kw.split()) >= 2]
        if not multi_word:
            return []

        # gdeltdoc wraps the keyword param in quotes automatically, so we
        # can't use OR syntax. Search each keyword separately and merge.
        import time
        all_articles = {}  # url -> NewsArticle (dedup by URL)

        for kw in multi_word:
            time.sleep(self._request_delay)
            try:
                filters = Filters(
                    keyword=kw,
                    start_date=start_date,
                    end_date=end_date,
                    num_records=min(max_records, 250),
                )
                df = self._client.article_search(filters)
            except Exception as e:
                logger.error("GDELT article search failed for %r: %s", kw, e)
                continue

            if df is None or df.empty:
                continue

            for _, row in df.iterrows():
                url = row.get("url", "")
                if url and url not in all_articles:
                    all_articles[url] = NewsArticle(
                        url=url,
                        title=row.get("title", ""),
                        seen_date=row.get("seendate", ""),
                        domain=row.get("domain"),
                        language=row.get("language"),
                        source_country=row.get("sourcecountry"),
                        tone=_parse_tone(row.get("tone")),
                    )

        return list(all_articles.values())

    def get_timeline(
        self,
        keyword: str,
        mode: str = "timelinevol",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get coverage volume or tone timeline for a keyword.

        Args:
            keyword: Search term.
            mode: One of "timelinevol", "timelinevolraw", "timelinetone".
            start_date: YYYY-MM-DD (optional).
            end_date: YYYY-MM-DD (optional).

        Returns:
            DataFrame with (datetime, value) columns.
        """
        if self._client is None:
            return pd.DataFrame()

        from gdeltdoc import Filters

        try:
            filter_kwargs = {"keyword": keyword}
            if start_date:
                filter_kwargs["start_date"] = start_date
            if end_date:
                filter_kwargs["end_date"] = end_date
            filters = Filters(**filter_kwargs)
            df = self._client.timeline_search(mode, filters)
        except Exception as e:
            logger.error("GDELT timeline search failed for %r: %s", keyword, e)
            return pd.DataFrame()

        return df if df is not None else pd.DataFrame()


def _parse_tone(value) -> Optional[float]:
    """Parse GDELT tone value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
