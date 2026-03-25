"""Query GDELT news data via Google BigQuery (no rate limits)."""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from kalshi_trader.data.models import NewsArticle

logger = logging.getLogger(__name__)


class BigQueryGDELTClient:
    """Fetches news articles from GDELT's GKG table via BigQuery.

    This is a drop-in replacement for GDELTClient.search_articles() that
    queries the public ``gdelt-bq.gdeltv2.gkg_partitioned`` table instead
    of the DOC 2.0 REST API, avoiding rate limits entirely.

    Requires:
        - ``google-cloud-bigquery`` package
        - A GCP project with billing enabled (queries are metered but the
          1 TiB/month free tier is more than enough for this use case)
        - Credentials: either ``GOOGLE_APPLICATION_CREDENTIALS`` env var
          pointing to a service-account JSON key, or ``gcloud auth
          application-default login`` for local dev.
    """

    # Only web sources have usable URLs.
    _SOURCE_WEB = 1

    def __init__(self, project: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Args:
            project: GCP project ID.  If *None*, the client infers it from
                     the environment / credentials.
            credentials_path: Path to a service-account JSON key.  If *None*,
                              uses Application Default Credentials.
        """
        try:
            from google.cloud import bigquery

            kwargs: dict = {}
            if credentials_path:
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                kwargs["credentials"] = creds
                kwargs["project"] = creds.project_id
            if project:
                kwargs["project"] = project
            self._client = bigquery.Client(**kwargs)
        except Exception as exc:
            logger.warning("BigQuery client init failed: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    # Public API (mirrors GDELTClient)
    # ------------------------------------------------------------------

    def search_articles(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
        max_records: int = 250,
    ) -> list[NewsArticle]:
        """Search GDELT GKG via BigQuery for articles matching *keywords*.

        Args:
            keywords: Search terms — matched against V2Organizations,
                      V2Persons, and AllNames columns.
            start_date: ``YYYY-MM-DD`` inclusive start.
            end_date:   ``YYYY-MM-DD`` inclusive end.
            max_records: Maximum rows to return.

        Returns:
            List of ``NewsArticle`` dataclasses.
        """
        if self._client is None:
            return []

        multi_word = [kw for kw in keywords if len(kw.split()) >= 2]
        if not multi_word:
            return []

        from google.cloud import bigquery

        # Build a regex pattern from keywords, escaping any regex metacharacters.
        escaped = [re.escape(kw) for kw in multi_word]
        pattern = "|".join(escaped)

        sql = """
        SELECT
            DocumentIdentifier,
            DATE,
            SourceCommonName,
            SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
            TranslationInfo
        FROM
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP(@start_date)
            AND _PARTITIONTIME <= TIMESTAMP(@end_date)
            AND SourceCollectionIdentifier = @source_web
            AND (
                REGEXP_CONTAINS(V2Organizations, @pattern)
                OR REGEXP_CONTAINS(AllNames, @pattern)
            )
        ORDER BY DATE DESC
        LIMIT @max_records
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ScalarQueryParameter("source_web", "INT64", self._SOURCE_WEB),
                bigquery.ScalarQueryParameter("pattern", "STRING", pattern),
                bigquery.ScalarQueryParameter("max_records", "INT64", max_records),
            ]
        )

        try:
            df = self._client.query(sql, job_config=job_config).result().to_dataframe()
        except Exception as exc:
            logger.error("BigQuery GDELT search failed: %s", exc)
            return []

        if df.empty:
            return []

        articles: dict[str, NewsArticle] = {}
        for _, row in df.iterrows():
            url = row.get("DocumentIdentifier", "")
            if not url or url in articles:
                continue
            articles[url] = NewsArticle(
                url=url,
                title="",  # GKG does not store article titles
                seen_date=_parse_gkg_date(row.get("DATE")),
                domain=row.get("SourceCommonName"),
                language=_parse_language(row.get("TranslationInfo")),
                source_country=None,  # not directly available in GKG
                tone=_safe_float(row.get("tone")),
            )

        logger.info(
            "BigQuery returned %d articles for keywords: %s",
            len(articles),
            multi_word,
        )
        return list(articles.values())

    def get_timeline(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Daily aggregated tone and volume for *keywords*.

        Returns a DataFrame with columns:
            date, article_count, avg_tone, avg_positive, avg_negative
        """
        if self._client is None:
            return pd.DataFrame()

        multi_word = [kw for kw in keywords if len(kw.split()) >= 2]
        if not multi_word:
            return pd.DataFrame()

        from google.cloud import bigquery

        escaped = [re.escape(kw) for kw in multi_word]
        pattern = "|".join(escaped)

        sql = """
        SELECT
            PARSE_DATE('%Y%m%d', CAST(CAST(FLOOR(DATE / 1000000) AS INT64) AS STRING)) AS date,
            COUNT(*) AS article_count,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_tone,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(1)] AS FLOAT64)) AS avg_positive,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(2)] AS FLOAT64)) AS avg_negative
        FROM
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME >= TIMESTAMP(@start_date)
            AND _PARTITIONTIME <= TIMESTAMP(@end_date)
            AND SourceCollectionIdentifier = @source_web
            AND (
                REGEXP_CONTAINS(V2Organizations, @pattern)
                OR REGEXP_CONTAINS(AllNames, @pattern)
            )
        GROUP BY date
        ORDER BY date
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ScalarQueryParameter("source_web", "INT64", self._SOURCE_WEB),
                bigquery.ScalarQueryParameter("pattern", "STRING", pattern),
            ]
        )

        try:
            return self._client.query(sql, job_config=job_config).result().to_dataframe()
        except Exception as exc:
            logger.error("BigQuery timeline search failed: %s", exc)
            return pd.DataFrame()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_gkg_date(value) -> str:
    """Convert GKG integer date (YYYYMMDDHHMMSS) to ISO-ish string."""
    if value is None:
        return ""
    s = str(int(value))
    if len(s) < 14:
        s = s.ljust(14, "0")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"


def _parse_language(translation_info) -> Optional[str]:
    """Extract source language from GKG TranslationInfo field.

    Blank means English-original.  Translated articles contain
    ``srclc:XXX`` where XXX is an ISO 639 code.
    """
    if not translation_info:
        return "en"
    match = re.search(r"srclc:(\w+)", str(translation_info))
    return match.group(1) if match else "en"


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
