from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from kalshi_trader.data import db
from kalshi_trader.features.base import FeaturePipeline
from kalshi_trader.models.base import ProbabilityModel

logger = logging.getLogger(__name__)


def build_training_dataset(
    conn: sqlite3.Connection,
    feature_pipeline: FeaturePipeline,
    category: str | None = None,
    sample_offsets_hours: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a training dataset from settled markets.

    For each settled market, extracts features at N hours before close time.
    Label = 1 if result == "yes", 0 if result == "no".

    Args:
        conn: SQLite connection.
        feature_pipeline: Feature pipeline to extract features.
        category: Optional market category filter.
        sample_offsets_hours: Hours before close to sample features.
            Defaults to [1.0, 6.0, 24.0].

    Returns:
        (X, y, feature_names) tuple.
    """
    if sample_offsets_hours is None:
        sample_offsets_hours = [1.0, 6.0, 24.0]

    settled = db.get_settled_markets(conn, category=category)
    logger.info("Found %d settled markets for training", len(settled))

    X_rows = []
    y_labels = []

    for market in settled:
        if market.result is None or market.close_ts is None:
            continue

        label = 1.0 if market.result == "yes" else 0.0

        # Get all snapshots for this market
        # Use a wide window: from open to close
        start_ts = market.open_ts or (market.close_ts - 30 * 86400)
        snapshots = db.get_snapshots(conn, market.ticker, start_ts, market.close_ts)

        if snapshots.empty:
            continue

        # Get news if available
        news = db.get_news_for_ticker(conn, market.ticker, start_ts, market.close_ts)

        for offset_hours in sample_offsets_hours:
            as_of_ts = market.close_ts - int(offset_hours * 3600)

            # Skip if as_of_ts is before market opened
            if market.open_ts and as_of_ts < market.open_ts:
                continue

            # Skip if no snapshots exist before this time
            if (snapshots["ts"] <= as_of_ts).sum() == 0:
                continue

            try:
                features = feature_pipeline.extract(
                    ticker=market.ticker,
                    as_of_ts=as_of_ts,
                    snapshots=snapshots,
                    news=news if not news.empty else None,
                    close_ts=market.close_ts,
                )
                X_rows.append(features)
                y_labels.append(label)
            except Exception as e:
                logger.warning(
                    "Feature extraction failed for %s at offset %.1fh: %s",
                    market.ticker,
                    offset_hours,
                    e,
                )

    if not X_rows:
        logger.warning("No training samples generated")
        return np.array([]), np.array([]), feature_pipeline.feature_names

    X = np.array(X_rows)
    y = np.array(y_labels)
    logger.info("Built training dataset: %d samples, %d features", len(X), X.shape[1])
    return X, y, feature_pipeline.feature_names


def cross_validate_model(
    model_cls: type[ProbabilityModel],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    **model_kwargs,
) -> dict:
    """
    Time-series aware cross-validation.

    Uses expanding window (TimeSeriesSplit) to avoid using future data.

    Returns:
        Dict with averaged metrics across folds.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    tscv = TimeSeriesSplit(n_splits=n_folds)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_cls(**model_kwargs)

        # Use 0 validation split since we're handling it ourselves
        model.fit(X_train, y_train, feature_names, validation_split=0.0)

        val_probs = model.predict_proba(X_val)

        fold_metrics.append(
            {
                "fold": fold,
                "val_brier": brier_score_loss(y_val, val_probs),
                "val_log_loss": log_loss(y_val, val_probs),
                "val_accuracy": ((val_probs >= 0.5) == y_val).mean(),
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )

    # Average metrics across folds
    df = pd.DataFrame(fold_metrics)
    avg = {
        "mean_val_brier": df["val_brier"].mean(),
        "std_val_brier": df["val_brier"].std(),
        "mean_val_log_loss": df["val_log_loss"].mean(),
        "mean_val_accuracy": df["val_accuracy"].mean(),
        "n_folds": n_folds,
        "fold_details": fold_metrics,
    }
    return avg
