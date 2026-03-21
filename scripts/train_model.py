#!/usr/bin/env python3
"""CLI script to train a model on historical data."""

import argparse
import json
import logging
from pathlib import Path

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.features.base import FeaturePipeline
from kalshi_trader.features.market_features import (
    MarketMomentumFeatures,
    MarketPriceFeatures,
)
from kalshi_trader.models.logistic import LogisticBaseline
from kalshi_trader.models.training import build_training_dataset, cross_validate_model
from kalshi_trader.models.xgboost_model import XGBoostModel


MODELS = {
    "logistic": LogisticBaseline,
    "xgboost": XGBoostModel,
}


def main():
    parser = argparse.ArgumentParser(description="Train a probability model")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="logistic",
        help="Model type to train",
    )
    parser.add_argument("--category", default=None, help="Market category filter")
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Cross-validation folds"
    )
    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="Directory to save trained model",
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

    # Build feature pipeline (market features only for now)
    pipeline = FeaturePipeline([
        MarketPriceFeatures(),
        MarketMomentumFeatures(),
    ])

    # Build training dataset
    X, y, feature_names = build_training_dataset(
        conn, pipeline, category=args.category
    )
    conn.close()

    if len(X) == 0:
        logger.error("No training data available. Run ingestion first.")
        return

    logger.info("Training dataset: %d samples, %d features", len(X), len(feature_names))
    logger.info("Label distribution: %.1f%% YES", y.mean() * 100)

    # Cross-validate
    model_cls = MODELS[args.model]
    cv_results = cross_validate_model(model_cls, X, y, feature_names, n_folds=args.cv_folds)
    logger.info("Cross-validation results:")
    logger.info("  Mean Brier:    %.4f (+/- %.4f)", cv_results["mean_val_brier"], cv_results["std_val_brier"])
    logger.info("  Mean Log Loss: %.4f", cv_results["mean_val_log_loss"])
    logger.info("  Mean Accuracy: %.4f", cv_results["mean_val_accuracy"])

    # Train on full dataset
    model = model_cls()
    metrics = model.fit(X, y, feature_names)
    logger.info("Full training metrics: %s", json.dumps(metrics, indent=2))

    # Save model
    output_path = Path(args.output_dir) / f"{args.model}.pkl"
    model.save(output_path)
    logger.info("Model saved to %s", output_path)

    # Print feature importance if available
    importance = model.feature_importance()
    if importance:
        logger.info("Feature importance:")
        for name, imp in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info("  %-20s %.4f", name, imp)


if __name__ == "__main__":
    main()
