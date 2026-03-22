#!/usr/bin/env python3
"""CLI script to run a backtest."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from kalshi_trader.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_trader.config import AppConfig
from kalshi_trader.features.base import FeaturePipeline
from kalshi_trader.features.market_features import (
    MarketMomentumFeatures,
    MarketPriceFeatures,
)
from kalshi_trader.models.logistic import LogisticBaseline
from kalshi_trader.models.naive import (
    ContrarianModel,
    MarketImpliedModel,
    MostLikelyModel,
    RandomModel,
)
from kalshi_trader.models.xgboost_model import XGBoostModel
from kalshi_trader.strategy.risk import RiskManager
from kalshi_trader.strategy.signal import SignalStrategy


MODELS = {
    "logistic": LogisticBaseline,
    "xgboost": XGBoostModel,
    "random": RandomModel,
    "most_likely": MostLikelyModel,
    "contrarian": ContrarianModel,
    "market_implied": MarketImpliedModel,
}


def main():
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="logistic",
        help="Model type",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained model file. If not provided, trains on data before start date.",
    )
    parser.add_argument(
        "--tickers", nargs="*", default=[], help="Specific tickers to backtest"
    )
    parser.add_argument(
        "--balance", type=int, default=10000, help="Initial balance in cents (default: $100)"
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.05, help="Minimum edge to trade"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=6,
        help="Only evaluate every N-th snapshot per market (default: 6, ~every 6 hours)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = AppConfig()
    config.min_edge = args.min_edge

    # Build components
    pipeline = FeaturePipeline([
        MarketPriceFeatures(),
        MarketMomentumFeatures(),
    ])

    model_cls = MODELS[args.model]
    model = model_cls()

    if args.model_path:
        model.load(Path(args.model_path))
        logger.info("Loaded model from %s", args.model_path)
    else:
        # Train on data before the backtest period
        from kalshi_trader.data import db
        from kalshi_trader.models.training import build_training_dataset

        conn = db.init_db(config.db_path)
        X, y, feature_names = build_training_dataset(conn, pipeline)
        conn.close()

        if len(X) == 0:
            logger.error("No training data. Run ingestion + wait for settlements.")
            return

        metrics = model.fit(X, y, feature_names)
        logger.info("Trained %s model: %s", args.model, json.dumps(metrics, indent=2))

    strategy = SignalStrategy(
        min_edge=args.min_edge,
        base_size=config.default_order_size,
        max_position=config.max_position_per_market,
    )

    risk_manager = RiskManager(config)

    bt_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        initial_balance_cents=args.balance,
        sample_interval=args.sample_interval,
    )

    # Run backtest
    engine = BacktestEngine(
        db_path=config.db_path,
        model=model,
        feature_pipeline=pipeline,
        strategy=strategy,
        risk_manager=risk_manager,
        config=bt_config,
    )

    logger.info("Running backtest: %s to %s", args.start, args.end)
    result = engine.run()

    # Print results
    m = result.metrics
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:          {args.start} to {args.end}")
    print(f"Model:           {args.model}")
    print(f"Strategy:        signal (min_edge={args.min_edge})")
    print(f"Initial Balance: ${args.balance / 100:.2f}")
    print("-" * 60)
    print(f"Total PnL:       ${m.total_pnl_cents / 100:.2f} ({m.total_return_pct:+.1%})")
    print(f"Gross PnL:       ${m.gross_pnl_cents / 100:.2f}")
    print(f"Total Fees:      ${m.total_fees_cents / 100:.2f}")
    print(f"Num Trades:      {m.num_trades}")
    print(f"Win Rate:        {m.win_rate:.1%}")
    print(f"Avg PnL/Trade:   ${m.avg_pnl_per_trade / 100:.2f}")
    print(f"Max Drawdown:    {m.max_drawdown_pct:.1%} (${m.max_drawdown_cents / 100:.2f})")
    print(f"Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"Brier Score:     {m.brier_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
