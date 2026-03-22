from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from kalshi_trader.backtest.metrics import BacktestMetrics, compute_metrics
from kalshi_trader.backtest.sim_exchange import SimulatedExchange
from kalshi_trader.data import db
from kalshi_trader.data.models import (
    Action,
    Market,
    MarketSnapshot,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from kalshi_trader.features.base import FeaturePipeline
from kalshi_trader.models.base import ProbabilityModel
from kalshi_trader.strategy.base import TradingStrategy
from kalshi_trader.strategy.risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str  # ISO date, e.g. "2025-01-01"
    end_date: str
    tickers: list[str] = field(default_factory=list)  # empty = all settled in range
    initial_balance_cents: int = 10000  # $100
    fee_per_contract_cents: int = 2
    sample_interval: int = 1  # evaluate every N-th snapshot per market


@dataclass
class BacktestResult:
    config: BacktestConfig
    trade_log: list[Order]
    equity_curve: list[tuple[int, int]]  # [(ts, balance_cents)]
    metrics: BacktestMetrics
    model_predictions: list[tuple[float, int]]  # [(model_prob, outcome)]


class BacktestEngine:
    """
    Replays historical market snapshots through a model + strategy
    and simulates order execution.
    """

    def __init__(
        self,
        db_path: str,
        model: ProbabilityModel,
        feature_pipeline: FeaturePipeline,
        strategy: TradingStrategy,
        risk_manager: RiskManager,
        config: BacktestConfig,
    ):
        self.db_path = db_path
        self.model = model
        self.pipeline = feature_pipeline
        self.strategy = strategy
        self.risk = risk_manager
        self.config = config

    def run(self) -> BacktestResult:
        """
        Main backtest loop.

        1. Load snapshots in [start_date, end_date], ordered by ts.
        2. Load market metadata.
        3. For each snapshot chronologically:
           a. Extract features.
           b. model.predict_proba -> model_prob.
           c. strategy.on_snapshot -> signals.
           d. Risk check each signal.
           e. Submit passing signals to SimulatedExchange.
           f. Check for fills.
           g. Record equity state.
        4. At market settlement, close positions.
        5. Compute and return metrics.
        """
        conn = db.init_db(self.db_path)
        exchange = SimulatedExchange(self.config.fee_per_contract_cents)

        # Convert dates to timestamps
        start_ts = int(
            datetime.strptime(self.config.start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        end_ts = int(
            datetime.strptime(self.config.end_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        # Load all snapshots in range
        all_snapshots = db.get_all_snapshots_in_range(
            conn, start_ts, end_ts, self.config.tickers or None
        )

        if all_snapshots.empty:
            logger.warning("No snapshots found in date range")
            conn.close()
            empty_metrics = compute_metrics([], [], None, self.config.initial_balance_cents)
            return BacktestResult(
                config=self.config,
                trade_log=[],
                equity_curve=[(start_ts, self.config.initial_balance_cents)],
                metrics=empty_metrics,
                model_predictions=[],
            )

        # Load market metadata
        all_markets = db.get_markets(conn)
        market_map: dict[str, Market] = {m.ticker: m for m in all_markets}

        # Track state
        balance = self.config.initial_balance_cents
        positions: dict[str, Position] = {}
        trade_log: list[Order] = []
        equity_curve: list[tuple[int, int]] = [(start_ts, balance)]
        model_predictions: list[tuple[float, int]] = []
        settled_tickers: set[str] = set()

        self.risk.set_balance(balance)

        # Get unique tickers in the dataset
        tickers_in_data = all_snapshots["ticker"].unique().tolist()

        # Pre-load snapshots per ticker for feature extraction
        ticker_snapshots: dict[str, object] = {}
        for ticker in tickers_in_data:
            ticker_snapshots[ticker] = all_snapshots[
                all_snapshots["ticker"] == ticker
            ].copy()

        # Subsample snapshots if configured (keep every N-th per ticker)
        if self.config.sample_interval > 1:
            sampled_indices = []
            for ticker in tickers_in_data:
                ticker_idx = all_snapshots[all_snapshots["ticker"] == ticker].index
                sampled_indices.extend(ticker_idx[:: self.config.sample_interval])
            iter_snapshots = all_snapshots.loc[sampled_indices].sort_values("ts")
            logger.info(
                "Subsampled %d -> %d snapshots (interval=%d)",
                len(all_snapshots), len(iter_snapshots), self.config.sample_interval,
            )
        else:
            iter_snapshots = all_snapshots

        # Process snapshots chronologically
        for _, snap_row in iter_snapshots.iterrows():
            ticker = snap_row["ticker"]
            ts = int(snap_row["ts"])

            if ticker in settled_tickers:
                continue

            market = market_map.get(ticker)
            if market is None:
                continue

            # In backtest, settle markets when we reach their close_ts,
            # not based on the result field (which is already known).
            if (
                market.result is not None
                and market.close_ts is not None
                and ts >= market.close_ts
                and ticker not in settled_tickers
            ):
                settled_tickers.add(ticker)
                pos = positions.get(ticker)
                if pos and pos.net_contracts != 0:
                    settlement_value = 100 if market.result == "yes" else 0
                    pnl = self._settle_position(pos, settlement_value)
                    balance += pnl
                    outcome = 1 if market.result == "yes" else 0
                    model_predictions.append((0.5, outcome))
                exchange.settle_market(ticker, market.result)
                equity_curve.append((ts, balance))
                continue

            snapshot = MarketSnapshot(
                ticker=ticker,
                ts=ts,
                yes_bid=_safe_int(snap_row.get("yes_bid")),
                yes_ask=_safe_int(snap_row.get("yes_ask")),
                last_price=_safe_int(snap_row.get("last_price")),
                volume=_safe_int(snap_row.get("volume")),
                open_interest=_safe_int(snap_row.get("open_interest")),
                yes_bid_size=_safe_int(snap_row.get("yes_bid_size")),
                yes_ask_size=_safe_int(snap_row.get("yes_ask_size")),
            )

            # Check for fills on existing orders
            fills = exchange.process_snapshot(snapshot)
            for filled_order in fills:
                balance, positions = self._process_fill(filled_order, balance, positions)
                trade_log.append(filled_order)

            # Extract features and get model prediction
            try:
                features = self.pipeline.extract(
                    ticker=ticker,
                    as_of_ts=ts,
                    snapshots=ticker_snapshots[ticker],
                    close_ts=market.close_ts,
                )
                import numpy as np

                X = features.reshape(1, -1)
                model_prob = float(self.model.predict_proba(X)[0])
            except Exception:
                model_prob = 0.5  # fallback

            # Get current position
            pos = positions.get(ticker, Position(ticker=ticker))

            # Generate signals
            signals = self.strategy.on_snapshot(
                market, snapshot, None, model_prob, pos
            )

            # Risk check and submit
            open_order_count = len(exchange.resting_orders)
            for signal in signals:
                check = self.risk.check_signal(signal, positions, open_order_count)
                if not check.allowed:
                    continue

                order = Order(
                    order_id=str(uuid.uuid4()),
                    ticker=signal.ticker,
                    ts=signal.ts,
                    side=signal.side,
                    action=signal.action,
                    price_cents=signal.price_cents,
                    quantity=signal.quantity,
                    order_type=OrderType.IOC,
                    status=OrderStatus.SUBMITTED,
                    source="backtest",
                    strategy_name=signal.strategy_name,
                    model_prob=signal.model_prob,
                )
                exchange.submit_order(order, snapshot)

                # If immediately filled (FOK/IOC or market order)
                if order.status == OrderStatus.FILLED:
                    balance, positions = self._process_fill(order, balance, positions)
                    trade_log.append(order)

            # Update balance tracking
            self.risk.set_balance(balance)
            equity_curve.append((ts, balance))

        # Final settlement pass: settle any remaining positions
        for ticker in tickers_in_data:
            if ticker in settled_tickers:
                continue
            market = market_map.get(ticker)
            if market and market.result is not None:
                pos = positions.get(ticker)
                if pos and pos.net_contracts != 0:
                    settlement_value = 100 if market.result == "yes" else 0
                    pnl = self._settle_position(pos, settlement_value)
                    balance += pnl
                    outcome = 1 if market.result == "yes" else 0
                    model_predictions.append((0.5, outcome))
                exchange.settle_market(ticker, market.result)

        equity_curve.append((end_ts, balance))

        conn.close()

        metrics = compute_metrics(
            trade_log, equity_curve, model_predictions, self.config.initial_balance_cents
        )

        return BacktestResult(
            config=self.config,
            trade_log=trade_log,
            equity_curve=equity_curve,
            metrics=metrics,
            model_predictions=model_predictions,
        )

    def _process_fill(
        self, order: Order, balance: int, positions: dict[str, Position]
    ) -> tuple[int, dict[str, Position]]:
        """Update balance and positions after a fill."""
        pos = positions.get(order.ticker, Position(ticker=order.ticker))

        cost = order.filled_quantity * order.price_cents + order.fees_cents

        if order.action == Action.BUY:
            if order.side == Side.YES:
                # Buying YES: pay price, gain YES contracts
                balance -= cost
                pos.net_contracts += order.filled_quantity
            else:
                # Buying NO: pay price, gain NO contracts (negative YES)
                balance -= cost
                pos.net_contracts -= order.filled_quantity
        else:
            # Selling
            if order.side == Side.YES:
                balance += order.filled_quantity * order.price_cents - order.fees_cents
                pos.net_contracts -= order.filled_quantity
            else:
                balance += order.filled_quantity * order.price_cents - order.fees_cents
                pos.net_contracts += order.filled_quantity

        pos.fees_paid += order.fees_cents
        positions[order.ticker] = pos
        return balance, positions

    def _settle_position(self, pos: Position, settlement_value: int) -> int:
        """
        Settle a position at the given settlement value (100 for YES, 0 for NO).
        Returns PnL in cents.
        """
        if pos.net_contracts > 0:
            # Long YES: receive settlement_value per contract
            pnl = pos.net_contracts * settlement_value
        elif pos.net_contracts < 0:
            # Long NO: receive (100 - settlement_value) per contract
            pnl = abs(pos.net_contracts) * (100 - settlement_value)
        else:
            pnl = 0

        pos.net_contracts = 0
        return pnl


def _safe_int(val) -> int | None:
    """Safely convert a value to int, returning None for NaN/None."""
    if val is None:
        return None
    try:
        import math

        if math.isnan(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None
