from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalshi_trader.data.models import Order


@dataclass
class BacktestMetrics:
    total_pnl_cents: int
    total_return_pct: float
    num_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    max_drawdown_pct: float
    max_drawdown_cents: int
    sharpe_ratio: float
    brier_score: float
    total_fees_cents: int
    gross_pnl_cents: int


def compute_metrics(
    trade_log: list[Order],
    equity_curve: list[tuple[int, int]],
    model_predictions: list[tuple[float, int]] | None = None,
    initial_balance: int = 10000,
) -> BacktestMetrics:
    """
    Compute backtest performance metrics.

    Args:
        trade_log: List of filled orders.
        equity_curve: List of (timestamp, balance_cents) tuples.
        model_predictions: List of (model_prob, outcome 0/1) for Brier score.
        initial_balance: Starting balance in cents.
    """
    total_fees = sum(o.fees_cents for o in trade_log)
    num_trades = len(trade_log)

    # Final PnL from equity curve
    if equity_curve:
        final_balance = equity_curve[-1][1]
        total_pnl = final_balance - initial_balance
    else:
        total_pnl = 0
        final_balance = initial_balance

    gross_pnl = total_pnl + total_fees
    total_return = total_pnl / initial_balance if initial_balance > 0 else 0.0

    # Win rate (trades that contributed positively)
    # This is approximate -- we track by order, not by round-trip
    wins = sum(1 for o in trade_log if o.fees_cents < o.price_cents)
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    avg_pnl = total_pnl / num_trades if num_trades > 0 else 0.0

    # Max drawdown from equity curve
    max_dd_pct, max_dd_cents = _compute_max_drawdown(equity_curve)

    # Sharpe ratio from equity curve
    sharpe = _compute_sharpe(equity_curve)

    # Brier score
    brier = 0.0
    if model_predictions:
        probs = np.array([p for p, _ in model_predictions])
        outcomes = np.array([o for _, o in model_predictions])
        brier = float(np.mean((probs - outcomes) ** 2))

    return BacktestMetrics(
        total_pnl_cents=total_pnl,
        total_return_pct=total_return,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_pnl_per_trade=avg_pnl,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_cents=max_dd_cents,
        sharpe_ratio=sharpe,
        brier_score=brier,
        total_fees_cents=total_fees,
        gross_pnl_cents=gross_pnl,
    )


def _compute_max_drawdown(
    equity_curve: list[tuple[int, int]],
) -> tuple[float, int]:
    """Compute max peak-to-trough drawdown."""
    if len(equity_curve) < 2:
        return 0.0, 0

    balances = [b for _, b in equity_curve]
    peak = balances[0]
    max_dd_cents = 0
    max_dd_pct = 0.0

    for balance in balances:
        peak = max(peak, balance)
        dd = peak - balance
        if dd > max_dd_cents:
            max_dd_cents = dd
            max_dd_pct = dd / peak if peak > 0 else 0.0

    return max_dd_pct, max_dd_cents


def _compute_sharpe(
    equity_curve: list[tuple[int, int]],
    periods_per_year: float = 252.0,
) -> float:
    """
    Compute annualized Sharpe ratio from equity curve.
    Assumes zero risk-free rate.
    """
    if len(equity_curve) < 3:
        return 0.0

    balances = np.array([b for _, b in equity_curve], dtype=np.float64)
    returns = np.diff(balances) / balances[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))
