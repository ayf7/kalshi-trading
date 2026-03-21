from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kalshi_trader.config import AppConfig
from kalshi_trader.data.models import Position, Signal


@dataclass
class RiskCheck:
    allowed: bool
    reason: str = ""


class RiskManager:
    """
    Enforces position limits and drawdown controls.
    Called before every signal is converted to an order.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._initial_balance: Optional[int] = None
        self._current_balance: Optional[int] = None
        self._peak_balance: Optional[int] = None

    def set_balance(self, balance_cents: int) -> None:
        """Update the current portfolio balance."""
        if self._initial_balance is None:
            self._initial_balance = balance_cents
            self._peak_balance = balance_cents
        self._current_balance = balance_cents
        if self._peak_balance is not None:
            self._peak_balance = max(self._peak_balance, balance_cents)

    def check_signal(
        self,
        signal: Signal,
        positions: dict[str, Position],
        open_order_count: int = 0,
    ) -> RiskCheck:
        """
        Check if a signal passes all risk limits.

        Args:
            signal: The trading signal to check.
            positions: Current positions keyed by ticker.
            open_order_count: Number of currently resting orders.

        Returns:
            RiskCheck with allowed=True if the signal passes, or reason if not.
        """
        # 1. Per-market position limit
        pos = positions.get(signal.ticker)
        current_size = abs(pos.net_contracts) if pos else 0
        if current_size + signal.quantity > self.config.max_position_per_market:
            return RiskCheck(
                False,
                f"Position limit: {current_size}+{signal.quantity} > "
                f"{self.config.max_position_per_market}",
            )

        # 2. Total exposure limit
        total_exposure = sum(
            abs(p.net_contracts) * p.avg_entry_price for p in positions.values()
        )
        new_exposure = signal.quantity * signal.price_cents
        if total_exposure + new_exposure > self.config.max_total_exposure_cents:
            return RiskCheck(
                False,
                f"Total exposure: {total_exposure}+{new_exposure} > "
                f"{self.config.max_total_exposure_cents}",
            )

        # 3. Open order limit
        if open_order_count >= self.config.max_open_orders:
            return RiskCheck(
                False,
                f"Open order limit: {open_order_count} >= {self.config.max_open_orders}",
            )

        # 4. Drawdown circuit breaker
        if self._peak_balance is not None and self._current_balance is not None:
            if self._peak_balance > 0:
                drawdown = 1.0 - (self._current_balance / self._peak_balance)
                if drawdown >= self.config.max_drawdown_fraction:
                    return RiskCheck(
                        False,
                        f"Drawdown {drawdown:.1%} >= limit "
                        f"{self.config.max_drawdown_fraction:.1%}",
                    )

        return RiskCheck(True)
