"""SMA crossover -- a classic dual moving-average trend follower."""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Action, Signal, Strategy
from src.strategy.indicators import sma


class SmaCrossover(Strategy):
    """Buy golden crosses, sell death crosses.

    A *golden cross* is the fast SMA crossing above the slow SMA between the
    previous and current bar -> BUY; a *death cross* is the reverse -> SELL;
    otherwise HOLD. Confidence scales with the normalised gap between the two
    SMAs, so a decisive separation reads as more conviction than a marginal one.
    """

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow}).")
        self.fast = fast
        self.slow = slow
        self.name = f"SMA({fast}/{slow}) crossover"

    @property
    def min_bars(self) -> int:
        # Need the slow SMA defined on two consecutive bars to detect a cross.
        return self.slow + 1

    def generate_signals(self, data: pd.DataFrame) -> Signal:
        self._check(data)
        if not self._has_enough_bars(data):
            return self.hold(
                reason=f"warming up: need {self.min_bars} bars, have {len(data)}"
            )

        close = data["Close"]
        fast_sma = sma(close, self.fast)
        slow_sma = sma(close, self.slow)

        fast_now, fast_prev = fast_sma.iloc[-1], fast_sma.iloc[-2]
        slow_now, slow_prev = slow_sma.iloc[-1], slow_sma.iloc[-2]

        if any(pd.isna(v) for v in (fast_now, fast_prev, slow_now, slow_prev)):
            return self.hold(reason="indicator warm-up incomplete")

        # Normalise the current gap by price -> a unitless, clamped confidence.
        confidence = min(abs(fast_now - slow_now) / slow_now, 1.0) if slow_now else 0.0

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up:
            return Signal(
                Action.BUY,
                confidence=confidence,
                reason=(
                    f"golden cross: SMA{self.fast} {fast_now:.2f} "
                    f"> SMA{self.slow} {slow_now:.2f}"
                ),
            )
        if crossed_down:
            return Signal(
                Action.SELL,
                confidence=confidence,
                reason=(
                    f"death cross: SMA{self.fast} {fast_now:.2f} "
                    f"< SMA{self.slow} {slow_now:.2f}"
                ),
            )
        return self.hold(
            reason=f"no cross (SMA{self.fast}={fast_now:.2f}, SMA{self.slow}={slow_now:.2f})"
        )
