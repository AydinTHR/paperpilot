"""RSI mean-reversion -- fade oversold/overbought extremes."""

from __future__ import annotations

import pandas as pd

from src.strategy.base import Action, Signal, Strategy
from src.strategy.indicators import rsi


class RsiMeanReversion(Strategy):
    """Buy oversold, sell overbought, using Wilder's RSI.

    When the latest RSI falls below ``oversold`` the move is considered
    stretched to the downside and likely to revert up -> BUY. Above
    ``overbought`` -> SELL. Confidence scales with how far past the threshold
    the RSI sits, so a deeper extreme reads as a stronger signal.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if not 0 < oversold < overbought < 100:
            raise ValueError(
                "require 0 < oversold < overbought < 100, got "
                f"oversold={oversold}, overbought={overbought}."
            )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI({period}) mean-reversion"

    @property
    def min_bars(self) -> int:
        return self.period + 1

    def generate_signals(self, data: pd.DataFrame) -> Signal:
        self._check(data)
        if not self._has_enough_bars(data):
            return self.hold(reason=f"warming up: need {self.min_bars} bars, have {len(data)}")

        latest = rsi(data["Close"], self.period).iloc[-1]
        if pd.isna(latest):
            return self.hold(reason="RSI warm-up incomplete")

        if latest < self.oversold:
            confidence = min((self.oversold - latest) / self.oversold, 1.0)
            return Signal(
                Action.BUY,
                confidence=confidence,
                reason=f"oversold: RSI {latest:.1f} < {self.oversold:.0f}",
            )
        if latest > self.overbought:
            confidence = min((latest - self.overbought) / (100.0 - self.overbought), 1.0)
            return Signal(
                Action.SELL,
                confidence=confidence,
                reason=f"overbought: RSI {latest:.1f} > {self.overbought:.0f}",
            )
        return self.hold(reason=f"neutral: RSI {latest:.1f}")
