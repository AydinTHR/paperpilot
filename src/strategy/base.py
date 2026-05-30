"""Strategy interface: the contract every trading strategy implements.

A strategy is a pure function of price history -> :class:`Signal`. It never
touches the broker, the network, or the wall clock; that keeps strategies
deterministic, unit-testable offline, and reusable by both the Phase 3
backtester and the Phase 5 live loop.

Look-ahead safety is part of the contract: :meth:`Strategy.generate_signals`
returns the decision *as of the last row* of ``data`` and may use only data up
to and including that row. Appending future bars must never change a past
decision (see the look-ahead test in ``tests/test_strategy.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class Action(str, Enum):
    """What the strategy wants to do on the most recent bar."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    """A strategy's probabilistic decision for the latest bar.

    Attributes:
        action: BUY / SELL / HOLD.
        confidence: Strength of the signal in ``[0, 1]``. ``HOLD`` is typically
            ``0.0``. This is a *relative* conviction score, not a probability of
            profit.
        size_hint: Optional suggested position size as a fraction of equity in
            ``[0, 1]``. Only a hint -- the Phase 4 risk manager has the final say
            and may shrink or veto it.
        reason: Short human-readable explanation, useful for logs and the
            preview script.
    """

    action: Action
    confidence: float = 0.0
    size_hint: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        if self.size_hint is not None and not 0.0 <= self.size_hint <= 1.0:
            raise ValueError(
                f"size_hint must be in [0, 1] or None, got {self.size_hint!r}"
            )


class Strategy(ABC):
    """Base class for all strategies.

    Subclasses implement :attr:`min_bars` (warm-up length) and
    :meth:`generate_signals`. The OHLCV schema is validated for them via
    :meth:`_check`, and :meth:`hold` provides the canonical no-op signal.
    """

    #: Columns every strategy expects -- matches the market-data layer's output
    #: (and ``backtesting.py``'s expected schema in Phase 3).
    REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

    #: Human-readable name; subclasses override in ``__init__``.
    name: str = "Strategy"

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars required before a non-HOLD signal is possible."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Signal:
        """Return the decision as of the last row of ``data``.

        Implementations MUST use only data up to and including the final row
        (no look-ahead) and MUST return ``HOLD`` when ``len(data) < min_bars``.
        """

    # --- helpers for subclasses ---

    def _check(self, data: pd.DataFrame) -> None:
        """Validate the OHLCV schema; raise ``ValueError`` on a bad frame."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: data is missing required column(s) "
                f"{missing}; expected {list(self.REQUIRED_COLUMNS)}."
            )

    def _has_enough_bars(self, data: pd.DataFrame) -> bool:
        return len(data) >= self.min_bars

    @staticmethod
    def hold(reason: str = "") -> Signal:
        """The canonical no-op signal."""
        return Signal(action=Action.HOLD, confidence=0.0, reason=reason)
