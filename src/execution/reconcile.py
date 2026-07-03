"""Order fill reconciliation: poll a submitted order to its terminal state.

The loop submits market orders and, until now, journaled the submit-time
snapshot ("accepted", no fill price). Real fills land asynchronously, and
Alpaca's paper simulator deliberately delivers random partial fills 10% of the
time, so the journal must record what actually happened. ``OrderReconciler``
polls with bounded exponential backoff and returns the best-known final state;
it never raises into the trading loop.

The clock and sleeper are injected callables so tests drive the whole schedule
with fake time (no real sleeping).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from config.logging_config import get_logger
from src.execution.broker import OrderInfo

logger = get_logger(__name__)

# Order statuses after which the fill state can no longer change.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day"}
)


class OrderReader(Protocol):
    """The single capability reconciliation needs from the broker."""

    def get_order(self, order_id: str) -> OrderInfo: ...


@dataclass(frozen=True)
class ReconcileResult:
    """Best-known final state of one order after polling."""

    order_id: str
    terminal: bool
    status: str
    filled_qty: float
    filled_avg_price: float | None
    attempts: int
    elapsed_s: float


class OrderReconciler:
    """Polls one order to a terminal status with bounded backoff."""

    def __init__(
        self,
        reader: OrderReader,
        *,
        max_wait_s: float = 30.0,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._reader = reader
        self.max_wait_s = max_wait_s
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self._clock = clock
        self._sleeper = sleeper

    def wait_for_terminal(self, order_id: str) -> ReconcileResult:
        """Poll until the order is terminal or ``max_wait_s`` elapses.

        Read errors are logged and count as an attempt; a partial fill's
        progress is captured in the returned snapshot even on timeout. This
        method never raises: a tick must survive a flaky order endpoint.
        """
        started = self._clock()
        attempts = 0
        delay = self.base_delay_s
        last: OrderInfo | None = None

        while True:
            attempts += 1
            try:
                last = self._reader.get_order(order_id)
                status = last.status.lower()
                if status in TERMINAL_STATUSES:
                    return self._result(order_id, last, True, attempts, started)
            except Exception as exc:
                logger.warning(
                    "Order %s status read failed (attempt %d): %s", order_id, attempts, exc
                )

            if self._clock() - started + delay > self.max_wait_s:
                logger.warning(
                    "Order %s not terminal after %d attempt(s); journaling last known state.",
                    order_id,
                    attempts,
                )
                return self._result(order_id, last, False, attempts, started)
            self._sleeper(delay)
            delay = min(delay * 2, self.max_delay_s)

    def _result(
        self,
        order_id: str,
        order: OrderInfo | None,
        terminal: bool,
        attempts: int,
        started: float,
    ) -> ReconcileResult:
        return ReconcileResult(
            order_id=order_id,
            terminal=terminal,
            status=order.status if order else "unknown",
            filled_qty=order.filled_qty if order else 0.0,
            filled_avg_price=order.filled_avg_price if order else None,
            attempts=attempts,
            elapsed_s=self._clock() - started,
        )
