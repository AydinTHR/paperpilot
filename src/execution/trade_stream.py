"""Optional websocket fill listener (Alpaca ``TradingStream``).

Off by default (``USE_TRADE_STREAM=false``). The synchronous loop already
reconciles fills by polling right after submit; this listener is an alternative
source that catches fills landing *between* ticks (for example a broker-held
stop leg firing mid-interval) and reconciles them into the journal via
``Journal.update_order_fill``.

``TradingStream.run()`` blocks and owns its own asyncio loop, so it runs in a
daemon thread beside the BlockingScheduler. The stream object is injectable for
tests; the real one is built lazily from settings.
"""

from __future__ import annotations

import threading
from typing import Any

from config.logging_config import get_logger
from config.settings import Settings
from src.journal.store import Journal

logger = get_logger(__name__)

# Trade-update events that carry fill state worth journaling.
_FILL_EVENTS: frozenset[str] = frozenset({"fill", "partial_fill", "canceled", "rejected"})


class TradeStreamListener:
    """Runs a trade-updates stream in a daemon thread, journaling fills."""

    def __init__(
        self,
        settings: Settings,
        journal: Journal,
        *,
        stream: Any | None = None,
    ) -> None:
        self.settings = settings
        self.journal = journal
        self._stream = stream
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Subscribe and run the stream in the background (idempotent)."""
        if self._thread is not None:
            return
        stream = self._get_stream()
        stream.subscribe_trade_updates(self._on_update)
        self._thread = threading.Thread(target=stream.run, name="trade-stream", daemon=True)
        self._thread.start()
        logger.info("Trade stream listener started.")

    async def _on_update(self, data: Any) -> None:
        """Journal fill state from one trade-update event; never raises."""
        try:
            event = str(getattr(data, "event", ""))
            if event not in _FILL_EVENTS:
                return
            order = getattr(data, "order", None)
            order_id = str(getattr(order, "id", ""))
            if not order_id:
                return
            self.journal.update_order_fill(
                order_id,
                status=str(getattr(order, "status", event)),
                filled_qty=float(getattr(order, "filled_qty", 0.0) or 0.0),
                filled_avg_price=(
                    float(price) if (price := getattr(order, "filled_avg_price", None)) else None
                ),
            )
            logger.info("Trade stream reconciled order %s (%s).", order_id, event)
        except Exception:
            logger.exception("Trade-update handling failed; stream continues.")

    def _get_stream(self) -> Any:
        if self._stream is None:  # pragma: no cover - real SDK path
            from alpaca.trading.stream import TradingStream

            self._stream = TradingStream(
                api_key=self.settings.alpaca_api_key.get_secret_value(),
                secret_key=self.settings.alpaca_secret_key.get_secret_value(),
                paper=self.settings.paper,
            )
        return self._stream
