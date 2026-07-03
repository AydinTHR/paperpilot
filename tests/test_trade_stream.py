"""Tests for the optional trade-stream fill listener (fake stream, no network)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from config.settings import Settings
from src.execution.trade_stream import TradeStreamListener
from src.journal.store import Journal


class _FakeStream:
    def __init__(self) -> None:
        self.handler = None
        self.ran = False

    def subscribe_trade_updates(self, handler) -> None:
        self.handler = handler

    def run(self) -> None:
        self.ran = True


def _update(event: str, **order_fields) -> SimpleNamespace:
    return SimpleNamespace(event=event, order=SimpleNamespace(**order_fields))


def _listener(journal: Journal) -> tuple[TradeStreamListener, _FakeStream]:
    stream = _FakeStream()
    listener = TradeStreamListener(Settings(), journal, stream=stream)
    return listener, stream


def test_start_subscribes_and_runs_in_background() -> None:
    listener, stream = _listener(Journal("sqlite:///:memory:"))
    listener.start()
    listener.start()  # idempotent: no second thread
    deadline = time.monotonic() + 2.0
    while not stream.ran and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream.ran is True
    assert stream.handler is not None


def test_fill_event_reconciles_journal_row() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_order(
        symbol="AAPL", side="buy", qty=10, status="accepted", broker_order_id="b-1"
    )
    listener, _ = _listener(journal)

    asyncio.run(
        listener._on_update(
            _update("fill", id="b-1", status="filled", filled_qty="10", filled_avg_price="101.25")
        )
    )
    row = journal.recent_orders()[0]
    assert row.status == "filled"
    assert row.filled_qty == 10.0
    assert row.filled_avg_price == 101.25


def test_non_fill_events_ignored() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_order(
        symbol="AAPL", side="buy", qty=10, status="accepted", broker_order_id="b-1"
    )
    listener, _ = _listener(journal)
    asyncio.run(listener._on_update(_update("new", id="b-1", status="new", filled_qty="0")))
    assert journal.recent_orders()[0].status == "accepted"  # untouched


def test_malformed_update_never_raises() -> None:
    listener, _ = _listener(Journal("sqlite:///:memory:"))
    asyncio.run(listener._on_update(SimpleNamespace()))  # no event/order at all
