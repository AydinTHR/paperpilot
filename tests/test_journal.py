"""Tests for the SQLite trade journal -- offline, no network or broker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.journal.store import Journal


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 5, 30, 14, 0, tzinfo=UTC) + timedelta(minutes=minute)


def test_signal_round_trip(tmp_path) -> None:
    journal = Journal(f"sqlite:///{tmp_path / 'j.db'}")
    journal.record_signal(
        symbol="aapl", strategy="SMA", action="BUY", confidence=0.4, reason="cross", ts=_ts()
    )
    rows = journal.recent_signals()
    assert len(rows) == 1
    assert rows[0].symbol == "AAPL"  # stored upper-cased
    assert rows[0].action == "BUY"
    assert rows[0].confidence == 0.4


def test_order_round_trip(tmp_path) -> None:
    journal = Journal(f"sqlite:///{tmp_path / 'j.db'}")
    journal.record_order(
        symbol="msft",
        side="buy",
        qty=12,
        status="accepted",
        broker_order_id="x1",
        reason="signal BUY",
        ts=_ts(),
    )
    rows = journal.recent_orders()
    assert len(rows) == 1
    assert rows[0].symbol == "MSFT"
    assert rows[0].qty == 12
    assert rows[0].reason == "signal BUY"


def test_equity_and_peak() -> None:
    journal = Journal("sqlite:///:memory:")
    assert journal.peak_equity() is None  # empty
    journal.record_equity(equity=100_000.0, cash=100_000.0, ts=_ts(0))
    journal.record_equity(equity=105_000.0, cash=20_000.0, ts=_ts(1))
    journal.record_equity(equity=98_000.0, cash=10_000.0, ts=_ts(2))
    assert journal.peak_equity() == 105_000.0


def test_recent_is_oldest_first_and_limited() -> None:
    journal = Journal("sqlite:///:memory:")
    for i in range(5):
        journal.record_equity(equity=100_000.0 + i, ts=_ts(i))
    rows = journal.recent_equity(limit=3)
    assert len(rows) == 3
    # The 3 most-recent rows, returned oldest-first for display.
    assert [r.equity for r in rows] == [100_002.0, 100_003.0, 100_004.0]


def test_counts() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_signal(symbol="AAPL", strategy="SMA", action="HOLD", ts=_ts())
    journal.record_order(symbol="AAPL", side="buy", qty=1, ts=_ts())
    journal.record_equity(equity=1.0, ts=_ts())
    journal.record_equity(equity=2.0, ts=_ts(1))
    counts = journal.counts()
    assert counts == {"signals": 1, "orders": 1, "equity_snapshots": 2}


def test_halt_fields_persist() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_equity(
        equity=80_000.0,
        cash=80_000.0,
        halted=True,
        halt_reason="max-drawdown halt active",
        ts=_ts(),
    )
    row = journal.recent_equity()[0]
    assert row.halted is True
    assert row.halt_reason == "max-drawdown halt active"
