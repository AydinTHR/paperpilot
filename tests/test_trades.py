"""Tests for FIFO trade pairing and per-strategy realized-P&L reporting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.journal.store import Journal
from src.journal.trades import build_trades, summarize_by_strategy

T0 = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)


def _journal_with(*orders) -> Journal:
    journal = Journal("sqlite:///:memory:")
    for kwargs in orders:
        journal.record_order(**kwargs)
    return journal


def _buy(symbol="AAPL", qty=10, px=100.0, ts=T0, strategy="sma", reason="signal BUY"):
    return {
        "symbol": symbol,
        "side": "buy",
        "qty": qty,
        "strategy": strategy,
        "status": "filled",
        "filled_qty": qty,
        "filled_avg_price": px,
        "reason": reason,
        "ts": ts,
    }


def _sell(symbol="AAPL", qty=10, px=110.0, ts=None, strategy="sma", reason="signal SELL"):
    return {
        "symbol": symbol,
        "side": "sell",
        "qty": qty,
        "strategy": strategy,
        "status": "filled",
        "filled_qty": qty,
        "filled_avg_price": px,
        "reason": reason,
        "ts": ts or (T0 + timedelta(days=2)),
    }


def test_simple_round_trip() -> None:
    journal = _journal_with(_buy(qty=10, px=100.0), _sell(qty=10, px=110.0))
    trades = build_trades(journal._all_orders())
    assert len(trades) == 1
    trade = trades[0]
    assert trade.pnl == pytest.approx(100.0)  # (110-100)*10
    assert trade.pnl_pct == pytest.approx(0.10)
    assert trade.holding_period_hours == pytest.approx(48.0)
    assert trade.entry_reason == "signal BUY"
    assert trade.exit_reason == "signal SELL"


def test_partial_exit_splits_lot() -> None:
    journal = _journal_with(_buy(qty=10, px=100.0), _sell(qty=4, px=105.0))
    trades = build_trades(journal._all_orders())
    assert len(trades) == 1
    assert trades[0].qty == 4
    assert trades[0].pnl == pytest.approx(20.0)  # 4 realized; 6 still open (no row)


def test_sell_crossing_multiple_lots_fifo_order() -> None:
    journal = _journal_with(
        _buy(qty=5, px=100.0, ts=T0),
        _buy(qty=5, px=110.0, ts=T0 + timedelta(days=1)),
        _sell(qty=8, px=120.0, ts=T0 + timedelta(days=3)),
    )
    trades = build_trades(journal._all_orders())
    assert [(t.qty, t.entry_px) for t in trades] == [(5, 100.0), (3, 110.0)]  # oldest first
    assert trades[0].pnl == pytest.approx(100.0)
    assert trades[1].pnl == pytest.approx(30.0)


def test_interleaved_symbols_and_strategies_never_cross() -> None:
    journal = _journal_with(
        _buy(symbol="AAPL", strategy="sma", qty=5, px=100.0),
        _buy(symbol="AAPL", strategy="rsi", qty=5, px=90.0),
        _buy(symbol="MSFT", strategy="sma", qty=5, px=200.0),
        _sell(symbol="AAPL", strategy="rsi", qty=5, px=99.0),
    )
    trades = build_trades(journal._all_orders())
    assert len(trades) == 1  # only the rsi AAPL pairing realizes
    assert trades[0].strategy == "rsi"
    assert trades[0].entry_px == 90.0


def test_unfilled_orders_ignored() -> None:
    journal = _journal_with(
        {"symbol": "AAPL", "side": "buy", "qty": 10, "status": "rejected", "filled_qty": 0.0},
        _buy(qty=5, px=100.0),
        _sell(qty=5, px=101.0),
    )
    trades = build_trades(journal._all_orders())
    assert len(trades) == 1
    assert trades[0].qty == 5


def test_partial_fill_uses_actual_quantity() -> None:
    # Requested 10, actually filled 6 (paper's random partial fills).
    journal = _journal_with(
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "strategy": "sma",
            "status": "partially_filled",
            "filled_qty": 6.0,
            "filled_avg_price": 100.0,
            "ts": T0,
        },
        _sell(qty=6, px=105.0),
    )
    trades = build_trades(journal._all_orders())
    assert trades[0].qty == 6
    assert trades[0].pnl == pytest.approx(30.0)


def test_rebuild_trades_is_idempotent() -> None:
    journal = _journal_with(_buy(), _sell())
    assert journal.rebuild_trades() == 1
    assert journal.rebuild_trades() == 1  # delete-and-rebuild, no duplication
    assert len(journal.recent_trades()) == 1


def test_strategy_report_stats() -> None:
    journal = _journal_with(
        _buy(strategy="sma", qty=10, px=100.0, ts=T0),
        _sell(strategy="sma", qty=10, px=110.0, ts=T0 + timedelta(days=1)),  # +100
        _buy(strategy="sma", qty=10, px=100.0, ts=T0 + timedelta(days=2)),
        _sell(strategy="sma", qty=10, px=95.0, ts=T0 + timedelta(days=3)),  # -50
    )
    report = journal.strategy_report()
    stats = report["sma"]
    assert stats.num_trades == 2
    assert stats.win_rate_pct == pytest.approx(50.0)
    assert stats.total_pnl == pytest.approx(50.0)
    assert stats.avg_win == pytest.approx(100.0)
    assert stats.avg_loss == pytest.approx(-50.0)
    assert stats.profit_factor == pytest.approx(2.0)


def test_summarize_handles_all_wins_profit_factor() -> None:
    journal = _journal_with(_buy(qty=10, px=100.0), _sell(qty=10, px=110.0))
    report = summarize_by_strategy(build_trades(journal._all_orders()))
    assert report["sma"].profit_factor == float("inf")
