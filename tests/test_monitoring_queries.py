"""Tests for the dashboard's headless query layer (seeded journal, no UI)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.journal.store import Journal
from src.monitoring import queries


@pytest.fixture
def seeded(tmp_path) -> str:
    """A file-backed journal with a few rows of everything."""
    db_url = f"sqlite:///{tmp_path}/journal.db"
    journal = Journal(db_url)
    start = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    for i, equity in enumerate([100_000.0, 104_000.0, 98_000.0]):
        journal.record_equity(equity=equity, cash=equity / 2, ts=start + timedelta(days=i))
    journal.record_signal(symbol="AAPL", strategy="sma", action="BUY", confidence=0.8)
    journal.record_order(
        symbol="AAPL", side="buy", qty=10, status="filled", filled_qty=10, filled_avg_price=100.0
    )
    return db_url


def test_equity_curve_is_oldest_first(seeded) -> None:
    df = queries.load_equity_curve(queries.read_only_engine(seeded))
    assert list(df["equity"]) == [100_000.0, 104_000.0, 98_000.0]


def test_read_only_engine_cannot_write(seeded) -> None:
    engine = queries.read_only_engine(seeded)
    from sqlalchemy import text

    with pytest.raises(Exception, match=r"readonly|attempt to write"), engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO signals (ts, symbol, strategy, action) VALUES "
                "('2026-01-01', 'X', 's', 'BUY')"
            )
        )
        conn.commit()


def test_recent_orders_and_signals(seeded) -> None:
    engine = queries.read_only_engine(seeded)
    orders = queries.load_recent_orders(engine)
    signals = queries.load_recent_signals(engine)
    assert orders.iloc[0]["symbol"] == "AAPL"
    assert orders.iloc[0]["filled_qty"] == 10
    assert signals.iloc[0]["action"] == "BUY"


def test_halt_status_reads_latest_transition(seeded) -> None:
    assert queries.load_halt_status(seeded).halted is False
    Journal(seeded).record_halt(halt_type="drawdown", active=True, reason="21% below peak")
    status = queries.load_halt_status(seeded)
    assert status.halted is True
    assert "drawdown" in status.detail


def test_compute_drawdown() -> None:
    df = pd.DataFrame({"equity": [100.0, 110.0, 99.0]})
    assert queries.compute_drawdown(df) == pytest.approx(0.1)  # 10% below the 110 peak
    assert queries.compute_drawdown(pd.DataFrame({"equity": []})) == 0.0


def test_equity_figure_builder() -> None:
    pytest.importorskip("plotly")
    df = pd.DataFrame({"ts": ["2026-06-01", "2026-06-02"], "equity": [100.0, 101.0]})
    figure = queries.build_equity_figure(df)
    assert figure.data[0].y[1] == 101.0
