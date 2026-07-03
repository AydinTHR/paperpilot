"""Tests for the live trading loop -- fully offline with injected fakes.

A fake broker (records orders/closes, mutable positions) and a fake data
provider (canned frames) let us drive every branch of ``run_once`` -- HOLD,
risk-sized BUY, SELL/close, the stop-loss exit, the halt-flatten path, and the
insufficient-bars skip -- without any network, SDK, or real money.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.agent.loop import TradingLoop
from src.execution.broker import AccountSnapshot, BrokerError, OrderInfo, PositionInfo
from src.journal.store import Journal
from src.risk.manager import RiskManager
from src.strategy.base import Action, Signal, Strategy

NOW = datetime(2026, 5, 30, 15, 0, tzinfo=UTC)


# --- fakes ------------------------------------------------------------------


class _FakeBroker:
    def __init__(self, equity: float, cash: float, positions=None) -> None:
        self.equity = equity
        self.cash = cash
        self._positions = list(positions or [])
        self.orders: list[OrderInfo] = []
        self.closed: list[str] = []
        self.stops: list[tuple[str, float]] = []
        self.open_orders: list[OrderInfo] = []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_number="TEST",
            status="ACTIVE",
            currency="USD",
            cash=self.cash,
            equity=self.equity,
            buying_power=self.cash * 2,
            portfolio_value=self.equity,
            pattern_day_trader=False,
        )

    def get_positions(self) -> list[PositionInfo]:
        return list(self._positions)

    def place_market_order(self, symbol: str, qty: float, side: str) -> OrderInfo:
        # Fills immediately (like Alpaca paper), so reconciliation is one poll.
        order = OrderInfo(
            id=f"ord-{len(self.orders) + 1}",
            symbol=symbol.upper(),
            qty=qty,
            side=side,
            order_type="market",
            status="filled",
            filled_qty=qty,
            filled_avg_price=100.0,
        )
        self.orders.append(order)
        return order

    def place_market_order_with_stop(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        ref_price: float | None = None,
    ) -> OrderInfo:
        self.stops.append((symbol.upper(), stop_price))
        return self.place_market_order(symbol, qty, side)

    def close_position(self, symbol: str) -> None:
        self.closed.append(symbol.upper())
        self._positions = [p for p in self._positions if p.symbol != symbol.upper()]

    def get_order(self, order_id: str) -> OrderInfo:
        for order in self.orders:
            if order.id == order_id:
                return order
        raise KeyError(order_id)

    def get_open_orders(self, symbol: str | None = None) -> list[OrderInfo]:
        return list(self.open_orders)


class _FakeProvider:
    def __init__(self, frames) -> None:
        self._frames = frames  # dict[str, DataFrame] or a single DataFrame

    def get_latest_bars(self, symbol: str, lookback: int, interval: str):
        if isinstance(self._frames, dict):
            return self._frames[symbol.upper()]
        return self._frames


class _StubStrategy(Strategy):
    name = "stub"

    def __init__(self, signal: Signal, min_bars: int = 5) -> None:
        self._signal = signal
        self._min_bars = min_bars

    @property
    def min_bars(self) -> int:
        return self._min_bars

    def generate_signals(self, data: pd.DataFrame) -> Signal:
        return self._signal


# --- helpers ----------------------------------------------------------------


def _bars(n: int = 60, last_close: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    close = np.full(n, last_close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _position(symbol: str, qty: float, entry: float, price: float) -> PositionInfo:
    return PositionInfo(
        symbol=symbol.upper(),
        qty=qty,
        side="long",
        avg_entry_price=entry,
        current_price=price,
        market_value=qty * price,
        unrealized_pl=(price - entry) * qty,
        unrealized_plpc=0.0,
    )


def _loop(broker, provider, strategy, *, symbols, starting_equity=100_000.0, journal=None):
    settings = Settings()  # defaults via autouse _clean_env
    risk = RiskManager.from_settings(settings, starting_equity=starting_equity)
    return TradingLoop(
        broker=broker,
        provider=provider,
        strategy=strategy,
        risk=risk,
        journal=journal or Journal("sqlite:///:memory:"),
        settings=settings,
        symbols=symbols,
    )


# --- tests ------------------------------------------------------------------


def test_hold_places_no_order() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.HOLD, reason="flat")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert broker.orders == []
    assert broker.closed == []
    assert [o.action for o in result.outcomes] == ["HOLD"]
    assert loop.journal.counts() == {"signals": 1, "orders": 0, "equity_snapshots": 1}


def test_buy_signal_places_risk_sized_order() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, confidence=0.6, reason="cross")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    # max_position_pct 0.10 of 100k = 10k budget / 100 price = 100 shares.
    assert len(broker.orders) == 1
    assert broker.orders[0].symbol == "AAPL"
    assert broker.orders[0].qty == 100
    assert broker.orders[0].side == "buy"
    out = result.outcomes[0]
    assert out.action == "BUY"
    assert out.qty == 100
    orders = loop.journal.recent_orders()
    assert orders[0].reason == "signal BUY"


def test_buy_when_already_holding_is_noop() -> None:
    broker = _FakeBroker(
        equity=100_000.0,
        cash=100_000.0,
        positions=[_position("AAPL", qty=50, entry=90.0, price=100.0)],
    )
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, reason="cross")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert broker.orders == []  # already long -> no second entry
    assert result.outcomes[0].action == "HOLD"


def test_sell_signal_closes_position() -> None:
    broker = _FakeBroker(
        equity=100_000.0,
        cash=50_000.0,
        positions=[_position("AAPL", qty=50, entry=90.0, price=100.0)],
    )
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.SELL, reason="death cross")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert broker.closed == ["AAPL"]
    assert result.outcomes[0].action == "SELL"
    assert loop.journal.recent_orders()[0].reason == "signal SELL"


def test_sell_when_flat_is_noop() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.SELL, reason="death cross")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert broker.closed == []
    assert result.outcomes[0].action == "HOLD"


def test_stop_loss_closes_before_signal() -> None:
    # Holding from 100; stop_loss_pct 0.05 -> stop 95; last close 90 -> breached.
    broker = _FakeBroker(
        equity=95_000.0,
        cash=0.0,
        positions=[_position("AAPL", qty=50, entry=100.0, price=90.0)],
    )
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=90.0)),
        _StubStrategy(Signal(Action.BUY, reason="ignored")),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert broker.closed == ["AAPL"]
    assert broker.orders == []
    assert result.outcomes[0].action == "STOP"
    orders = loop.journal.recent_orders()
    assert orders[0].reason == "stop-loss"
    # Stop short-circuits before the signal is generated/recorded.
    assert loop.journal.counts()["signals"] == 0


def test_halt_flattens_and_blocks_entries() -> None:
    broker = _FakeBroker(
        equity=100_000.0,
        cash=100_000.0,
        positions=[_position("AAPL", qty=50, entry=90.0, price=100.0)],
    )
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.BUY, reason="would-buy")),
        symbols=["AAPL"],
    )
    # Peak was 200k; equity now 100k -> 50% drawdown >> 20% limit -> halt.
    loop.risk.seed_peak(200_000.0)
    result = loop.run_once(now=NOW)

    assert result.halted is True
    assert "drawdown" in result.halt_reason
    assert broker.closed == ["AAPL"]  # flattened
    assert broker.orders == []  # no new entries despite the BUY signal
    assert result.outcomes[0].action == "FLATTEN"
    # Halt short-circuits per-symbol work: no signals recorded this tick.
    assert loop.journal.counts()["signals"] == 0


def test_insufficient_bars_skips() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars(n=10)),
        _StubStrategy(Signal(Action.BUY, reason="cross"), min_bars=60),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert result.outcomes[0].action == "SKIP"
    assert broker.orders == []
    assert loop.journal.counts()["signals"] == 0


def test_equity_snapshot_recorded() -> None:
    broker = _FakeBroker(equity=123_456.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.HOLD)),
        symbols=["AAPL"],
    )
    loop.run_once(now=NOW)
    snaps = loop.journal.recent_equity()
    assert len(snaps) == 1
    assert snaps[0].equity == 123_456.0
    assert snaps[0].halted is False


def test_cash_budget_decrements_across_symbols() -> None:
    # Tiny cash: first symbol affords 1 share, leaving too little for the second.
    broker = _FakeBroker(equity=100_000.0, cash=150.0)
    frames = {"AAA": _bars(last_close=100.0), "BBB": _bars(last_close=100.0)}
    loop = _loop(
        broker,
        _FakeProvider(frames),
        _StubStrategy(Signal(Action.BUY, reason="cross")),
        symbols=["AAA", "BBB"],
    )
    result = loop.run_once(now=NOW)

    actions = {o.symbol: o.action for o in result.outcomes}
    assert actions["AAA"] == "BUY"
    assert actions["BBB"] == "HOLD"  # cash exhausted -> risk-sized qty < 1
    assert len(broker.orders) == 1


def test_from_settings_rejects_zero_equity(monkeypatch) -> None:
    # A zeroed/empty paper account must fail fast with a clear BrokerError
    # (caught cleanly by the CLI), not bubble up the RiskManager ValueError.
    class _ZeroBroker:
        def __init__(self, settings, announce: bool = True) -> None:
            pass

        def get_account(self) -> AccountSnapshot:
            return AccountSnapshot(
                account_number="TEST",
                status="ACTIVE",
                currency="USD",
                cash=0.0,
                equity=0.0,
                buying_power=0.0,
                portfolio_value=0.0,
                pattern_day_trader=False,
            )

    # Patch the broker + data provider the factory builds so no network/SDK runs.
    monkeypatch.setattr("src.agent.loop.Broker", _ZeroBroker)
    monkeypatch.setattr("src.agent.loop.build_provider", lambda settings: object())

    with pytest.raises(BrokerError, match="equity"):
        TradingLoop.from_settings(
            Settings(),
            strategy=_StubStrategy(Signal(Action.HOLD)),
            journal=Journal("sqlite:///:memory:"),
        )


# --- scheduled_tick: market-hours gate + crash containment -------------------


class _StubCalendar:
    def __init__(self, is_open: bool) -> None:
        self._open = is_open

    def is_market_open(self, now=None) -> bool:
        return self._open

    def next_session_open(self, now=None):
        return None


class _ExplodingBroker:
    def get_account(self):
        raise RuntimeError("boom")

    def get_positions(self):  # pragma: no cover - never reached
        return []


def test_scheduled_tick_skipped_when_market_closed() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    journal = Journal("sqlite:///:memory:")
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.BUY, confidence=1.0)),
        symbols=["AAPL"],
        journal=journal,
    )
    loop.market_calendar = _StubCalendar(is_open=False)

    assert loop.scheduled_tick() is None
    assert broker.orders == []  # nothing traded
    assert journal.counts()["equity_snapshots"] == 0  # run_once never started


def test_scheduled_tick_runs_when_market_open() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.HOLD)),
        symbols=["AAPL"],
    )
    loop.market_calendar = _StubCalendar(is_open=True)

    result = loop.scheduled_tick()
    assert result is not None
    assert result.halted is False


def test_scheduled_tick_ungated_without_calendar() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.HOLD)),
        symbols=["AAPL"],
    )
    assert loop.market_calendar is None
    assert loop.scheduled_tick() is not None


def test_scheduled_tick_contains_exceptions() -> None:
    loop = _loop(
        _ExplodingBroker(),
        _FakeProvider(_bars()),
        _StubStrategy(Signal(Action.HOLD)),
        symbols=["AAPL"],
    )
    # A tick failure is logged and swallowed so the scheduler keeps running.
    assert loop.scheduled_tick() is None


# --- broker-held stops + reconciliation ---------------------------------------


def _stops_settings(**kw):
    return Settings(
        _env_file=None,
        alpaca_api_key="k",
        alpaca_secret_key="s",
        **kw,
    )


def _loop_with_settings(broker, provider, strategy, settings, *, symbols):
    risk = RiskManager.from_settings(settings, starting_equity=100_000.0)
    return TradingLoop(
        broker=broker,
        provider=provider,
        strategy=strategy,
        risk=risk,
        journal=Journal("sqlite:///:memory:"),
        settings=settings,
        symbols=symbols,
    )


def test_entry_routes_via_broker_stop_when_enabled() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop_with_settings(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, confidence=1.0)),
        _stops_settings(),  # creds present -> use_broker_stops resolves true
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)

    assert result.outcomes[0].action == "BUY"
    assert broker.stops == [("AAPL", 95.0)]  # stop_loss_pct 0.05 below entry 100


def test_entry_plain_market_order_when_stops_disabled() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop_with_settings(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, confidence=1.0)),
        _stops_settings(use_broker_stops=False),
        symbols=["AAPL"],
    )
    loop.run_once(now=NOW)
    assert broker.stops == []
    assert len(broker.orders) == 1


def test_journal_records_reconciled_fill() -> None:
    broker = _FakeBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, confidence=1.0)),
        symbols=["AAPL"],
    )
    loop.run_once(now=NOW)
    order = loop.journal.recent_orders()[0]
    assert order.status == "filled"
    assert order.filled_qty == 100
    assert order.filled_avg_price == 100.0


def test_rejected_entry_degrades_to_skip() -> None:
    class _RejectingBroker(_FakeBroker):
        def place_market_order(self, symbol: str, qty: float, side: str) -> OrderInfo:
            raise BrokerError("rejected as potential wash trade")

    broker = _RejectingBroker(equity=100_000.0, cash=100_000.0)
    loop = _loop(
        broker,
        _FakeProvider(_bars(last_close=100.0)),
        _StubStrategy(Signal(Action.BUY, confidence=1.0)),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)  # must not raise
    assert result.outcomes[0].action == "SKIP"
    assert "rejected" in result.outcomes[0].detail


def test_loop_stop_skipped_when_broker_stop_live() -> None:
    # Holding from 100, price 90 -> loop stop WOULD fire, but a live broker
    # stop order protects the position, so the loop leaves it alone.
    broker = _FakeBroker(
        equity=95_000.0,
        cash=0.0,
        positions=[_position("AAPL", qty=50, entry=100.0, price=90.0)],
    )
    broker.open_orders = [
        OrderInfo(
            id="stop-1",
            symbol="AAPL",
            qty=50,
            side="sell",
            order_type="stop",
            status="new",
            filled_qty=0.0,
            filled_avg_price=None,
        )
    ]
    loop = _loop_with_settings(
        broker,
        _FakeProvider(_bars(last_close=90.0)),
        _StubStrategy(Signal(Action.HOLD)),
        _stops_settings(),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)
    assert broker.closed == []  # not closed by the loop
    assert result.outcomes[0].action == "HOLD"


def test_loop_stop_backstops_when_no_live_stop_order() -> None:
    # Same losing position, but no live stop order (e.g. DAY stop expired at
    # the close) -> the loop-side backstop must fire.
    broker = _FakeBroker(
        equity=95_000.0,
        cash=0.0,
        positions=[_position("AAPL", qty=50, entry=100.0, price=90.0)],
    )
    loop = _loop_with_settings(
        broker,
        _FakeProvider(_bars(last_close=90.0)),
        _StubStrategy(Signal(Action.HOLD)),
        _stops_settings(),
        symbols=["AAPL"],
    )
    result = loop.run_once(now=NOW)
    assert broker.closed == ["AAPL"]
    assert result.outcomes[0].action == "STOP"


# --- persisted halt restoration on startup -------------------------------------


class _HealthyBroker:
    def __init__(self, settings, announce: bool = True) -> None:
        pass

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_number="TEST",
            status="ACTIVE",
            currency="USD",
            cash=100_000.0,
            equity=100_000.0,
            buying_power=200_000.0,
            portfolio_value=100_000.0,
        )


def _patched_from_settings(monkeypatch, journal):
    monkeypatch.setattr("src.agent.loop.Broker", _HealthyBroker)
    monkeypatch.setattr("src.agent.loop.build_provider", lambda settings: object())
    return TradingLoop.from_settings(
        Settings(),
        strategy=_StubStrategy(Signal(Action.HOLD)),
        journal=journal,
        market_hours_gate=False,
    )


def test_from_settings_restores_persisted_drawdown_halt(monkeypatch) -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_halt(halt_type="drawdown", active=True, reason="21% below peak")
    loop = _patched_from_settings(monkeypatch, journal)
    assert loop.risk.halted
    assert loop.risk.can_enter().reason == "max-drawdown halt active"


def test_from_settings_ignores_cleared_halt(monkeypatch) -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_halt(halt_type="drawdown", active=True, reason="tripped")
    journal.record_halt(halt_type="drawdown", active=False, reason="manual reset")
    loop = _patched_from_settings(monkeypatch, journal)
    assert not loop.risk.halted


def test_from_settings_ignores_stale_daily_trip(monkeypatch) -> None:
    journal = Journal("sqlite:///:memory:")
    yesterday = datetime.now(UTC).replace(hour=12) - timedelta(days=1)
    journal.record_halt(
        halt_type="daily_loss", active=True, reason="down 4%", triggered_at=yesterday
    )
    loop = _patched_from_settings(monkeypatch, journal)
    assert not loop.risk.halted  # yesterday's trip does not carry over


def test_from_settings_restores_todays_daily_trip(monkeypatch) -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_halt(
        halt_type="daily_loss", active=True, reason="down 4%", triggered_at=datetime.now(UTC)
    )
    loop = _patched_from_settings(monkeypatch, journal)
    assert loop.risk.halted
