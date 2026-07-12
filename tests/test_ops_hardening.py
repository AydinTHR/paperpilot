"""Tests for the week-2 ops hardening: exit identities, the per-tick
reconciliation sweep, GTC stop legs, per-arm trade streams, and the weekly
report. Fully offline, as ever."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
from alpaca.trading.enums import TimeInForce

from config.settings import Settings
from src.agent.loop import TradingLoop
from src.execution.broker import Broker, OrderInfo
from src.execution.reconcile import TERMINAL_STATUSES
from src.experiments.harness import ExperimentArm, ExperimentHarness, arm_report_from_journal
from src.journal.store import Journal
from src.risk.manager import RiskManager
from src.strategy.base import Action, Signal, Strategy

# --- fakes ---------------------------------------------------------------------


def _order(
    order_id: str = "close-1",
    *,
    status: str = "filled",
    qty: float = 50,
    filled_qty: float = 50,
    price: float | None = 99.5,
    order_type: str = "market",
) -> OrderInfo:
    return OrderInfo(
        id=order_id,
        symbol="AAPL",
        qty=qty,
        side="sell",
        order_type=order_type,
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=price,
    )


class _Broker:
    """Loop-facing fake: closes return a real closing order; get_order scripted."""

    def __init__(self, equity: float = 100_000.0, cash: float = 0.0, positions=None) -> None:
        self.equity = equity
        self.cash = cash
        self._positions = list(positions or [])
        self.orders_by_id: dict[str, OrderInfo] = {}
        self.closed: list[str] = []

    def get_account(self):
        from src.execution.broker import AccountSnapshot

        return AccountSnapshot(
            account_number="T",
            status="ACTIVE",
            currency="USD",
            cash=self.cash,
            equity=self.equity,
            buying_power=self.cash,
            portfolio_value=self.equity,
        )

    def get_positions(self):
        return list(self._positions)

    def place_market_order(self, symbol, qty, side):
        raise AssertionError("not used in these tests")

    def place_market_order_with_stop(self, symbol, qty, side, stop_price, *, ref_price=None):
        raise AssertionError("not used in these tests")

    def close_position(self, symbol: str) -> OrderInfo | None:
        self.closed.append(symbol.upper())
        self._positions = [p for p in self._positions if p.symbol != symbol.upper()]
        order = _order(f"close-{len(self.closed)}")
        self.orders_by_id[order.id] = order
        return order

    def get_order(self, order_id: str) -> OrderInfo:
        return self.orders_by_id[order_id]

    def get_open_orders(self, symbol=None):
        return []


class _SellStrategy(Strategy):
    name = "seller"

    @property
    def min_bars(self) -> int:
        return 5

    def generate_signals(self, data) -> Signal:
        return Signal(Action.SELL, reason="exit now")


def _bars(n: int = 60, close: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000},
        index=idx,
    )


def _position(symbol: str, qty: float, entry: float, price: float):
    from src.execution.broker import PositionInfo

    return PositionInfo(
        symbol=symbol,
        qty=qty,
        side="long",
        avg_entry_price=entry,
        current_price=price,
        market_value=qty * price,
        unrealized_pl=0.0,
        unrealized_plpc=0.0,
    )


class _Provider:
    def __init__(self, frame) -> None:
        self._frame = frame

    def get_latest_bars(self, symbol, lookback, interval):
        return self._frame


def _loop(broker, strategy, *, journal=None) -> TradingLoop:
    settings = Settings()
    return TradingLoop(
        broker=broker,
        provider=_Provider(_bars()),
        strategy=strategy,
        risk=RiskManager.from_settings(settings, starting_equity=100_000.0),
        journal=journal or Journal("sqlite:///:memory:"),
        settings=settings,
        symbols=["AAPL"],
    )


# --- exits carry identities and reconciled fills ---------------------------------


def test_sell_exit_journals_order_id_and_fill() -> None:
    broker = _Broker(positions=[_position("AAPL", 50, 90.0, 100.0)])
    loop = _loop(broker, _SellStrategy())
    result = loop.run_once()

    assert result.outcomes[0].action == "SELL"
    row = loop.journal.recent_orders()[0]
    assert row.broker_order_id == "close-1"
    assert row.status == "filled"
    assert row.filled_qty == 50
    assert row.filled_avg_price == 99.5


def test_halt_flatten_journals_order_id() -> None:
    broker = _Broker(equity=70_000.0, positions=[_position("AAPL", 50, 90.0, 100.0)])
    loop = _loop(broker, _SellStrategy())
    loop.risk.update_equity(100_000.0)  # set peak; next tick's 70k equity trips drawdown
    result = loop.run_once()

    assert result.halted
    row = loop.journal.recent_orders()[0]
    assert row.reason.startswith("halt:")
    assert row.broker_order_id == "close-1"
    assert row.filled_qty == 50


def test_close_returning_none_still_journals() -> None:
    broker = _Broker(positions=[_position("AAPL", 50, 90.0, 100.0)])
    broker.close_position = lambda symbol: None  # e.g. older SDK / odd client
    loop = _loop(broker, _SellStrategy())
    loop.run_once()
    row = loop.journal.recent_orders()[0]
    assert row.status == "submitted"
    assert row.broker_order_id == ""


# --- the per-tick reconciliation sweep --------------------------------------------


def test_sweep_repairs_stale_order_row() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_order(
        symbol="AAPL",
        strategy="llm",
        side="buy",
        qty=32,
        status="accepted",  # the week-1 scenario: fill landed while we were down
        broker_order_id="stale-1",
    )
    broker = _Broker()
    broker.orders_by_id["stale-1"] = _order(
        "stale-1", status="filled", qty=32, filled_qty=32, price=308.41
    )
    loop = _loop(broker, _SellStrategy(), journal=journal)

    loop.run_once()

    row = journal.recent_orders()[0]
    assert row.status == "filled"
    assert row.filled_qty == 32
    assert row.filled_avg_price == 308.41


def test_sweep_survives_broker_errors() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_order(
        symbol="AAPL",
        strategy="llm",
        side="buy",
        qty=1,
        status="accepted",
        broker_order_id="gone-1",
    )
    broker = _Broker()  # knows nothing about gone-1 -> get_order raises KeyError
    loop = _loop(broker, _SellStrategy(), journal=journal)
    loop.run_once()  # must not raise
    assert journal.recent_orders()[0].status == "accepted"  # untouched, not corrupted


def test_unreconciled_orders_reader_filters() -> None:
    journal = Journal("sqlite:///:memory:")
    journal.record_order(
        symbol="A", strategy="s", side="buy", qty=1, status="accepted", broker_order_id="open-1"
    )
    journal.record_order(
        symbol="B", strategy="s", side="buy", qty=1, status="filled", broker_order_id="done-1"
    )
    journal.record_order(symbol="C", strategy="s", side="sell", qty=1, status="submitted")

    rows = journal.unreconciled_orders(terminal_statuses=TERMINAL_STATUSES)
    assert [r.broker_order_id for r in rows] == ["open-1"]  # no id -> skipped; terminal -> skipped


# --- GTC stop legs -----------------------------------------------------------------


def test_oto_stop_leg_is_gtc() -> None:
    captured: dict = {}

    class _Client:
        def submit_order(self, order_data):
            captured["request"] = order_data
            return SimpleNamespace(
                id="o-1",
                symbol=order_data.symbol,
                qty=order_data.qty,
                side=order_data.side,
                order_type="market",
                status="accepted",
                filled_qty="0",
                filled_avg_price=None,
            )

    settings = Settings(_env_file=None, alpaca_api_key="k", alpaca_secret_key="s")  # type: ignore[arg-type]
    broker = Broker(settings, client=_Client(), announce=False)  # type: ignore[arg-type]
    broker.place_market_order_with_stop("AAPL", 10, "buy", 95.0, ref_price=100.0)
    assert captured["request"].time_in_force == TimeInForce.GTC


# --- per-arm trade streams ----------------------------------------------------------


class _FakeListener:
    instances: ClassVar[list[_FakeListener]] = []

    def __init__(self, settings, journal, *, api_key, secret_key, name) -> None:
        self.kwargs = {"api_key": api_key, "secret_key": secret_key, "name": name}
        self.started = False
        _FakeListener.instances.append(self)

    def start(self) -> None:
        self.started = True


def _harness(mode: str, settings: Settings) -> ExperimentHarness:
    arms = [
        ExperimentArm(name="sma", loop=None, journal=None, credentials=("k1", "s1")),  # type: ignore[arg-type]
        ExperimentArm(name="llm", loop=None, journal=None, credentials=("k3", "s3")),  # type: ignore[arg-type]
    ]
    return ExperimentHarness(arms, mode, settings)


def test_streams_started_per_arm_with_own_credentials() -> None:
    _FakeListener.instances = []
    settings = Settings(_env_file=None, use_trade_stream=True)  # type: ignore[arg-type]
    listeners = _harness("accounts", settings).start_trade_streams(listener_factory=_FakeListener)
    assert len(listeners) == 2
    assert all(fake.started for fake in _FakeListener.instances)
    assert _FakeListener.instances[0].kwargs["api_key"] == "k1"
    assert _FakeListener.instances[1].kwargs["api_key"] == "k3"
    assert _FakeListener.instances[1].kwargs["name"] == "trade-stream-llm"


def test_streams_skipped_when_disabled_or_virtual() -> None:
    _FakeListener.instances = []
    off = Settings(_env_file=None)  # type: ignore[arg-type]
    assert _harness("accounts", off).start_trade_streams(listener_factory=_FakeListener) == []
    on = Settings(_env_file=None, use_trade_stream=True)  # type: ignore[arg-type]
    assert _harness("virtual", on).start_trade_streams(listener_factory=_FakeListener) == []
    assert _FakeListener.instances == []


# --- weekly report -------------------------------------------------------------------


def test_weekly_report_message_renders(tmp_path) -> None:
    from scripts.weekly_report import build_message

    base = f"sqlite:///{tmp_path}/paperpilot.db"
    arm_journal = Journal(f"sqlite:///{tmp_path}/experiments/sma.db")
    arm_journal.record_equity(equity=100_000.0)
    arm_journal.record_equity(equity=101_500.0)

    message = build_message(base, arms=("sma", "llm"))
    assert "sma: $101,500 (+1.50%)" in message
    assert "llm: no journal yet" in message


def test_arm_report_from_journal_missing_is_none(tmp_path) -> None:
    assert arm_report_from_journal(f"sqlite:///{tmp_path}/paperpilot.db", "ghost") is None
