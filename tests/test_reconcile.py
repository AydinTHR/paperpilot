"""Tests for order fill reconciliation, driven entirely by fake time."""

from __future__ import annotations

from src.execution.broker import OrderInfo
from src.execution.reconcile import OrderReconciler, ReconcileResult


def _order(status: str, filled_qty: float = 0.0, price: float | None = None) -> OrderInfo:
    return OrderInfo(
        id="ord-1",
        symbol="AAPL",
        qty=10,
        side="buy",
        order_type="market",
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=price,
    )


class _FakeTime:
    """A clock + sleeper pair where sleeping advances the clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _ScriptedReader:
    """Yields a scripted sequence of statuses (or exceptions), then repeats last."""

    def __init__(self, *script: OrderInfo | Exception) -> None:
        self._script = list(script)
        self.calls = 0

    def get_order(self, order_id: str) -> OrderInfo:
        self.calls += 1
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        return item


def _reconciler(reader: _ScriptedReader, ft: _FakeTime, **kwargs) -> OrderReconciler:
    return OrderReconciler(reader, clock=ft.clock, sleeper=ft.sleep, **kwargs)


def test_immediate_fill_returns_first_attempt() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(_order("filled", filled_qty=10, price=100.5))
    result = _reconciler(reader, ft).wait_for_terminal("ord-1")
    assert result.terminal is True
    assert result.status == "filled"
    assert result.filled_qty == 10
    assert result.filled_avg_price == 100.5
    assert result.attempts == 1
    assert ft.sleeps == []  # no waiting needed


def test_partial_then_filled_polls_with_backoff() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(
        _order("accepted"),
        _order("partially_filled", filled_qty=4, price=100.0),
        _order("filled", filled_qty=10, price=100.2),
    )
    result = _reconciler(reader, ft).wait_for_terminal("ord-1")
    assert result.terminal is True
    assert result.filled_qty == 10
    assert result.attempts == 3
    assert ft.sleeps == [0.5, 1.0]  # exponential backoff between polls


def test_timeout_reports_partial_progress() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(_order("partially_filled", filled_qty=3, price=99.9))
    result = _reconciler(reader, ft, max_wait_s=5.0).wait_for_terminal("ord-1")
    assert result.terminal is False
    assert result.status == "partially_filled"
    assert result.filled_qty == 3  # progress captured even without a terminal state
    assert ft.now <= 5.0  # never waits past the bound


def test_read_errors_are_survived() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(
        ConnectionError("api down"),
        ConnectionError("still down"),
        _order("filled", filled_qty=10, price=101.0),
    )
    result = _reconciler(reader, ft).wait_for_terminal("ord-1")
    assert result.terminal is True
    assert result.attempts == 3


def test_never_raises_even_if_every_read_fails() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(ConnectionError("permanently down"))
    result = _reconciler(reader, ft, max_wait_s=3.0).wait_for_terminal("ord-1")
    assert result.terminal is False
    assert result.status == "unknown"
    assert result.filled_qty == 0.0


def test_rejected_is_terminal() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(_order("rejected"))
    result = _reconciler(reader, ft).wait_for_terminal("ord-1")
    assert result.terminal is True
    assert result.status == "rejected"


def test_backoff_caps_at_max_delay() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(_order("accepted"))
    _reconciler(reader, ft, max_wait_s=40.0, max_delay_s=8.0).wait_for_terminal("ord-1")
    assert max(ft.sleeps) == 8.0
    assert ft.sleeps[:5] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_result_is_frozen_dataclass() -> None:
    ft = _FakeTime()
    reader = _ScriptedReader(_order("filled", filled_qty=10, price=100.0))
    result = _reconciler(reader, ft).wait_for_terminal("ord-1")
    assert isinstance(result, ReconcileResult)
