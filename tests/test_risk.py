"""Tests for the risk manager -- pure, offline, no broker or network.

Covers position sizing (risk cap vs. size_hint vs. cash bound), per-trade
stops, and the two latching halts: the daily-loss kill switch (resets at day
rollover) and the sticky max-drawdown halt (cleared only by reset()).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from config.settings import Settings
from src.risk.manager import RiskDecision, RiskLimits, RiskManager


def _limits(
    *,
    max_position_pct: float = 0.10,
    max_daily_loss_pct: float = 0.03,
    max_drawdown_pct: float = 0.20,
    stop_loss_pct: float = 0.05,
) -> RiskLimits:
    return RiskLimits(
        max_position_pct=max_position_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        stop_loss_pct=stop_loss_pct,
    )


# --- construction -----------------------------------------------------------


def test_limits_from_settings_maps_all_fields() -> None:
    settings = Settings()  # defaults (env cleaned by autouse fixture)
    limits = RiskLimits.from_settings(settings)
    assert limits.max_position_pct == settings.max_position_pct
    assert limits.max_daily_loss_pct == settings.max_daily_loss_pct
    assert limits.max_drawdown_pct == settings.max_drawdown_pct
    assert limits.stop_loss_pct == settings.stop_loss_pct


def test_manager_rejects_nonpositive_starting_equity() -> None:
    with pytest.raises(ValueError):
        RiskManager(_limits(), starting_equity=0)
    with pytest.raises(ValueError):
        RiskManager(_limits(), starting_equity=-100.0)


def test_from_settings_builds_manager() -> None:
    rm = RiskManager.from_settings(Settings(), starting_equity=10_000.0)
    assert rm.limits.max_position_pct == Settings().max_position_pct
    assert not rm.halted


# --- position sizing --------------------------------------------------------


def test_position_size_capped_by_max_position_pct() -> None:
    rm = RiskManager(_limits(max_position_pct=0.10), 10_000.0)
    # 10% of 10_000 = 1_000 budget; 1_000 // 100 = 10 shares.
    assert rm.position_size(equity=10_000.0, price=100.0, cash=10_000.0) == 10


def test_position_size_floors_to_whole_shares() -> None:
    rm = RiskManager(_limits(max_position_pct=0.10), 10_000.0)
    # budget 1_000, price 150 -> 6 shares (6*150=900 <= 1000).
    assert rm.position_size(equity=10_000.0, price=150.0, cash=10_000.0) == 6


def test_size_hint_smaller_than_cap_is_used() -> None:
    rm = RiskManager(_limits(max_position_pct=0.10), 10_000.0)
    # hint 0.05 < cap 0.10 -> budget 500 -> 5 shares.
    assert rm.position_size(10_000.0, 100.0, 10_000.0, size_hint=0.05) == 5


def test_size_hint_larger_than_cap_is_clamped() -> None:
    rm = RiskManager(_limits(max_position_pct=0.10), 10_000.0)
    # hint 0.50 > cap 0.10 -> clamped to cap -> 10 shares.
    assert rm.position_size(10_000.0, 100.0, 10_000.0, size_hint=0.50) == 10


def test_position_size_bounded_by_cash() -> None:
    rm = RiskManager(_limits(max_position_pct=0.10), 10_000.0)
    # cap budget is 1_000 but only 300 cash on hand -> 3 shares.
    assert rm.position_size(10_000.0, 100.0, cash=300.0) == 3


@pytest.mark.parametrize(
    "equity,price,cash",
    [
        (0.0, 100.0, 10_000.0),
        (10_000.0, 0.0, 10_000.0),
        (10_000.0, 100.0, 0.0),
        (10_000.0, -5.0, 10_000.0),
    ],
)
def test_position_size_zero_on_degenerate_inputs(equity: float, price: float, cash: float) -> None:
    rm = RiskManager(_limits(), 10_000.0)
    assert rm.position_size(equity, price, cash) == 0


# --- per-trade stop ---------------------------------------------------------


def test_stop_price_and_breach() -> None:
    rm = RiskManager(_limits(stop_loss_pct=0.05), 10_000.0)
    assert rm.stop_price(100.0) == pytest.approx(95.0)
    assert rm.stop_breached(entry=100.0, current=95.0) is True  # touch == breach
    assert rm.stop_breached(entry=100.0, current=94.0) is True
    assert rm.stop_breached(entry=100.0, current=96.0) is False


def test_stop_disabled_when_zero() -> None:
    rm = RiskManager(_limits(stop_loss_pct=0.0), 10_000.0)
    assert rm.stop_price(100.0) == pytest.approx(100.0)
    assert rm.stop_breached(entry=100.0, current=1.0) is False


# --- daily-loss kill switch -------------------------------------------------


def test_daily_kill_switch_trips_on_cumulative_loss() -> None:
    rm = RiskManager(_limits(max_daily_loss_pct=0.03), 10_000.0)
    day = date(2026, 1, 2)
    rm.update_equity(10_000.0, now=day)
    rm.update_equity(9_800.0, now=day)  # -2% -> still ok
    assert rm.can_enter().allowed is True
    rm.update_equity(9_700.0, now=day)  # -3% -> trips
    assert rm.halted is True
    decision = rm.can_enter()
    assert decision.allowed is False
    assert "daily" in decision.reason


def test_daily_kill_switch_latches_within_day_despite_recovery() -> None:
    rm = RiskManager(_limits(max_daily_loss_pct=0.03), 10_000.0)
    day = date(2026, 1, 2)
    rm.update_equity(10_000.0, now=day)
    rm.update_equity(9_600.0, now=day)  # -4% -> trips
    assert rm.halted is True
    rm.update_equity(9_950.0, now=day)  # recovers, but stays tripped for the day
    assert rm.halted is True


def test_daily_kill_switch_resets_next_day() -> None:
    rm = RiskManager(_limits(max_daily_loss_pct=0.03, max_drawdown_pct=0.20), 10_000.0)
    rm.update_equity(10_000.0, now=date(2026, 1, 2))
    rm.update_equity(9_600.0, now=date(2026, 1, 2))  # trips
    assert rm.halted is True
    rm.update_equity(9_600.0, now=date(2026, 1, 3))  # new day -> kill switch clears
    assert rm.halted is False
    assert rm.can_enter().allowed is True


def test_daily_kill_switch_accepts_datetime_and_timestamp() -> None:
    rm = RiskManager(_limits(max_daily_loss_pct=0.03), 10_000.0)
    rm.update_equity(10_000.0, now=datetime(2026, 1, 2, 9, 30))
    rm.update_equity(9_600.0, now=pd.Timestamp("2026-01-02 15:00"))  # same day -> trips
    assert rm.halted is True
    rm.update_equity(9_600.0, now=pd.Timestamp("2026-01-03 09:30"))  # next day clears
    assert rm.halted is False


# --- max-drawdown halt ------------------------------------------------------


def test_drawdown_halt_trips_from_peak() -> None:
    # Step each reading into its own day so the daily switch never masks the
    # drawdown halt we are isolating.
    rm = RiskManager(_limits(max_drawdown_pct=0.20), 10_000.0)
    rm.update_equity(10_000.0, now=date(2026, 1, 2))
    rm.update_equity(12_000.0, now=date(2026, 1, 3))  # new peak
    rm.update_equity(9_600.0, now=date(2026, 1, 4))  # -20% from peak -> halt
    assert rm.halted is True
    assert rm.can_enter().reason == "max-drawdown halt active"


def test_drawdown_halt_is_sticky_until_reset() -> None:
    rm = RiskManager(_limits(max_drawdown_pct=0.20), 10_000.0)
    rm.update_equity(10_000.0, now=date(2026, 1, 2))
    rm.update_equity(12_000.0, now=date(2026, 1, 3))
    rm.update_equity(9_600.0, now=date(2026, 1, 4))  # trips
    assert rm.halted is True
    rm.update_equity(11_500.0, now=date(2026, 1, 5))  # recovers, still halted
    assert rm.halted is True
    rm.reset()
    rm.update_equity(11_500.0, now=date(2026, 1, 6))  # cleared, no re-trip
    assert rm.halted is False
    assert rm.can_enter().allowed is True


def test_drawdown_reason_takes_priority_over_daily() -> None:
    # A single big down-bar trips both; can_enter reports the more severe one.
    rm = RiskManager(_limits(max_daily_loss_pct=0.03, max_drawdown_pct=0.20), 10_000.0)
    rm.update_equity(10_000.0, now=date(2026, 1, 2))
    rm.update_equity(7_500.0, now=date(2026, 1, 2))  # -25% day & -25% drawdown
    assert rm.can_enter().reason == "max-drawdown halt active"


# --- misc -------------------------------------------------------------------


def test_risk_decision_is_truthy() -> None:
    assert bool(RiskDecision(True, "ok")) is True
    assert bool(RiskDecision(False, "nope")) is False


def test_state_snapshot_keys() -> None:
    rm = RiskManager(_limits(), 10_000.0)
    rm.update_equity(10_000.0, now=date(2026, 1, 2))
    state = rm.state()
    for key in ("peak_equity", "day_start_equity", "current_day", "halted"):
        assert key in state
    assert state["halted"] is False
    assert state["current_day"] == "2026-01-02"


# --- halt persistence hooks ----------------------------------------------------


class _RecordingHaltStore:
    def __init__(self, *, raises: bool = False) -> None:
        self.events: list[dict] = []
        self.raises = raises

    def record_halt(self, **kwargs) -> int:
        if self.raises:
            raise ConnectionError("db down")
        self.events.append(kwargs)
        return len(self.events)


def test_daily_trip_is_persisted() -> None:
    store = _RecordingHaltStore()
    manager = RiskManager(_limits(), 100_000.0, halt_store=store)
    manager.update_equity(100_000.0, now=datetime(2026, 1, 5, 15, 0))
    manager.update_equity(96_000.0, now=datetime(2026, 1, 5, 16, 0))  # -4% > 3%
    assert manager.halted
    assert len(store.events) == 1
    event = store.events[0]
    assert event["halt_type"] == "daily_loss"
    assert event["active"] is True
    assert event["equity_at_halt"] == 96_000.0


def test_drawdown_trip_is_persisted() -> None:
    store = _RecordingHaltStore()
    manager = RiskManager(_limits(), 100_000.0, halt_store=store)
    manager.update_equity(100_000.0, now=datetime(2026, 1, 5))
    manager.update_equity(79_000.0, now=datetime(2026, 1, 6))  # -21% > 20%
    types = [e["halt_type"] for e in store.events]
    assert "drawdown" in types


def test_reset_persists_both_types_cleared() -> None:
    store = _RecordingHaltStore()
    manager = RiskManager(_limits(), 100_000.0, halt_store=store)
    manager.reset()
    cleared = {e["halt_type"]: e["active"] for e in store.events}
    assert cleared == {"daily_loss": False, "drawdown": False}


def test_store_failure_never_breaks_risk_logic() -> None:
    manager = RiskManager(_limits(), 100_000.0, halt_store=_RecordingHaltStore(raises=True))
    manager.update_equity(100_000.0, now=datetime(2026, 1, 5))
    manager.update_equity(70_000.0, now=datetime(2026, 1, 5))  # trips both
    assert manager.halted  # halt logic intact despite the store exploding


def test_restore_relatches_halts() -> None:
    manager = RiskManager(_limits(), 100_000.0)
    manager.restore(drawdown_halt=True)
    assert manager.halted
    assert manager.can_enter().reason == "max-drawdown halt active"


def test_restored_daily_trip_survives_same_day_update() -> None:
    manager = RiskManager(_limits(), 100_000.0)
    day = datetime(2026, 1, 5)
    manager.restore(daily_tripped=True, day=day.date())
    manager.update_equity(100_000.0, now=day)  # same day -> must stay tripped
    assert manager.halted


def test_restored_daily_trip_clears_on_next_day() -> None:
    manager = RiskManager(_limits(), 100_000.0)
    manager.restore(daily_tripped=True, day=datetime(2026, 1, 5).date())
    manager.update_equity(100_000.0, now=datetime(2026, 1, 6))  # rollover
    assert not manager.halted
