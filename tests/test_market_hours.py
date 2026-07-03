"""Tests for market-hours awareness and the scheduled-tick gate.

pandas_market_calendars ships its calendar data locally, so every check here
runs offline with frozen datetimes. NYSE regular hours are 09:30-16:00 ET
(14:30-21:00 UTC in January).
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.agent.market_hours import MarketCalendar

# Frozen instants (UTC). 2024-01-10 is a regular Wednesday session.
MID_SESSION = datetime(2024, 1, 10, 15, 0, tzinfo=UTC)  # 10:00 ET, open
PRE_OPEN = datetime(2024, 1, 10, 13, 0, tzinfo=UTC)  # 08:00 ET, closed
SATURDAY = datetime(2024, 1, 6, 15, 0, tzinfo=UTC)
NEW_YEARS_DAY = datetime(2024, 1, 1, 15, 0, tzinfo=UTC)  # NYSE holiday
# 2024-07-03 closed early at 13:00 ET (17:00 UTC in July).
HALF_DAY_AFTER_CLOSE = datetime(2024, 7, 3, 17, 30, tzinfo=UTC)
HALF_DAY_MORNING = datetime(2024, 7, 3, 14, 30, tzinfo=UTC)  # 10:30 ET, open


class _FakeClock:
    def __init__(self, is_open: bool, *, raises: bool = False) -> None:
        self.is_open = is_open
        self.raises = raises

    def __call__(self) -> _FakeClock:
        if self.raises:
            raise ConnectionError("clock endpoint down")
        return self


def test_open_mid_session() -> None:
    assert MarketCalendar().is_market_open(MID_SESSION)


def test_closed_pre_open() -> None:
    assert not MarketCalendar().is_market_open(PRE_OPEN)


def test_closed_on_saturday() -> None:
    assert not MarketCalendar().is_market_open(SATURDAY)


def test_closed_on_holiday() -> None:
    assert not MarketCalendar().is_market_open(NEW_YEARS_DAY)


def test_half_day_early_close_respected() -> None:
    cal = MarketCalendar()
    assert cal.is_market_open(HALF_DAY_MORNING)
    assert not cal.is_market_open(HALF_DAY_AFTER_CLOSE)


def test_next_session_open_from_weekend() -> None:
    nxt = MarketCalendar().next_session_open(SATURDAY)
    assert nxt is not None
    assert nxt.date().isoformat() == "2024-01-08"  # Monday
    assert (nxt.hour, nxt.minute) == (14, 30)  # 09:30 ET in UTC


def test_live_clock_overrides_offline_answer() -> None:
    # Saturday offline says closed; a live clock saying open wins.
    cal = MarketCalendar(clock_fn=_FakeClock(is_open=True))
    assert cal.is_market_open(SATURDAY)


def test_live_clock_failure_degrades_to_offline() -> None:
    cal = MarketCalendar(clock_fn=_FakeClock(is_open=True, raises=True))
    assert not cal.is_market_open(SATURDAY)  # offline answer stands
    assert cal.is_market_open(MID_SESSION)


def test_naive_datetime_treated_as_utc() -> None:
    assert MarketCalendar().is_market_open(MID_SESSION.replace(tzinfo=None))
