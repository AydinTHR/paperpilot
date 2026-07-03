"""Market-hours awareness for the scheduled loop.

Offline-first: session times (including holidays and early closes) come from
``pandas_market_calendars``, which ships its calendar data locally, so the
check needs no network and works in tests with frozen datetimes. An optional
``clock_fn`` (wrapping Alpaca's ``TradingClient.get_clock``) can confirm the
answer live; any failure there degrades silently to the offline result.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)

# How far around "now" to materialise the session schedule. A week each way
# comfortably covers weekends and holiday clusters.
_SCHEDULE_PAD = timedelta(days=7)


class MarketCalendar:
    """Answers "is the market open?" for one exchange calendar (default NYSE)."""

    def __init__(
        self,
        calendar_name: str = "NYSE",
        *,
        clock_fn: Callable[[], Any] | None = None,
    ) -> None:
        self.calendar_name = calendar_name
        self._clock_fn = clock_fn
        self._calendar: Any = None

    def is_market_open(self, now: datetime | None = None) -> bool:
        """True while the exchange is in a regular session at ``now``."""
        now = _as_utc(now)
        if self._clock_fn is not None:
            try:
                return bool(self._clock_fn().is_open)
            except Exception as exc:
                logger.warning("Live market clock failed (%s); using offline calendar.", exc)
        return self._is_open_offline(now)

    def next_session_open(self, now: datetime | None = None) -> datetime | None:
        """The next session-open timestamp strictly after ``now`` (UTC)."""
        now = _as_utc(now)
        schedule = self._schedule(now)
        opens = schedule["market_open"]
        future = opens[opens > pd.Timestamp(now)]
        if future.empty:
            return None
        return future.iloc[0].to_pydatetime()

    # --- internals ---

    def _is_open_offline(self, now: datetime) -> bool:
        schedule = self._schedule(now)
        ts = pd.Timestamp(now)
        open_col, close_col = schedule["market_open"], schedule["market_close"]
        return bool(((open_col <= ts) & (ts < close_col)).any())

    def _schedule(self, now: datetime) -> pd.DataFrame:
        cal = self._get_calendar()
        return cal.schedule(
            start_date=(now - _SCHEDULE_PAD).date(),
            end_date=(now + _SCHEDULE_PAD).date(),
        )

    def _get_calendar(self) -> Any:
        if self._calendar is None:
            import pandas_market_calendars as pmc  # local calendar data; no network

            self._calendar = pmc.get_calendar(self.calendar_name)
        return self._calendar


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    return now if now.tzinfo else now.replace(tzinfo=UTC)
