"""Risk management: position sizing, per-trade stops, and trading halts.

This module is the safety core of PaperPilot. It is deliberately **pure and
broker-agnostic** -- every method takes plain numbers (equity, price, cash) and
returns plain numbers or decisions, so the exact same logic drives the backtest
engine (Phase 3) and the live paper loop (Phase 5), and the whole thing is
unit-testable offline with no network or broker.

Three controls, escalating in severity:

* **Position sizing** -- never risk more than ``max_position_pct`` of equity in a
  single entry (a per-strategy ``size_hint`` may only make this *smaller*).
* **Daily-loss kill switch** -- once the loss *within a calendar day* reaches
  ``max_daily_loss_pct`` of that day's starting equity, refuse new entries for
  the rest of the day. Resets automatically at the next day rollover.
* **Max-drawdown halt** -- once equity falls ``max_drawdown_pct`` below its
  all-time peak, halt entirely. This is *sticky*: it stays tripped until an
  operator calls :meth:`RiskManager.reset`, because a deep drawdown is a signal
  that something is wrong, not a routine daily event.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

from config.logging_config import get_logger

if TYPE_CHECKING:
    from config.settings import Settings

logger = get_logger(__name__)


class HaltStore(Protocol):
    """Where halt transitions are persisted (the Journal satisfies this).

    A structural Protocol keeps the risk core pure: it never imports SQLAlchemy
    or the journal package, and tests use an in-memory fake.
    """

    def record_halt(
        self,
        *,
        halt_type: str,
        active: bool,
        reason: str,
        triggered_at: datetime,
        equity_at_halt: float,
    ) -> int: ...


@dataclass(frozen=True)
class RiskLimits:
    """Immutable bundle of the four risk thresholds (all fractions of equity)."""

    max_position_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    stop_loss_pct: float

    @classmethod
    def from_settings(cls, settings: Settings) -> RiskLimits:
        """Build limits from a ``config.settings.Settings`` instance."""
        return cls(
            max_position_pct=settings.max_position_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            stop_loss_pct=settings.stop_loss_pct,
        )


@dataclass(frozen=True)
class RiskDecision:
    """The answer to 'may I open a new position right now?' plus a reason."""

    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


def _as_date(now: datetime | date | None) -> date:
    """Coerce a timestamp/datetime/date (or None -> today) to a calendar date."""
    if now is None:
        return datetime.now().date()
    if isinstance(now, datetime):  # also matches pandas.Timestamp
        return now.date()
    return now


class RiskManager:
    """Stateful guardian that sizes positions and trips trading halts.

    The manager tracks two running statistics from :meth:`update_equity`:
    the all-time peak equity (for the drawdown halt) and the current day's
    starting equity (for the daily kill switch). Both halts *latch* once
    tripped -- the daily one until the day rolls over, the drawdown one until
    :meth:`reset` is called.
    """

    def __init__(
        self,
        limits: RiskLimits,
        starting_equity: float,
        *,
        halt_store: HaltStore | None = None,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError(f"starting_equity must be positive, got {starting_equity}")
        self.limits = limits
        self._peak_equity = starting_equity
        self._current_day: date | None = None
        self._day_start_equity = starting_equity
        self._daily_tripped = False
        self._drawdown_halt = False
        self._halt_store = halt_store

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        starting_equity: float,
        *,
        halt_store: HaltStore | None = None,
    ) -> RiskManager:
        return cls(RiskLimits.from_settings(settings), starting_equity, halt_store=halt_store)

    def restore(
        self,
        *,
        drawdown_halt: bool = False,
        daily_tripped: bool = False,
        day: date | None = None,
    ) -> None:
        """Re-latch persisted halt state after a process restart.

        Restoring the daily trip must also set the trip's calendar day,
        otherwise the next :meth:`update_equity` sees a "new day" and clears
        it immediately.
        """
        if drawdown_halt:
            self._drawdown_halt = True
        if daily_tripped:
            self._daily_tripped = True
            self._current_day = day or datetime.now().date()

    def seed_peak(self, equity: float) -> None:
        """Raise the tracked peak to ``equity`` if higher (never lowers it).

        Used on live-loop startup to restore the all-time peak from the trade
        journal, so a process restart cannot quietly forget a drawdown and
        re-arm the halt from a lower baseline.
        """
        if equity > self._peak_equity:
            self._peak_equity = equity

    # --- sizing --------------------------------------------------------------

    def position_size(
        self,
        equity: float,
        price: float,
        cash: float,
        size_hint: float | None = None,
    ) -> int:
        """Whole number of shares to buy, capped by risk *and* available cash.

        The cap is ``max_position_pct`` of equity; a strategy ``size_hint`` may
        only tighten it, never loosen it. The result is additionally bounded by
        ``cash`` so the order can actually be afforded. Returns 0 on any
        degenerate input (non-positive price/equity/cash) or when even one share
        is unaffordable.
        """
        if price <= 0 or equity <= 0 or cash <= 0:
            return 0
        frac = self.limits.max_position_pct
        if size_hint is not None:
            frac = min(size_hint, frac)
        frac = max(frac, 0.0)
        budget = min(equity * frac, cash)
        return max(int(budget // price), 0)

    # --- per-trade stop ------------------------------------------------------

    def stop_price(self, entry: float) -> float:
        """Stop-loss price a fixed fraction below the entry price."""
        return entry * (1.0 - self.limits.stop_loss_pct)

    def stop_breached(self, entry: float, current: float) -> bool:
        """True if ``current`` has fallen to/through the stop (no-op if disabled)."""
        if self.limits.stop_loss_pct <= 0:
            return False
        return current <= self.stop_price(entry)

    # --- halts ---------------------------------------------------------------

    def update_equity(self, equity: float, *, now: datetime | date | None = None) -> None:
        """Feed the latest equity reading; updates peaks and trips halts.

        Call this once per bar (backtest) or per loop tick (live) *before*
        consulting :meth:`can_enter`. ``now`` controls the calendar-day logic
        for the daily kill switch; in a backtest pass the bar's timestamp.
        """
        day = _as_date(now)
        if self._current_day is None or day != self._current_day:
            self._current_day = day
            self._day_start_equity = equity
            self._daily_tripped = False

        self._peak_equity = max(self._peak_equity, equity)

        if self._day_start_equity > 0 and not self._daily_tripped:
            daily_loss = (self._day_start_equity - equity) / self._day_start_equity
            if daily_loss >= self.limits.max_daily_loss_pct:
                self._daily_tripped = True
                logger.warning(
                    "Daily-loss kill switch TRIPPED: down %.2f%% on the day "
                    "(limit %.2f%%). No new entries until tomorrow.",
                    daily_loss * 100.0,
                    self.limits.max_daily_loss_pct * 100.0,
                )
                self._persist(
                    "daily_loss",
                    active=True,
                    reason=f"down {daily_loss:.2%} on the day",
                    now=now,
                    equity=equity,
                )

        if self._peak_equity > 0 and not self._drawdown_halt:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.limits.max_drawdown_pct:
                self._drawdown_halt = True
                logger.warning(
                    "Max-drawdown halt TRIPPED: %.2f%% below peak (limit %.2f%%). "
                    "Trading halted until manual reset.",
                    drawdown * 100.0,
                    self.limits.max_drawdown_pct * 100.0,
                )
                self._persist(
                    "drawdown",
                    active=True,
                    reason=f"{drawdown:.2%} below peak",
                    now=now,
                    equity=equity,
                )

    def can_enter(self) -> RiskDecision:
        """Whether a new position may be opened right now, with a reason."""
        if self._drawdown_halt:
            return RiskDecision(False, "max-drawdown halt active")
        if self._daily_tripped:
            return RiskDecision(False, "daily-loss kill switch tripped")
        return RiskDecision(True, "ok")

    @property
    def halted(self) -> bool:
        """True if either halt is currently active."""
        return self._drawdown_halt or self._daily_tripped

    @property
    def halt_reason(self) -> str:
        """Human-readable reason for the active halt, or '' if not halted."""
        return self.can_enter().reason if self.halted else ""

    def reset(self, *, reason: str = "manual reset") -> None:
        """Clear both halts (operator action after reviewing a drawdown event).

        Both types are journaled as cleared so the persisted latest-state view
        converges with memory.
        """
        self._daily_tripped = False
        self._drawdown_halt = False
        for halt_type in ("daily_loss", "drawdown"):
            self._persist(halt_type, active=False, reason=reason, now=None, equity=0.0)

    def _persist(
        self,
        halt_type: str,
        *,
        active: bool,
        reason: str,
        now: datetime | date | None,
        equity: float,
    ) -> None:
        """Best-effort halt persistence; a store failure never breaks risk logic."""
        if self._halt_store is None:
            return
        triggered = now if isinstance(now, datetime) else datetime.now()
        try:
            self._halt_store.record_halt(
                halt_type=halt_type,
                active=active,
                reason=reason,
                triggered_at=triggered,
                equity_at_halt=equity,
            )
        except Exception as exc:
            logger.warning("Could not persist %s halt state: %s", halt_type, exc)

    def state(self) -> dict[str, object]:
        """A flat, log-friendly snapshot of the manager's internal state."""
        return {
            "peak_equity": round(self._peak_equity, 2),
            "day_start_equity": round(self._day_start_equity, 2),
            "current_day": self._current_day.isoformat() if self._current_day else None,
            "daily_tripped": self._daily_tripped,
            "drawdown_halt": self._drawdown_halt,
            "halted": self.halted,
        }
