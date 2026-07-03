"""SQLAlchemy ORM models for the trade journal.

Three append-only tables capture everything the agent decides and does, so any
run is fully auditable after the fact:

* :class:`SignalRecord`  -- every strategy signal evaluated (incl. HOLDs).
* :class:`OrderRecord`   -- every order the agent submitted, with the reason.
* :class:`EquitySnapshot` -- account equity each loop tick, with halt state.

Timestamps are stored as timezone-aware UTC datetimes supplied by the caller,
so the journal is deterministic in tests and consistent across machines.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all journal tables."""


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(String(256), default="")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SignalRecord(ts={self.ts!r}, symbol={self.symbol!r}, "
            f"action={self.action!r}, confidence={self.confidence})"
        )


class OrderRecord(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="")
    broker_order_id: Mapped[str] = mapped_column(String(64), default="")
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)
    filled_avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(String(64), default="")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"OrderRecord(ts={self.ts!r}, symbol={self.symbol!r}, "
            f"side={self.side!r}, qty={self.qty}, reason={self.reason!r})"
        )


class HaltStateRecord(Base):
    """Append-only halt transitions; the latest row per type is authoritative."""

    __tablename__ = "halt_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    halt_type: Mapped[str] = mapped_column(String(16), index=True)  # daily_loss | drawdown
    active: Mapped[bool] = mapped_column(default=False)
    reason: Mapped[str] = mapped_column(String(128), default="")
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    equity_at_halt: Mapped[float] = mapped_column(Float, default=0.0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"HaltStateRecord(halt_type={self.halt_type!r}, active={self.active}, "
            f"triggered_at={self.triggered_at!r})"
        )


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float, default=0.0)
    halted: Mapped[bool] = mapped_column(default=False)
    halt_reason: Mapped[str] = mapped_column(String(64), default="")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EquitySnapshot(ts={self.ts!r}, equity={self.equity}, halted={self.halted})"
