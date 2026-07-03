"""The trade journal: a thin SQLAlchemy wrapper over the journal tables.

``Journal`` owns the engine and session factory and exposes small, explicit
``record_*`` writers and ``recent_*`` readers. It defaults to a local sqlite
file (gitignored) but accepts any SQLAlchemy URL; ``sqlite:///:memory:`` is
supported for fast, isolated tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from config.logging_config import get_logger
from src.journal.models import Base, EquitySnapshot, HaltStateRecord, OrderRecord, SignalRecord

logger = get_logger(__name__)

DEFAULT_DB_URL = "sqlite:///data/paperpilot.db"

# Any journal table class. Bound to Base so readers keep their concrete row type.
ModelT = TypeVar("ModelT", bound=Base)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _ensure_column(engine: Engine, table: str, column: str, ddl_type: str) -> None:
    """Idempotent sqlite mini-migration: add ``column`` if the table lacks it.

    The project has no Alembic; this covers the narrow case of adding nullable
    or defaulted columns to a journal file created by an older version. No-op
    for fresh databases (create_all already made the full schema) and for
    non-sqlite engines.
    """
    from sqlalchemy import text

    if engine.dialect.name != "sqlite":
        return
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
        if column not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            conn.commit()
            logger.info("Migrated journal: added %s.%s (%s).", table, column, ddl_type)


def _build_engine(db_url: str) -> Engine:
    """Create an engine, making the parent dir for file-based sqlite URLs.

    In-memory sqlite needs a ``StaticPool`` so every session shares the one
    connection (otherwise each session would see a fresh, empty database).
    """
    url = make_url(db_url)
    if url.get_backend_name() == "sqlite":
        database = url.database
        if database and database != ":memory:":
            Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            return create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
    return create_engine(db_url)


class Journal:
    """Append-only audit log of signals, orders, and equity snapshots."""

    def __init__(self, db_url: str = DEFAULT_DB_URL, *, engine: Engine | None = None) -> None:
        self.db_url = db_url
        self._engine = engine or _build_engine(db_url)
        Base.metadata.create_all(self._engine)
        # create_all never ALTERs existing tables, so columns added after a
        # user's journal file was created must be migrated explicitly.
        _ensure_column(self._engine, "orders", "filled_qty", "FLOAT DEFAULT 0.0")
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("Trade journal ready at %s", db_url)

    # --- writers -------------------------------------------------------------

    def record_signal(
        self,
        *,
        symbol: str,
        strategy: str,
        action: str,
        confidence: float = 0.0,
        reason: str = "",
        ts: datetime | None = None,
    ) -> int:
        row = SignalRecord(
            ts=ts or _utcnow(),
            symbol=symbol.upper(),
            strategy=strategy,
            action=action,
            confidence=confidence,
            reason=reason,
        )
        return self._add(row)

    def record_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        status: str = "",
        broker_order_id: str = "",
        filled_qty: float = 0.0,
        filled_avg_price: float | None = None,
        reason: str = "",
        ts: datetime | None = None,
    ) -> int:
        row = OrderRecord(
            ts=ts or _utcnow(),
            symbol=symbol.upper(),
            side=side,
            qty=qty,
            status=status,
            broker_order_id=broker_order_id,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            reason=reason,
        )
        return self._add(row)

    def update_order_fill(
        self,
        broker_order_id: str,
        *,
        status: str,
        filled_qty: float,
        filled_avg_price: float | None,
    ) -> bool:
        """Reconcile an order row's fill fields in place; True if a row matched.

        The journal is append-only by doctrine; this is the ONE deliberate
        exception, used by asynchronous fill sources (the optional trade
        stream) that learn the real fill after the row was written. Only the
        fill fields ever change; the order row itself is never re-created.
        """
        with self._session_factory() as session:
            row = session.scalars(
                select(OrderRecord)
                .where(OrderRecord.broker_order_id == broker_order_id)
                .order_by(OrderRecord.id.desc())
                .limit(1)
            ).first()
            if row is None:
                logger.warning("No journal order with broker id %s to update.", broker_order_id)
                return False
            row.status = status
            row.filled_qty = filled_qty
            row.filled_avg_price = filled_avg_price
            session.commit()
        return True

    def record_halt(
        self,
        *,
        halt_type: str,
        active: bool,
        reason: str = "",
        triggered_at: datetime | None = None,
        equity_at_halt: float = 0.0,
        ts: datetime | None = None,
    ) -> int:
        """Append one halt transition (trip or clear) for ``halt_type``."""
        now = ts or _utcnow()
        row = HaltStateRecord(
            ts=now,
            halt_type=halt_type,
            active=active,
            reason=reason,
            triggered_at=triggered_at or now,
            equity_at_halt=equity_at_halt,
        )
        return self._add(row)

    def record_equity(
        self,
        *,
        equity: float,
        cash: float = 0.0,
        halted: bool = False,
        halt_reason: str = "",
        ts: datetime | None = None,
    ) -> int:
        row = EquitySnapshot(
            ts=ts or _utcnow(),
            equity=equity,
            cash=cash,
            halted=halted,
            halt_reason=halt_reason,
        )
        return self._add(row)

    def _add(self, row: object) -> int:
        with self._session_factory() as session:
            session.add(row)
            session.commit()
            return int(row.id)  # type: ignore[attr-defined]

    # --- readers -------------------------------------------------------------

    def recent_signals(self, limit: int = 20) -> list[SignalRecord]:
        return self._recent(SignalRecord, limit)

    def recent_orders(self, limit: int = 20) -> list[OrderRecord]:
        return self._recent(OrderRecord, limit)

    def recent_equity(self, limit: int = 20) -> list[EquitySnapshot]:
        return self._recent(EquitySnapshot, limit)

    def latest_halt_states(self) -> dict[str, HaltStateRecord]:
        """The most recent transition per halt type (the authoritative state)."""
        with self._session_factory() as session:
            rows = session.scalars(select(HaltStateRecord).order_by(HaltStateRecord.id.asc()))
            return {row.halt_type: row for row in rows}

    def _recent(self, model: type[ModelT], limit: int) -> list[ModelT]:
        with self._session_factory() as session:
            # Every journal table declares ``id``; mypy cannot see it through the
            # generic ``type[ModelT]``, so this one access is annotated.
            stmt = select(model).order_by(model.id.desc()).limit(limit)  # type: ignore[attr-defined]
            rows = list(session.scalars(stmt))
        rows.reverse()  # oldest-first for display
        return rows

    def peak_equity(self) -> float | None:
        """Highest equity ever recorded, for seeding the risk manager's peak."""
        with self._session_factory() as session:
            value = session.scalar(select(func.max(EquitySnapshot.equity)))
        return float(value) if value is not None else None

    def counts(self) -> dict[str, int]:
        """Row counts per table -- handy for a quick journal summary."""
        with self._session_factory() as session:
            return {
                "signals": session.scalar(select(func.count(SignalRecord.id))) or 0,
                "orders": session.scalar(select(func.count(OrderRecord.id))) or 0,
                "equity_snapshots": session.scalar(select(func.count(EquitySnapshot.id))) or 0,
            }

    def close(self) -> None:
        self._engine.dispose()
