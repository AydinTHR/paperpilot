"""Realized-trade pairing and per-strategy P&L attribution.

The orders table is an event log; this module turns it into a discrete
``trades`` table by FIFO-pairing BUY fills against SELL fills per
``(symbol, strategy)``: each SELL consumes the oldest open BUY lots first,
emitting one realized trade row per (partial) lot match. Actual fill prices
and quantities from reconciliation (Phase 2) drive the math, so the numbers
reflect what really executed, not what was requested.

``build_trades`` is a pure function over order rows; ``Journal`` gains an
idempotent ``rebuild_trades()`` (delete-and-rebuild, safe to re-run) and a
``strategy_report()`` reader.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.journal.models import OrderRecord


@dataclass(frozen=True)
class TradeInput:
    """One realized (entry, exit) pairing, ready to insert as a trades row."""

    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_px: float
    exit_px: float
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    entry_reason: str
    exit_reason: str


@dataclass
class _Lot:
    qty: float
    price: float
    time: datetime
    reason: str


def _fill_qty(order: OrderRecord) -> float:
    """Actual filled quantity; legacy filled rows fall back to requested qty."""
    if order.filled_qty and order.filled_qty > 0:
        return float(order.filled_qty)
    status = (order.status or "").lower()
    if status in ("filled", "submitted", "accepted", ""):
        return float(order.qty)
    return 0.0


def _fill_price(order: OrderRecord) -> float | None:
    if order.filled_avg_price is not None and order.filled_avg_price > 0:
        return float(order.filled_avg_price)
    return None


def build_trades(orders: list[OrderRecord]) -> list[TradeInput]:
    """FIFO-pair buy/sell fills into realized trades, grouped per (symbol, strategy).

    Orders with no usable fill (rejected/canceled, zero quantity, or no fill
    price on either side of a pairing) are skipped. A SELL larger than the open
    lots realizes only what was actually held (long-only: no short lots).
    """
    open_lots: dict[tuple[str, str], deque[_Lot]] = {}
    trades: list[TradeInput] = []

    for order in sorted(orders, key=lambda o: o.id):
        qty = _fill_qty(order)
        price = _fill_price(order)
        if qty <= 0:
            continue
        key = (order.symbol, order.strategy or "")
        side = (order.side or "").lower()

        if side == "buy":
            if price is None:
                continue  # cannot open a lot without a real entry price
            open_lots.setdefault(key, deque()).append(
                _Lot(qty=qty, price=price, time=order.ts, reason=order.reason or "")
            )
            continue

        if side != "sell":
            continue
        lots = open_lots.get(key)
        remaining = qty
        while lots and remaining > 0:
            lot = lots[0]
            matched = min(lot.qty, remaining)
            if price is not None:
                pnl = (price - lot.price) * matched
                held = (order.ts - lot.time).total_seconds() / 3600.0
                trades.append(
                    TradeInput(
                        symbol=order.symbol,
                        strategy=order.strategy or "",
                        entry_time=lot.time,
                        exit_time=order.ts,
                        qty=matched,
                        entry_px=lot.price,
                        exit_px=price,
                        pnl=pnl,
                        pnl_pct=pnl / (lot.price * matched) if lot.price > 0 else 0.0,
                        holding_period_hours=max(held, 0.0),
                        entry_reason=lot.reason,
                        exit_reason=order.reason or "",
                    )
                )
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= 1e-9:
                lots.popleft()

    return trades


@dataclass(frozen=True)
class StrategyStats:
    """Realized-P&L summary for one strategy."""

    strategy: str
    num_trades: int
    win_rate_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    avg_holding_hours: float


def summarize_by_strategy(trades: list[TradeInput]) -> dict[str, StrategyStats]:
    """Aggregate realized trades into per-strategy statistics."""
    by_strategy: dict[str, list[TradeInput]] = {}
    for trade in trades:
        by_strategy.setdefault(trade.strategy or "(untagged)", []).append(trade)

    report: dict[str, StrategyStats] = {}
    for name, rows in sorted(by_strategy.items()):
        wins = [t.pnl for t in rows if t.pnl > 0]
        losses = [t.pnl for t in rows if t.pnl <= 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        report[name] = StrategyStats(
            strategy=name,
            num_trades=len(rows),
            win_rate_pct=100.0 * len(wins) / len(rows) if rows else 0.0,
            avg_win=gross_win / len(wins) if wins else 0.0,
            avg_loss=-gross_loss / len(losses) if losses else 0.0,
            profit_factor=gross_win / gross_loss if gross_loss > 0 else float("inf"),
            total_pnl=sum(t.pnl for t in rows),
            avg_holding_hours=(
                sum(t.holding_period_hours for t in rows) / len(rows) if rows else 0.0
            ),
        )
    return report
