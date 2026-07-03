"""The autonomous paper-trading loop -- where every layer comes together.

One :meth:`TradingLoop.run_once` iteration is the whole agent in miniature:

1. Read the account; feed equity to the risk manager and journal a snapshot.
2. If a halt is active (daily kill switch or max-drawdown), **flatten every
   position and place no new entries** -- the safety layer overrides signals.
3. Otherwise, for each symbol: fetch bars, enforce the live stop-loss on any
   open position, generate a signal, journal it, and -- subject to the risk
   manager's sizing and entry gate -- place a long-only market order.

Everything the agent sees and does is written to the trade journal, so a run is
fully auditable. :meth:`run_scheduled` simply calls :meth:`run_once` on an
APScheduler cadence; the loop body is identical, which keeps it unit-testable
offline with fakes (see ``tests/test_live_loop.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.data.market_data import build_provider
from src.execution.broker import AccountSnapshot, Broker, BrokerError, OrderInfo, PositionInfo
from src.execution.reconcile import OrderReconciler
from src.journal.store import Journal
from src.risk.manager import RiskManager
from src.strategy.base import Action, Strategy

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class BrokerLike(Protocol):
    """The slice of :class:`~src.execution.broker.Broker` the loop relies on."""

    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[PositionInfo]: ...
    def place_market_order(self, symbol: str, qty: float, side: str) -> OrderInfo: ...
    def place_market_order_with_stop(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        ref_price: float | None = None,
    ) -> OrderInfo: ...
    def close_position(self, symbol: str) -> None: ...
    def get_order(self, order_id: str) -> OrderInfo: ...
    def get_open_orders(self, symbol: str | None = None) -> list[OrderInfo]: ...


class DataProviderLike(Protocol):
    def get_latest_bars(self, symbol: str, lookback: int, interval: str) -> pd.DataFrame: ...


class MarketCalendarLike(Protocol):
    """The slice of :class:`~src.agent.market_hours.MarketCalendar` the loop uses."""

    def is_market_open(self, now: datetime | None = None) -> bool: ...
    def next_session_open(self, now: datetime | None = None) -> datetime | None: ...


@dataclass(frozen=True)
class SymbolOutcome:
    """What the loop decided (and did) for one symbol this tick."""

    symbol: str
    action: str  # BUY / SELL / HOLD / STOP / FLATTEN / SKIP
    detail: str = ""
    qty: float | None = None
    price: float | None = None
    order_id: str | None = None


@dataclass(frozen=True)
class LoopResult:
    """Summary of one loop iteration."""

    ts: datetime
    equity: float
    cash: float
    halted: bool
    halt_reason: str
    outcomes: list[SymbolOutcome] = field(default_factory=list)


class TradingLoop:
    """Orchestrates data -> strategy -> risk -> broker -> journal each tick."""

    def __init__(
        self,
        *,
        broker: BrokerLike,
        provider: DataProviderLike,
        strategy: Strategy,
        risk: RiskManager,
        journal: Journal,
        settings: Settings,
        symbols: list[str],
        lookback: int | None = None,
        market_calendar: MarketCalendarLike | None = None,
        reconciler: OrderReconciler | None = None,
    ) -> None:
        self.broker = broker
        self.provider = provider
        self.strategy = strategy
        self.risk = risk
        self.journal = journal
        self.settings = settings
        self.symbols = [s.upper() for s in symbols]
        self.interval = settings.default_interval
        # Enough history to warm the strategy up, with headroom.
        self.lookback = lookback or max(strategy.min_bars + 50, 200)
        # When set, scheduled ticks are skipped while the market is closed.
        # run_once is never gated, so --once and tests always execute.
        self.market_calendar = market_calendar
        self._reconciler = reconciler or OrderReconciler(broker)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        strategy: Strategy,
        symbols: list[str] | None = None,
        lookback: int | None = None,
        journal: Journal | None = None,
        announce: bool = True,
        market_hours_gate: bool | None = None,
    ) -> TradingLoop:
        """Wire the loop to the real broker, data provider, risk, and journal.

        The risk manager starts from the live account equity and has its peak
        seeded from the journal, so a process restart cannot forget a prior
        drawdown and silently re-arm the halt from a lower baseline.
        ``market_hours_gate`` overrides ``settings.market_hours_only`` (the CLI
        flag ``--ignore-market-hours`` passes False).
        """
        from src.agent.market_hours import MarketCalendar
        from src.data.universe import get_universe

        settings = settings or get_settings()
        broker = Broker(settings, announce=announce)
        provider = build_provider(settings)
        journal = journal or Journal(settings.db_url)

        account = broker.get_account()
        if not (account.equity > 0):  # 0, negative, or NaN -> no usable baseline
            mode = "live" if settings.is_live else "paper"
            raise BrokerError(
                f"Account equity is {account.equity:,.2f} {account.currency} -- cannot "
                f"start the trading loop on an empty account. Reset your Alpaca {mode} "
                f"account balance in the dashboard (https://app.alpaca.markets) first."
            )
        risk = RiskManager.from_settings(settings, starting_equity=account.equity)
        peak = journal.peak_equity()
        if peak is not None:
            risk.seed_peak(peak)

        gate = settings.market_hours_only if market_hours_gate is None else market_hours_gate
        return cls(
            broker=broker,
            provider=provider,
            strategy=strategy,
            risk=risk,
            journal=journal,
            settings=settings,
            symbols=symbols or get_universe(),
            lookback=lookback,
            market_calendar=MarketCalendar() if gate else None,
        )

    # --- one iteration -------------------------------------------------------

    def run_once(self, *, now: datetime | None = None) -> LoopResult:
        """Execute a single trading iteration and return what happened."""
        now = now or datetime.now(UTC)
        account = self.broker.get_account()
        equity, cash = account.equity, account.cash

        self.risk.update_equity(equity, now=now)
        halted = self.risk.halted
        halt_reason = self.risk.halt_reason
        self.journal.record_equity(
            equity=equity, cash=cash, halted=halted, halt_reason=halt_reason, ts=now
        )

        positions = {p.symbol.upper(): p for p in self.broker.get_positions()}

        outcomes: list[SymbolOutcome] = []
        if halted:
            outcomes = self._flatten_all(positions, halt_reason, now)
            logger.warning(
                "Risk halt active (%s): flattened %d position(s), no new entries.",
                halt_reason,
                sum(1 for o in outcomes if o.action == "FLATTEN"),
            )
            return LoopResult(now, equity, cash, True, halt_reason, outcomes)

        available_cash = cash
        for symbol in self.symbols:
            outcome = self._process_symbol(
                symbol, positions.get(symbol), equity, available_cash, now
            )
            if outcome.action == "BUY" and outcome.qty and outcome.price:
                available_cash = max(available_cash - outcome.qty * outcome.price, 0.0)
            outcomes.append(outcome)

        logger.info(
            "Loop tick done: equity=%.2f cash=%.2f actions=%s",
            equity,
            cash,
            {o.symbol: o.action for o in outcomes},
        )
        return LoopResult(now, equity, cash, False, "", outcomes)

    # --- per-symbol decision -------------------------------------------------

    def _process_symbol(
        self,
        symbol: str,
        position: PositionInfo | None,
        equity: float,
        cash: float,
        now: datetime,
    ) -> SymbolOutcome:
        try:
            bars = self.provider.get_latest_bars(
                symbol, lookback=self.lookback, interval=self.interval
            )
        except Exception as exc:
            logger.error("Data fetch failed for %s: %s", symbol, exc)
            return SymbolOutcome(symbol, "SKIP", f"data error: {exc}")

        if len(bars) < self.strategy.min_bars:
            return SymbolOutcome(
                symbol,
                "SKIP",
                f"insufficient bars ({len(bars)} < {self.strategy.min_bars})",
            )

        price = float(bars["Close"].iloc[-1])
        holding = position is not None and position.qty > 0

        # Loop-side stop-loss. With broker-held stops this is a BACKSTOP that
        # only engages when no live stop order protects the position (Alpaca
        # DAY stop legs expire at the close, leaving overnight holds bare).
        if (
            holding
            and position is not None
            and not self._has_live_stop(symbol)
            and self.risk.stop_breached(position.avg_entry_price, price)
        ):
            self.broker.close_position(symbol)
            self.journal.record_order(
                symbol=symbol,
                side="sell",
                qty=position.qty,
                status="submitted",
                reason="stop-loss",
                ts=now,
            )
            logger.warning("Stop-loss hit on %s at %.2f; closed position.", symbol, price)
            return SymbolOutcome(
                symbol, "STOP", f"stop-loss @ {price:.2f}", qty=position.qty, price=price
            )

        signal = self.strategy.generate_signals(bars)
        self.journal.record_signal(
            symbol=symbol,
            strategy=self.strategy.name,
            action=signal.action.value,
            confidence=signal.confidence,
            reason=signal.reason,
            ts=now,
        )

        if signal.action is Action.BUY and not holding:
            return self._try_enter(
                symbol, price, equity, cash, signal.size_hint, signal.reason, now
            )

        if signal.action is Action.SELL and holding and position is not None:
            self.broker.close_position(symbol)
            self.journal.record_order(
                symbol=symbol,
                side="sell",
                qty=position.qty,
                status="submitted",
                reason="signal SELL",
                ts=now,
            )
            return SymbolOutcome(symbol, "SELL", signal.reason, qty=position.qty, price=price)

        return SymbolOutcome(symbol, "HOLD", signal.reason or signal.action.value, price=price)

    def _try_enter(
        self,
        symbol: str,
        price: float,
        equity: float,
        cash: float,
        size_hint: float | None,
        reason: str,
        now: datetime,
    ) -> SymbolOutcome:
        decision = self.risk.can_enter()
        if not decision:
            return SymbolOutcome(symbol, "HOLD", f"risk blocked: {decision.reason}", price=price)

        qty = self.risk.position_size(equity, price, cash, size_hint)
        if qty < 1:
            return SymbolOutcome(symbol, "HOLD", "risk-sized qty < 1 share", price=price)

        use_stop = self.settings.resolved_use_broker_stops and self.risk.limits.stop_loss_pct > 0
        try:
            if use_stop:
                order = self.broker.place_market_order_with_stop(
                    symbol, qty, "buy", self.risk.stop_price(price), ref_price=price
                )
            else:
                order = self.broker.place_market_order(symbol, qty, "buy")
        except BrokerError as exc:
            # A rejected entry (wash trade, transient API failure) must not
            # kill the tick; the signal simply goes unfilled this round.
            logger.warning("Entry rejected for %s: %s", symbol, exc)
            return SymbolOutcome(symbol, "SKIP", f"order rejected: {exc}", price=price)

        recon = self._reconciler.wait_for_terminal(order.id)
        self.journal.record_order(
            symbol=symbol,
            side="buy",
            qty=qty,
            status=recon.status,
            broker_order_id=order.id,
            filled_qty=recon.filled_qty,
            filled_avg_price=recon.filled_avg_price,
            reason="signal BUY",
            ts=now,
        )
        return SymbolOutcome(symbol, "BUY", reason, qty=qty, price=price, order_id=order.id)

    def _has_live_stop(self, symbol: str) -> bool:
        """True when a broker-held stop order currently protects ``symbol``."""
        if not self.settings.resolved_use_broker_stops:
            return False
        try:
            orders = self.broker.get_open_orders(symbol)
        except Exception as exc:
            logger.warning("Open-order check failed for %s (%s); using loop stop.", symbol, exc)
            return False
        return any("stop" in o.order_type.lower() for o in orders)

    def _flatten_all(
        self,
        positions: dict[str, PositionInfo],
        halt_reason: str,
        now: datetime,
    ) -> list[SymbolOutcome]:
        outcomes: list[SymbolOutcome] = []
        for symbol, pos in positions.items():
            if pos.qty <= 0:
                continue
            self.broker.close_position(symbol)
            self.journal.record_order(
                symbol=symbol,
                side="sell",
                qty=pos.qty,
                status="submitted",
                reason=f"halt: {halt_reason}",
                ts=now,
            )
            outcomes.append(SymbolOutcome(symbol, "FLATTEN", halt_reason, qty=pos.qty))
        return outcomes

    # --- scheduled mode ------------------------------------------------------

    def scheduled_tick(self, *, now: datetime | None = None) -> LoopResult | None:
        """One scheduler tick: the market-hours gate plus a crash-contained run.

        A plain method rather than a closure so both behaviours are testable
        without a running scheduler. Returns None when the tick was skipped
        (market closed) or failed (logged; the scheduler retries next interval).
        """
        now = now or datetime.now(UTC)
        if self.market_calendar is not None and not self.market_calendar.is_market_open(now):
            logger.info(
                "Market closed; skipping tick. Next open: %s",
                self.market_calendar.next_session_open(now),
            )
            return None
        try:
            result = self.run_once(now=now)
        except Exception:
            logger.exception("Loop tick failed; will retry next interval.")
            return None
        logger.info("Scheduled tick complete: equity=%.2f halted=%s", result.equity, result.halted)
        return result

    def run_scheduled(self, interval_minutes: int | None = None) -> None:
        """Run :meth:`run_once` forever on an APScheduler interval (blocking)."""
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = interval_minutes or self.settings.loop_interval_minutes

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(
            self.scheduled_tick, "interval", minutes=interval, next_run_time=datetime.now(UTC)
        )
        logger.info(
            "Starting scheduled loop: %s on %s every %d min. Ctrl-C to stop.",
            self.strategy.name,
            self.symbols,
            interval,
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user.")
