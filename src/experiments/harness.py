"""Run SMA vs RSI vs LLM concurrently and compare them fairly.

A fair comparison needs isolation: one strategy's fills must not move another
strategy's equity. Two modes, picked automatically:

* **accounts** -- each arm gets its own Alpaca paper account (Alpaca allows up
  to 3; configure ``ALPACA_API_KEY_2/3``). Cleanest: real per-arm equity curves.
* **virtual** -- one real account, each arm wrapped in a
  :class:`VirtualPortfolio` that implements the loop's ``BrokerLike`` Protocol:
  it starts with ``equity / N`` virtual cash, forwards buys as real orders, and
  translates ``close_position`` into a *sized* sell so one arm can never
  liquidate another arm's shares. Caveat: same-symbol positions from different
  arms share real-account netting and wash-trade rejections, so prefer the
  accounts mode (or disjoint universes) for publishable results.

Every arm journals to its own sqlite file (``data/experiments/{name}.db``), so
the Phase 5 trades/report readers work per-arm unchanged. All arms share ONE
market data provider (one cache; rate-limit friendly), and the LLM arm gets a
per-bar response cache so re-runs cost zero API dollars.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.agent.loop import LoopResult, TradingLoop
from src.data.market_data import build_provider
from src.execution.broker import AccountSnapshot, Broker, OrderInfo, PositionInfo
from src.journal.store import Journal
from src.risk.manager import RiskManager
from src.strategy.base import Strategy
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover
from src.strategy.llm.strategy import LlmStrategy

logger = get_logger(__name__)

MIN_TICKS_FOR_SHARPE = 20


class VirtualPortfolio:
    """One arm's slice of a shared real broker, as a ``BrokerLike``.

    Keeps a virtual cash balance and position book; only *orders* touch the
    real account. The ledger books fills at the order's fill price when the
    (near-instant) paper fill reports one, else at the caller's price via the
    real position's latest mark.
    """

    def __init__(self, broker: Broker, fraction: float, name: str) -> None:
        self._broker = broker
        self.name = name
        account = broker.get_account()
        self._cash = account.equity * fraction
        self._book: dict[str, PositionInfo] = {}

    # --- BrokerLike surface ---

    def get_account(self) -> AccountSnapshot:
        real = self._broker.get_account()
        equity = self._cash + sum(
            p.qty * self._mark(p.symbol, p.avg_entry_price) for p in self._book.values()
        )
        return AccountSnapshot(
            account_number=f"{real.account_number}/{self.name}",
            status=real.status,
            currency=real.currency,
            cash=self._cash,
            equity=equity,
            buying_power=self._cash,
            portfolio_value=equity,
        )

    def get_positions(self) -> list[PositionInfo]:
        positions = []
        for p in self._book.values():
            price = self._mark(p.symbol, p.avg_entry_price)
            positions.append(
                PositionInfo(
                    symbol=p.symbol,
                    qty=p.qty,
                    side="long",
                    avg_entry_price=p.avg_entry_price,
                    current_price=price,
                    market_value=p.qty * price,
                    unrealized_pl=(price - p.avg_entry_price) * p.qty,
                    unrealized_plpc=(price / p.avg_entry_price - 1.0) if p.avg_entry_price else 0.0,
                )
            )
        return positions

    def place_market_order(self, symbol: str, qty: float, side: str) -> OrderInfo:
        order = self._broker.place_market_order(symbol, qty, side)
        self._book_fill(order, side)
        return order

    def place_market_order_with_stop(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        *,
        ref_price: float | None = None,
    ) -> OrderInfo:
        order = self._broker.place_market_order_with_stop(
            symbol, qty, side, stop_price, ref_price=ref_price
        )
        self._book_fill(order, side)
        return order

    def close_position(self, symbol: str) -> OrderInfo | None:
        """Sell only THIS arm's shares -- never the whole real position."""
        symbol = symbol.upper()
        held = self._book.get(symbol)
        if held is None or held.qty <= 0:
            return None
        order = self._broker.place_market_order(symbol, held.qty, "sell")
        self._book_fill(order, "sell")
        return order

    def get_order(self, order_id: str) -> OrderInfo:
        return self._broker.get_order(order_id)

    def get_open_orders(self, symbol: str | None = None) -> list[OrderInfo]:
        return self._broker.get_open_orders(symbol)

    # --- ledger internals ---

    def _book_fill(self, order: OrderInfo, side: str) -> None:
        symbol = order.symbol.upper()
        qty = order.filled_qty or order.qty
        price = order.filled_avg_price or self._mark(symbol, 0.0)
        if qty <= 0 or price <= 0:
            logger.warning("%s: unusable fill for %s; ledger unchanged.", self.name, symbol)
            return
        if side.lower() == "buy":
            self._cash -= qty * price
            held = self._book.get(symbol)
            if held is None:
                self._book[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=qty,
                    side="long",
                    avg_entry_price=price,
                    current_price=price,
                    market_value=qty * price,
                    unrealized_pl=0.0,
                    unrealized_plpc=0.0,
                )
            else:
                total = held.qty + qty
                avg = (held.avg_entry_price * held.qty + price * qty) / total
                self._book[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=total,
                    side="long",
                    avg_entry_price=avg,
                    current_price=price,
                    market_value=total * price,
                    unrealized_pl=(price - avg) * total,
                    unrealized_plpc=0.0,
                )
        else:  # sell
            held = self._book.get(symbol)
            if held is None:
                return
            sold = min(qty, held.qty)
            self._cash += sold * price
            remaining = held.qty - sold
            if remaining <= 1e-9:
                self._book.pop(symbol, None)
            else:
                self._book[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=remaining,
                    side="long",
                    avg_entry_price=held.avg_entry_price,
                    current_price=price,
                    market_value=remaining * price,
                    unrealized_pl=(price - held.avg_entry_price) * remaining,
                    unrealized_plpc=0.0,
                )

    def _mark(self, symbol: str, fallback: float) -> float:
        """Latest price for a held symbol, from the real account's position."""
        for p in self._broker.get_positions():
            if p.symbol.upper() == symbol.upper() and p.current_price > 0:
                return p.current_price
        return fallback


@dataclass
class ExperimentArm:
    name: str
    loop: TradingLoop
    journal: Journal
    # (api_key, secret_key) for this arm's account; None in virtual mode.
    # Lets run_scheduled route a fill-stream listener to the right account.
    credentials: tuple[str, str] | None = None


@dataclass(frozen=True)
class ArmReport:
    """One arm's headline comparison numbers, from its own journal."""

    name: str
    ticks: int
    start_equity: float
    last_equity: float
    return_pct: float
    max_drawdown_pct: float
    sharpe: float | None  # None when there are too few ticks to be meaningful
    realized_pnl: float
    num_trades: int
    win_rate_pct: float


class ExperimentHarness:
    """Drives N isolated TradingLoops in lockstep and reports per-arm results."""

    def __init__(self, arms: list[ExperimentArm], mode: str, settings: Settings) -> None:
        self.arms = arms
        self.mode = mode
        self.settings = settings

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        strategies: tuple[str, ...] = ("sma", "rsi", "llm"),
        mode: str = "auto",
        symbols: list[str] | None = None,
    ) -> ExperimentHarness:
        settings = settings or get_settings()
        if mode == "auto":
            mode = "accounts" if settings.account_credentials(2) else "virtual"
        if mode not in ("accounts", "virtual"):
            raise ValueError(f"mode must be 'auto', 'accounts', or 'virtual', got {mode!r}")

        from src.agent.market_hours import MarketCalendar
        from src.data.universe import get_universe

        provider = build_provider(settings)  # ONE provider: shared cache across arms
        symbols = symbols or get_universe()
        # One shared calendar gates every arm's scheduled ticks (same rule as
        # the single live loop); None disables the gate.
        calendar = MarketCalendar() if settings.market_hours_only else None
        arms: list[ExperimentArm] = []

        shared_broker: Broker | None = None
        if mode == "virtual":
            shared_broker = Broker(settings, announce=True)

        for i, key in enumerate(strategies, start=1):
            journal = Journal(_arm_db_url(settings.db_url, key))
            creds: tuple[str, str] | None = None
            if mode == "accounts":
                creds = settings.account_credentials(i)
                if creds is None:
                    raise ValueError(
                        f"accounts mode needs credentials for paper account {i} "
                        f"(ALPACA_API_KEY_{i}/ALPACA_SECRET_KEY_{i}); "
                        "or run with --mode virtual."
                    )
                broker = Broker(settings, announce=(i == 1), api_key=creds[0], secret_key=creds[1])
            else:
                assert shared_broker is not None
                broker = VirtualPortfolio(  # type: ignore[assignment]
                    shared_broker, 1.0 / len(strategies), key
                )

            account = broker.get_account()
            risk = RiskManager.from_settings(
                settings, starting_equity=account.equity, halt_store=journal
            )
            peak = journal.peak_equity()
            if peak is not None:
                risk.seed_peak(peak)

            arms.append(
                ExperimentArm(
                    name=key,
                    loop=TradingLoop(
                        broker=broker,
                        provider=provider,
                        strategy=_build_strategy(key, settings, journal),
                        risk=risk,
                        journal=journal,
                        settings=settings,
                        symbols=symbols,
                        market_calendar=calendar,
                    ),
                    journal=journal,
                    credentials=creds,
                )
            )

        logger.info("Experiment harness ready: %s mode, arms=%s", mode, [a.name for a in arms])
        return cls(arms, mode, settings)

    def run_once(self) -> dict[str, LoopResult]:
        """One tick per arm, sequentially (shared provider cache keeps it cheap)."""
        return {arm.name: arm.loop.run_once() for arm in self.arms}

    def start_trade_streams(
        self, *, listener_factory: Callable[..., Any] | None = None
    ) -> list[Any]:
        """One fill-stream listener per arm when USE_TRADE_STREAM is enabled.

        Each arm streams its own account's trade updates into its own journal,
        catching fills that land between ticks (a stop leg firing mid-interval,
        a queued order filling at the open). Accounts mode only: in virtual
        mode all arms share one account, and the per-tick reconciliation sweep
        covers it without three duplicate websockets.
        """
        if not self.settings.use_trade_stream:
            return []
        if self.mode != "accounts":
            logger.info("Trade streams require accounts mode; relying on the per-tick sweep.")
            return []
        from src.execution.trade_stream import TradeStreamListener

        factory = listener_factory or TradeStreamListener
        listeners: list[Any] = []
        for arm in self.arms:
            if arm.credentials is None:
                continue
            listener = factory(
                self.settings,
                arm.journal,
                api_key=arm.credentials[0],
                secret_key=arm.credentials[1],
                name=f"trade-stream-{arm.name}",
            )
            listener.start()
            listeners.append(listener)
        logger.info("Started %d trade-stream listener(s).", len(listeners))
        return listeners

    def run_scheduled(self, interval_minutes: int | None = None) -> None:
        """All arms per tick on one blocking scheduler (Ctrl-C to stop)."""
        from datetime import UTC, datetime

        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = interval_minutes or self.settings.loop_interval_minutes
        self.start_trade_streams()

        def _tick() -> None:
            for arm in self.arms:
                arm.loop.scheduled_tick()

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(_tick, "interval", minutes=interval, next_run_time=datetime.now(UTC))
        logger.info(
            "Experiment running: %s every %d min on %s. Ctrl-C to stop.",
            [a.name for a in self.arms],
            interval,
            self.arms[0].loop.symbols if self.arms else [],
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Experiment stopped by user.")

    def report(self) -> list[ArmReport]:
        """Per-arm comparison from each arm's own journal."""
        reports = []
        for arm in self.arms:
            reports.append(_arm_report(arm.name, arm.journal))
        return reports


def _arm_db_url(base_db_url: str, arm_name: str) -> str:
    """data/paperpilot.db -> data/experiments/{arm}.db (same directory root)."""
    if base_db_url.startswith("sqlite:///"):
        base = Path(base_db_url.removeprefix("sqlite:///"))
        target = base.parent / "experiments" / f"{arm_name}.db"
        return f"sqlite:///{target}"
    return base_db_url  # non-sqlite: shared db; strategy tags keep rows apart


def archive_arm_journals(settings: Settings, strategies: Sequence[str]) -> list[Path]:
    """Move existing arm journals aside so a new experiment starts clean.

    Arm reports baseline against the journal's first equity snapshot, so mixing
    runs in one file corrupts the numbers (a virtual-mode allocation followed by
    an accounts-mode tick reads as a huge return). Files are renamed with a
    timestamped ``.bak`` suffix, never deleted.
    """
    archived: list[Path] = []
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    for key in strategies:
        url = _arm_db_url(settings.db_url, key)
        if not url.startswith("sqlite:///"):
            continue
        path = Path(url.removeprefix("sqlite:///"))
        if path.exists():
            target = path.with_name(f"{path.name}.{stamp}.bak")
            path.rename(target)
            archived.append(target)
            logger.info("Archived arm journal %s -> %s", path, target)
    return archived


def _build_strategy(key: str, settings: Settings, journal: Journal) -> Strategy:
    if key == "sma":
        return SmaCrossover()
    if key == "rsi":
        return RsiMeanReversion()
    if key == "llm":
        return LlmStrategy(settings=settings, response_store=journal)
    raise ValueError(f"unknown strategy key {key!r}; expected sma, rsi, or llm.")


def arm_report_from_journal(base_db_url: str, arm_name: str) -> ArmReport | None:
    """Build one arm's report straight from its journal file, broker-free.

    For read-only consumers (the weekly report cron) that must not open broker
    connections or place anything. None when the arm has no journal yet.
    """
    url = _arm_db_url(base_db_url, arm_name)
    if url.startswith("sqlite:///") and not Path(url.removeprefix("sqlite:///")).exists():
        return None
    return _arm_report(arm_name, Journal(url))


def _arm_report(name: str, journal: Journal) -> ArmReport:
    equity_rows = journal.recent_equity(limit=100_000)
    equities = pd.Series([row.equity for row in equity_rows], dtype=float)

    start = float(equities.iloc[0]) if len(equities) else 0.0
    last = float(equities.iloc[-1]) if len(equities) else 0.0
    return_pct = (last / start - 1.0) * 100.0 if start > 0 else 0.0

    drawdown = 0.0
    if len(equities):
        peak = equities.cummax()
        drawdown = float(((peak - equities) / peak).max()) * 100.0

    sharpe: float | None = None
    if len(equities) >= MIN_TICKS_FOR_SHARPE:
        returns = equities.pct_change().dropna()
        if len(returns) and float(returns.std()) > 0:
            sharpe = float(returns.mean() / returns.std() * (252**0.5))

    stats = journal.strategy_report()
    realized = sum(s.total_pnl for s in stats.values())
    trades = sum(s.num_trades for s in stats.values())
    win_rate = (
        sum(s.win_rate_pct * s.num_trades for s in stats.values()) / trades if trades else 0.0
    )

    return ArmReport(
        name=name,
        ticks=len(equities),
        start_equity=start,
        last_equity=last,
        return_pct=return_pct,
        max_drawdown_pct=drawdown,
        sharpe=sharpe,
        realized_pnl=realized,
        num_trades=trades,
        win_rate_pct=win_rate,
    )
