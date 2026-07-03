#!/usr/bin/env python3
"""Run the autonomous paper-trading loop (Phase 5 deliverable).

    # one iteration against the paper account, then exit (safe to demo):
    python scripts/run_live.py --once

    # run continuously every 60 minutes (Ctrl-C to stop):
    python scripts/run_live.py --interval 60 --strategy sma

    # inspect what the agent has done (read-only, no broker connection):
    python scripts/run_live.py --report

Trading is ALWAYS paper unless PAPER=false AND ALLOW_LIVE_TRADING=true (the
loud, deliberate gate enforced in config + broker). Every signal, order, and
equity snapshot is written to the gitignored sqlite trade journal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.agent.loop import LoopResult, TradingLoop
from src.execution.broker import BrokerError
from src.journal.store import Journal
from src.strategy.base import Strategy
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover
from src.strategy.llm.strategy import LlmStrategy

logger = get_logger("paperpilot.live")

_STRATEGIES: dict[str, type[Strategy]] = {
    "sma": SmaCrossover,
    "rsi": RsiMeanReversion,
    "llm": LlmStrategy,
}


def _print_result(result: LoopResult) -> None:
    print("\n--- Loop iteration ---")
    print(f"  Time            : {result.ts.isoformat(timespec='seconds')}")
    print(f"  Equity          : {result.equity:,.2f}")
    print(f"  Cash            : {result.cash:,.2f}")
    if result.halted:
        print(f"  RISK HALT       : {result.halt_reason}  (no new entries)")
    print(f"\n  {'SYMBOL':<8}{'ACTION':<9}{'QTY':>8}  DETAIL")
    for o in result.outcomes:
        qty = f"{o.qty:.0f}" if o.qty is not None else "-"
        print(f"  {o.symbol:<8}{o.action:<9}{qty:>8}  {o.detail}")
    print()


def _print_report(journal: Journal, limit: int) -> None:
    counts = journal.counts()
    print("\n--- Trade journal report ---")
    print(f"  DB              : {journal.db_url}")
    print(
        f"  Rows            : {counts['signals']} signals, "
        f"{counts['orders']} orders, {counts['equity_snapshots']} equity snapshots"
    )

    orders = journal.recent_orders(limit)
    print(f"\n  Recent orders ({len(orders)}):")
    if not orders:
        print("    (none)")
    for o in orders:
        ts = o.ts.isoformat(timespec="seconds") if o.ts else "?"
        print(f"    {ts}  {o.symbol:<6} {o.side:<4} {o.qty:>8.0f}  [{o.reason}]")

    signals = journal.recent_signals(limit)
    print(f"\n  Recent signals ({len(signals)}):")
    if not signals:
        print("    (none)")
    for s in signals:
        ts = s.ts.isoformat(timespec="seconds") if s.ts else "?"
        print(f"    {ts}  {s.symbol:<6} {s.action:<4} conf={s.confidence:.2f}  {s.reason}")

    equity = journal.recent_equity(limit)
    print(f"\n  Recent equity ({len(equity)}):")
    if not equity:
        print("    (none)")
    for e in equity:
        ts = e.ts.isoformat(timespec="seconds") if e.ts else "?"
        flag = f"  HALTED:{e.halt_reason}" if e.halted else ""
        print(f"    {ts}  equity={e.equity:,.2f}  cash={e.cash:,.2f}{flag}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_live",
        description="Run PaperPilot's autonomous paper-trading loop.",
    )
    parser.add_argument("--once", action="store_true", help="Run one iteration, then exit.")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        metavar="MIN",
        help="Run continuously every MIN minutes (BlockingScheduler).",
    )
    parser.add_argument(
        "--strategy", choices=sorted(_STRATEGIES), default=None, help="Strategy key."
    )
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated tickers (overrides universe)."
    )
    parser.add_argument("--lookback", type=int, default=None, help="Bars to fetch per symbol.")
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Run scheduled ticks even while the market is closed.",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        type=int,
        const=20,
        default=None,
        metavar="N",
        help="Print the last N journal rows (read-only) and exit. Default N=20.",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:  # validation / live-gate errors
        print(f"\n[config error] {exc}\n")
        return 2

    setup_logging(level=settings.log_level, log_dir=settings.log_dir)

    # Read-only journal report needs no broker / network.
    if args.report is not None:
        _print_report(Journal(settings.db_url), args.report)
        return 0

    if not args.once and args.interval is None:
        parser.print_help()
        print("\nPass --once for a single iteration, or --interval MIN to run on a schedule.")
        return 0

    strategy_key = args.strategy or settings.default_strategy
    strategy = _STRATEGIES[strategy_key]()
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )

    try:
        loop = TradingLoop.from_settings(
            settings,
            strategy=strategy,
            symbols=symbols,
            lookback=args.lookback,
            market_hours_gate=False if args.ignore_market_hours else None,
        )
    except BrokerError as exc:
        print(f"\n[live loop FAILED] {exc}\n")
        logger.error("Loop setup failed: %s", exc)
        return 1

    if args.once:
        try:
            result = loop.run_once()
        except Exception as exc:
            print(f"\n[live loop FAILED] {exc}\n")
            logger.exception("run_once failed")
            return 1
        _print_result(result)
        return 0

    # Optional websocket fill listener (off by default), then scheduled mode
    # (blocks until Ctrl-C).
    if settings.use_trade_stream:
        from src.execution.trade_stream import TradeStreamListener

        TradeStreamListener(settings, loop.journal).start()
    loop.run_scheduled(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
