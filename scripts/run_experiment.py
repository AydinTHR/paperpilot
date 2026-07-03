"""Run the LLM-vs-classical strategy comparison.

Each arm (SMA, RSI, LLM) trades in isolation -- separate Alpaca paper accounts
when ALPACA_API_KEY_2/3 are configured, virtual sub-portfolios on one account
otherwise -- and journals to its own database under data/experiments/.

Examples:
    python scripts/run_experiment.py --once            # one tick of every arm
    python scripts/run_experiment.py --interval 60     # run until Ctrl-C
    python scripts/run_experiment.py --report          # compare the arms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.experiments.harness import MIN_TICKS_FOR_SHARPE, ArmReport, ExperimentHarness

logger = get_logger("paperpilot.experiment")


def _print_report(reports: list[ArmReport], mode: str) -> None:
    print(f"\n--- Strategy comparison ({mode} isolation) ---\n")
    print(
        f"  {'ARM':<8}{'TICKS':>6}{'EQUITY':>13}{'RETURN%':>9}"
        f"{'MAX DD%':>9}{'SHARPE':>8}{'REAL P&L':>11}{'TRADES':>8}{'WIN%':>7}"
    )
    for r in reports:
        sharpe = f"{r.sharpe:.2f}" if r.sharpe is not None else "n/a*"
        print(
            f"  {r.name:<8}{r.ticks:>6}{r.last_equity:>13,.2f}{r.return_pct:>9.2f}"
            f"{r.max_drawdown_pct:>9.2f}{sharpe:>8}{r.realized_pnl:>11.2f}"
            f"{r.num_trades:>8}{r.win_rate_pct:>7.1f}"
        )
    if any(r.sharpe is None for r in reports):
        print(f"\n  * Sharpe needs >= {MIN_TICKS_FOR_SHARPE} equity ticks to be meaningful.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="Fair concurrent comparison of SMA vs RSI vs LLM strategies.",
    )
    parser.add_argument("--once", action="store_true", help="One tick per arm, then exit.")
    parser.add_argument(
        "--interval", type=int, default=None, metavar="MIN", help="Run every MIN minutes."
    )
    parser.add_argument(
        "--report", action="store_true", help="Print the per-arm comparison and exit."
    )
    parser.add_argument(
        "--strategies",
        default="sma,rsi,llm",
        help="Comma-separated arm keys (default: sma,rsi,llm).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "accounts", "virtual"],
        default="auto",
        help="Arm isolation: separate paper accounts, virtual sub-portfolios, "
        "or auto (accounts when extra credentials exist).",
    )
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated tickers (overrides universe)."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive existing arm journals first, so the comparison starts "
        "from a clean baseline (mixing runs in one journal skews RETURN%%).",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n[config error] {exc}\n")
        return 2
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)

    strategies = tuple(s.strip().lower() for s in args.strategies.split(",") if s.strip())
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )

    if args.fresh:
        from src.experiments.harness import archive_arm_journals

        for path in archive_arm_journals(settings, strategies):
            print(f"  archived {path}")

    try:
        harness = ExperimentHarness.from_settings(
            settings, strategies=strategies, mode=args.mode, symbols=symbols
        )
    except ValueError as exc:
        print(f"\n[experiment error] {exc}\n")
        return 1

    if args.report:
        _print_report(harness.report(), harness.mode)
        return 0

    if args.once:
        results = harness.run_once()
        for name, result in results.items():
            actions = {o.symbol: o.action for o in result.outcomes}
            print(f"  {name:<8} equity={result.equity:>12,.2f}  {actions}")
        _print_report(harness.report(), harness.mode)
        return 0

    if args.interval is None:
        parser.print_help()
        print("\nPass --once, --interval MIN, or --report.")
        return 0

    harness.run_scheduled(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
