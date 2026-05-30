#!/usr/bin/env python3
"""PaperPilot agent entry point.

Phase 1 implements ``--check``: connect to the (paper) Alpaca account and print
the balance and open positions. The scheduled trading loop arrives in Phase 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging  # noqa: E402
from config.settings import Settings, get_settings  # noqa: E402
from src.execution.broker import Broker, BrokerError  # noqa: E402

logger = get_logger("paperpilot.cli")


def _fmt_money(value: float, currency: str = "USD") -> str:
    return f"{value:,.2f} {currency}"


def cmd_check(settings: Settings) -> int:
    """Connect to the account and print balance + positions."""
    try:
        broker = Broker(settings)
    except BrokerError as exc:
        print(f"\n[connection check FAILED] {exc}\n")
        logger.error("Connection check failed: %s", exc)
        return 1

    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except BrokerError as exc:
        print(f"\n[connection check FAILED] {exc}\n")
        logger.error("Connection check failed: %s", exc)
        return 1

    mode = "LIVE" if settings.is_live else "PAPER"
    print("\n--- Account ---")
    print(f"  Mode            : {mode}")
    print(f"  Account #       : {account.account_number}")
    print(f"  Status          : {account.status}")
    print(f"  Cash            : {_fmt_money(account.cash, account.currency)}")
    print(f"  Equity          : {_fmt_money(account.equity, account.currency)}")
    print(f"  Buying power    : {_fmt_money(account.buying_power, account.currency)}")
    print(
        f"  Portfolio value : {_fmt_money(account.portfolio_value, account.currency)}"
    )

    print(f"\n--- Positions ({len(positions)}) ---")
    if not positions:
        print("  (no open positions)")
    else:
        print(f"  {'SYMBOL':<8}{'QTY':>10}{'AVG ENTRY':>14}{'PRICE':>12}{'UNREAL P/L':>14}")
        for p in positions:
            print(
                f"  {p.symbol:<8}{p.qty:>10.4f}{p.avg_entry_price:>14.2f}"
                f"{p.current_price:>12.2f}{p.unrealized_pl:>14.2f}"
            )
    print()
    logger.info("Connection check OK (mode=%s, positions=%d).", mode, len(positions))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_agent",
        description="PaperPilot autonomous paper-trading agent.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Connect to the Alpaca paper account and print balance + positions.",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:  # validation / live-gate errors
        print(f"\n[config error] {exc}\n")
        return 2

    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    logger.debug("Resolved settings: %s", settings.safe_summary())

    if args.check:
        return cmd_check(settings)

    parser.print_help()
    print("\nThe scheduled trading loop is implemented in Phase 5. Use --check for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
