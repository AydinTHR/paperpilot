"""Send the weekly strategy-comparison report to the configured alert channel.

Reads each arm's journal (no broker connection, no orders placed) and pushes a
compact per-arm summary through the same Telegram/Discord alerter the loop
uses. Run from cron on the deployment box, e.g. Sundays:

    0 20 * * 0  cd /opt/paperpilot && docker compose exec -T experiment \
        python scripts/weekly_report.py

Exit code 0 when the report was sent (or printed with --dry-run), 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.experiments.harness import MIN_TICKS_FOR_SHARPE, arm_report_from_journal
from src.monitoring.alerts import NullAlerter, build_alerter

logger = get_logger("paperpilot.weekly_report")

DEFAULT_ARMS = ("sma", "rsi", "llm")


def build_message(db_url: str, arms: tuple[str, ...] = DEFAULT_ARMS) -> str:
    lines = ["PaperPilot weekly report", ""]
    needs_footnote = False
    for name in arms:
        report = arm_report_from_journal(db_url, name)
        if report is None:
            lines.append(f"{name}: no journal yet")
            continue
        sharpe = f"{report.sharpe:.2f}" if report.sharpe is not None else "n/a*"
        if report.sharpe is None:
            needs_footnote = True
        lines.append(
            f"{name}: ${report.last_equity:,.0f} ({report.return_pct:+.2f}%) | "
            f"maxDD {report.max_drawdown_pct:.2f}% | sharpe {sharpe} | "
            f"{report.num_trades} closed trade(s), P&L ${report.realized_pnl:,.2f}"
        )
    if needs_footnote:
        lines += ["", f"* sharpe needs >= {MIN_TICKS_FOR_SHARPE} ticks"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weekly_report",
        description="Send the per-arm comparison to Telegram/Discord.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the report instead of sending it."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)

    message = build_message(settings.db_url)
    if args.dry_run:
        print(message)
        return 0

    alerter = build_alerter(settings)
    if isinstance(alerter, NullAlerter):
        print("No alert channel configured (set TELEGRAM_* or DISCORD_WEBHOOK_URL).")
        print(message)
        return 1
    sent = alerter.send(message)
    print("Report sent." if sent else "Report could not be delivered (see logs).")
    return 0 if sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
