#!/usr/bin/env python3
"""Preview the current signal from each example strategy for one symbol.

Phase 2 deliverable, made tangible: fetch recent bars via the market-data layer
and print what each strategy decides on the latest bar. Read-only -- no orders
are ever placed here.

    python scripts/preview_signals.py --symbol AAPL [--interval 1d] [--lookback 250]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging  # noqa: E402
from config.settings import get_settings  # noqa: E402
from src.data.market_data import YFinanceProvider  # noqa: E402
from src.strategy.base import Strategy  # noqa: E402
from src.strategy.examples.mean_reversion import RsiMeanReversion  # noqa: E402
from src.strategy.examples.sma_crossover import SmaCrossover  # noqa: E402

logger = get_logger("paperpilot.preview")


def _build_strategies() -> list[Strategy]:
    return [SmaCrossover(), RsiMeanReversion()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="preview_signals",
        description="Preview each strategy's current signal for a symbol.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker, e.g. AAPL.")
    parser.add_argument(
        "--interval",
        default=None,
        choices=list(("1d", "1h")),
        help="Bar interval (defaults to settings.default_interval).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=250,
        help="Number of recent bars to fetch.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    interval = args.interval or settings.default_interval
    symbol = args.symbol.upper()

    provider = YFinanceProvider(settings)
    try:
        bars = provider.get_latest_bars(symbol, lookback=args.lookback, interval=interval)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure cleanly
        print(f"\n[preview FAILED] could not fetch {symbol}: {exc}\n")
        logger.error("Fetch failed for %s: %s", symbol, exc)
        return 1

    if bars.empty:
        print(f"\n[preview] no data returned for {symbol}.\n")
        return 1

    last = bars.iloc[-1]
    print(f"\n--- {symbol} ({interval}) ---")
    print(
        f"  Bars            : {len(bars)}  "
        f"({bars.index[0].date()} .. {bars.index[-1].date()})"
    )
    print(f"  Last close      : {last['Close']:.2f}")

    print("\n--- Signals ---")
    for strat in _build_strategies():
        sig = strat.generate_signals(bars)
        print(
            f"  {strat.name:<26} {sig.action.value:<5} "
            f"conf={sig.confidence:0.2f}  {sig.reason}"
        )
    print()
    logger.info("Previewed signals for %s (%s, %d bars).", symbol, interval, len(bars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
