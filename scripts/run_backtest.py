#!/usr/bin/env python3
"""Backtest a strategy on historical bars and print metrics + an equity curve.

Phase 3 deliverable. Fetches OHLCV via the market-data layer, runs the chosen
strategy through the event-driven engine (with commission + slippage), prints
the headline metrics and an ASCII equity curve, and saves the full curve to CSV.

    python scripts/run_backtest.py --symbol AAPL --strategy sma
    python scripts/run_backtest.py --symbol NVDA --strategy rsi --start 2022-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.data.market_data import build_provider
from src.risk.manager import RiskManager
from src.strategy.base import Strategy
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover

logger = get_logger("paperpilot.backtest")

_STRATEGIES: dict[str, type[Strategy]] = {
    "sma": SmaCrossover,
    "rsi": RsiMeanReversion,
}


def _sparkline(series: object, width: int = 60) -> str:
    """Render a numeric series as a compact unicode block sparkline."""
    blocks = "▁▂▃▄▅▆▇█"
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return ""
    if values.size > width:
        idx = np.linspace(0, values.size - 1, width).astype(int)
        values = values[idx]
    lo, hi = values.min(), values.max()
    if hi == lo:
        return blocks[0] * len(values)
    scaled = ((values - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in scaled)


def _print_report(
    result: BacktestResult, *, saved_to: Path | None, risk: RiskManager | None
) -> None:
    edge = result.return_pct - result.buy_hold_return_pct
    print(f"\n--- Backtest: {result.symbol} / {result.strategy} ({result.interval}) ---")
    if risk is None:
        print("  Risk manager    : OFF")
    else:
        lim = risk.limits
        print(
            f"  Risk manager    : ON  (max_pos {lim.max_position_pct:.0%}, "
            f"stop {lim.stop_loss_pct:.0%}, daily {lim.max_daily_loss_pct:.0%}, "
            f"maxDD {lim.max_drawdown_pct:.0%})"
        )
        if risk.halted:
            print(f"  Risk halt       : {risk.halt_reason}")
    print(f"  Period          : {result.start.date()} .. {result.end.date()}")
    print(f"  Trades          : {result.num_trades}")
    print(f"  Exposure        : {result.exposure_pct:6.2f} %")
    print(f"  Win rate        : {result.win_rate_pct:6.2f} %")
    print(f"  Return          : {result.return_pct:+7.2f} %")
    print(f"  Buy & hold      : {result.buy_hold_return_pct:+7.2f} %")
    print(f"  Edge vs B&H     : {edge:+7.2f} %")
    print(f"  Sharpe          : {result.sharpe:6.2f}")
    print(f"  Max drawdown    : {result.max_drawdown_pct:+7.2f} %")
    print(f"  Final equity    : {result.final_equity:,.2f}")
    print(f"\n  Equity curve    : {_sparkline(result.equity_curve.to_numpy())}")
    if saved_to is not None:
        print(f"  Curve saved to  : {saved_to}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description="Backtest a strategy on historical bars.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker, e.g. AAPL.")
    parser.add_argument(
        "--strategy", choices=sorted(_STRATEGIES), default="sma", help="Strategy key."
    )
    parser.add_argument("--interval", choices=["1d", "1h"], default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (explicit range).")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (explicit range).")
    parser.add_argument("--lookback", type=int, default=500, help="Bars if no range.")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--size", type=float, default=0.95, help="Equity fraction/entry.")
    parser.add_argument(
        "--risk",
        action="store_true",
        help="Apply the risk manager (position sizing, stop-loss, halts).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write the CSV.")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    interval = args.interval or settings.default_interval
    symbol = args.symbol.upper()

    config = BacktestConfig(
        cash=args.cash,
        commission=args.commission,
        slippage=args.slippage,
        position_size=args.size,
    )

    provider = build_provider(settings)
    try:
        if args.start and args.end:
            bars = provider.get_bars(symbol, args.start, args.end, interval)
        else:
            bars = provider.get_latest_bars(symbol, lookback=args.lookback, interval=interval)
    except Exception as exc:
        print(f"\n[backtest FAILED] could not fetch {symbol}: {exc}\n")
        logger.error("Fetch failed for %s: %s", symbol, exc)
        return 1

    strategy = _STRATEGIES[args.strategy]()
    if len(bars) < strategy.min_bars:
        print(
            f"\n[backtest] not enough bars for {strategy.name}: "
            f"have {len(bars)}, need >= {strategy.min_bars}.\n"
        )
        return 1

    risk = RiskManager.from_settings(settings, config.cash) if args.risk else None

    try:
        result = run_backtest(bars, strategy, config, symbol=symbol, interval=interval, risk=risk)
    except Exception as exc:
        print(f"\n[backtest FAILED] {exc}\n")
        logger.error("Backtest failed: %s", exc)
        return 1

    saved_to: Path | None = None
    if not args.no_save:
        out_dir = Path(config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_to = out_dir / f"{symbol}_{args.strategy}_{interval}_equity.csv"
        result.equity_curve.to_csv(saved_to, header=["equity"])

    _print_report(result, saved_to=saved_to, risk=risk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
