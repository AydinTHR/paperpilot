"""Walk-forward analysis CLI.

Optimizes a strategy's parameters in-sample, validates them on unseen
out-of-sample windows, and reports the Walk-Forward Efficiency (WFE) with an
overfit warning below 0.5. See ``src/backtest/walkforward.py`` for the method.

Examples:
    python scripts/run_walkforward.py --symbol AAPL --strategy sma
    python scripts/run_walkforward.py --symbol SPY --strategy rsi --folds 6 --mode rolling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.backtest.engine import BacktestConfig
from src.backtest.walkforward import (
    WFE_OVERFIT_THRESHOLD,
    StrategySpec,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)
from src.data.market_data import build_provider
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover

logger = get_logger("paperpilot.walkforward")

# Default parameter spaces per optimizable strategy. Deliberately small grids:
# walk-forward multiplies every combination by every fold.
_SPECS: dict[str, StrategySpec] = {
    "sma": StrategySpec(
        name="sma",
        factory=lambda fast, slow: SmaCrossover(fast=fast, slow=slow),
        param_grid={"fast": [5, 10, 20], "slow": [30, 50, 100]},
        constraint=lambda p: p.fast < p.slow,
        defaults={"fast": 20, "slow": 50},
    ),
    "rsi": StrategySpec(
        name="rsi",
        factory=lambda period, oversold, overbought: RsiMeanReversion(
            period=period, oversold=oversold, overbought=overbought
        ),
        param_grid={"period": [7, 14, 21], "oversold": [20.0, 30.0], "overbought": [70.0, 80.0]},
        constraint=lambda p: p.oversold < p.overbought,
        defaults={"period": 14, "oversold": 30.0, "overbought": 70.0},
    ),
}


def _print_result(result: WalkForwardResult) -> None:
    print(f"\n--- Walk-forward analysis: {result.spec_name} ({result.mode}) ---\n")
    header = (
        f"  {'FOLD':<5}{'IS RANGE':<26}{'OOS RANGE':<26}"
        f"{'PARAMS':<28}{'IS ANN%':>9}{'OOS ANN%':>10}"
    )
    print(header)
    for fold in result.folds:
        is_range = f"{fold.is_start.date()}..{fold.is_end.date()}"
        oos_range = f"{fold.oos_start.date()}..{fold.oos_end.date()}"
        params = ",".join(f"{k}={v}" for k, v in fold.best_params.items())
        print(
            f"  {fold.fold:<5}{is_range:<26}{oos_range:<26}"
            f"{params:<28}{fold.is_return_ann_pct:>9.1f}{fold.oos_return_ann_pct:>10.1f}"
        )

    print(f"\n  Walk-Forward Efficiency : {result.wfe:.2f}")
    if result.wfe_flagged:
        print(
            f"  WARNING: WFE < {WFE_OVERFIT_THRESHOLD} -- the optimized performance "
            "does not survive out-of-sample. Treat these parameters as OVERFIT; "
            "do not deploy them."
        )
    else:
        print(f"  (>= {WFE_OVERFIT_THRESHOLD} is conventionally considered healthy)")

    if result.holdout is not None:
        print(
            f"\n  Hold-out (never optimized): return {result.holdout.return_pct:.1f}%, "
            f"sharpe {result.holdout.sharpe:.2f}, "
            f"max drawdown {result.holdout.max_drawdown_pct:.1f}%"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_walkforward",
        description="Walk-forward validation of a strategy's parameters.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker, e.g. AAPL.")
    parser.add_argument("--strategy", choices=sorted(_SPECS), default="sma")
    parser.add_argument("--interval", choices=["1d", "1h"], default=None)
    parser.add_argument("--lookback", type=int, default=1250, help="Bars of history (~5y daily).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--oos", type=float, default=0.20, help="OOS fraction of working range.")
    parser.add_argument("--mode", choices=["anchored", "rolling"], default="anchored")
    parser.add_argument("--holdout", type=float, default=0.10, help="Untouched final fraction.")
    parser.add_argument(
        "--method",
        choices=["grid", "sambo"],
        default="grid",
        help="Optimizer ('sambo' needs: pip install -r requirements-research.txt).",
    )
    parser.add_argument("--max-tries", type=int, default=None)
    parser.add_argument("--cash", type=float, default=10_000.0)
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:
        print(f"\n[config error] {exc}\n")
        return 2
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)

    interval = args.interval or settings.default_interval
    provider = build_provider(settings)
    print(f"Fetching {args.lookback} {interval} bars for {args.symbol}...")
    data = provider.get_latest_bars(args.symbol, lookback=args.lookback, interval=interval)

    try:
        result = run_walk_forward(
            data,
            _SPECS[args.strategy],
            WalkForwardConfig(
                n_folds=args.folds,
                oos_fraction=args.oos,
                mode=args.mode,
                holdout_fraction=args.holdout,
                method=args.method,
                max_tries=args.max_tries,
            ),
            BacktestConfig(cash=args.cash),
            symbol=args.symbol.upper(),
            interval=interval,
        )
    except ValueError as exc:
        print(f"\n[walk-forward error] {exc}\n")
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
