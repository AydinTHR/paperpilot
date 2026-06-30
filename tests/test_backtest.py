"""Tests for the event-driven backtest engine -- fully offline, no network.

Synthetic OHLCV frames exercise the adapter (BUY opens, SELL closes, HOLD is a
no-op), the cost wiring, and the structured result/equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.risk.manager import RiskLimits, RiskManager
from src.strategy.base import Signal, Strategy
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover

# --- helpers ---------------------------------------------------------------


def _sine_ohlcv(n: int = 300, start: str = "2023-01-02") -> pd.DataFrame:
    """A drifting sine -> guaranteed crossovers and RSI extremes, valid OHLC."""
    t = np.arange(n)
    close = 100 + 15 * np.sin(t / 12.0) + t * 0.03
    idx = pd.date_range(start, periods=n, freq="B")
    c = pd.Series(close, index=idx)
    o = c.shift(1).bfill()
    hi = pd.concat([o, c], axis=1).max(axis=1) * 1.005
    lo = pd.concat([o, c], axis=1).min(axis=1) * 0.995
    return pd.DataFrame(
        {"Open": o, "High": hi, "Low": lo, "Close": c, "Volume": 1_000_000}, index=idx
    )


class _AlwaysHold(Strategy):
    name = "always-hold"

    @property
    def min_bars(self) -> int:
        return 1

    def generate_signals(self, data: pd.DataFrame) -> Signal:
        return self.hold()


# --- config validation -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cash": 0},
        {"commission": 1.0},
        {"slippage": -0.1},
        {"position_size": 0},
        {"position_size": 1.0},
    ],
)
def test_config_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


# --- engine behaviour ------------------------------------------------------


def test_sma_backtest_produces_trades_and_curve() -> None:
    df = _sine_ohlcv()
    result = run_backtest(df, SmaCrossover(), symbol="TEST", interval="1d")

    assert isinstance(result, BacktestResult)
    assert result.strategy == "SMA(20/50) crossover"
    assert result.num_trades > 0
    assert result.final_equity > 0
    assert 0.0 <= result.exposure_pct <= 100.0
    assert result.max_drawdown_pct <= 0.0
    assert np.isfinite(result.return_pct)
    # Equity curve is one point per bar, aligned to the input index.
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.index.equals(df.index)


def test_rsi_backtest_runs() -> None:
    df = _sine_ohlcv()
    result = run_backtest(df, RsiMeanReversion(), symbol="TEST", interval="1d")
    assert result.num_trades >= 0
    assert np.isfinite(result.final_equity)


def test_hold_only_strategy_makes_no_trades() -> None:
    df = _sine_ohlcv()
    config = BacktestConfig(cash=10_000.0)
    result = run_backtest(df, _AlwaysHold(), config)
    assert result.num_trades == 0
    # No trades -> no costs -> equity stays exactly at starting cash.
    assert result.final_equity == pytest.approx(config.cash)
    assert result.return_pct == pytest.approx(0.0)


def test_commission_drags_return() -> None:
    df = _sine_ohlcv()
    free = run_backtest(df, SmaCrossover(), BacktestConfig(commission=0.0, slippage=0.0))
    costly = run_backtest(df, SmaCrossover(), BacktestConfig(commission=0.02, slippage=0.0))
    # Same signals, but trading costs can only reduce (or not improve) the return.
    assert costly.return_pct <= free.return_pct + 1e-9


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), SmaCrossover())


def test_missing_columns_raise() -> None:
    df = _sine_ohlcv().drop(columns=["Close"])
    with pytest.raises(ValueError):
        run_backtest(df, SmaCrossover())


def test_summary_has_headline_keys() -> None:
    result = run_backtest(_sine_ohlcv(), SmaCrossover(), symbol="TEST", interval="1d")
    summary = result.summary()
    for key in ("symbol", "strategy", "return_pct", "num_trades", "final_equity", "sharpe"):
        assert key in summary


# --- risk integration ------------------------------------------------------


def test_backtest_with_risk_reduces_drawdown() -> None:
    df = _sine_ohlcv()
    limits = RiskLimits(
        max_position_pct=0.10,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.20,
        stop_loss_pct=0.05,
    )
    config = BacktestConfig(cash=10_000.0)

    base = run_backtest(df, SmaCrossover(), config, symbol="TEST", interval="1d")
    risk = RiskManager(limits, starting_equity=config.cash)
    risked = run_backtest(df, SmaCrossover(), config, symbol="TEST", interval="1d", risk=risk)

    # Drawdowns are <= 0; capping exposure at 10% (vs the default 95%) plus the
    # stop-loss can only pull the worst drawdown closer to zero, never deeper.
    assert risked.max_drawdown_pct <= 0.0
    assert risked.max_drawdown_pct >= base.max_drawdown_pct
    assert np.isfinite(risked.final_equity)
