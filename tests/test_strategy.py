"""Tests for the strategy interface, indicators, and example strategies.

Everything here is deterministic and offline: synthetic OHLCV frames drive each
strategy into BUY / SELL / HOLD / warm-up, plus a look-ahead (causality) check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import Action, Signal
from src.strategy.examples.mean_reversion import RsiMeanReversion
from src.strategy.examples.sma_crossover import SmaCrossover
from src.strategy.indicators import rsi, sma


# --- helpers ---------------------------------------------------------------


def _ohlcv(close_values: object, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV frame from a sequence of closes."""
    close = pd.Series(np.asarray(close_values, dtype=float))
    idx = pd.date_range(start, periods=len(close), freq="D")
    close.index = idx
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _find_cross(close: pd.Series, fast: int, slow: int, *, up: bool) -> int | None:
    f, s = sma(close, fast), sma(close, slow)
    for i in range(1, len(close)):
        vals = (f.iloc[i], f.iloc[i - 1], s.iloc[i], s.iloc[i - 1])
        if any(pd.isna(v) for v in vals):
            continue
        if up and f.iloc[i - 1] <= s.iloc[i - 1] and f.iloc[i] > s.iloc[i]:
            return i
        if not up and f.iloc[i - 1] >= s.iloc[i - 1] and f.iloc[i] < s.iloc[i]:
            return i
    return None


# --- Signal validation -----------------------------------------------------


def test_signal_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        Signal(Action.BUY, confidence=1.5)
    with pytest.raises(ValueError):
        Signal(Action.BUY, confidence=-0.01)


def test_signal_rejects_out_of_range_size_hint() -> None:
    with pytest.raises(ValueError):
        Signal(Action.BUY, confidence=0.5, size_hint=2.0)


def test_signal_defaults_are_valid() -> None:
    s = Signal(Action.HOLD)
    assert s.action is Action.HOLD
    assert s.confidence == 0.0
    assert s.size_hint is None


# --- indicators ------------------------------------------------------------


def test_sma_matches_rolling_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 2)
    assert pd.isna(out.iloc[0])
    pd.testing.assert_series_equal(out, s.rolling(2, min_periods=2).mean())


def test_rsi_rising_series_saturates_high() -> None:
    rising = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_falling_series_saturates_low() -> None:
    falling = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0)


def test_indicators_are_causal_no_lookahead() -> None:
    # Truncating the input must not change the last computed value: a value at
    # position t depends only on data up to t.
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    n = 150
    assert sma(series.iloc[:n], 20).iloc[-1] == pytest.approx(sma(series, 20).iloc[n - 1])
    assert rsi(series.iloc[:n], 14).iloc[-1] == pytest.approx(rsi(series, 14).iloc[n - 1])


# --- SMA crossover ---------------------------------------------------------


def _sine_close(n: int = 200) -> pd.DataFrame:
    # A smooth sine guarantees clean, well-separated golden/death crosses.
    t = np.arange(n)
    return _ohlcv(100 + 10 * np.sin(t / 10.0))


def test_sma_insufficient_bars_holds() -> None:
    df = _sine_close()
    assert SmaCrossover().generate_signals(df.iloc[:10]).action is Action.HOLD


def test_sma_golden_cross_buys() -> None:
    df = _sine_close()
    i = _find_cross(df["Close"], 20, 50, up=True)
    assert i is not None
    sig = SmaCrossover().generate_signals(df.iloc[: i + 1])
    assert sig.action is Action.BUY
    assert 0.0 <= sig.confidence <= 1.0


def test_sma_death_cross_sells() -> None:
    df = _sine_close()
    i = _find_cross(df["Close"], 20, 50, up=False)
    assert i is not None
    sig = SmaCrossover().generate_signals(df.iloc[: i + 1])
    assert sig.action is Action.SELL


def test_sma_one_bar_after_cross_holds() -> None:
    # One bar past a golden cross there is no fresh cross -> HOLD.
    df = _sine_close()
    i = _find_cross(df["Close"], 20, 50, up=True)
    assert i is not None
    assert SmaCrossover().generate_signals(df.iloc[: i + 2]).action is Action.HOLD


def test_sma_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        SmaCrossover(fast=50, slow=20)


# --- RSI mean reversion ----------------------------------------------------


def test_rsi_strategy_insufficient_bars_holds() -> None:
    df = _ohlcv(np.arange(1, 6, dtype=float))
    assert RsiMeanReversion().generate_signals(df).action is Action.HOLD


def test_rsi_strategy_oversold_buys() -> None:
    df = _ohlcv(np.arange(60, 1, -1, dtype=float))  # falling -> RSI ~0
    sig = RsiMeanReversion().generate_signals(df)
    assert sig.action is Action.BUY
    assert sig.confidence == pytest.approx(1.0)


def test_rsi_strategy_overbought_sells() -> None:
    df = _ohlcv(np.arange(1, 60, dtype=float))  # rising -> RSI ~100
    assert RsiMeanReversion().generate_signals(df).action is Action.SELL


def test_rsi_strategy_neutral_holds() -> None:
    # Alternating equal up/down moves keep RSI near 50.
    closes = 100 + (np.arange(40) % 2)
    sig = RsiMeanReversion().generate_signals(_ohlcv(closes))
    assert sig.action is Action.HOLD


def test_rsi_strategy_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        RsiMeanReversion(oversold=70, overbought=30)


# --- look-ahead safety at the strategy level -------------------------------


def test_strategy_decision_stable_under_future_bars() -> None:
    df = _sine_close()
    n = 120
    strat = RsiMeanReversion()
    first = strat.generate_signals(df.iloc[:n])
    _ = strat.generate_signals(df)  # computing on the full frame must not leak state
    again = strat.generate_signals(df.iloc[:n])
    assert first == again
