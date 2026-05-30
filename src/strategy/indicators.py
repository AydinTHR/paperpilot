"""Vectorised technical indicators built on pandas (no TA-Lib).

Each function takes a price ``Series`` and returns a new ``Series`` aligned to
the same index, so indicators compose cleanly and never silently drop rows.
All are causal: a value at position ``t`` depends only on data up to ``t``,
which is what keeps the strategies look-ahead-safe.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` periods.

    Leading positions with fewer than ``window`` observations are ``NaN``.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in ``[0, 100]``.

    Uses Wilder's smoothing (an EWM with ``alpha = 1/period`` and
    ``adjust=False``) of average gains and losses -- the standard RSI
    formulation. Returns a Series aligned to ``series``; the warm-up region is
    ``NaN``. A run with no losses yields 100, a run with no gains yields 0.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    # No losses -> RSI 100; no gains -> RSI 0. (Leaves the NaN warm-up intact,
    # since NaN != 0.0 is True and so those positions keep their NaN value.)
    rsi_values = rsi_values.where(avg_loss != 0.0, 100.0)
    rsi_values = rsi_values.where(avg_gain != 0.0, 0.0)
    return rsi_values
