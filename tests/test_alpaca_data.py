"""Tests for the Alpaca-backed market data provider.

Hermetic: a duck-typed fake stands in for ``StockHistoricalDataClient`` (only
``.get_stock_bars(request).df`` is consumed), so no network or credentials are
needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.data.alpaca_data import AlpacaDataProvider
from src.data.market_data import OHLCV_COLUMNS

# --- helpers -----------------------------------------------------------------


def _alpaca_frame(symbols: list[str], n: int = 120) -> pd.DataFrame:
    """A MultiIndex (symbol, timestamp) frame shaped like ``BarSet.df``."""
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    parts = []
    for symbol in symbols:
        close = 100 + np.arange(n) * 0.1
        parts.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.full(n, 1_000_000, dtype="int64"),
                    "trade_count": np.full(n, 5_000, dtype="int64"),
                    "vwap": close,
                }
            )
        )
    return pd.concat(parts).set_index(["symbol", "timestamp"])


class _FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class _FakeDataClient:
    """Returns a canned BarSet-like object; records requests; can fail first."""

    def __init__(self, df: pd.DataFrame, *, fail_times: int = 0) -> None:
        self._df = df
        self.fail_times = fail_times
        self.calls = 0
        self.requests: list[object] = []

    def get_stock_bars(self, request: object) -> _FakeBarSet:
        self.calls += 1
        self.requests.append(request)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated 429 rate limit")
        return _FakeBarSet(self._df.copy())


class _FakeSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_cache_dir=str(tmp_path))  # type: ignore[arg-type]


def _provider(tmp_path, client: _FakeDataClient, **kwargs) -> AlpacaDataProvider:
    return AlpacaDataProvider(_settings(tmp_path), client=client, **kwargs)


UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY"]


# --- batched fetch + cache -----------------------------------------------------


def test_one_batched_fetch_warms_every_cache(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(UNIVERSE))
    provider = _provider(tmp_path, client, batch_symbols=UNIVERSE)

    df = provider.get_latest_bars("aapl", lookback=50, interval="1d")
    assert client.calls == 1
    assert len(df) == 50
    assert list(df.columns) == list(OHLCV_COLUMNS)
    for symbol in UNIVERSE:
        assert (tmp_path / f"{symbol}_1d_alpaca.parquet").exists()

    # Every other symbol in the batch is now a cache hit -> still one call.
    provider.get_latest_bars("MSFT", lookback=50, interval="1d")
    provider.get_latest_bars("SPY", lookback=50, interval="1d")
    assert client.calls == 1


def test_daily_index_is_tz_naive_and_normalized(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(["AAPL"]))
    provider = _provider(tmp_path, client, batch_symbols=["AAPL"])
    df = provider.get_latest_bars("AAPL", lookback=10, interval="1d")
    assert df.index.tz is None  # matches yfinance daily frames
    assert df.index.is_monotonic_increasing


def test_get_bars_always_fetches(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(["AAPL"], n=30))
    provider = _provider(tmp_path, client, batch_symbols=["AAPL"])
    provider.get_bars("AAPL", "2024-01-01", "2024-02-01", interval="1d")
    provider.get_bars("AAPL", "2024-01-01", "2024-02-01", interval="1d")
    assert client.calls == 2  # explicit-range fetches bypass the cache


# --- retry/backoff -------------------------------------------------------------


def test_retries_with_backoff_then_succeeds(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(["AAPL"]), fail_times=2)
    sleeper = _FakeSleeper()
    provider = _provider(tmp_path, client, batch_symbols=["AAPL"], sleeper=sleeper)
    df = provider.get_latest_bars("AAPL", lookback=20, interval="1d")
    assert len(df) == 20
    assert client.calls == 3
    assert sleeper.delays == [0.5, 1.0]  # exponential backoff before each retry


def test_raises_after_exhausted_retries(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(["AAPL"]), fail_times=10)
    sleeper = _FakeSleeper()
    provider = _provider(tmp_path, client, batch_symbols=["AAPL"], sleeper=sleeper)
    with pytest.raises(RuntimeError, match="429"):
        provider.get_latest_bars("AAPL", lookback=20, interval="1d")
    assert client.calls == 4  # 1 try + 3 retries
    assert sleeper.delays == [0.5, 1.0, 2.0]


# --- validation ----------------------------------------------------------------


def test_unsupported_interval_rejected(tmp_path) -> None:
    provider = _provider(tmp_path, _FakeDataClient(_alpaca_frame(["AAPL"])))
    with pytest.raises(ValueError):
        provider.get_latest_bars("AAPL", lookback=10, interval="5m")


def test_missing_symbol_raises(tmp_path) -> None:
    client = _FakeDataClient(_alpaca_frame(["MSFT"]))  # AAPL absent from response
    provider = _provider(tmp_path, client, batch_symbols=["MSFT"])
    with pytest.raises(ValueError, match="no bars"):
        provider.get_latest_bars("AAPL", lookback=10, interval="1d")


def test_lookback_must_be_positive(tmp_path) -> None:
    provider = _provider(tmp_path, _FakeDataClient(_alpaca_frame(["AAPL"])))
    with pytest.raises(ValueError, match="lookback"):
        provider.get_latest_bars("AAPL", lookback=0)
