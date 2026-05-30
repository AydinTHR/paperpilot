"""Tests for the market-data layer and the universe helper.

Hermetic by default: an injected fake downloader stands in for yfinance so no
network is touched. A real-fetch smoke test is marked ``network`` and is
deselected by default (run with ``pytest -m network``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.data.market_data import OHLCV_COLUMNS, YFinanceProvider, _normalize
from src.data.universe import DEFAULT_UNIVERSE, get_universe


# --- helpers ---------------------------------------------------------------


def _flat_frame(n: int = 120) -> pd.DataFrame:
    """A clean flat-column frame, as yfinance returns with multi_level_index=False."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000, dtype="int64"),
        },
        index=idx,
    )


def _multiindex_frame(n: int = 5) -> pd.DataFrame:
    """A (PriceField, Ticker) MultiIndex frame, like yfinance's default output."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = pd.DataFrame(
        {
            "Open": np.arange(n, dtype=float),
            "High": np.arange(n, dtype=float),
            "Low": np.arange(n, dtype=float),
            "Close": np.arange(n, dtype=float),
            "Volume": np.arange(n, dtype=float) * 10,
        },
        index=idx,
    )
    base.columns = pd.MultiIndex.from_product([list(base.columns), ["AAPL"]])
    return base


class _FakeDownloader:
    """Records call count and returns a canned frame -- never hits the network."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def __call__(self, **kwargs: object) -> pd.DataFrame:
        self.calls += 1
        return self.frame.copy()


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_cache_dir=str(tmp_path))  # type: ignore[arg-type]


# --- _normalize ------------------------------------------------------------


def test_normalize_flattens_multiindex() -> None:
    out = _normalize(_multiindex_frame())
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out["Close"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_normalize_sorts_and_drops_nan() -> None:
    df = _flat_frame(10).iloc[::-1]  # reversed -> must come back ascending
    df.iloc[0, df.columns.get_loc("Close")] = np.nan  # a NaN row must be dropped
    out = _normalize(df)
    assert out.index.is_monotonic_increasing
    assert not out.isna().any().any()


# --- YFinanceProvider cache ------------------------------------------------


def test_get_latest_bars_caches(tmp_path) -> None:
    dl = _FakeDownloader(_flat_frame(120))
    provider = YFinanceProvider(_settings(tmp_path), downloader=dl)

    first = provider.get_latest_bars("aapl", lookback=50, interval="1d")
    assert dl.calls == 1
    assert len(first) == 50
    assert list(first.columns) == list(OHLCV_COLUMNS)
    assert (tmp_path / "AAPL_1d.parquet").exists()

    # Second call is served from the fresh cache -> no new download.
    second = provider.get_latest_bars("AAPL", lookback=50, interval="1d")
    assert dl.calls == 1
    assert first.shape == second.shape
    assert (first["Close"].to_numpy() == second["Close"].to_numpy()).all()


def test_get_bars_always_fetches(tmp_path) -> None:
    dl = _FakeDownloader(_flat_frame(30))
    provider = YFinanceProvider(_settings(tmp_path), downloader=dl)
    provider.get_bars("AAPL", "2024-01-01", "2024-02-01", interval="1d")
    provider.get_bars("AAPL", "2024-01-01", "2024-02-01", interval="1d")
    assert dl.calls == 2  # explicit-range fetches bypass the cache


def test_unsupported_interval_rejected(tmp_path) -> None:
    provider = YFinanceProvider(_settings(tmp_path), downloader=_FakeDownloader(_flat_frame()))
    with pytest.raises(ValueError):
        provider.get_latest_bars("AAPL", lookback=10, interval="5m")


# --- universe --------------------------------------------------------------


def test_default_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNIVERSE", raising=False)
    assert get_universe() == DEFAULT_UNIVERSE


def test_universe_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIVERSE", " aapl, tsla ,, msft ")
    assert get_universe() == ["AAPL", "TSLA", "MSFT"]


# --- optional real network smoke test --------------------------------------


@pytest.mark.network
def test_real_fetch_smoke() -> None:
    provider = YFinanceProvider()
    df = provider.get_latest_bars("AAPL", lookback=30, interval="1d")
    assert not df.empty
    assert list(df.columns) == list(OHLCV_COLUMNS)
