"""Market-data layer: fetch and cache OHLCV bars.

Defines a provider :class:`~typing.Protocol` (so Alpaca/Polygon can swap in
later) and a yfinance-backed implementation with a local parquet cache. Every
frame is normalised to a flat ``Open/High/Low/Close/Volume`` schema on an
ascending ``DatetimeIndex`` -- exactly the shape the strategies and (in Phase 3)
``backtesting.py`` expect.

The download callable is injectable, so tests run fully offline with a fake.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.data.cache import ParquetBarCache

logger = get_logger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
SUPPORTED_INTERVALS: tuple[str, ...] = ("1d", "1h")


class MarketDataProvider(Protocol):
    """The market-data surface the rest of PaperPilot depends on.

    Declaring it as a Protocol lets later phases swap in an Alpaca- or
    Polygon-backed provider without changing any caller.
    """

    def get_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame: ...

    def get_latest_bars(self, symbol: str, lookback: int, interval: str = "1d") -> pd.DataFrame: ...


def build_provider(settings: Settings | None = None) -> MarketDataProvider:
    """Return the provider selected by ``settings.data_provider``.

    ``auto`` resolves to Alpaca when Alpaca credentials are configured and
    yfinance otherwise, so a fresh clone with no keys still works out of the
    box. ``DATA_PROVIDER=yfinance`` is the one-line rollback.
    """
    settings = settings or get_settings()
    if settings.resolved_data_provider == "alpaca":
        # Imported lazily: alpaca_data imports from this module.
        from src.data.alpaca_data import AlpacaDataProvider

        return AlpacaDataProvider(settings)
    return YFinanceProvider(settings)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw yfinance frame to the canonical OHLCV schema.

    Flattens MultiIndex columns (yfinance returns these for a single symbol
    unless told otherwise), title-cases names, keeps only OHLCV, sorts ascending
    by date, de-duplicates the index, and drops rows with missing values.
    Idempotent: re-normalising an already-clean frame is a no-op.
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance uses (PriceField, Ticker); the field we want is level 0.
        out.columns = out.columns.get_level_values(0)

    out = out.rename(columns={c: str(c).strip().title() for c in out.columns})

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"data missing required column(s) {missing}; got {list(out.columns)}")

    out = out.loc[:, list(OHLCV_COLUMNS)]
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna()


class YFinanceProvider:
    """yfinance-backed :class:`MarketDataProvider` with a local parquet cache.

    ``get_bars`` always fetches the exact ``[start, end]`` window (no cache, so
    backtests get precisely the range they ask for). ``get_latest_bars`` is the
    cached convenience path used by the live loop and the preview script.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        downloader: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(self.settings.data_cache_dir)
        self._cache = ParquetBarCache(self.cache_dir)
        self._downloader = downloader

    # --- public API ---

    def get_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch a specific ``[start, end]`` window straight from the source."""
        interval = self._validate_interval(interval)
        symbol = symbol.upper()
        raw = self._download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        df = _normalize(raw)
        logger.info("Fetched %d %s bars for %s [%s..%s].", len(df), interval, symbol, start, end)
        return df

    def get_latest_bars(self, symbol: str, lookback: int, interval: str = "1d") -> pd.DataFrame:
        """Return roughly the most recent ``lookback`` bars, using the cache.

        Cache file: ``{data_cache_dir}/{SYMBOL}_{interval}.parquet``. When the
        file is present and fresh it is loaded from disk; otherwise we refetch a
        window comfortably larger than ``lookback``, rewrite the cache, and
        return the last ``lookback`` rows.
        """
        interval = self._validate_interval(interval)
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        symbol = symbol.upper()
        cache_path = self._cache_path(symbol, interval)

        df = self._load_cache_if_fresh(cache_path, interval)
        if df is None or len(df) < lookback:
            start, end = self._lookback_window(lookback, interval)
            df = self.get_bars(symbol, start, end, interval)
            self._write_cache(cache_path, df)
        else:
            logger.info(
                "Loaded %d %s bars for %s from cache %s.",
                len(df),
                interval,
                symbol,
                cache_path,
            )
        return df.tail(lookback)

    # --- internals ---

    def _download(self, **kwargs: object) -> pd.DataFrame:
        if self._downloader is not None:
            return self._downloader(**kwargs)
        import yfinance as yf  # lazy import keeps offline tests light

        return yf.download(**kwargs)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self._cache.path(symbol, interval)

    def _load_cache_if_fresh(self, path: Path, interval: str) -> pd.DataFrame | None:
        return self._cache.load_if_fresh(path, interval, transform=_normalize)

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        self._cache.write(path, df)

    @staticmethod
    def _lookback_window(lookback: int, interval: str) -> tuple[str, str]:
        """Pick a calendar window wide enough to yield ``lookback`` bars.

        Padded generously for weekends/holidays/non-trading hours; the caller
        trims to ``lookback`` afterwards.
        """
        end = datetime.now(UTC)
        if interval == "1d":
            start = end - timedelta(days=int(lookback * 1.6) + 10)
        else:  # "1h": ~6.5 trading hours per day
            start = end - timedelta(days=int(lookback / 5) + 10)
        return start.date().isoformat(), end.date().isoformat()

    @staticmethod
    def _validate_interval(interval: str) -> str:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"interval {interval!r} not supported; choose from {list(SUPPORTED_INTERVALS)}."
            )
        return interval
