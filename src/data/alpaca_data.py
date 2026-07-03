"""Alpaca-backed market data provider.

Implements the same :class:`~src.data.market_data.MarketDataProvider` Protocol
as ``YFinanceProvider``, so the loop, backtests, and scripts can swap sources
via settings. Two Alpaca-specific behaviours matter:

* **Batched fetches.** Alpaca's Basic plan allows 200 requests/min per key, so
  a cache miss fetches the *whole universe* in one ``StockBarsRequest`` and
  warms every symbol's cache. The loop's remaining symbols then hit disk.
* **Provider-suffixed cache files.** Alpaca bars (IEX feed, ``Adjustment.ALL``)
  are not byte-identical to yfinance's, so cache files carry an ``alpaca``
  suffix and the two providers never serve each other's frames.

The data client is injectable (duck-typed: only ``.get_stock_bars(req).df`` is
consumed), so tests run fully offline.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.data.cache import ParquetBarCache
from src.data.market_data import SUPPORTED_INTERVALS, _normalize

logger = get_logger(__name__)

CACHE_SUFFIX = "alpaca"

# IEX on the free tier rejects queries into the most recent ~15 minutes for
# intraday data; clamp the request end a little further back to be safe.
_INTRADAY_RECENCY_CLAMP = timedelta(minutes=16)

_RETRY_DELAYS_S: tuple[float, ...] = (0.5, 1.0, 2.0)


class AlpacaDataProvider:
    """Alpaca ``StockHistoricalDataClient``-backed provider with shared cache."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        batch_symbols: list[str] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(self.settings.data_cache_dir)
        self._cache = ParquetBarCache(self.cache_dir)
        self._client = client
        self._batch_symbols = batch_symbols
        self._sleeper = sleeper

    # --- public API (MarketDataProvider Protocol) ---

    def get_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch a specific ``[start, end]`` window straight from Alpaca."""
        interval = _validate_interval(interval)
        symbol = symbol.upper()
        frames = self._fetch([symbol], start, end, interval)
        if symbol not in frames:
            raise ValueError(f"Alpaca returned no bars for {symbol!r} in [{start}..{end}].")
        df = frames[symbol]
        logger.info("Fetched %d %s bars for %s [%s..%s].", len(df), interval, symbol, start, end)
        return df

    def get_latest_bars(self, symbol: str, lookback: int, interval: str = "1d") -> pd.DataFrame:
        """Return roughly the most recent ``lookback`` bars, using the cache.

        A miss fetches the whole batch universe in one request (rate-limit
        friendly) and warms every symbol's cache, not just the one asked for.
        """
        interval = _validate_interval(interval)
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        symbol = symbol.upper()
        cache_path = self._cache.path(symbol, interval, suffix=CACHE_SUFFIX)

        df = self._cache.load_if_fresh(cache_path, interval, transform=_normalize)
        if df is None or len(df) < lookback:
            start, end = _lookback_window(lookback, interval)
            batch = sorted({symbol, *(s.upper() for s in self._batch())})
            frames = self._fetch(batch, start, end, interval)
            for sym, frame in frames.items():
                self._cache.write(self._cache.path(sym, interval, suffix=CACHE_SUFFIX), frame)
            if symbol not in frames:
                raise ValueError(f"Alpaca returned no bars for {symbol!r}.")
            df = frames[symbol]
        else:
            logger.info(
                "Loaded %d %s bars for %s from cache %s.", len(df), interval, symbol, cache_path
            )
        return df.tail(lookback)

    # --- internals ---

    def _batch(self) -> list[str]:
        if self._batch_symbols is not None:
            return self._batch_symbols
        from src.data.universe import get_universe

        return get_universe()

    def _fetch(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        interval: str,
    ) -> dict[str, pd.DataFrame]:
        """One batched bars request, split into per-symbol normalised frames."""
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=_timeframe(interval),
            start=_as_datetime(start),
            end=_clamp_end(_as_datetime(end), interval),
            feed=DataFeed(self.settings.alpaca_data_feed),
            adjustment=Adjustment.ALL,
        )

        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS_S, None)):
            try:
                barset = self._get_client().get_stock_bars(request)
                break
            except Exception as exc:
                last_exc = exc
                if delay is None:
                    raise
                logger.warning(
                    "Alpaca bars fetch failed (attempt %d: %s); retrying in %.1fs.",
                    attempt + 1,
                    exc,
                    delay,
                )
                self._sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise last_exc or RuntimeError("unreachable")

        return _split_barset(barset.df, symbols, interval)

    def _get_client(self) -> Any:
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            if not self.settings.has_credentials:
                raise ValueError(
                    "Missing Alpaca credentials: set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                    "(or use DATA_PROVIDER=yfinance)."
                )
            self._client = StockHistoricalDataClient(
                api_key=self.settings.alpaca_api_key.get_secret_value(),
                secret_key=self.settings.alpaca_secret_key.get_secret_value(),
            )
        return self._client

    def _sleep(self, seconds: float) -> None:
        if self._sleeper is not None:
            self._sleeper(seconds)
        else:  # pragma: no cover - real sleep exercised only against the live API
            import time

            time.sleep(seconds)


# --- module helpers ----------------------------------------------------------


def _validate_interval(interval: str) -> str:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"interval {interval!r} not supported; choose from {list(SUPPORTED_INTERVALS)}."
        )
    return interval


def _timeframe(interval: str) -> Any:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if interval == "1d":
        return TimeFrame.Day
    return TimeFrame(1, TimeFrameUnit.Hour)


def _as_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _clamp_end(end: datetime, interval: str) -> datetime:
    if interval == "1d":
        return end
    limit = datetime.now(UTC) - _INTRADAY_RECENCY_CLAMP
    return min(end, limit)


def _lookback_window(lookback: int, interval: str) -> tuple[datetime, datetime]:
    """A calendar window comfortably wider than ``lookback`` bars."""
    end = datetime.now(UTC)
    if interval == "1d":
        start = end - timedelta(days=int(lookback * 1.6) + 10)
    else:  # "1h": ~6.5 trading hours per day
        start = end - timedelta(days=int(lookback / 5) + 10)
    return start, end


def _split_barset(df: pd.DataFrame, symbols: list[str], interval: str) -> dict[str, pd.DataFrame]:
    """Split Alpaca's MultiIndex (symbol, timestamp) frame into canonical frames."""
    if df is None or df.empty:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            sym_df = df.xs(symbol, level="symbol") if isinstance(df.index, pd.MultiIndex) else df
        except KeyError:
            logger.warning("Alpaca returned no bars for %s; skipping.", symbol)
            continue
        frames[symbol] = _from_alpaca(sym_df, interval)
    return frames


def _from_alpaca(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Rename Alpaca's lowercase columns and route through ``_normalize``."""
    out = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    out = out.drop(columns=[c for c in ("trade_count", "vwap") if c in out.columns])
    idx = pd.to_datetime(out.index)
    if interval == "1d" and getattr(idx, "tz", None) is not None:
        # Daily bars: match yfinance's tz-naive date index so strategies and
        # backtesting.py see the same shape from either provider.
        idx = idx.tz_convert("UTC").tz_localize(None).normalize()
    out.index = idx
    return _normalize(out)
