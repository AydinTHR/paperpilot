"""Shared parquet bar cache used by every market-data provider.

Extracted from ``YFinanceProvider`` so Alpaca (and any later provider) reuses
the exact same TTL/staleness semantics and on-disk layout instead of growing a
second cache implementation. Filenames are ``{SYMBOL}_{interval}.parquet`` by
default; providers whose data differs from yfinance's (different feed or
adjustment) pass a ``suffix`` so the two never serve each other stale frames.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)

# How long a cached file stays "fresh" before we refetch. Daily bars settle
# once per session; hourly change far more often.
DEFAULT_CACHE_TTL: dict[str, timedelta] = {
    "1d": timedelta(hours=12),
    "1h": timedelta(minutes=30),
}


class ParquetBarCache:
    """A TTL-guarded parquet cache of normalised OHLCV frames."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        ttl: Mapping[str, timedelta] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl = dict(ttl) if ttl is not None else dict(DEFAULT_CACHE_TTL)

    def path(self, symbol: str, interval: str, *, suffix: str = "") -> Path:
        name = f"{symbol}_{interval}_{suffix}" if suffix else f"{symbol}_{interval}"
        return self.cache_dir / f"{name}.parquet"

    def load_if_fresh(
        self,
        path: Path,
        interval: str,
        *,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> pd.DataFrame | None:
        """Return the cached frame if present and inside its TTL, else None.

        ``transform`` (normally the provider's normaliser) runs inside the same
        try/except as the parquet read, so a corrupt or schema-drifted file is
        treated as a cache miss rather than an error.
        """
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        age = datetime.now(UTC) - mtime
        ttl = self.ttl.get(interval, timedelta(hours=12))
        if age > ttl:
            logger.info("Cache %s stale (age %s > ttl %s); refetching.", path, age, ttl)
            return None
        try:
            df = pd.read_parquet(path)
            return transform(df) if transform is not None else df
        except Exception as exc:
            logger.warning("Could not read cache %s (%s); refetching.", path, exc)
            return None

    def write(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path)
            logger.info("Wrote %d rows to cache %s.", len(df), path)
        except Exception as exc:
            logger.warning("Could not write cache %s (%s).", path, exc)
