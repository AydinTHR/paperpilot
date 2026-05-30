"""The tradable universe and the benchmark symbol.

Default is a small, liquid mix plus SPY as a benchmark. An optional ``UNIVERSE``
environment variable (comma-separated tickers) overrides the default.
"""

from __future__ import annotations

import os

DEFAULT_UNIVERSE: list[str] = ["AAPL", "MSFT", "NVDA", "SPY"]
BENCHMARK: str = "SPY"


def get_universe() -> list[str]:
    """Return the trading universe.

    Defaults to :data:`DEFAULT_UNIVERSE`; ``UNIVERSE=AAPL,TSLA`` overrides it.
    Tickers are upper-cased and blanks are dropped.
    """
    raw = os.getenv("UNIVERSE", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return symbols or list(DEFAULT_UNIVERSE)
