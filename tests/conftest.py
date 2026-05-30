"""Shared test fixtures."""

from __future__ import annotations

import pytest

_ENV_VARS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "PAPER",
    "ALLOW_LIVE_TRADING",
    "MAX_POSITION_PCT",
    "MAX_DAILY_LOSS_PCT",
    "MAX_DRAWDOWN_PCT",
    "STOP_LOSS_PCT",
    "LOG_LEVEL",
    "LOG_DIR",
    "DEFAULT_INTERVAL",
    "DATA_CACHE_DIR",
    "UNIVERSE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip PaperPilot env vars so settings tests are deterministic."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
