"""Tests for the optional LLM signal layer -- fully offline with a fake client.

A fake :class:`LlmClient` returns canned text (or raises), so every branch of
the defensive parser and the graceful-degradation paths is exercised without any
network, SDK, or API key. ``Settings(_env_file=None)`` keeps the suite
deterministic even once a real ANTHROPIC_API_KEY lives in the developer's .env.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from src.strategy.base import Action
from src.strategy.llm.client import LlmError, build_llm_client
from src.strategy.llm.strategy import (
    LlmStrategy,
    _clamp_confidence,
    _extract_json,
    _parse_signal,
)

# --- fakes / helpers --------------------------------------------------------


class _FakeClient:
    """Canned LLM backend that records the prompts it received."""

    def __init__(self, response: str = "", *, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.system: str | None = None
        self.user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.system, self.user = system, user
        if self.raises is not None:
            raise self.raises
        return self.response


def _settings(**kw) -> Settings:
    # Disable .env so these tests are hermetic regardless of the local .env.
    return Settings(_env_file=None, **kw)


def _bars(n: int = 120, trend: float = 0.0, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    close = np.maximum(start + np.arange(n) * trend, 1.0)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _strategy(client) -> LlmStrategy:
    return LlmStrategy(client=client, settings=_settings())


# --- parsing happy paths ----------------------------------------------------


def test_buy_signal_parsed() -> None:
    client = _FakeClient('{"action": "BUY", "confidence": 0.7, "reason": "uptrend"}')
    sig = _strategy(client).generate_signals(_bars())
    assert sig.action is Action.BUY
    assert sig.confidence == 0.7
    assert sig.reason == "uptrend"


def test_sell_signal_parsed() -> None:
    client = _FakeClient('{"action": "SELL", "confidence": 0.4, "reason": "rollover"}')
    sig = _strategy(client).generate_signals(_bars())
    assert sig.action is Action.SELL
    assert sig.confidence == 0.4


def test_hold_forces_zero_confidence() -> None:
    client = _FakeClient('{"action": "HOLD", "confidence": 0.9, "reason": "mixed"}')
    sig = _strategy(client).generate_signals(_bars())
    assert sig.action is Action.HOLD
    assert sig.confidence == 0.0  # HOLD carries no conviction by convention


def test_confidence_clamped_high() -> None:
    client = _FakeClient('{"action": "BUY", "confidence": 2.5, "reason": "very sure"}')
    assert _strategy(client).generate_signals(_bars()).confidence == 1.0


def test_code_fenced_json_parsed() -> None:
    client = _FakeClient('```json\n{"action": "BUY", "confidence": 0.5, "reason": "x"}\n```')
    sig = _strategy(client).generate_signals(_bars())
    assert sig.action is Action.BUY
    assert sig.confidence == 0.5


def test_json_with_surrounding_prose_parsed() -> None:
    client = _FakeClient('Sure: {"action": "SELL", "confidence": 0.3, "reason": "weak"} ok')
    assert _strategy(client).generate_signals(_bars()).action is Action.SELL


# --- fail-safe paths --------------------------------------------------------


def test_malformed_output_holds() -> None:
    sig = _strategy(_FakeClient("I think you should buy!")).generate_signals(_bars())
    assert sig.action is Action.HOLD


def test_unknown_action_holds() -> None:
    client = _FakeClient('{"action": "YOLO", "confidence": 1.0, "reason": "moon"}')
    assert _strategy(client).generate_signals(_bars()).action is Action.HOLD


def test_client_error_holds() -> None:
    client = _FakeClient(raises=RuntimeError("network down"))
    sig = _strategy(client).generate_signals(_bars())
    assert sig.action is Action.HOLD
    assert "failed" in sig.reason


def test_no_client_holds() -> None:
    # No injected client + clean env (no key) -> LLM unavailable -> HOLD.
    strat = LlmStrategy(client=None, settings=_settings())
    assert strat.available is False
    sig = strat.generate_signals(_bars())
    assert sig.action is Action.HOLD
    assert "unavailable" in sig.reason


def test_insufficient_bars_holds_without_calling_model() -> None:
    client = _FakeClient('{"action": "BUY", "confidence": 0.9, "reason": "x"}')
    sig = _strategy(client).generate_signals(_bars(n=10))
    assert sig.action is Action.HOLD
    assert client.user is None  # short-circuited before any model call


def test_unwired_provider_degrades_to_hold() -> None:
    # 'openai' validates but isn't wired -> strategy HOLDs instead of crashing.
    strat = LlmStrategy(settings=_settings(llm_provider="openai"))
    assert strat.available is False
    assert strat.generate_signals(_bars()).action is Action.HOLD


# --- prompt contract --------------------------------------------------------


def test_prompt_includes_indicators_and_schema() -> None:
    client = _FakeClient('{"action": "HOLD", "confidence": 0, "reason": "ok"}')
    _strategy(client).generate_signals(_bars(trend=0.3))
    assert client.user is not None
    for token in ("Last close", "SMA20", "SMA50", "RSI14", "Recent closes"):
        assert token in client.user
    assert client.system is not None and "JSON" in client.system


# --- client factory + unit helpers -----------------------------------------


def test_build_llm_client_no_key_returns_none() -> None:
    assert build_llm_client(_settings()) is None


def test_build_llm_client_returns_injected() -> None:
    fake = _FakeClient("{}")
    assert build_llm_client(_settings(), client=fake) is fake


def test_build_llm_client_unwired_provider_raises() -> None:
    with pytest.raises(LlmError):
        build_llm_client(_settings(llm_provider="openai"))


def test_extract_json_variants() -> None:
    assert _extract_json('{"a": 1}') == '{"a": 1}'
    assert _extract_json('prefix {"a": 1} suffix') == '{"a": 1}'
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json("no json here") is None
    assert _extract_json("") is None


def test_clamp_confidence() -> None:
    assert _clamp_confidence(0.5) == 0.5
    assert _clamp_confidence(2.0) == 1.0
    assert _clamp_confidence(-1.0) == 0.0
    assert _clamp_confidence("nan") == 0.0
    assert _clamp_confidence("abc") == 0.0
    assert _clamp_confidence(None) == 0.0


def test_parse_signal_non_object_holds() -> None:
    assert _parse_signal("[1, 2, 3]").action is Action.HOLD
