"""Tests for the LLM response cache (fake inner client, in-memory journal)."""

from __future__ import annotations

from config.settings import Settings
from src.journal.store import Journal
from src.strategy.llm.cache import CachedLlmClient, params_hash
from src.strategy.llm.strategy import LlmStrategy


class _CountingClient:
    def __init__(self, response: str = '{"action": "HOLD", "confidence": 0.0}') -> None:
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response


class _DictStore:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get_llm_response(self, key: str) -> str | None:
        return self.data.get(key)

    def put_llm_response(self, key: str, response: str, *, model: str = "") -> None:
        self.data[key] = response


def test_second_identical_call_never_reaches_inner_client() -> None:
    inner = _CountingClient()
    cached = CachedLlmClient(inner, _DictStore(), params_hash("m", 0.0, 512, 30))
    first = cached.complete("sys", "user prompt")
    second = cached.complete("sys", "user prompt")
    assert first == second
    assert inner.calls == 1  # the whole point: re-runs don't re-bill


def test_different_prompt_is_a_different_key() -> None:
    inner = _CountingClient()
    cached = CachedLlmClient(inner, _DictStore(), params_hash("m", 0.0, 512, 30))
    cached.complete("sys", "prompt A")
    cached.complete("sys", "prompt B")
    assert inner.calls == 2


def test_params_change_invalidates_key() -> None:
    store = _DictStore()
    a = CachedLlmClient(_CountingClient("A"), store, params_hash("m", 0.0, 512, 30))
    b = CachedLlmClient(_CountingClient("B"), store, params_hash("m", 0.5, 512, 30))
    assert a.complete("s", "u") == "A"
    assert b.complete("s", "u") == "B"  # different temperature -> not A's cache


def test_journal_store_round_trip() -> None:
    journal = Journal("sqlite:///:memory:")
    assert journal.get_llm_response("k1") is None
    journal.put_llm_response("k1", '{"action": "BUY"}', model="claude-haiku-4-5")
    assert journal.get_llm_response("k1") == '{"action": "BUY"}'
    journal.put_llm_response("k1", "overwrite attempt")
    assert journal.get_llm_response("k1") == '{"action": "BUY"}'  # first write wins


def test_llm_strategy_wires_cache_when_store_given() -> None:
    inner = _CountingClient('{"action": "HOLD", "confidence": 0.0, "reason": "flat"}')
    strategy = LlmStrategy(
        client=inner,
        settings=Settings(),
        response_store=_DictStore(),
        min_bars=5,
        context_bars=5,
    )
    assert isinstance(strategy._client, CachedLlmClient)

    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=10, freq="B")
    close = 100 + np.arange(10, dtype=float)
    bars = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx
    )
    strategy.generate_signals(bars)
    strategy.generate_signals(bars)  # identical bars -> cache hit
    assert inner.calls == 1
