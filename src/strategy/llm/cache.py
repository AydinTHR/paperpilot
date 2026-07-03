"""Per-bar LLM response caching for reproducible, re-billable-free runs.

An experiment re-run over the same bars should produce the same signals and
cost zero API dollars. The cache key is a hash of the call parameters plus the
exact prompts; the user prompt embeds the recent closes, so the same market
snapshot maps to the same key while any change to data, model, or prompt
invalidates it naturally. (The strategy interface never sees the symbol, so it
is deliberately not part of the key; the prompt's price series disambiguates
symbols in practice.)
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from config.logging_config import get_logger
from src.strategy.llm.client import LlmClient

logger = get_logger(__name__)


class LlmResponseStore(Protocol):
    """Key-value persistence for LLM responses (the Journal satisfies this)."""

    def get_llm_response(self, key: str) -> str | None: ...
    def put_llm_response(self, key: str, response: str, *, model: str = "") -> None: ...


def params_hash(model: str, temperature: float, max_tokens: int, context_bars: int) -> str:
    """A short stable digest of everything that shapes an LLM call's output."""
    raw = f"{model}|{temperature}|{max_tokens}|{context_bars}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CachedLlmClient:
    """Wraps any :class:`LlmClient` with read-through response caching."""

    def __init__(
        self, inner: LlmClient, store: LlmResponseStore, params: str, model: str = ""
    ) -> None:
        self._inner = inner
        self._store = store
        self._params = params
        self._model = model

    def complete(self, system: str, user: str) -> str:
        key = self._key(system, user)
        cached = self._store.get_llm_response(key)
        if cached is not None:
            logger.debug("LLM cache hit (%s).", key[:12])
            return cached
        response = self._inner.complete(system, user)
        try:
            self._store.put_llm_response(key, response, model=self._model)
        except Exception as exc:
            logger.warning("Could not cache LLM response: %s", exc)
        return response

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self._params}|{system}|{user}".encode()).hexdigest()
