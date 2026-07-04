"""Tests for the OpenRouter LLM client -- fake HTTP transport, no network."""

from __future__ import annotations

import pytest

from config.settings import Settings
from src.strategy.llm.client import (
    OPENROUTER_API_URL,
    AnthropicClient,
    LlmConfig,
    LlmError,
    OpenRouterClient,
    _openrouter_model,
    build_llm_client,
)


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakePost:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kwargs) -> _Response:
        self.calls.append((url, kwargs))
        return self.response


def _ok_response(content: str) -> _Response:
    return _Response(200, {"choices": [{"message": {"content": content}}]})


def _client(post: _FakePost, model: str = "anthropic/claude-haiku-4.5") -> OpenRouterClient:
    config = LlmConfig(provider="openrouter", model=model, max_tokens=64, temperature=0.0)
    return OpenRouterClient(config, api_key="sk-or-test", post=post)


def test_complete_sends_expected_payload_and_auth() -> None:
    post = _FakePost(_ok_response('{"action":"HOLD"}'))
    result = _client(post).complete("system prompt", "user prompt")

    assert result == '{"action":"HOLD"}'
    url, kwargs = post.calls[0]
    assert url == OPENROUTER_API_URL
    assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"
    assert kwargs["timeout"] == 30.0
    payload = kwargs["json"]
    assert payload["model"] == "anthropic/claude-haiku-4.5"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0] == {"role": "system", "content": "system prompt"}
    assert payload["messages"][1] == {"role": "user", "content": "user prompt"}


def test_http_error_raises_llm_error_with_detail() -> None:
    post = _FakePost(_Response(402, {"error": {"message": "Insufficient credits"}}))
    with pytest.raises(LlmError, match=r"402.*Insufficient credits"):
        _client(post).complete("s", "u")


def test_malformed_body_raises_llm_error() -> None:
    post = _FakePost(_Response(200, {"unexpected": True}))
    with pytest.raises(LlmError, match="response shape"):
        _client(post).complete("s", "u")


def test_model_mapping() -> None:
    # OpenRouter-native ids pass through; the Anthropic-style default is
    # aliased; unknown bare names get a best-effort anthropic/ prefix.
    assert _openrouter_model("anthropic/claude-haiku-4.5") == "anthropic/claude-haiku-4.5"
    assert _openrouter_model("claude-haiku-4-5") == "anthropic/claude-haiku-4.5"
    assert _openrouter_model("claude-sonnet-x") == "anthropic/claude-sonnet-x"


# --- factory routing --------------------------------------------------------------


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_factory_builds_openrouter_client_with_key() -> None:
    client = build_llm_client(_settings(llm_provider="openrouter", openrouter_api_key="sk-or-x"))
    assert isinstance(client, OpenRouterClient)


def test_factory_returns_none_without_openrouter_key() -> None:
    assert build_llm_client(_settings(llm_provider="openrouter")) is None


def test_factory_still_builds_anthropic_by_default() -> None:
    client = build_llm_client(_settings(anthropic_api_key="sk-ant-x"))
    assert isinstance(client, AnthropicClient)


def test_has_llm_key_is_provider_aware() -> None:
    assert _settings(llm_provider="openrouter", openrouter_api_key="k").has_llm_key
    assert not _settings(llm_provider="openrouter", anthropic_api_key="k").has_llm_key
    assert _settings(llm_provider="anthropic", anthropic_api_key="k").has_llm_key
    assert not _settings(llm_provider="anthropic", openrouter_api_key="k").has_llm_key
