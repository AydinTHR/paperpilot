"""Provider-agnostic LLM client for the optional signal layer.

The strategy depends only on the :class:`LlmClient` protocol -- a single
``complete(system, user) -> str`` call -- so the backend is swappable and tests
inject a fake (no network, no SDK, no API key). Two backends are wired:
:class:`AnthropicClient` (the ``anthropic`` SDK, imported lazily so it stays a
*soft* dependency) and :class:`OpenRouterClient` (raw HTTP to OpenRouter's
gateway, which resells Claude and other models under one API and accepts more
payment methods than Anthropic direct).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from config.logging_config import get_logger
from config.settings import Settings

logger = get_logger(__name__)


class LlmError(RuntimeError):
    """Raised when an LLM backend is misconfigured or unavailable."""


class LlmClient(Protocol):
    """The single capability the LLM strategy needs from any backend."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for the given prompts."""
        ...


@dataclass(frozen=True)
class LlmConfig:
    """Backend-neutral LLM call parameters."""

    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 512
    temperature: float = 0.0
    timeout_seconds: float = 30.0

    @classmethod
    def from_settings(cls, settings: Settings) -> LlmConfig:
        return cls(
            provider=settings.llm_provider,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )


class AnthropicClient:
    """:class:`LlmClient` backed by Anthropic's Claude API.

    The ``anthropic`` SDK is imported lazily so it stays optional. A
    ``sdk_client`` can be injected for tests, though the strategy tests use a
    fake :class:`LlmClient` directly and never construct this.
    """

    def __init__(self, config: LlmConfig, api_key: str, *, sdk_client: Any = None) -> None:
        self.config = config
        if sdk_client is not None:
            self._client = sdk_client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise LlmError(
                "The 'anthropic' package is not installed. Run "
                "`pip install anthropic` to use the LLM signal layer."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=config.timeout_seconds)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate any text blocks; ignore non-text content defensively.
        parts = [
            getattr(block, "text", "")
            for block in getattr(resp, "content", [])
            if getattr(block, "type", "text") == "text"
        ]
        return "".join(parts).strip()


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Anthropic-style model names mapped to OpenRouter's catalogue ids. Ids with a
# "/" are already OpenRouter-native and pass through untouched.
_OPENROUTER_ALIASES: dict[str, str] = {
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
}


def _openrouter_model(model: str) -> str:
    if "/" in model:
        return model
    return _OPENROUTER_ALIASES.get(model, f"anthropic/{model}")


class OpenRouterClient:
    """:class:`LlmClient` backed by OpenRouter's OpenAI-compatible chat API.

    Raw ``requests`` (no SDK): one POST, one JSON shape. The HTTP transport is
    injectable for offline tests, mirroring the alerters' seam.
    """

    def __init__(
        self,
        config: LlmConfig,
        api_key: str,
        *,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._api_key = api_key
        self._post = post

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": _openrouter_model(self.config.model),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = self._do_post(
            OPENROUTER_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.config.timeout_seconds,
        )
        status = getattr(response, "status_code", 0)
        if status != 200:
            raise LlmError(f"OpenRouter returned HTTP {status}: {_error_detail(response)}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LlmError(f"Unexpected OpenRouter response shape: {exc}") from exc
        return str(content).strip()

    def _do_post(self, url: str, **kwargs: Any) -> Any:
        if self._post is not None:
            return self._post(url, **kwargs)
        import requests  # lazy: only needed when the layer is active

        return requests.post(url, **kwargs)


def _error_detail(response: Any) -> str:
    """Best-effort error text from a failed response; never raises."""
    try:
        return str(response.json().get("error", {}).get("message", ""))[:200]
    except Exception:
        return ""


def build_llm_client(settings: Settings, *, client: LlmClient | None = None) -> LlmClient | None:
    """Return a ready LLM client, or ``None`` when no API key is configured.

    Returning ``None`` (rather than raising) is what makes the layer *optional*:
    the strategy treats a missing client as "LLM unavailable -> HOLD". An
    explicitly injected ``client`` is always returned as-is (used by tests).
    """
    if client is not None:
        return client

    provider = settings.llm_provider
    if provider == "anthropic":
        api_key = settings.anthropic_api_key.get_secret_value()
        if not api_key:
            logger.info("No ANTHROPIC_API_KEY set; LLM signal layer will HOLD.")
            return None
        return AnthropicClient(LlmConfig.from_settings(settings), api_key=api_key)

    if provider == "openrouter":
        api_key = settings.openrouter_api_key.get_secret_value()
        if not api_key:
            logger.info("No OPENROUTER_API_KEY set; LLM signal layer will HOLD.")
            return None
        return OpenRouterClient(LlmConfig.from_settings(settings), api_key=api_key)

    # Provider-agnostic by design: the protocol + factory make another backend
    # (e.g. OpenAI) a drop-in. It just isn't wired yet.
    raise LlmError(
        f"LLM provider {provider!r} is not wired yet; "
        "set LLM_PROVIDER=anthropic or LLM_PROVIDER=openrouter."
    )
