"""Provider-agnostic LLM client for the optional signal layer.

The strategy depends only on the :class:`LlmClient` protocol -- a single
``complete(system, user) -> str`` call -- so the backend is swappable and tests
inject a fake (no network, no SDK, no API key). The concrete
:class:`AnthropicClient` wraps the ``anthropic`` SDK and imports it lazily, so
the package stays a *soft* dependency: PaperPilot runs fine without it until the
LLM strategy is actually used with a key.
"""

from __future__ import annotations

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
    model: str = "claude-3-5-haiku-latest"
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

    # Provider-agnostic by design: the protocol + factory make another backend
    # (e.g. OpenAI) a drop-in. It just isn't wired yet.
    raise LlmError(f"LLM provider {provider!r} is not wired yet; set LLM_PROVIDER=anthropic.")
