"""Optional LLM signal layer (Phase 6).

Provider-agnostic by design: strategies depend only on the :class:`LlmClient`
protocol, so swapping Claude for another backend is a one-class change and tests
inject a fake with no network, SDK, or API key.
"""

from src.strategy.llm.client import (
    AnthropicClient,
    LlmClient,
    LlmConfig,
    LlmError,
    build_llm_client,
)
from src.strategy.llm.strategy import LlmStrategy

__all__ = [
    "AnthropicClient",
    "LlmClient",
    "LlmConfig",
    "LlmError",
    "build_llm_client",
    "LlmStrategy",
]
