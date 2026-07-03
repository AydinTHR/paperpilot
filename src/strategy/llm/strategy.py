"""LlmStrategy -- an LLM as a first-class, risk-gated trading strategy.

The model is handed a compact, *factual* snapshot of recent price action and
computed indicators (SMA20/50, RSI14, recent returns) and asked for a single
structured decision: BUY / SELL / HOLD with a confidence and a one-line reason.
It is explicitly NOT asked to forecast prices.

Safety properties (consistent with the rest of PaperPilot):
  * The LLM only *emits a Signal*. The RiskManager still sizes and gates every
    entry, the agent stays long-only, and trading stays paper-only.
  * Output is parsed defensively -- anything malformed, off-schema, or missing
    degrades to HOLD. The model can never crash a tick or force a trade.
  * The layer is optional: with no API key (or a backend build error) the
    strategy HOLDs and logs why, so the rest of the agent runs unaffected.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.strategy.base import Action, Signal, Strategy
from src.strategy.indicators import rsi, sma
from src.strategy.llm.cache import CachedLlmClient, LlmResponseStore, params_hash
from src.strategy.llm.client import LlmClient, LlmConfig, LlmError, build_llm_client

logger = get_logger(__name__)

_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}

_SYSTEM_PROMPT = (
    "You are a disciplined, risk-aware trading analyst for an educational, "
    "paper-only, long-only stock agent. Given a factual snapshot of recent price "
    "action and technical indicators for ONE symbol, decide whether to BUY "
    "(open or keep a long), SELL (close a long), or HOLD.\n"
    "Rules:\n"
    "- Judge only the data provided. Do NOT predict exact prices or invent news.\n"
    "- Prefer HOLD when the picture is mixed or weak; act only on a clear setup.\n"
    "- Long-only: SELL means 'exit a long', never 'go short'.\n"
    "- Respond with STRICT JSON only -- no prose, no markdown -- exactly:\n"
    '  {"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "<= 20 words"}'
)


class LlmStrategy(Strategy):
    """Turn an LLM's read of the indicators into a :class:`Signal`."""

    def __init__(
        self,
        *,
        client: LlmClient | None = None,
        settings: Settings | None = None,
        context_bars: int = 30,
        min_bars: int = 50,
        response_store: LlmResponseStore | None = None,
    ) -> None:
        self.name = "LLM signal"
        self.settings = settings or get_settings()
        self._context_bars = context_bars
        self._min_bars = min_bars
        self._unavailable = ""
        try:
            self._client = build_llm_client(self.settings, client=client)
        except LlmError as exc:
            self._client = None
            self._unavailable = str(exc)
            logger.warning("LLM strategy unavailable: %s", exc)
        if self._client is None and not self._unavailable:
            self._unavailable = "no API key configured"

        if self._client is not None and response_store is not None:
            # Read-through cache: re-runs over the same bars hit the store
            # instead of the API (reproducible experiments, zero re-billing).
            config = LlmConfig.from_settings(self.settings)
            self._client = CachedLlmClient(
                self._client,
                response_store,
                params_hash(config.model, config.temperature, config.max_tokens, context_bars),
                model=config.model,
            )

    @property
    def min_bars(self) -> int:
        return self._min_bars

    @property
    def available(self) -> bool:
        """True when a usable LLM client was constructed."""
        return self._client is not None

    def generate_signals(self, data: pd.DataFrame) -> Signal:
        self._check(data)
        if not self._has_enough_bars(data):
            return self.hold(reason=f"warming up: need {self.min_bars} bars, have {len(data)}")
        if self._client is None:
            return self.hold(reason=f"LLM unavailable: {self._unavailable}")

        prompt = self._build_user_prompt(data)
        try:
            raw = self._client.complete(_SYSTEM_PROMPT, prompt)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return self.hold(reason=f"LLM call failed: {exc}")

        signal = _parse_signal(raw)
        logger.info(
            "LLM signal: %s conf=%.2f (%s)",
            signal.action.value,
            signal.confidence,
            signal.reason,
        )
        return signal

    # --- prompt construction -------------------------------------------------

    def _build_user_prompt(self, data: pd.DataFrame) -> str:
        close = data["Close"]
        last = float(close.iloc[-1])
        sma20 = sma(close, 20).iloc[-1]
        sma50 = sma(close, 50).iloc[-1]
        rsi14 = rsi(close, 14).iloc[-1]
        ret5 = _pct_change(close, 5)
        ret20 = _pct_change(close, 20)
        recent = close.tail(self._context_bars)
        closes = ", ".join(f"{v:.2f}" for v in recent)
        return (
            "Symbol snapshot (oldest -> newest).\n"
            f"Last close: {last:.2f}\n"
            f"SMA20: {_fmt(sma20)}   SMA50: {_fmt(sma50)}\n"
            f"RSI14: {_fmt(rsi14)}\n"
            f"Return over last 5 bars: {_fmt_pct(ret5)}   over last 20 bars: {_fmt_pct(ret20)}\n"
            f"Recent closes ({len(recent)}): {closes}\n"
            "Decide BUY, SELL, or HOLD. Respond with strict JSON only."
        )


# --- module-level helpers (easy to unit-test) -------------------------------


def _fmt(value: float) -> str:
    return f"{value:.2f}" if pd.notna(value) else "n/a"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}%" if pd.notna(value) else "n/a"


def _pct_change(series: pd.Series, n: int) -> float:
    if len(series) <= n:
        return float("nan")
    prev = series.iloc[-n - 1]
    last = series.iloc[-1]
    if pd.isna(prev) or prev == 0:
        return float("nan")
    return (last / prev - 1.0) * 100.0


def _parse_signal(text: str) -> Signal:
    """Defensively turn raw model text into a Signal; HOLD on anything off."""
    raw = _extract_json(text)
    if raw is None:
        logger.warning("LLM returned no JSON object: %r", (text or "")[:200])
        return Signal(Action.HOLD, reason="unparseable LLM output")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM JSON did not parse: %r", raw[:200])
        return Signal(Action.HOLD, reason="invalid LLM JSON")
    if not isinstance(data, dict):
        return Signal(Action.HOLD, reason="LLM JSON was not an object")

    action_str = str(data.get("action", "")).strip().upper()
    if action_str not in _VALID_ACTIONS:
        return Signal(Action.HOLD, reason=f"unknown LLM action {action_str!r}")

    reason = str(data.get("reason", "")).strip()[:200] or "LLM decision"
    if action_str == "HOLD":
        return Signal(Action.HOLD, confidence=0.0, reason=reason)
    return Signal(
        Action(action_str),
        confidence=_clamp_confidence(data.get("confidence", 0.0)),
        reason=reason,
    )


def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of the text, tolerating ``` fences."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _clamp_confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN
        return 0.0
    return max(0.0, min(1.0, conf))
