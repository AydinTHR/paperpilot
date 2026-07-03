"""Outbound alerting: Telegram and Discord, fire-and-forget.

An alert is a courtesy signal to the operator's phone, never a load-bearing
step: ``send`` returns a bool and **must not raise into the trading loop**,
no matter how broken the network or the webhook is. Both senders use raw
``requests.post`` (no bot-framework dependency), retry with exponential
backoff, honor Discord's ``retry_after`` on 429, and give up quietly.

The HTTP transport and the sleeper are injectable (the same seam pattern as
``YFinanceProvider``'s downloader), so tests exercise every path offline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from config.logging_config import get_logger
from config.settings import Settings

logger = get_logger(__name__)

_MAX_TRIES = 3
_BACKOFF_S = (1.0, 2.0)  # between tries


class Alerter(Protocol):
    """Anything that can push one short operator-facing message."""

    def send(self, message: str) -> bool: ...


class NullAlerter:
    """The default when no channel is configured: swallow everything."""

    def send(self, message: str) -> bool:
        logger.debug("Alert (no channel configured): %s", message)
        return False


class MultiAlerter:
    """Fan one message out to several channels; success if any accepted it."""

    def __init__(self, alerters: list[Alerter]) -> None:
        self.alerters = alerters

    def send(self, message: str) -> bool:
        # Every channel gets the message (no short-circuit on first success).
        results = [alerter.send(message) for alerter in self.alerters]
        return any(results)


class _HttpAlerter:
    """Shared retry/backoff shell around one HTTP alert channel."""

    name = "http"

    def __init__(
        self,
        *,
        post: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._post = post
        self._sleeper = sleeper

    def send(self, message: str) -> bool:
        for attempt in range(_MAX_TRIES):
            try:
                response = self._request(message)
                if self._accepted(response):
                    return True
                retry_after = self._retry_after(response)
                logger.warning(
                    "%s alert got HTTP %s (attempt %d).",
                    self.name,
                    getattr(response, "status_code", "?"),
                    attempt + 1,
                )
                if getattr(response, "status_code", None) == 404:
                    logger.warning("%s alert endpoint is gone (404); not retrying.", self.name)
                    return False
            except Exception as exc:
                logger.warning("%s alert failed (attempt %d): %s", self.name, attempt + 1, exc)
                retry_after = None
            if attempt < _MAX_TRIES - 1:
                self._sleep(retry_after or _BACKOFF_S[min(attempt, len(_BACKOFF_S) - 1)])
        logger.warning("%s alert dropped after %d attempts.", self.name, _MAX_TRIES)
        return False

    # --- channel specifics (overridden) ---

    def _request(self, message: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def _accepted(response: Any) -> bool:
        return 200 <= getattr(response, "status_code", 0) < 300

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        if getattr(response, "status_code", None) != 429:
            return None
        try:
            value = response.json().get("retry_after")
            return float(value) if value is not None else None
        except Exception:
            return None

    # --- plumbing ---

    def _do_post(self, url: str, **kwargs: Any) -> Any:
        if self._post is not None:
            return self._post(url, **kwargs)
        import requests  # lazy: alerting is optional

        return requests.post(url, timeout=10, **kwargs)

    def _sleep(self, seconds: float) -> None:
        if self._sleeper is not None:
            self._sleeper(seconds)
        else:  # pragma: no cover - real sleep only outside tests
            import time

            time.sleep(seconds)


class TelegramAlerter(_HttpAlerter):
    """Bot-API ``sendMessage``. The token stays out of every log line."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        post: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(post=post, sleeper=sleeper)
        self._token = token
        self._chat_id = chat_id

    def _request(self, message: str) -> Any:
        return self._do_post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            json={"chat_id": self._chat_id, "text": message},
        )


class DiscordAlerter(_HttpAlerter):
    """Webhook POST; Discord answers 204 on success."""

    name = "discord"

    def __init__(
        self,
        webhook_url: str,
        *,
        post: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(post=post, sleeper=sleeper)
        self._webhook_url = webhook_url

    def _request(self, message: str) -> Any:
        return self._do_post(self._webhook_url, json={"content": message})


def build_alerter(settings: Settings) -> Alerter:
    """Assemble the configured channels; ``NullAlerter`` when none are set."""
    alerters: list[Alerter] = []
    token = settings.telegram_bot_token.get_secret_value()
    chat_id = settings.telegram_chat_id.get_secret_value()
    if token and chat_id:
        alerters.append(TelegramAlerter(token, chat_id))
    webhook = settings.discord_webhook_url.get_secret_value()
    if webhook:
        alerters.append(DiscordAlerter(webhook))
    if not alerters:
        return NullAlerter()
    if len(alerters) == 1:
        return alerters[0]
    return MultiAlerter(alerters)
