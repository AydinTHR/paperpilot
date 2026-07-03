"""Tests for the alerting layer -- fake HTTP transport, no network."""

from __future__ import annotations

from config.settings import Settings
from src.monitoring.alerts import (
    DiscordAlerter,
    MultiAlerter,
    NullAlerter,
    TelegramAlerter,
    build_alerter,
)


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakePost:
    """Records calls; yields a scripted sequence of responses/exceptions."""

    def __init__(self, *script) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kwargs) -> _Response:
        self.calls.append((url, kwargs))
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        return item


class _FakeSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --- Telegram ------------------------------------------------------------------


def test_telegram_sends_token_url_and_payload() -> None:
    post = _FakePost(_Response(200))
    alerter = TelegramAlerter("SECRET-TOKEN", "42", post=post, sleeper=_FakeSleeper())
    assert alerter.send("halt tripped") is True
    url, kwargs = post.calls[0]
    assert url == "https://api.telegram.org/botSECRET-TOKEN/sendMessage"
    assert kwargs["json"] == {"chat_id": "42", "text": "halt tripped"}


def test_telegram_retries_then_succeeds() -> None:
    post = _FakePost(ConnectionError("down"), _Response(200))
    sleeper = _FakeSleeper()
    alerter = TelegramAlerter("t", "c", post=post, sleeper=sleeper)
    assert alerter.send("x") is True
    assert len(post.calls) == 2
    assert sleeper.delays == [1.0]


# --- Discord ---------------------------------------------------------------------


def test_discord_204_is_success() -> None:
    post = _FakePost(_Response(204))
    alerter = DiscordAlerter("https://discord/webhook", post=post, sleeper=_FakeSleeper())
    assert alerter.send("hello") is True
    assert post.calls[0][1]["json"] == {"content": "hello"}


def test_discord_429_honors_retry_after() -> None:
    post = _FakePost(_Response(429, {"retry_after": 3.5}), _Response(204))
    sleeper = _FakeSleeper()
    alerter = DiscordAlerter("https://discord/webhook", post=post, sleeper=sleeper)
    assert alerter.send("x") is True
    assert sleeper.delays == [3.5]  # server-provided delay, not our backoff


def test_dead_webhook_404_stops_retrying() -> None:
    post = _FakePost(_Response(404))
    alerter = DiscordAlerter("https://discord/webhook", post=post, sleeper=_FakeSleeper())
    assert alerter.send("x") is False
    assert len(post.calls) == 1  # no pointless retries on a dead endpoint


# --- the never-raise contract ------------------------------------------------------


def test_send_never_raises_even_when_transport_always_explodes() -> None:
    post = _FakePost(RuntimeError("catastrophic"))
    sleeper = _FakeSleeper()
    alerter = TelegramAlerter("t", "c", post=post, sleeper=sleeper)
    assert alerter.send("x") is False  # swallowed, logged, done
    assert len(post.calls) == 3  # all tries exhausted
    assert sleeper.delays == [1.0, 2.0]


# --- selection matrix --------------------------------------------------------------


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_build_alerter_defaults_to_null() -> None:
    assert isinstance(build_alerter(_settings()), NullAlerter)


def test_build_alerter_telegram_only() -> None:
    alerter = build_alerter(_settings(telegram_bot_token="t", telegram_chat_id="c"))
    assert isinstance(alerter, TelegramAlerter)


def test_build_alerter_requires_both_telegram_fields() -> None:
    assert isinstance(build_alerter(_settings(telegram_bot_token="t")), NullAlerter)


def test_build_alerter_discord_only() -> None:
    alerter = build_alerter(_settings(discord_webhook_url="https://d/w"))
    assert isinstance(alerter, DiscordAlerter)


def test_build_alerter_both_channels_fans_out() -> None:
    alerter = build_alerter(
        _settings(
            telegram_bot_token="t",
            telegram_chat_id="c",
            discord_webhook_url="https://d/w",
        )
    )
    assert isinstance(alerter, MultiAlerter)
    assert len(alerter.alerters) == 2


def test_multi_alerter_true_if_any_channel_accepts() -> None:
    ok = TelegramAlerter("t", "c", post=_FakePost(_Response(200)), sleeper=_FakeSleeper())
    dead = DiscordAlerter("https://d/w", post=_FakePost(_Response(404)), sleeper=_FakeSleeper())
    assert MultiAlerter([dead, ok]).send("x") is True


def test_null_alerter_returns_false() -> None:
    assert NullAlerter().send("anything") is False
