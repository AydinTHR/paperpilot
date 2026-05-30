"""Tests for the configuration layer and the live-trading safety gate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def make_settings(**kw: object) -> Settings:
    # _env_file=None keeps the test hermetic (ignores any local .env).
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_defaults_to_paper() -> None:
    s = make_settings()
    assert s.paper is True
    assert s.is_live is False


def test_live_requires_explicit_override() -> None:
    # PAPER=false alone must NOT be enough to enable real-money trading.
    with pytest.raises(ValidationError):
        make_settings(paper=False, allow_live_trading=False)


def test_live_allowed_only_with_both_flags() -> None:
    s = make_settings(paper=False, allow_live_trading=True)
    assert s.is_live is True


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(log_level="LOUD")


def test_risk_fractions_bounded() -> None:
    with pytest.raises(ValidationError):
        make_settings(max_position_pct=1.5)
    with pytest.raises(ValidationError):
        make_settings(max_daily_loss_pct=0)


def test_safe_summary_hides_secrets() -> None:
    s = make_settings(alpaca_api_key="KEY_ABC", alpaca_secret_key="SECRET_XYZ")
    summary = s.safe_summary()
    rendered = str(summary)
    assert "KEY_ABC" not in rendered
    assert "SECRET_XYZ" not in rendered
    assert summary["mode"] == "PAPER"
    assert s.has_credentials is True


def test_secrets_not_in_repr() -> None:
    s = make_settings(alpaca_api_key="KEY_ABC", alpaca_secret_key="SECRET_XYZ")
    assert "KEY_ABC" not in repr(s)
    assert "SECRET_XYZ" not in repr(s)
