"""Tests for the --reset-halt CLI path (journal only, no broker)."""

from __future__ import annotations

import pytest

from config.settings import get_settings
from src.journal.store import Journal


@pytest.fixture
def _tmp_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/journal.db")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_reset_halt_clears_persisted_state(_tmp_env, tmp_path, capsys) -> None:
    from scripts.run_live import main

    db_url = f"sqlite:///{tmp_path}/journal.db"
    Journal(db_url).record_halt(halt_type="drawdown", active=True, reason="21% below peak")

    assert main(["--reset-halt"]) == 0
    out = capsys.readouterr().out
    assert "drawdown" in out
    assert "cleared" in out

    states = Journal(db_url).latest_halt_states()
    assert states["drawdown"].active is False
    assert states["daily_loss"].active is False


def test_reset_halt_with_no_prior_state(_tmp_env, capsys) -> None:
    from scripts.run_live import main

    assert main(["--reset-halt"]) == 0
    assert "nothing to clear" in capsys.readouterr().out
