"""Tests for the experiment harness and virtual-portfolio isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import Settings
from src.execution.broker import AccountSnapshot, OrderInfo, PositionInfo
from src.experiments.harness import (
    ExperimentHarness,
    VirtualPortfolio,
    _arm_db_url,
    _arm_report,
    _build_strategy,
)
from src.journal.store import Journal

# --- fakes ---------------------------------------------------------------------


class _FakeRealBroker:
    """The shared real account under the virtual portfolios."""

    def __init__(self, equity: float = 90_000.0) -> None:
        self.equity = equity
        self.orders: list[tuple[str, float, str]] = []
        self.positions: list[PositionInfo] = []

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_number="REAL",
            status="ACTIVE",
            currency="USD",
            cash=self.equity,
            equity=self.equity,
            buying_power=self.equity,
            portfolio_value=self.equity,
        )

    def get_positions(self) -> list[PositionInfo]:
        return list(self.positions)

    def place_market_order(self, symbol: str, qty: float, side: str) -> OrderInfo:
        self.orders.append((symbol.upper(), qty, side))
        return OrderInfo(
            id=f"ord-{len(self.orders)}",
            symbol=symbol.upper(),
            qty=qty,
            side=side,
            order_type="market",
            status="filled",
            filled_qty=qty,
            filled_avg_price=100.0,
        )

    def place_market_order_with_stop(self, symbol, qty, side, stop_price, *, ref_price=None):
        return self.place_market_order(symbol, qty, side)

    def get_order(self, order_id: str) -> OrderInfo:  # pragma: no cover - passthrough
        raise KeyError(order_id)

    def get_open_orders(self, symbol: str | None = None) -> list[OrderInfo]:
        return []


# --- VirtualPortfolio ledger -----------------------------------------------------


def test_virtual_allocation_splits_equity() -> None:
    real = _FakeRealBroker(equity=90_000.0)
    arm = VirtualPortfolio(real, fraction=1 / 3, name="sma")
    account = arm.get_account()
    assert account.equity == pytest.approx(30_000.0)
    assert account.cash == pytest.approx(30_000.0)


def test_virtual_buy_moves_cash_into_position() -> None:
    arm = VirtualPortfolio(_FakeRealBroker(), fraction=1 / 3, name="sma")
    arm.place_market_order("AAPL", 100, "buy")  # fills @ 100.0
    account = arm.get_account()
    assert account.cash == pytest.approx(20_000.0)
    assert account.equity == pytest.approx(30_000.0)  # cash + position value
    assert arm.get_positions()[0].qty == 100


def test_virtual_close_sells_only_own_shares() -> None:
    real = _FakeRealBroker()
    arm_a = VirtualPortfolio(real, fraction=0.5, name="sma")
    arm_b = VirtualPortfolio(real, fraction=0.5, name="rsi")
    arm_a.place_market_order("AAPL", 100, "buy")
    arm_b.place_market_order("AAPL", 40, "buy")

    arm_b.close_position("AAPL")

    sells = [(s, q) for s, q, side in real.orders if side == "sell"]
    assert sells == [("AAPL", 40)]  # sized sell: arm_a's 100 shares untouched
    assert arm_a.get_positions()[0].qty == 100
    assert arm_b.get_positions() == []


def test_virtual_close_when_flat_is_noop() -> None:
    real = _FakeRealBroker()
    arm = VirtualPortfolio(real, fraction=1.0, name="sma")
    arm.close_position("AAPL")
    assert real.orders == []


def test_virtual_sell_realizes_pnl_into_cash() -> None:
    real = _FakeRealBroker()
    arm = VirtualPortfolio(real, fraction=1.0, name="sma")
    arm.place_market_order("AAPL", 10, "buy")  # -1000
    cash_after_buy = arm.get_account().cash
    arm.close_position("AAPL")  # +1000 (same fill price)
    assert arm.get_account().cash == pytest.approx(cash_after_buy + 1_000.0)


# --- harness construction ----------------------------------------------------------


def _patched(monkeypatch) -> None:
    monkeypatch.setattr("src.experiments.harness.Broker", lambda *a, **k: _FakeRealBroker())
    monkeypatch.setattr("src.experiments.harness.build_provider", lambda settings: object())


def test_auto_mode_falls_back_to_virtual_without_second_account(monkeypatch) -> None:
    _patched(monkeypatch)
    harness = ExperimentHarness.from_settings(
        Settings(), strategies=("sma", "rsi"), symbols=["AAPL"]
    )
    assert harness.mode == "virtual"
    assert [arm.name for arm in harness.arms] == ["sma", "rsi"]
    assert isinstance(harness.arms[0].loop.broker, VirtualPortfolio)


def test_auto_mode_picks_accounts_with_second_credentials(monkeypatch) -> None:
    _patched(monkeypatch)
    settings = Settings(
        alpaca_api_key="k1",
        alpaca_secret_key="s1",
        alpaca_api_key_2="k2",
        alpaca_secret_key_2="s2",
    )
    harness = ExperimentHarness.from_settings(settings, strategies=("sma", "rsi"), symbols=["AAPL"])
    assert harness.mode == "accounts"


def test_accounts_mode_requires_credentials_per_arm(monkeypatch) -> None:
    _patched(monkeypatch)
    with pytest.raises(ValueError, match="paper account 2"):
        ExperimentHarness.from_settings(
            Settings(alpaca_api_key="k1", alpaca_secret_key="s1"),
            strategies=("sma", "rsi"),
            mode="accounts",
            symbols=["AAPL"],
        )


def test_arm_journals_are_separate_files() -> None:
    assert _arm_db_url("sqlite:///data/paperpilot.db", "sma").endswith("experiments/sma.db")
    assert _arm_db_url("sqlite:///data/paperpilot.db", "llm") != _arm_db_url(
        "sqlite:///data/paperpilot.db", "sma"
    )


def test_build_strategy_keys() -> None:
    settings = Settings()
    journal = Journal("sqlite:///:memory:")
    assert _build_strategy("sma", settings, journal).name.startswith("SMA")
    assert _build_strategy("rsi", settings, journal).name.startswith("RSI")
    with pytest.raises(ValueError, match="unknown strategy"):
        _build_strategy("hodl", settings, journal)


# --- reporting ----------------------------------------------------------------------


def test_arm_report_aggregates_journal() -> None:
    journal = Journal("sqlite:///:memory:")
    t0 = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    for i, equity in enumerate([100_000.0, 102_000.0, 101_000.0]):
        journal.record_equity(equity=equity, cash=equity, ts=t0 + timedelta(days=i))
    journal.record_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        strategy="sma",
        status="filled",
        filled_qty=10,
        filled_avg_price=100.0,
        ts=t0,
    )
    journal.record_order(
        symbol="AAPL",
        side="sell",
        qty=10,
        strategy="sma",
        status="filled",
        filled_qty=10,
        filled_avg_price=110.0,
        ts=t0 + timedelta(days=1),
    )

    report = _arm_report("sma", journal)
    assert report.ticks == 3
    assert report.return_pct == pytest.approx(1.0)
    assert report.realized_pnl == pytest.approx(100.0)
    assert report.num_trades == 1
    assert report.win_rate_pct == pytest.approx(100.0)
    assert report.sharpe is None  # < 20 ticks -> insufficient data


# --- --fresh: archiving stale arm journals ---------------------------------------


def test_archive_arm_journals_renames_existing(tmp_path) -> None:
    from src.experiments.harness import archive_arm_journals

    settings = Settings(
        _env_file=None,  # type: ignore[arg-type]
        db_url=f"sqlite:///{tmp_path}/paperpilot.db",
    )
    arm_dir = tmp_path / "experiments"
    arm_dir.mkdir()
    (arm_dir / "sma.db").write_bytes(b"stale")
    (arm_dir / "rsi.db").write_bytes(b"stale")

    archived = archive_arm_journals(settings, ["sma", "rsi", "llm"])

    assert len(archived) == 2  # llm had no journal -> nothing to archive
    assert not (arm_dir / "sma.db").exists()
    assert not (arm_dir / "rsi.db").exists()
    for target in archived:
        assert target.exists()
        assert target.name.endswith(".bak")
        assert target.read_bytes() == b"stale"  # renamed, never deleted


def test_archive_arm_journals_noop_when_clean(tmp_path) -> None:
    from src.experiments.harness import archive_arm_journals

    settings = Settings(
        _env_file=None,  # type: ignore[arg-type]
        db_url=f"sqlite:///{tmp_path}/paperpilot.db",
    )
    assert archive_arm_journals(settings, ["sma", "rsi", "llm"]) == []


def test_arms_share_market_hours_gate(monkeypatch) -> None:
    _patched(monkeypatch)
    harness = ExperimentHarness.from_settings(
        Settings(), strategies=("sma", "rsi"), symbols=["AAPL"]
    )
    calendars = [arm.loop.market_calendar for arm in harness.arms]
    assert all(c is not None for c in calendars)
    assert calendars[0] is calendars[1]  # one shared calendar, not one per arm


def test_market_hours_gate_disabled_via_settings(monkeypatch) -> None:
    _patched(monkeypatch)
    harness = ExperimentHarness.from_settings(
        Settings(market_hours_only=False), strategies=("sma",), symbols=["AAPL"]
    )
    assert harness.arms[0].loop.market_calendar is None
