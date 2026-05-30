"""Tests for the Alpaca broker wrapper using an injected fake client (no network)."""

from __future__ import annotations

import pytest
from alpaca.trading.enums import OrderSide

from config.settings import Settings
from src.execution.broker import (
    AccountSnapshot,
    Broker,
    BrokerError,
    OrderInfo,
    PositionInfo,
    _to_float,
)


# --- fakes -----------------------------------------------------------------


class _FakeAccount:
    account_number = "PA12345"
    status = "ACTIVE"
    currency = "USD"
    cash = "10000.00"
    equity = "10500.50"
    buying_power = "20000.00"
    portfolio_value = "10500.50"
    pattern_day_trader = False


class _FakePosition:
    symbol = "AAPL"
    qty = "10"
    side = "long"
    avg_entry_price = "100.00"
    current_price = "105.00"
    market_value = "1050.00"
    unrealized_pl = "50.00"
    unrealized_plpc = "0.05"


class _FakeOrder:
    def __init__(self, req: object) -> None:
        self.id = "order-1"
        self.symbol = getattr(req, "symbol")
        self.qty = getattr(req, "qty")
        self.side = getattr(req, "side")  # OrderSide enum
        self.order_type = "market"
        self.status = "accepted"
        self.filled_qty = "0"
        self.filled_avg_price = None


class FakeClient:
    def __init__(self) -> None:
        self.cancelled = False
        self.closed: list[str] = []
        self.orders: list[_FakeOrder] = []

    def get_account(self) -> _FakeAccount:
        return _FakeAccount()

    def get_all_positions(self) -> list[_FakePosition]:
        return [_FakePosition()]

    def submit_order(self, order_data: object) -> _FakeOrder:
        order = _FakeOrder(order_data)
        self.orders.append(order)
        return order

    def cancel_orders(self) -> None:
        self.cancelled = True

    def close_position(self, symbol: str) -> None:
        self.closed.append(symbol)


# --- helpers ---------------------------------------------------------------


def paper_settings(**kw: object) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[arg-type]
        alpaca_api_key="k",
        alpaca_secret_key="s",
        **kw,
    )


def make_broker(client: FakeClient | None = None) -> Broker:
    return Broker(paper_settings(), client=client or FakeClient(), announce=False)


# --- tests -----------------------------------------------------------------


def test_get_account_maps_fields() -> None:
    acct = make_broker().get_account()
    assert isinstance(acct, AccountSnapshot)
    assert acct.account_number == "PA12345"
    assert acct.cash == 10000.00
    assert acct.equity == 10500.50
    assert acct.pattern_day_trader is False


def test_get_positions_maps_fields() -> None:
    positions = make_broker().get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert isinstance(pos, PositionInfo)
    assert pos.symbol == "AAPL"
    assert pos.qty == 10.0
    assert pos.side == "long"
    assert pos.unrealized_pl == 50.0


def test_place_market_order_buy_uppercases_and_maps_side() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    info = broker.place_market_order("aapl", 5, "buy")
    assert isinstance(info, OrderInfo)
    assert info.symbol == "AAPL"
    assert client.orders[0].side == OrderSide.BUY
    assert client.orders[0].symbol == "AAPL"


def test_place_market_order_sell_side() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    broker.place_market_order("MSFT", 2, "sell")
    assert client.orders[0].side == OrderSide.SELL


def test_place_market_order_rejects_nonpositive_qty() -> None:
    broker = make_broker()
    with pytest.raises(BrokerError):
        broker.place_market_order("AAPL", 0, "buy")


def test_unknown_side_rejected() -> None:
    broker = make_broker()
    with pytest.raises(BrokerError):
        broker.place_market_order("AAPL", 1, "hodl")


def test_cancel_and_close_delegate_to_client() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    broker.cancel_all_orders()
    broker.close_position("aapl")
    assert client.cancelled is True
    assert client.closed == ["AAPL"]


def test_missing_credentials_raises() -> None:
    # No injected client and no keys -> cannot build a real client.
    settings = Settings(_env_file=None)  # type: ignore[arg-type]
    with pytest.raises(BrokerError):
        Broker(settings, announce=False)


def test_broker_refuses_live_without_override() -> None:
    # Bypass settings validation to forge the forbidden combo, then prove the
    # broker boundary refuses it as defense in depth.
    forbidden = Settings.model_construct(paper=False, allow_live_trading=False)
    with pytest.raises(BrokerError):
        Broker(forbidden, client=FakeClient(), announce=False)


def test_to_float_is_robust() -> None:
    assert _to_float(None) == 0.0
    assert _to_float("abc", default=1.0) == 1.0
    assert _to_float("3.5") == 3.5
    assert _to_float("12") == 12.0
