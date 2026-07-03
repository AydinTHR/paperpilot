"""Tests for the Alpaca broker wrapper using an injected fake client (no network)."""

from __future__ import annotations

import pytest
from alpaca.trading.enums import OrderClass, OrderSide

from config.settings import Settings
from src.execution.broker import (
    AccountSnapshot,
    Broker,
    BrokerError,
    OrderInfo,
    PositionInfo,
    WashTradeError,
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
    def __init__(self, req: object, order_id: str = "order-1") -> None:
        self.id = order_id
        self.symbol = req.symbol
        self.qty = req.qty
        self.side = req.side  # OrderSide enum
        self.order_type = "market"
        self.status = "accepted"
        self.filled_qty = "0"
        self.filled_avg_price = None
        # OTO/bracket fields captured for assertions.
        self.order_class = getattr(req, "order_class", None)
        self.stop_loss = getattr(req, "stop_loss", None)


class FakeClient:
    def __init__(self, *, submit_raises: Exception | None = None) -> None:
        self.cancelled = False
        self.closed: list[str] = []
        self.orders: list[_FakeOrder] = []
        self.cancelled_ids: list[str] = []
        self.open_orders: list[_FakeOrder] = []
        self._submit_raises = submit_raises

    def get_account(self) -> _FakeAccount:
        return _FakeAccount()

    def get_all_positions(self) -> list[_FakePosition]:
        return [_FakePosition()]

    def submit_order(self, order_data: object) -> _FakeOrder:
        if self._submit_raises is not None:
            raise self._submit_raises
        order = _FakeOrder(order_data, order_id=f"order-{len(self.orders) + 1}")
        self.orders.append(order)
        return order

    def cancel_orders(self) -> None:
        self.cancelled = True

    def close_position(self, symbol: str) -> None:
        self.closed.append(symbol)

    def get_order_by_id(self, order_id: str) -> _FakeOrder:
        for order in self.orders:
            if order.id == order_id:
                return order
        raise KeyError(order_id)

    def get_orders(self, filter: object = None) -> list[_FakeOrder]:
        return list(self.open_orders)

    def cancel_order_by_id(self, order_id: str) -> None:
        self.cancelled_ids.append(order_id)


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


# --- broker-held stops (OTO) -------------------------------------------------


def test_market_order_with_stop_builds_oto_request() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    info = broker.place_market_order_with_stop("aapl", 10, "buy", 95.0, ref_price=100.0)
    assert isinstance(info, OrderInfo)
    order = client.orders[0]
    assert order.symbol == "AAPL"
    assert order.order_class == OrderClass.OTO
    assert order.stop_loss is not None
    assert order.stop_loss.stop_price == 95.0


def test_stop_clamped_below_ref_price() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    # A stop at/above the base price is invalid at Alpaca; clamp to ref - 0.01.
    broker.place_market_order_with_stop("AAPL", 5, "buy", 101.0, ref_price=100.0)
    assert client.orders[0].stop_loss.stop_price == 99.99


def test_oto_rejects_fractional_shares() -> None:
    broker = make_broker()
    with pytest.raises(BrokerError, match="whole shares"):
        broker.place_market_order_with_stop("AAPL", 1.5, "buy", 95.0)


def test_oto_rejects_nonpositive_stop() -> None:
    broker = make_broker()
    with pytest.raises(BrokerError, match="Stop price"):
        broker.place_market_order_with_stop("AAPL", 1, "buy", 0.0)


def test_oto_reasserts_live_gate() -> None:
    forbidden = Settings.model_construct(paper=False, allow_live_trading=False)
    broker = Broker.__new__(Broker)  # bypass __init__ gate to test the method's own
    broker.settings = forbidden
    broker._client = FakeClient()
    with pytest.raises(BrokerError, match="live order"):
        broker.place_market_order_with_stop("AAPL", 1, "buy", 95.0)


def test_wash_trade_rejection_is_typed() -> None:
    client = FakeClient(submit_raises=RuntimeError('{"code": 40310000, "message": "wash"}'))
    broker = Broker(paper_settings(), client=client, announce=False)
    with pytest.raises(WashTradeError):
        broker.place_market_order("AAPL", 1, "buy")


# --- order status + cancel-before-close ---------------------------------------


def test_get_order_returns_current_state() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    placed = broker.place_market_order("AAPL", 5, "buy")
    client.orders[0].status = "filled"
    client.orders[0].filled_qty = "5"
    client.orders[0].filled_avg_price = "100.5"
    fetched = broker.get_order(placed.id)
    assert fetched.status == "filled"
    assert fetched.filled_qty == 5.0
    assert fetched.filled_avg_price == 100.5


def test_close_position_cancels_open_orders_first() -> None:
    client = FakeClient()
    broker = Broker(paper_settings(), client=client, announce=False)
    stop_leg = broker.place_market_order_with_stop("AAPL", 5, "buy", 95.0)
    client.open_orders = [client.orders[0]]  # the stop leg is still live
    broker.close_position("AAPL")
    assert client.cancelled_ids == [stop_leg.id]
    assert client.closed == ["AAPL"]


def test_get_open_orders_degrades_to_empty_on_error() -> None:
    class _Exploding(FakeClient):
        def get_orders(self, filter: object = None) -> list[_FakeOrder]:
            raise ConnectionError("api down")

    broker = Broker(paper_settings(), client=_Exploding(), announce=False)
    assert broker.get_open_orders("AAPL") == []
