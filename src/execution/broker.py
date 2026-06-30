"""Alpaca trading client wrapper.

Exposes a small, broker-agnostic surface (account, positions, place/cancel
orders) over Alpaca's ``alpaca-py`` SDK so the rest of PaperPilot never imports
the SDK directly. Defaults to the paper endpoint; real-money trading is refused
at this boundary unless deliberately enabled in settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config.logging_config import get_logger
from config.settings import (
    DISCLAIMER_BANNER,
    LIVE_TRADING_WARNING,
    Settings,
    get_settings,
)

logger = get_logger(__name__)


class BrokerError(RuntimeError):
    """Raised when a broker operation fails or is disallowed."""


# --- Plain, SDK-agnostic data carriers returned to the rest of the app ---


@dataclass(frozen=True)
class AccountSnapshot:
    account_number: str
    status: str
    currency: str
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass(frozen=True)
class OrderInfo:
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str
    filled_qty: float
    filled_avg_price: float | None


class TradingClientProtocol(Protocol):
    """The subset of ``alpaca.trading.client.TradingClient`` we depend on.

    Declaring it as a Protocol lets tests inject a fake client without touching
    the network.
    """

    def get_account(self): ...
    def get_all_positions(self): ...
    def submit_order(self, order_data): ...
    def cancel_orders(self): ...
    def close_position(self, symbol_or_asset_id): ...


def _to_float(value: object, default: float = 0.0) -> float:
    """Alpaca returns numerics as strings; coerce safely."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _enum_str(value: object) -> str:
    """Return an enum-like value's ``.value``, else ``str(value)``.

    Alpaca returns fields like account status / order side as enums; this
    yields the clean string ("ACTIVE") instead of the repr ("AccountStatus.ACTIVE").
    """
    return str(getattr(value, "value", value))


class Broker:
    """Thin, safe wrapper around the Alpaca trading client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: TradingClientProtocol | None = None,
        announce: bool = True,
    ) -> None:
        self.settings = settings or get_settings()

        if announce:
            self._announce_mode()

        # Defense in depth: settings already refuses PAPER=false without the
        # override, but re-assert at the execution boundary so a live client can
        # never be constructed by accident.
        if not self.settings.paper and not self.settings.allow_live_trading:
            raise BrokerError(
                "Refusing to create a live trading client: PAPER=false requires "
                "ALLOW_LIVE_TRADING=true."
            )

        if client is not None:
            self._client: TradingClientProtocol = client
        else:
            if not self.settings.has_credentials:
                raise BrokerError(
                    "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                    "ALPACA_SECRET_KEY in your environment or .env file."
                )
            self._client = TradingClient(
                api_key=self.settings.alpaca_api_key.get_secret_value(),
                secret_key=self.settings.alpaca_secret_key.get_secret_value(),
                paper=self.settings.paper,
            )

        logger.info(
            "Broker initialised in %s mode.",
            "LIVE" if self.settings.is_live else "PAPER",
        )

    # --- startup banners ---

    def _announce_mode(self) -> None:
        if self.settings.is_live:
            print(LIVE_TRADING_WARNING)
            logger.warning("LIVE real-money trading is ENABLED.")
        print(DISCLAIMER_BANNER)

    # --- read operations ---

    def get_account(self) -> AccountSnapshot:
        try:
            acct = self._client.get_account()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch account: {exc}") from exc
        return AccountSnapshot(
            account_number=str(getattr(acct, "account_number", "")),
            status=_enum_str(getattr(acct, "status", "")),
            currency=str(getattr(acct, "currency", "USD")),
            cash=_to_float(getattr(acct, "cash", None)),
            equity=_to_float(getattr(acct, "equity", None)),
            buying_power=_to_float(getattr(acct, "buying_power", None)),
            portfolio_value=_to_float(getattr(acct, "portfolio_value", None)),
            pattern_day_trader=bool(getattr(acct, "pattern_day_trader", False)),
        )

    def get_positions(self) -> list[PositionInfo]:
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc
        return [
            PositionInfo(
                symbol=str(getattr(p, "symbol", "")),
                qty=_to_float(getattr(p, "qty", None)),
                side=_enum_str(getattr(p, "side", "")),
                avg_entry_price=_to_float(getattr(p, "avg_entry_price", None)),
                current_price=_to_float(getattr(p, "current_price", None)),
                market_value=_to_float(getattr(p, "market_value", None)),
                unrealized_pl=_to_float(getattr(p, "unrealized_pl", None)),
                unrealized_plpc=_to_float(getattr(p, "unrealized_plpc", None)),
            )
            for p in positions
        ]

    # --- write operations ---

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> OrderInfo:
        """Submit a market order. ``side`` is ``"buy"`` or ``"sell"``.

        v1 is long-only: ``"sell"`` is intended to reduce or close an existing
        long position, never to open a short.
        """
        if qty <= 0:
            raise BrokerError(f"Order qty must be positive, got {qty}.")
        order_side = self._coerce_side(side)
        request = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=order_side,
            time_in_force=time_in_force,
        )
        try:
            order = self._client.submit_order(order_data=request)
        except Exception as exc:
            raise BrokerError(f"Failed to submit {side} order for {qty} {symbol}: {exc}") from exc
        logger.info("Submitted %s order: %s %s", side, qty, symbol)
        return self._to_order_info(order)

    def cancel_all_orders(self) -> None:
        try:
            self._client.cancel_orders()
        except Exception as exc:
            raise BrokerError(f"Failed to cancel open orders: {exc}") from exc
        logger.info("Cancelled all open orders.")

    def close_position(self, symbol: str) -> None:
        """Liquidate an entire position in ``symbol``."""
        try:
            self._client.close_position(symbol.upper())
        except Exception as exc:
            raise BrokerError(f"Failed to close position {symbol}: {exc}") from exc
        logger.info("Closed position: %s", symbol)

    # --- helpers ---

    @staticmethod
    def _coerce_side(side: str) -> OrderSide:
        normalized = side.strip().lower()
        if normalized in ("buy", "b"):
            return OrderSide.BUY
        if normalized in ("sell", "s"):
            return OrderSide.SELL
        raise BrokerError(f"Unknown order side {side!r}; expected 'buy' or 'sell'.")

    @staticmethod
    def _to_order_info(order: object) -> OrderInfo:
        def _enum_val(attr: str) -> str:
            return _enum_str(getattr(order, attr, ""))

        filled_price = getattr(order, "filled_avg_price", None)
        return OrderInfo(
            id=str(getattr(order, "id", "")),
            symbol=str(getattr(order, "symbol", "")),
            qty=_to_float(getattr(order, "qty", None)),
            side=_enum_val("side"),
            order_type=_enum_val("order_type") or _enum_val("type"),
            status=_enum_val("status"),
            filled_qty=_to_float(getattr(order, "filled_qty", None)),
            filled_avg_price=(_to_float(filled_price) if filled_price is not None else None),
        )
