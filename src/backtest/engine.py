"""Event-driven backtesting on top of ``backtesting.py``.

The whole point of the Phase 2 strategy interface lands here: a thin adapter
wraps any PaperPilot :class:`~src.strategy.base.Strategy` as a
``backtesting.Strategy`` and, on every bar, feeds the *expanding window up to
that bar* into ``generate_signals`` -- so the exact code that will run live also
drives the backtest, with no look-ahead.

Costs are modelled two ways: a per-trade ``commission`` (fees) and ``slippage``
applied as a bid-ask ``spread`` on every fill (folded into commission on older
``backtesting.py`` builds that lack the ``spread`` argument). The result is a
structured :class:`BacktestResult` with headline metrics and the equity curve.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from backtesting import Backtest
from backtesting import Strategy as BtStrategy

from config.logging_config import get_logger
from src.strategy.base import Action, Strategy

if TYPE_CHECKING:
    from src.risk.manager import RiskManager

logger = get_logger(__name__)

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


@dataclass(frozen=True)
class BacktestConfig:
    """Run-time backtest knobs (not environment config)."""

    cash: float = 10_000.0
    commission: float = 0.001  # 0.1% per trade (fees)
    slippage: float = 0.0005  # 0.05% bid-ask spread per fill
    position_size: float = 0.95  # equity fraction per entry when no size_hint
    results_dir: str = "backtest_results"

    def __post_init__(self) -> None:
        if self.cash <= 0:
            raise ValueError(f"cash must be positive, got {self.cash}")
        if not 0 <= self.commission < 1:
            raise ValueError(f"commission must be in [0, 1), got {self.commission}")
        if not 0 <= self.slippage < 1:
            raise ValueError(f"slippage must be in [0, 1), got {self.slippage}")
        if not 0 < self.position_size < 1:
            raise ValueError(f"position_size must be in (0, 1), got {self.position_size}")


@dataclass(frozen=True)
class BacktestResult:
    """Headline metrics + equity curve from a single backtest run."""

    symbol: str
    strategy: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp
    return_pct: float
    buy_hold_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    exposure_pct: float
    final_equity: float
    equity_curve: pd.Series
    stats: pd.Series

    def summary(self) -> dict[str, object]:
        """A flat, log-friendly view of the headline numbers."""
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "interval": self.interval,
            "period": f"{self.start.date()}..{self.end.date()}",
            "return_pct": round(self.return_pct, 2),
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
            "sharpe": round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "num_trades": self.num_trades,
            "exposure_pct": round(self.exposure_pct, 2),
            "final_equity": round(self.final_equity, 2),
        }


def _step(
    bt_self: BtStrategy,
    pp_strategy: Strategy,
    full_df: pd.DataFrame,
    default_size: float,
    risk: RiskManager | None,
) -> None:
    """One bar of the adapter loop, shared by both adapter flavours.

    Long-only (v1): BUY opens a position when flat, SELL closes it, HOLD does
    nothing. The window handed to ``generate_signals`` is sliced to the current
    bar count, which guarantees no look-ahead regardless of engine internals.
    """
    price = float(bt_self.data.Close[-1])

    if risk is not None:
        risk.update_equity(bt_self.equity, now=bt_self.data.index[-1])
        if risk.halted:
            if bt_self.position:
                bt_self.position.close()
            return

    window = full_df.iloc[: len(bt_self.data)]
    signal = pp_strategy.generate_signals(window)

    if signal.action is Action.BUY and not bt_self.position.is_long:
        _enter(bt_self, price, signal.size_hint, default_size, risk)
    elif signal.action is Action.SELL and bt_self.position:
        bt_self.position.close()
    # Action.HOLD -> do nothing


def _enter(
    bt_self: BtStrategy,
    price: float,
    size_hint: float | None,
    default_size: float,
    risk: RiskManager | None,
) -> None:
    if risk is None:
        frac = size_hint if size_hint is not None else default_size
        bt_self.buy(size=max(min(frac, 0.99), 0.001))
        return

    # When flat, equity is effectively all cash, so it doubles as the
    # cash bound for sizing.
    qty = risk.position_size(bt_self.equity, price, bt_self.equity, size_hint)
    if qty < 1:
        return
    if risk.limits.stop_loss_pct > 0:
        bt_self.buy(size=qty, sl=risk.stop_price(price))
    else:
        bt_self.buy(size=qty)


def _build_adapter(
    pp_strategy: Strategy,
    full_df: pd.DataFrame,
    default_size: float,
    risk: RiskManager | None = None,
) -> type[BtStrategy]:
    """Create a ``backtesting.Strategy`` subclass that delegates to ``pp_strategy``.

    When a :class:`~src.risk.manager.RiskManager` is supplied, it governs every
    entry: it updates its equity statistics each bar, force-closes the position
    and blocks new entries while halted, sizes entries by ``max_position_pct``,
    and attaches a protective stop-loss order at entry.
    """

    class _Adapter(BtStrategy):
        def init(self) -> None:
            pass

        def next(self) -> None:
            _step(self, pp_strategy, full_df, default_size, risk)

    _Adapter.__name__ = f"Adapter[{pp_strategy.name}]"
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


def _build_param_adapter(
    factory: Callable[..., Strategy],
    full_df: pd.DataFrame,
    default_size: float,
    risk: RiskManager | None,
    params: dict[str, object],
) -> type[BtStrategy]:
    """An adapter whose strategy is rebuilt from class-attribute params.

    ``Backtest.optimize`` varies parameters by mutating class attributes, so
    each declared param lives on the class and ``init()`` constructs a fresh
    PaperPilot strategy from their current values. The exact live
    ``generate_signals`` code still runs each bar via the same expanding-window
    slice, preserving the no-look-ahead guarantee.
    """
    param_names = tuple(params)

    class _ParamAdapter(BtStrategy):
        def init(self) -> None:
            values = {name: getattr(self, name) for name in param_names}
            self._pp_strategy = factory(**values)

        def next(self) -> None:
            _step(self, self._pp_strategy, full_df, default_size, risk)

    for name, default in params.items():
        setattr(_ParamAdapter, name, default)
    _ParamAdapter.__name__ = "ParamAdapter"
    _ParamAdapter.__qualname__ = _ParamAdapter.__name__
    return _ParamAdapter


def _coerce_float(value: object, default: float = 0.0) -> float:
    """backtesting.py reports NaN for some stats (e.g. Sharpe with 0 trades)."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    *,
    symbol: str = "",
    interval: str = "",
    risk: RiskManager | None = None,
) -> BacktestResult:
    """Backtest ``strategy`` over ``data`` (a normalised OHLCV frame).

    Pass a :class:`~src.risk.manager.RiskManager` to enforce position sizing,
    per-trade stops, and the daily/drawdown halts during the run.
    """
    config = config or BacktestConfig()

    if data.empty:
        raise ValueError("cannot backtest on an empty frame.")
    missing = [c for c in _REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"data missing required column(s) {missing}.")

    adapter = _build_adapter(strategy, data, config.position_size, risk)

    bt_kwargs: dict[str, object] = {
        "cash": config.cash,
        "commission": config.commission,
        "exclusive_orders": True,
    }
    # Model slippage as a bid-ask spread where supported; otherwise fold it into
    # the commission so older backtesting.py builds still account for it.
    if "spread" in inspect.signature(Backtest.__init__).parameters:
        bt_kwargs["spread"] = config.slippage
    elif config.slippage:
        bt_kwargs["commission"] = config.commission + config.slippage
        logger.info(
            "backtesting.py lacks 'spread'; folding slippage into commission (%.4f + %.4f).",
            config.commission,
            config.slippage,
        )

    bt = Backtest(data, adapter, **bt_kwargs)
    stats = bt.run()

    equity_curve = stats["_equity_curve"]["Equity"].copy()
    result = BacktestResult(
        symbol=symbol,
        strategy=strategy.name,
        interval=interval,
        start=data.index[0],
        end=data.index[-1],
        return_pct=_coerce_float(stats.get("Return [%]")),
        buy_hold_return_pct=_coerce_float(stats.get("Buy & Hold Return [%]")),
        sharpe=_coerce_float(stats.get("Sharpe Ratio")),
        max_drawdown_pct=_coerce_float(stats.get("Max. Drawdown [%]")),
        win_rate_pct=_coerce_float(stats.get("Win Rate [%]")),
        num_trades=int(_coerce_float(stats.get("# Trades"))),
        exposure_pct=_coerce_float(stats.get("Exposure Time [%]")),
        final_equity=_coerce_float(stats.get("Equity Final [$]"), config.cash),
        equity_curve=equity_curve,
        stats=stats,
    )
    logger.info("Backtest complete: %s", result.summary())
    return result
