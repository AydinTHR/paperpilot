"""Walk-forward analysis: optimize in-sample, validate out-of-sample, roll on.

A single optimized backtest answers "what parameters WOULD have worked" -- a
curve-fit by construction. Walk-forward analysis (Pardo) is the standard
defense: optimize on an in-sample (IS) window, lock the parameters, evaluate
on the following unseen out-of-sample (OOS) window, then roll forward. The
concatenated OOS results are the honest performance estimate, and the
Walk-Forward Efficiency (annualized OOS return / annualized IS return)
quantifies how much of the optimized performance survives contact with unseen
data; below ~0.5 is conventionally read as overfit.

Two windowing modes:

* ``anchored`` (default): IS always starts at bar 0 and expands -- matching
  PaperPilot's expanding-window design and suited to regime-stable strategies.
* ``rolling``: fixed-length IS window that slides -- drops old data, suited
  when regime change is suspected.

Each OOS window is prefixed with warmup bars so indicators are primed; the
strategy HOLDs during warmup via ``min_bars``. A final hold-out slice is never
touched by any optimization. Every fold gets a FRESH risk manager (when used):
``RiskManager`` latches halts, so sharing one across folds would silently halt
later folds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from backtesting import Backtest

from config.logging_config import get_logger
from src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    _build_param_adapter,
    _coerce_float,
    run_backtest,
)
from src.strategy.base import Strategy

if TYPE_CHECKING:
    from src.risk.manager import RiskManager

logger = get_logger(__name__)

WFE_OVERFIT_THRESHOLD = 0.5


@dataclass(frozen=True)
class StrategySpec:
    """An optimizable strategy: a factory plus the parameter space to search."""

    name: str
    factory: Callable[..., Strategy]
    param_grid: dict[str, list]
    constraint: Callable[..., bool] | None = None
    defaults: dict[str, object] = field(default_factory=dict)

    def default_params(self) -> dict[str, object]:
        if self.defaults:
            return dict(self.defaults)
        return {name: values[0] for name, values in self.param_grid.items()}


@dataclass(frozen=True)
class WalkForwardConfig:
    """Windowing and optimization knobs."""

    n_folds: int = 5
    oos_fraction: float = 0.20  # of the working range, split across folds
    mode: str = "anchored"  # or "rolling"
    warmup_bars: int | None = None  # default: strategy min_bars + 10
    holdout_fraction: float = 0.10  # final untouched slice
    maximize: str = "Sharpe Ratio"
    method: str = "grid"  # or "sambo" (pip install sambo)
    max_tries: int | None = None
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if not 0 < self.oos_fraction < 1:
            raise ValueError(f"oos_fraction must be in (0, 1), got {self.oos_fraction}")
        if self.mode not in ("anchored", "rolling"):
            raise ValueError(f"mode must be 'anchored' or 'rolling', got {self.mode!r}")
        if not 0 <= self.holdout_fraction < 1:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {self.holdout_fraction}")


@dataclass(frozen=True)
class FoldResult:
    """One IS-optimize / OOS-evaluate step."""

    fold: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: dict[str, object]
    is_return_ann_pct: float
    oos_return_ann_pct: float
    is_sharpe: float
    oos_sharpe: float
    oos_result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """All folds plus the WFE verdict and the untouched hold-out evaluation."""

    spec_name: str
    mode: str
    folds: list[FoldResult]
    wfe: float
    wfe_flagged: bool  # True when WFE < WFE_OVERFIT_THRESHOLD
    holdout: BacktestResult | None


def run_walk_forward(
    data: pd.DataFrame,
    spec: StrategySpec,
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    *,
    symbol: str = "",
    interval: str = "",
    risk_factory: Callable[[], RiskManager] | None = None,
) -> WalkForwardResult:
    """Run anchored/rolling walk-forward analysis of ``spec`` over ``data``."""
    wf = wf_config or WalkForwardConfig()
    bt_config = bt_config or BacktestConfig()

    warmup = wf.warmup_bars
    if warmup is None:
        warmup = spec.factory(**spec.default_params()).min_bars + 10

    n = len(data)
    holdout_len = int(n * wf.holdout_fraction)
    working = n - holdout_len
    oos_total = int(working * wf.oos_fraction)
    oos_len = oos_total // wf.n_folds
    is_len = working - oos_total  # initial IS length (and fixed length in rolling mode)

    if oos_len < 5 or is_len <= warmup + 10:
        raise ValueError(
            f"not enough data for {wf.n_folds} folds: {n} bars -> IS {is_len}, "
            f"OOS/fold {oos_len} (warmup {warmup}). Provide more history."
        )

    folds: list[FoldResult] = []
    best_params: dict[str, object] = spec.default_params()

    for k in range(wf.n_folds):
        oos_start = is_len + k * oos_len
        oos_end = min(oos_start + oos_len, working)
        is_start = 0 if wf.mode == "anchored" else oos_start - is_len
        is_df = data.iloc[is_start:oos_start]

        best_params, is_stats = _optimize(is_df, spec, wf, bt_config, risk_factory)

        # OOS: prefix warmup bars so indicators are primed; the strategy HOLDs
        # through them (min_bars), so only genuine OOS bars produce trades.
        oos_df = data.iloc[max(oos_start - warmup, 0) : oos_end]
        oos_result = run_backtest(
            oos_df,
            spec.factory(**best_params),
            bt_config,
            symbol=symbol,
            interval=interval,
            risk=risk_factory() if risk_factory else None,
        )

        folds.append(
            FoldResult(
                fold=k + 1,
                is_start=data.index[is_start],
                is_end=data.index[oos_start - 1],
                oos_start=data.index[oos_start],
                oos_end=data.index[oos_end - 1],
                best_params=dict(best_params),
                is_return_ann_pct=_coerce_float(is_stats.get("Return (Ann.) [%]")),
                oos_return_ann_pct=_coerce_float(oos_result.stats.get("Return (Ann.) [%]")),
                is_sharpe=_coerce_float(is_stats.get("Sharpe Ratio")),
                oos_sharpe=oos_result.sharpe,
                oos_result=oos_result,
            )
        )
        logger.info(
            "Fold %d/%d: params=%s IS ann %.1f%% -> OOS ann %.1f%%",
            k + 1,
            wf.n_folds,
            best_params,
            folds[-1].is_return_ann_pct,
            folds[-1].oos_return_ann_pct,
        )

    wfe = _walk_forward_efficiency(folds)

    holdout_result: BacktestResult | None = None
    if holdout_len > 0:
        holdout_df = data.iloc[max(working - warmup, 0) :]
        holdout_result = run_backtest(
            holdout_df,
            spec.factory(**best_params),  # the most recent fold's parameters
            bt_config,
            symbol=symbol,
            interval=interval,
            risk=risk_factory() if risk_factory else None,
        )

    return WalkForwardResult(
        spec_name=spec.name,
        mode=wf.mode,
        folds=folds,
        wfe=wfe,
        wfe_flagged=wfe < WFE_OVERFIT_THRESHOLD,
        holdout=holdout_result,
    )


def _optimize(
    is_df: pd.DataFrame,
    spec: StrategySpec,
    wf: WalkForwardConfig,
    bt_config: BacktestConfig,
    risk_factory: Callable[[], RiskManager] | None,
) -> tuple[dict[str, object], pd.Series]:
    """Grid/sambo-optimize the spec on one IS window; return (params, stats)."""
    adapter = _build_param_adapter(
        spec.factory,
        is_df,
        bt_config.position_size,
        risk_factory() if risk_factory else None,
        spec.default_params(),
    )
    bt = Backtest(
        is_df,
        adapter,
        cash=bt_config.cash,
        commission=bt_config.commission,
        exclusive_orders=True,
    )
    optimize_kwargs: dict[str, object] = {
        "maximize": wf.maximize,
        "method": wf.method,
        "random_state": wf.random_state,
        **spec.param_grid,
    }
    if spec.constraint is not None:
        optimize_kwargs["constraint"] = spec.constraint
    if wf.max_tries is not None:
        optimize_kwargs["max_tries"] = wf.max_tries
    stats = bt.optimize(**optimize_kwargs)
    best = {name: getattr(stats._strategy, name) for name in spec.param_grid}
    return best, stats


def _walk_forward_efficiency(folds: list[FoldResult]) -> float:
    """Mean per-fold OOS/IS annualized-return ratio, sign-guarded.

    Folds whose IS annualized return is not meaningfully positive cannot form
    a sane ratio and are skipped; if every fold is skipped the WFE is 0.0
    (nothing survived optimization), which flags as overfit.
    """
    ratios = [
        f.oos_return_ann_pct / f.is_return_ann_pct for f in folds if f.is_return_ann_pct > 1e-9
    ]
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)
