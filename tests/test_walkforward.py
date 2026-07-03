"""Walk-forward analysis tests -- synthetic frames, tiny grids, fully offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig
from src.backtest.walkforward import (
    FoldResult,
    StrategySpec,
    WalkForwardConfig,
    run_walk_forward,
)
from src.strategy.examples.sma_crossover import SmaCrossover


def _wavy_frame(n: int = 420) -> pd.DataFrame:
    """A noisy sine wave: cyclic enough for SMA crossovers to trade."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 + 12 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.5, n).cumsum() * 0.2
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _sma_spec() -> StrategySpec:
    return StrategySpec(
        name="sma",
        factory=lambda fast, slow: SmaCrossover(fast=fast, slow=slow),
        param_grid={"fast": [5, 10], "slow": [20, 30]},
        constraint=lambda p: p.fast < p.slow,
    )


def _config(**kw) -> WalkForwardConfig:
    defaults = {"n_folds": 3, "oos_fraction": 0.3, "holdout_fraction": 0.1}
    defaults.update(kw)
    return WalkForwardConfig(**defaults)


@pytest.fixture(scope="module")
def anchored_result():
    return run_walk_forward(
        _wavy_frame(),
        _sma_spec(),
        _config(),
        BacktestConfig(cash=100_000.0),
        symbol="TEST",
    )


def test_produces_all_folds_and_holdout(anchored_result) -> None:
    assert len(anchored_result.folds) == 3
    assert anchored_result.holdout is not None
    assert anchored_result.mode == "anchored"


def test_anchored_is_windows_expand_from_bar_zero(anchored_result) -> None:
    first_bar = anchored_result.folds[0].is_start
    for fold in anchored_result.folds:
        assert fold.is_start == first_bar  # anchored: IS always starts at bar 0
    ends = [f.is_end for f in anchored_result.folds]
    assert ends == sorted(ends)  # and expands


def test_oos_never_overlaps_is(anchored_result) -> None:
    for fold in anchored_result.folds:
        assert fold.oos_start > fold.is_end  # OOS strictly after its IS window


def test_oos_windows_are_disjoint_and_ordered(anchored_result) -> None:
    for prev, cur in zip(anchored_result.folds, anchored_result.folds[1:], strict=False):
        assert cur.oos_start > prev.oos_end


def test_best_params_respect_grid_and_constraint(anchored_result) -> None:
    for fold in anchored_result.folds:
        params = fold.best_params
        assert params["fast"] in (5, 10)
        assert params["slow"] in (20, 30)
        assert params["fast"] < params["slow"]


def test_rolling_is_windows_have_fixed_length() -> None:
    result = run_walk_forward(
        _wavy_frame(),
        _sma_spec(),
        _config(mode="rolling"),
        BacktestConfig(cash=100_000.0),
    )
    spans = [(f.is_end - f.is_start) for f in result.folds]
    # Constant bar-count IS windows; calendar spans may differ by a weekend.
    assert max(spans) - min(spans) <= pd.Timedelta(days=5)
    starts = [f.is_start for f in result.folds]
    assert starts == sorted(starts)
    assert starts[0] != starts[-1]  # the window slides (anchored would pin it)


def test_wfe_math_and_flag() -> None:
    def _fold(k: int, is_ann: float, oos_ann: float) -> FoldResult:
        ts = pd.Timestamp("2023-01-01")
        return FoldResult(
            fold=k,
            is_start=ts,
            is_end=ts,
            oos_start=ts,
            oos_end=ts,
            best_params={},
            is_return_ann_pct=is_ann,
            oos_return_ann_pct=oos_ann,
            is_sharpe=0.0,
            oos_sharpe=0.0,
            oos_result=None,  # type: ignore[arg-type]
        )

    from src.backtest.walkforward import _walk_forward_efficiency

    # 10% IS -> 6% OOS and 20% IS -> 16% OOS: mean(0.6, 0.8) = 0.7 (healthy)
    assert _walk_forward_efficiency([_fold(1, 10.0, 6.0), _fold(2, 20.0, 16.0)]) == pytest.approx(
        0.7
    )
    # Negative-IS folds are skipped; nothing left -> 0.0 (flags as overfit)
    assert _walk_forward_efficiency([_fold(1, -5.0, 3.0)]) == 0.0


def test_insufficient_data_raises() -> None:
    with pytest.raises(ValueError, match="not enough data"):
        run_walk_forward(_wavy_frame(50), _sma_spec(), _config())


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        WalkForwardConfig(n_folds=1)
    with pytest.raises(ValueError, match="mode"):
        WalkForwardConfig(mode="sideways")
