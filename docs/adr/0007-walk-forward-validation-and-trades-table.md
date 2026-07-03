# 7. Walk-forward validation and a realized-trades table

- Status: accepted
- Date: 2026-07-02

## Context

A single optimized backtest is a curve-fit by construction: it answers "what
parameters WOULD have worked", not "what will work". PaperPilot had no way to
optimize parameters at all, let alone honestly. Separately, the journal logged
orders but never paired them into realized trades, so "which strategy actually
made money" had no grounded answer.

## Decision

**Walk-forward analysis** (`src/backtest/walkforward.py`), hand-rolled around
the existing expanding-window adapter per Pardo's method: optimize on an
in-sample window (`Backtest.optimize`; grid by default, sambo optional via
`requirements-research.txt`), lock parameters, evaluate on the next unseen
out-of-sample window, roll forward. Anchored (expanding, the default, matching
PaperPilot's design) and rolling modes. Each OOS window is prefixed with
warmup bars (the strategy HOLDs through them via ``min_bars``); a final
hold-out slice is never touched by optimization; each fold gets a fresh
RiskManager because halts latch. The Walk-Forward Efficiency (mean OOS/IS
annualized-return ratio, sign-guarded) is reported and flagged below 0.5 as
overfit. Parameter optimization required a second adapter flavour
(`_build_param_adapter`): parameters live as class attributes (the surface
`Backtest.optimize` mutates) and each run constructs a fresh strategy from
them, so the exact live `generate_signals` code still drives every bar.

**Realized trades** (`src/journal/trades.py` + `trades` table): a pure FIFO
pairer walks the order fills per (symbol, strategy) -- each SELL consumes the
oldest BUY lots first, splitting lots on partial exits, using actual reconciled
fill prices/quantities. `Journal.rebuild_trades()` is idempotent
(delete-and-rebuild) and `strategy_report()` aggregates win rate, average
win/loss, profit factor, total P&L, and holding period per strategy. Orders
now carry a `strategy` tag (with a sqlite mini-migration for existing files).
CLI: `scripts/run_walkforward.py` and `run_live.py --trades-report`.

## Consequences

- Parameter choices can be validated honestly; the first live run on 5 years
  of AAPL immediately flagged the optimized SMA crossover as overfit
  (WFE -0.72), which is the tool working as intended.
- Strategy comparison (Phase 6) gets grounded, per-strategy realized P&L from
  actual fills rather than signal-time estimates.
- Grid sizes multiply across folds; the default grids are deliberately small,
  and `sambo` is available for larger spaces.
