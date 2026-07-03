# 8. LLM-vs-classical experiment harness

- Status: accepted
- Date: 2026-07-02

## Context

The LLM signal layer existed but there was no fair way to answer the question
it raises: does an LLM-gated strategy actually beat plain SMA/RSI on the same
data, costs, and risk limits? Running strategies one at a time on one account
confounds their P&L; a fair comparison needs isolation, identical inputs, and
grounded per-arm results.

## Decision

`ExperimentHarness` runs one `TradingLoop` per strategy arm in lockstep with
two isolation modes, chosen automatically. **accounts**: each arm gets its own
Alpaca paper account (Alpaca allows 3; `ALPACA_API_KEY_2/3` are first-class
SecretStr settings, and the Broker accepts credential overrides -- only the
credentials, never the paper/live gate). **virtual**: one account, each arm
wrapped in a `VirtualPortfolio` implementing the loop's `BrokerLike` Protocol
with `equity/N` virtual cash; orders forward to the real broker and
`close_position` becomes a *sized* sell, so one arm can never liquidate
another arm's shares. Same-symbol netting across virtual arms is a documented
caveat; accounts mode is preferred for publishable results.

Fairness mechanics: all arms share one market data provider (identical bars,
one cache, rate-limit friendly); each arm journals to its own database under
`data/experiments/`, so the Phase 5 trades/report readers work per-arm
unchanged; and the LLM arm's client is wrapped in a read-through response
cache keyed by (params hash, prompts) in the journal, so re-runs over the
same bars are reproducible and cost zero API dollars. The default model is
`claude-haiku-4-5` (from settings, never hard-coded); at roughly 1-2k input
tokens per symbol per daily tick the marginal cost is around a cent per day.

The comparison report reads each arm's journal: equity curve, return, max
drawdown, Sharpe (labelled insufficient below 20 ticks rather than printing
noise), and realized P&L / win rate from the FIFO trades table.

## Consequences

- `python scripts/run_experiment.py --interval 60` runs the comparison
  unattended for weeks; `--report` prints a per-arm table grounded in actual
  fills, suitable for publishing.
- The virtual mode makes the experiment available with a single account at
  the cost of shared-account artifacts; the caveat is documented in the
  module and the report header names the mode used.
- One more journal file per arm under `data/experiments/` (gitignored).
