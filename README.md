# PaperPilot

[![CI](https://github.com/AydinTHR/paperpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/AydinTHR/paperpilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org)

Autonomous, risk-controlled paper-trading agent for backtesting strategies and running a
simulated live loop against Alpaca.

> **Disclaimer:** PaperPilot is an educational project. It defaults to simulated (paper)
> trading only. Nothing here is financial advice, and past or simulated performance does
> not predict future results. Real-money trading is gated off by default and requires two
> explicit opt-ins (see [Safety](#safety-and-risk-controls)). Use at your own risk.

## Why this exists

Most trading tutorials stop at a backtest that looks great and ignores costs, risk, and
the gap between a notebook and a process that runs unattended. PaperPilot is the opposite:
a small, honest end-to-end agent that fetches data, evaluates a strategy, sizes positions
against hard risk limits, places simulated orders through a broker, journals every fill,
and can run on a schedule. It is built for learning how the pieces fit together, not for
making money.

## Features

- **Config and broker connection.** Typed settings via `pydantic-settings`, a safe Alpaca
  paper connection, and a one-command account check.
- **Market data and strategies.** Cached OHLCV bars (local parquet), reusable indicators,
  and a small strategy interface with SMA-crossover and RSI mean-reversion examples.
- **Event-driven backtesting.** Backtests that account for commission and slippage, with
  results written to CSV.
- **Risk management and kill switch.** Per-position cap, a daily-loss kill switch, a sticky
  max-drawdown halt, and per-trade stop-loss distance.
- **Scheduled live paper loop.** A blocking scheduler runs the strategy on an interval and
  records every decision and fill to a SQLite trade journal.
- **Optional LLM signal layer.** A provider-agnostic adapter (Claude wired) that can supply
  a signal, risk-gated and disabled unless an API key is set.

## Quickstart

```bash
git clone https://github.com/AydinTHR/paperpilot.git
cd paperpilot

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env      # then fill in your Alpaca PAPER keys
```

Get free paper-trading keys from the [Alpaca dashboard](https://app.alpaca.markets) and
paste them into `.env`. The file is gitignored; never commit real keys.

## Usage

```bash
# Verify the config and Alpaca paper connection (prints balance + positions)
python scripts/run_agent.py --check

# Backtest a strategy on historical bars (costs and slippage included)
python scripts/run_backtest.py --symbol AAPL --strategy sma --lookback 500

# Preview the current signal each strategy would emit for a symbol
python scripts/preview_signals.py --symbol AAPL

# Run one iteration of the live paper loop, then exit
python scripts/run_live.py --once --strategy sma

# Run the loop continuously, every 60 minutes
python scripts/run_live.py --interval 60

# Read the last 20 rows of the trade journal (read-only) and exit
python scripts/run_live.py --report
```

## Architecture

A layered `src/` package keeps each concern testable in isolation:

```
src/
  data/        market data fetch + local cache, trading universe
  strategy/    strategy interface, indicators, SMA/RSI examples, optional LLM layer
  backtest/    event-driven engine with costs and slippage
  execution/   Alpaca broker adapter (paper by default)
  risk/        position sizing, kill switch, drawdown + stop-loss limits
  journal/     SQLAlchemy models + SQLite store for trades
  agent/       the loop that ties data, strategy, risk, execution, and journal together
config/        typed settings (pydantic-settings) and logging
scripts/       command-line entry points (check, backtest, preview, live)
```

Design decisions are recorded under [docs/adr](./docs/adr).

## Safety and risk controls

PaperPilot is paper-first by design. Real-money trading is impossible unless **both**
`PAPER=false` and `ALLOW_LIVE_TRADING=true` are set, and even then a warning banner prints
on startup. The risk layer enforces, on every cycle:

- a maximum fraction of equity per single position (`MAX_POSITION_PCT`),
- a daily-loss kill switch that halts new trades (`MAX_DAILY_LOSS_PCT`),
- a sticky max-drawdown halt measured from the equity peak (`MAX_DRAWDOWN_PCT`),
- a per-trade stop-loss distance below entry (`STOP_LOSS_PCT`).

All of these are configurable in `.env`. See [.env.example](./.env.example) for the full
list with defaults.

## Development

```bash
pre-commit install                       # enables the local checks and commit-msg hook
pre-commit run --all-files               # lint, format, and the house-rule checks
pytest                                   # run the test suite (network tests deselected)
```

- Branching: short-lived feature branches off `main`, merged via pull request.
- Commits: [Conventional Commits](https://www.conventionalcommits.org).
- Tests and lint run in CI on every pull request and must pass before merge.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). By participating you agree to the
[Code of Conduct](./CODE_OF_CONDUCT.md).

## License

[MIT](./LICENSE) (c) 2026 Aydin.
