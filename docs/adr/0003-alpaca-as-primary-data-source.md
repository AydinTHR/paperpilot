# 3. Alpaca as the primary market data source

- Status: accepted
- Date: 2026-07-02

## Context

PaperPilot fetched bars from yfinance while executing orders through Alpaca. Two
different sources means the price a signal saw and the price an order filled at
could drift apart, and yfinance is an unofficial API with no service guarantee.
Alpaca's Basic (free) plan includes the IEX real-time feed with the same SDK the
broker already uses, and allows 200 requests per minute per key.

## Decision

We will fetch market data from Alpaca by default whenever Alpaca credentials are
configured, keeping yfinance as the fallback and as a one-line rollback
(`DATA_PROVIDER=yfinance`). The provider is selected by a `build_provider`
factory behind the existing `MarketDataProvider` Protocol, so no caller changes.
Cache files are provider-suffixed (`{SYMBOL}_{interval}_alpaca.parquet`) because
IEX bars with `Adjustment.ALL` are not byte-identical to yfinance's
auto-adjusted bars, and the two providers must never serve each other's frames.
A cache miss fetches the whole trading universe in one batched request to stay
far inside the rate limit.

The scheduled loop also gains a market-hours gate: session times come offline
from `pandas_market_calendars` (NYSE calendar, including holidays and early
closes), optionally confirmed by Alpaca's live clock. `run_once` and `--once`
are never gated so demos and tests work after hours;
`--ignore-market-hours` disables the gate for scheduled runs.

## Consequences

- Signals and executions read the same source, removing data drift, and ticks
  no longer burn API calls or log noise while the market is closed.
- IEX is a single-venue feed: volume numbers are thinner than consolidated SIP
  data, and free-tier intraday queries cannot touch the most recent ~15 minutes
  (the provider clamps hourly requests accordingly). Users who need SIP can set
  `ALPACA_DATA_FEED=sip` with a paid subscription.
- Anyone who ran the agent before this change switches sources automatically;
  the change is loud in the README and reversible with one setting.
