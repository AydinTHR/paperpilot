# 6. Fire-and-forget alerting and a read-only dashboard

- Status: accepted
- Date: 2026-07-02

## Context

An unattended loop needs two things an operator can rely on: a push signal
when something safety-relevant happens (halt tripped, stop-loss fired, tick
crashed), and a way to see current state without ssh-ing into log files. The
`src/monitoring/` package had been an empty stub since the kickoff.

## Decision

Alerts are raw HTTP calls to Telegram (bot sendMessage) and/or Discord
(webhook), assembled by `build_alerter` from optional SecretStr settings and
defaulting to a `NullAlerter` when unconfigured. The design contract is that
`send()` **never raises**: alerts are a courtesy, not a load-bearing step, so
every failure path (timeouts, 429 with retry_after honored, dead 404 webhook,
transport exceptions) degrades to a logged False after bounded retries. The
loop fires alerts on exactly three events: a halt transition, a stop-loss
execution, and a scheduled-tick failure. Bot-framework dependencies were
deliberately avoided; `requests` is all it takes.

The dashboard is Streamlit over the journal, strictly read-only (sqlite
`mode=ro` URI, so it can never lock the journal the loop is writing). All
queries and figure-building live in `src/monitoring/queries.py` as pure
functions tested headlessly; `dashboard.py` is a thin shell and the only
module that imports streamlit, installed via the optional
`requirements-dashboard.txt` (streamlit + plotly stay out of the core and CI).

## Consequences

- Halts and failures reach the operator's phone within seconds, and the
  journal is inspectable at a glance (`streamlit run
  src/monitoring/dashboard.py`).
- A misconfigured or down channel costs at most three bounded retries per
  alert and can never crash a tick.
- The dashboard reads at a 30s cache TTL, so it lags the loop by up to half a
  minute by design.
