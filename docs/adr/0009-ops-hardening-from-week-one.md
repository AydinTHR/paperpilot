# 9. Ops hardening from week-one findings

- Status: accepted
- Date: 2026-07-11

## Context

Week one of the live experiment surfaced four gaps. The laptop-bound process
slept through most market hours, so (1) the position's DAY stop leg expired at
Monday's close and the position traded unprotected for two days, and (2) the
journal never learned the entry's real fill, because the submit-time
reconciliation window had closed days earlier and nothing ever looked again.
Code review during planning found (3) exits (signal sells, stop closes, halt
flattens) were journaled with no broker order id at all, so they could never be
reconciled and FIFO trade pairing would silently miss every exit, and (4) the
optional fill stream could not be pointed at the experiment's per-arm accounts.

## Decision

Four changes, one theme: the journal converges to broker truth no matter when
fills land.

- **Exits carry identities.** `Broker.close_position` returns the closing
  order Alpaca creates; all three exit sites journal its id and reconcile it
  with the same bounded poll entries use (one shared `_close_and_journal`).
- **A per-tick reconciliation sweep.** Each `run_once` re-checks journaled
  orders whose status is not terminal (`Journal.unreconciled_orders`) against
  the broker and back-fills status/fill in place. Downtime no longer loses
  fills; the week-one stale row repairs itself on the first tick after deploy.
- **Stop legs are GTC.** OTO protective stops persist across sessions instead
  of expiring at the close. The loop-side stop check remains as a backstop.
- **Per-arm fill streams.** `TradeStreamListener` accepts credential overrides
  (same pattern as `Broker`), and the experiment harness starts one listener
  per arm in accounts mode when `USE_TRADE_STREAM=true`.

A `scripts/weekly_report.py` (journal-only, broker-free) sends the per-arm
comparison through the existing alerter, for a weekly cron on the deployment
box.

## Consequences

- Realized-P&L attribution can trust the orders table: every entry AND exit
  converges to its real fill price and quantity.
- Positions stay broker-protected overnight and across agent downtime.
- Each tick spends a few extra API reads (the sweep, bounded at 20 rows) and
  accounts-mode scheduled runs hold up to three websocket connections.
- `close_position` changed its return type from None to `OrderInfo | None`;
  callers that ignored the return value are unaffected.
