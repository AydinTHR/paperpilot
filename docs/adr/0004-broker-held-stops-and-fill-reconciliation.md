# 4. Broker-held stops and fill reconciliation

- Status: accepted
- Date: 2026-07-02

## Context

The loop enforced stop-losses itself, once per tick. Between ticks (up to an
hour) and while the agent was down, a losing position had no protection at all.
Separately, the journal recorded orders at submit time ("accepted", no fill
price), but fills land asynchronously, and Alpaca's paper simulator
deliberately delivers random partial fills 10% of the time, so the journal's
picture of what actually happened was incomplete. Realized-P&L attribution
(Phase 5) needs actual fill prices and quantities.

## Decision

Entries go in as OTO orders (`OrderClass.OTO`): a market buy with a broker-held
protective stop attached, priced by the same pure risk core
(`RiskManager.stop_price`). The stop lives at Alpaca, protecting the position
between ticks. Alpaca's constraints are enforced at the broker boundary: whole
shares only, stop clamped at least $0.01 below the reference price, and the
live-trading double gate re-asserted per submit.

Because OTO stop legs are DAY orders that expire at the close, the loop-side
stop check remains as a backstop: it engages only when a held position has no
live stop order protecting it (e.g. overnight holds).

After each submit, an `OrderReconciler` polls the order to a terminal state
with bounded exponential backoff (injectable clock/sleeper, so tests use fake
time) and the journal records the reconciled status, `filled_qty`, and
`filled_avg_price` in one append. `Journal.update_order_fill` exists as the one
deliberate exception to the append-only doctrine, used by the optional
websocket trade-stream listener (`USE_TRADE_STREAM`, off by default) that
reconciles fills landing between ticks, such as a stop leg firing.

Rejections degrade instead of crashing: a wash-trade rejection (HTTP 403,
code 40310000) is a typed `WashTradeError`, and any entry rejection becomes a
SKIP outcome for that symbol while the tick continues. `close_position` cancels
the symbol's open orders first, since Alpaca refuses to close a position with a
live stop leg. No PDT logic is added anywhere (PDT is deprecated as of June
2026; the `pattern_day_trader` field is now defaulted and marked deprecated).

`USE_BROKER_STOPS` is unset by default and resolves to true when Alpaca
credentials are present; `USE_BROKER_STOPS=false` restores loop-only stops.

## Consequences

- Positions are protected continuously, not once per tick, and the journal
  reflects real fills, which makes FIFO trade pairing and per-strategy P&L
  (Phase 5) trustworthy.
- Each entry costs a few status polls (bounded at ~30s worst case; paper fills
  are near-instant), and each held symbol costs one open-orders check per tick.
- The stop leg and the loop backstop can in principle both fire in a fast
  market; the loop's cancel-before-close ordering makes the exit safe rather
  than doubled.
