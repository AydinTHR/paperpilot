# 5. Persist halt state across restarts

- Status: accepted
- Date: 2026-07-02

## Context

The max-drawdown halt is deliberately sticky: a deep drawdown means something
is wrong, so trading stops until an operator reviews and resets it. But the
latch lived only in process memory. A crash or an innocent restart silently
re-armed the agent, defeating the whole point of a sticky halt. The equity
peak was already restored from the journal on startup; the halt flags were not.

## Decision

Halt transitions are journaled in an append-only `halt_state` table (one row
per trip or clear; the latest row per type is authoritative). The risk core
stays pure: it takes an optional `HaltStore` Protocol (which the Journal
satisfies structurally, with no import in either direction), calls it
best-effort on every trip and reset, and never lets a store failure affect
risk logic. A pure `restore()` method re-latches flags on startup.

`TradingLoop.from_settings` restores the drawdown halt unconditionally while
it is active; the daily-loss trip is restored only when it fired on the current
calendar day, since it belongs to that day by design. Restoring the daily trip
also restores its calendar day, otherwise the next equity update would treat
it as a new day and clear it.

`run_live.py --reset-halt` is the operator's reset: journal-only (no broker or
network), prints the prior state, and records both types as cleared.

## Consequences

- A restart can no longer silently resume trading after a drawdown halt; the
  first tick after a restored halt flattens positions and refuses entries,
  exactly as if the process had never died.
- Resetting a halt is an explicit, journaled operator action with an audit
  trail, rather than "restart the process".
- The halt table adds one small write per trip/reset; the read cost is one
  query at startup.
