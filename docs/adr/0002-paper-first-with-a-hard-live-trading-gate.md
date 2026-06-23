# 2. Paper-first with a hard live-trading gate

- Status: accepted
- Date: 2026-06-22

## Context

PaperPilot can place real orders through the same broker adapter it uses for simulated
ones. An autonomous loop that trades unattended is exactly the kind of program where a
single misconfiguration (a stray environment variable, a copied production key) can move
real money. The project is educational, so the default posture must make accidental
real-money trading effectively impossible, while still leaving a deliberate path for a
user who genuinely wants it.

## Decision

We will default to paper trading and require two independent opt-ins before any real order
is possible:

1. `PAPER=false` selects the live endpoint, and
2. `ALLOW_LIVE_TRADING=true` arms the gate.

Either one alone keeps the agent in paper mode. When both are set, startup prints a loud
warning banner so the mode is never a surprise. The risk layer (position cap, daily-loss
kill switch, drawdown halt, stop-loss) runs in both modes, so the same controls protect a
paper run and a live run.

## Consequences

- The safe path is the default path: cloning, configuring paper keys, and running can
  never reach the live endpoint without a deliberate, two-step change.
- Any pull request that touches the gate is security-sensitive and should be reviewed as
  such (see `SECURITY.md`).
- The two-flag design adds a small amount of branching in config and a test that the gate
  holds. That cost is trivial next to the downside it prevents.
