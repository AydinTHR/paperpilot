# 1. Record architecture decisions

- Status: accepted
- Date: 2026-06-22

## Context

We want to capture the significant decisions made on PaperPilot: the ones
that are expensive or awkward to reverse, or that future contributors (including
our future selves) will want the reasoning behind. Without a record, the "why"
behind a choice is lost and gets relitigated.

## Decision

We will keep Architecture Decision Records (ADRs) in `docs/adr`, one Markdown file
per decision, numbered in sequence. We use the lightweight format introduced by
Michael Nygard. An ADR captures the context, the decision, and its consequences.
Once accepted, an ADR is immutable: to change a decision we add a new ADR that
supersedes the old one, rather than editing history.

See `docs/adr/adr-template.md` for the template.

## Consequences

- The reasoning behind structural choices is preserved and easy to find.
- New decisions cost a few minutes to write up, which is cheap next to the cost
  of forgetting why something was done.
- Reviewers can point to an ADR instead of repeating an explanation.
