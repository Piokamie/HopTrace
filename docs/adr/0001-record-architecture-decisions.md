---
status: accepted
date: 2026-08-13
---

# 1. Record Architecture Decisions

## Context

We need to record the architectural decisions made on this project so that
future developers (and our future selves) understand why things are the way
they are.

## Decision

We will use Architecture Decision Records (ADRs) as described by Michael
Nygard. Each ADR is a short Markdown file in `docs/adr/`. ADRs are numbered
sequentially and are immutable once accepted — changed decisions get a new
ADR that supersedes the old one.

## Consequences

- Decisions are discoverable and reviewable in version control alongside the
  code they affect.
- New readers can follow the decision log to understand the project's
  architectural evolution.
- Writing an ADR forces explicit consideration of context, alternatives, and
  trade-offs before committing to a choice.
