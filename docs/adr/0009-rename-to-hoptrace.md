---
status: accepted
date: 2026-08-14
---

# 9. Rename LRAG to HopTrace

## Context

The project shipped v1 under the name LRAG ("Learned Retrieval-Augmented
Generation"). Both halves of that expansion are misleading. There is no
augmented-generation component and never will be: the engine retrieves
and returns evidence with provenance; generation is explicitly the
external model's job (a founding non-goal). And the learned component is
a v2 ranking layer — measured to be necessary (the selection trilemma,
ADR 0008) but not yet shipped — which forced a release gate: the repo
could not honestly publish under a name promising learning it did not
contain.

## Decision

The project is named **HopTrace**: hop retrieval, with the trace
attached — the two things every retrieval actually returns. The rename is
total for living surfaces: package (`hoptrace`), CLI (`hoptrace`), MCP
server name, env var (`HOPTRACE_DATA_DIR`), default data dir
(`.hoptrace/`), system labels in reports (`hoptrace@1hop`), and all
prose documentation. Accepted ADRs keep their historical "LRAG" wording —
they are immutable records of decisions made under the old name. The
PyPI name `hoptrace` was verified unclaimed at decision time.

Consequence for the release gate: it dissolves. The gate existed solely
because the old name promised a learned component; HopTrace promises hop
retrieval with traces, which v1 delivers in full. Publication timing
becomes a product choice, not a naming obligation.

Alternatives rejected:

- **Keep LRAG, publish after v2** (the previous decision): ties
  publication of finished, externally-validated work to an unshipped
  component, purely to make an acronym truthful.
- **Keep LRAG, reposition the expansion**: any backronym for "RAG" still
  claims a generation component the non-goals explicitly disclaim.

## Consequences

- The name states the differentiator (the trace) instead of overclaiming
  the category (RAG) — every retrieve result is a small advertisement for
  the name.
- v1 is publishable on its own merits whenever its owner chooses.
- Historical artifacts (ADRs 0001–0008, archived eval reports, the plan
  file) retain "LRAG"/"lrag" wording and labels; readers of the decision
  log will encounter both names, with this ADR as the bridge.
- The working-directory name on the author's machine may lag the rename;
  the repository name at publication time should be `hoptrace`.
