---
status: accepted
date: 2026-08-21
---

# 13. Rename HopTrace to HopPath

## Context

Immediately before first publication, a name-collision check found
"HopTrace" in active use by an unrelated product: a visual-traceroute and
latency-mapping application for Android, publicly listed and documented.
Sharing a name with a networking diagnostic tool would misdirect search
traffic in both directions and read as affiliation where none exists.

Candidate replacements were checked against PyPI, GitHub and general web
search. "HopPath" has no PyPI package, no repository of that name, and
no product presence; it also names the two things the tool actually
returns — hops, and the recorded path.

## Decision

The project, package, CLI, environment variables and data directory are
renamed: `hoppath`, `HOPPATH_DATA_DIR`, `HOPPATH_RERANK_MODEL`,
`./.hoppath/`, model artifact `hoppath-rerank-minilm-l6`, repository
`Piokamie/HopPath`. ADRs 0001–0012 keep their historical
"LRAG"/"HopTrace" wording (the convention set by ADR 0009); this ADR is
the bridge. Git history is not rewritten: the trained model manifests
pin the trainer commit by hash, and a rewrite would orphan them.

## Consequences

- The rename lands before the first release, so no released artifact,
  URL or package name ever carried the colliding name.
- The bundled model manifest's `model` field is updated to the new
  artifact name and its registry checksum re-pinned; the graph and
  tokenizer files (and their pins) are unchanged.
- Readers of the decision log encounter three names; ADRs 0009 and 0013
  mark the two renames.
- The GitHub repository rename leaves a redirect from the old URL.
