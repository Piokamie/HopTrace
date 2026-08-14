---
status: superseded by 0006
date: 2026-08-13
---

> Superseded by [0006. Per-Ring Interleave, Stratified Reporting, and the Displacement Audit](0006-per-ring-interleave-and-stratified-reporting.md).

# 5. Score Hop Candidates by Path Propagation, Not Query Similarity

## Context

The first deterministic scorer ranked every candidate on one scale:
`final = bm25_norm + w_hop · decay^hop · path_strength`. Measured on
MuSiQue (pooled dev, 2,417 questions, budget-matched): hop expansion put
85.9% of gold evidence into the candidate pool, but recall@20 reached only
0.61, with 39.6% of all misses being ranking misses — gold chunks present
in the pool and scored below k. Adding a second hop changed recall by
roughly nothing (recall@10 0.5297 → 0.5279) despite MuSiQue questions
being 2–4-hop by construction.

The cause is structural: a genuinely multi-hop target chunk shares no
lexical material with the query *by construction* — that is what makes it
multi-hop — so its BM25 component is ~0 and it competes against
query-relevant seeds on a scale it can never score on. The additive
formula prices hop-2 evidence as "irrelevant chunk plus a small bonus."

## Decision

Two populations, two scales, no shared formula:

1. **Seeds (hop 0)** are scored by query relevance: BM25 ratio-normalized
   to the best seed; mention seeds take the stronger of that and their
   entity-match strength.
2. **Hop-reached chunks** are scored by propagation:
   `score(child) = score(parent) × bridge_strength(edge)`, where
   `bridge_strength = entity_specificity × count/(count+1)` is a property
   of the traversed edge — the bridge entity's IDF (which also encodes
   link uniqueness: an entity connecting two chunks scores higher than
   one connecting two hundred) and a saturating mention count. The
   parent's query relevance flows through the bridge instead of being
   recomputed against a chunk that cannot have it.
3. **Ranking interleaves** the two populations (alternating positions,
   seeds first, each population ordered by its own score, deterministic
   tie-breaks) rather than mapping both onto one scale.

Consequently hop decay is emergent (a product of bridge strengths < 1)
and the tuned `hop_decay`/`w_hop` constants are removed. Expansion also
uses the propagated score for frontier survival, so beam pruning and
ranking optimize the same quantity.

Alternatives rejected:

- **Keep the additive formula, tune weights**: no weight setting fixes a
  structurally zero feature; tuning against the eval sets would also be
  test-set fitting.
- **Score hop chunks by BM25 against an expanded query** (query + bridge
  entities): reintroduces lexical scoring of chunks selected precisely
  for non-lexical reachability, and blurs provenance.
- **Defer to the v2 reranker**: the learned ranker must beat the best
  deterministic baseline; crediting it for fixing what a multiplication
  fixes would overstate the case for learning.

## Consequences

- Hop-2 candidates are ranked by evidence that actually exists for them
  (path quality), and the v2 reranker gets a fair baseline.
- Interleaving guarantees hop candidates ranked positions even on corpora
  where the floor is sufficient (e.g. shortcut-prone datasets); if the
  hop population is junk there, alternation costs floor positions — this
  shows up in the calibration datasets and is reported, not hidden.
- One fewer pair of magic constants; bridge strength is inspectable per
  edge in `explain`.
- All Phase 3 hop measurements taken under the additive scorer are
  superseded; they are kept in the results archive as the scorer ablation
  ("additive" vs "propagated").
