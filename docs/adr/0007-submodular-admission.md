---
status: superseded by 0008
date: 2026-08-14
---

> Superseded by [0008. Submodular Admission Demoted to Experimental; Interleave Restored as Default](0008-submodular-demoted-to-experimental.md).

# 7. Greedy Submodular Admission Replaces Fixed Interleave Quotas

## Context

The displacement audit's first reading (MuSiQue, 2,417 questions, k=20)
showed the fixed 1:1 interleave is break-even in raw gold (hop-held slots
carried 1.7% gold; the seeds they evicted carried 1.8%; net −17) while
tripling all-gold@20 in the effectively-multi-hop stratum. The
reconciliation: slot value is not marginal gold, it is **marginal
coverage** — evicted gold seeds were mostly redundant copies of evidence
already selected, admitted hop gold was the missing complement. That
quantity is submodular, and greedy selection over a submodular objective
is a fifty-year-old, guarantee-carrying method (facility-location
coverage; MMR, Carbonell & Goldstein 1998; greedy (1−1/e) bound, Nemhauser
et al. 1978).

Separately, ADR 0006's fixed ring-2 reservation cost aggregate recall on a
corpus where ring 2 rarely pays (ag@20 0.279 vs 0.300) — the expected
price of any fixed quota.

## Decision

This supersedes ADR 0006. Unchanged from 0005/0006: propagated path
scoring (seeds carry query relevance; hop chunks carry
parent × bridge_strength; decay emergent; frontier pruning by propagated
score), stratified reporting wherever diagnostics run, and the
displacement audit as an instrument.

What changes — selection:

1. **Greedy marginal-coverage admission, no quotas.** The top-k list is
   built greedily; each slot goes to the candidate maximizing
   `gain(c | S) = relevance(c) × novelty(c | S)`, where `relevance` is the
   candidate's population-native score and
   `novelty(c | S) = |F(c) \ cov(S)| / |F(c)|` over coverage features
   `F(c)` = query terms present in the chunk ∪ bridge entities on the
   chunk's hop path. Redundant candidates collapse (their features are
   covered); a ring-2 candidate that completes coverage wins on merit, one
   that duplicates loses on merit. Per-ring reservations and the seed/hop
   alternation are removed — subsumed, not layered under. If all
   remaining gains are zero, remaining slots fill by relevance.
   Deterministic ties: (gain, relevance, lower chunk id).
2. **The quantity and its evil twin, by law.** The runtime objective is
   **content complementarity** — gold-blind by definition: query terms and
   bridge paths the selected list does not yet cover. Its evil twin is
   **gold complementarity** — what the offline displacement audit
   measures. The audit exists to validate that the gold-blind proxy tracks
   the gold-aware truth; the proxy must never be tuned on the audit's gold
   labels, or the eval stops measuring retrieval and starts measuring
   leakage. Any change to the coverage-feature definition must be
   justified by construction (what counts as a query aspect), not by gold
   deltas on the eval sets.

## Consequences

- Seed/hop scale mixing becomes benign: seeds outscore hop candidates in
  raw relevance, but after the first seed is taken, its lexical twins
  lose novelty and complementary hop candidates win slots — coverage
  mediates the comparison instead of a quota.
- Ring 2 needs no reservation to surface, making the hop-bound/beam sweep
  meaningful.
- Cost: O(k · |pool|) greedy loop plus per-pool-member feature extraction
  (one chunk fetch + analyze each); bounded by the frontier caps.
- The (1−1/e) guarantee applies to the coverage objective, not to gold
  recall; the gap between the two is exactly what the displacement audit
  reports, per dataset and stratum.
- The learned v2 reranker inherits a stronger deterministic baseline: it
  must beat greedy submodular selection over propagated scores, not a
  fixed quota.
- Prior hop measurements are superseded a third time; the archive keeps
  all generations (additive; propagated+interleave; propagated+per-ring;
  propagated+submodular) as the selection-evolution ablation.
