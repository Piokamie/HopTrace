---
status: accepted
date: 2026-08-14
---

# 8. Submodular Admission Demoted to Experimental; Interleave Restored as Default

## Context

ADR 0007 replaced fixed interleave quotas with greedy submodular admission
(gain = relevance × novelty over coverage features). Its first full
gold-aware validation — the displacement audit, doing exactly the job it
was built for — measured a collapse on MuSiQue (2,417 questions, k=20):

- all-gold@20 fell to 0.099, below the BM25 floor (0.210) and far below
  the per-ring interleave (0.279);
- hop-derived candidates seized ~85% of top-20 slots carrying 0.5% gold,
  while the evicted seeds carried 2.9% (net −988 gold).

Root cause: the coverage universe was unbounded. Path-entity features are
minted by the candidate pool itself, so every distinct route earns
novelty, and junk routes outnumber gold routes by orders of magnitude.
The obvious repair — restricting features to a query-anchored universe —
fails by construction: a genuinely multi-hop target contains no query
content (that is what makes it multi-hop), so any query-anchored novelty
assigns it zero and starves it. Between those two poles there is no
gold-blind static coverage rule that separates gold routes from junk
routes; that separation is a learned-ranking problem.

## Decision

1. **Selection reverts to the per-ring interleave** (ADR 0006 mechanics)
   as the default — the best measured deterministic policy. The
   submodular path remains implemented behind
   `RetrievalConfig.selection = "submodular"` (CLI `--selection`) as an
   explicitly experimental mode, always run with the displacement audit.
2. The stratified reporting, displacement audit, and the
   quantity/evil-twin law from ADR 0006/0007 are unchanged and remain
   mandatory.
3. The v2 learned reranker inherits the submodular framing as its
   *objective shape* — slot value is marginal completion — but supplies
   what no static rule can: a gold-trained (on external benchmarks, never
   self-benchmark, never the audit) estimate of route quality. The
   deterministic v1 baseline it must beat is the per-ring interleave.

Alternatives rejected:

- **Bounded (query-anchored) coverage universe**: starves
  no-lexical-overlap targets by construction — see above.
- **MMR with a tuned λ, or a novelty floor**: introduces constants that
  could only be set by optimizing against the eval sets — the leakage the
  house law forbids.

## Consequences

- The measured-best configuration ships; an elegant-but-worse one does
  not. This is the bracket's second live catch (the first was the
  aggregate-vs-stratified misread).
- The audit → proxy → revert loop is now demonstrated end to end and
  documented; future selection experiments have a template and a
  baseline.
- ADR 0007's decision stands only as the experimental mode's definition;
  its "subsumes per-ring machinery" claim is withdrawn.
- Carrying two selection paths costs some code surface; both are
  deterministic and share the pool, so the cost is bounded.
