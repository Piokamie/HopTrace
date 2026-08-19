---
status: superseded by 0012
date: 2026-08-14
---

> Superseded by [0012. Path-Aware Reranker: Shipped Form](0012-path-aware-reranker-shipped-form.md).

# 10. Path-Aware Cross-Encoder Rescoring of the Deterministic Pool

## Context

ADR 0008 established that bounded selection over the hop pool needs a
quantity no static rule supplies — route quality — and reserved it for a
learned ranker trained on external benchmarks. Two designs could deliver
that: (a) a conventional reranker scoring (query, candidate) pairs over
the pool, which learns text relevance but is blind to how a candidate
was reached; or (b) learned admission, which learns the selection policy
itself but is a novel credit-assignment problem and would break the
audit guarantee that ranking cannot reach outside the deterministically
generated pool.

The retriever already records everything a route-aware model needs:
every candidate carries its full hop path — bridge entity, parent chunk,
entity specificity, mention count (`expand.py` `Candidate`,
`provenance.py` `HopPath`).

Latency constraint: v1 retrieval runs 3–13 ms per query on demo corpora
and ~70–200 ms at 5.2M-chunk scale. CPU cross-encoding of a full pool
(150+ candidates) costs 100–500 ms and would dominate the budget.

## Decision

The v2 learned component is a cross-encoder whose input is (query,
candidate, route-context) — the bridge entity of the last traversed edge
plus a snippet of the hop-1 parent chunk, prefixed to the candidate
text. The model thereby learns route quality — the exact quantity ADR
0008 proved unsupplied — inside the standard reranker training recipe,
not a novel admission-policy learning problem. The apparent fork between
"rerank the pool" and "learn admission" dissolves under this one design
move.

Constraints that hold:

1. **Admission stays deterministic.** The ranker rescores the
   deterministically generated pool; "the ranker cannot reach outside
   the pool" holds verbatim. At inference, only the top-N candidates in
   deterministic interleave order (N≈50) are rescored, batched — the
   latency cost is bounded and printed (with/without reranker) in every
   eval table.
2. **Interleave remains the shipped default.** Reranking is opt-in
   (`selection="rerank"`) until the measurement harness earns the flip —
   the same law that reverted submodular (ADR 0008).
3. **Inference is ONNX Runtime, not torch.** MiniLM-class backbone
   initialized from `cross-encoder/ms-marco-MiniLM-L6-v2` (official ONNX
   exports exist: 91 MB fp32, 23 MB qint8), behind a
   `hoptrace[rerank]` extra. The no-GPU, local-first story survives; the
   base install stays dependency-light. Training code (torch) lives
   in-repo under a dev-only dependency group, never a published extra.
4. **The model artifact is hosted on a model hub**, downloaded on first
   use with a recorded checksum (the dataset-download pattern), never
   committed to git.
5. Greedy selection over learned scores may adopt the
   marginal-completion objective shape from ADR 0007/0008 — the learned
   score becomes the λ no static rule could set; both plain-sort and
   marginal-completion variants are measured before either ships.

Alternatives rejected:

- **Route-blind (query, candidate) reranking**: discards the recorded
  path; if route quality is learnable, this design cannot learn it. The
  path-context ablation (route prefix dropped) is retained in eval, so
  "path features carry no weight" would surface as a finding rather
  than an assumption.
- **Learned admission policy**: hairy credit assignment, no established
  training recipe, and it would replace the deterministic pool guarantee
  with model behavior — weakening the audit story ADR 0004/0008 depend
  on.
- **torch inference**: framework-sized dependency for a 23 MB model;
  ONNX Runtime delivers the same forward pass at ~40 MB.

## Consequences

> **Status at ship (2026-08-17; not a revision of the decision):** points
> 1–4 shipped as written. Point 5's marginal-completion variant did not
> ship and was not measured — the reranker reaches 95%+ of the ceiling
> for the candidates it scores (docs/results.md, pool oracle), so it
> moves to the V3 roadmap. The artifact is not hub-hosted: the registry
> ships the zero-shot base, and the fine-tuned model is built via
> `training/`.

- The trilemma's missing quantity gets a learned estimator while every
  v1 guarantee (deterministic pool, bounded budget, full provenance)
  survives intact; traces now show both the deterministic and the
  learned score per candidate.
- The reranker has a fair, budget-matched deterministic baseline
  (per-ring interleave) and inherits the displacement audit as its
  gold-aware validator.
- Two selection philosophies ship side by side; the code carries a third
  selection arm and an optional-dependency seam (mitigated by the
  existing numpy/spacy isolation patterns).
- CPU rerank adds a measured latency cost (bounded by top-N batching);
  the brand becomes "3–13 ms deterministic, +printed-cost learned",
  which every table must state.
- Fine-tuning requires a one-off training pipeline and hub-hosted
  artifact — reproducibility depends on the training manifest (ADR
  0011).
