---
status: superseded by 0007
date: 2026-08-13
---

> Superseded by [0007. Greedy Submodular Admission Replaces Fixed Interleave Quotas](0007-submodular-admission.md).

# 6. Per-Ring Interleave, Stratified Reporting, and the Displacement Audit

## Context

ADR 0005 replaced query-similarity scoring of hop candidates with path
propagation and interleaved the pool as two populations (seeds, hops).
Full MuSiQue results under that scheme showed: all-gold@20 up from 0.210
(floor) to 0.300, but recall@2 *below* the floor (0.281 vs 0.323), and
hop 2 still contributing nothing over hop 1.

Three structural causes, in order of importance:

1. **Aggregate reporting hides opposing effects.** The corpus is measured
   at 65.4% effectively multi-hop — meaning ~35% of questions are ones
   where seeds always sufficed and any hop admission can only displace.
   One aggregate number cannot summarize a corpus that is one-third
   single-hop; a scorer tuned on the aggregate would win back single-hop
   recall by destroying the multi-hop gains — optimizing away the
   product's reason to exist.
2. **Fixed 1:1 interleave admits hop candidates blind.** Whether the
   admitted hop candidate carries gold, and what the evicted seed
   carried, was unmeasured.
3. **A single hop queue starves ring 2.** A ring-2 score is a product of
   two sub-1 bridge strengths; it always loses to ring-1 scores in the
   same queue, so ring-2 candidates could never surface regardless of
   quality — explaining hop2 ≈ hop1.

## Decision

This supersedes ADR 0005. Propagated path scoring is unchanged: seeds
carry query relevance (BM25 ratio-normalized; mention seeds take the
stronger of that and entity-match strength); hop-reached chunks carry
`score(parent) × bridge_strength(edge)` with
`bridge_strength = entity_specificity × count/(count+1)`; hop decay stays
emergent; expansion prunes frontiers by propagated score. What changes:

1. **Per-ring queues.** Ranking interleaves ring queues, not two
   populations: seeds hold every other slot, and the remaining slots
   round-robin across rings 1..n (exhausted queues cede their turns).
   Ring 2 competes only against ring 2.
2. **Stratified reporting is mandatory wherever diagnostics run.** Every
   metric is additionally reported per stratum — effectively-single-hop
   vs effectively-multi-hop (hybrid rule, ADR 0004; yes/no and comparison
   excluded) — because the two strata are expected to move in opposite
   directions under any hop-admission policy.
3. **Displacement audit as a first-class instrument**: of the top-k slots
   held by hop-derived candidates, how many contained gold, versus how
   much gold the evicted seeds held. Net gold is the interleave's
   measured value and the input to any future admission policy
   (score-aware admission — a hop candidate clearing a bar relative to
   what it displaces — is the anticipated v1.x/v2 refinement; the fixed
   quota stays until the audit justifies replacing it).

## Consequences

- Ring-2 candidates can surface, making the hop-bound/beam sweep worth
  running at all — and making hop-2's contribution measurable rather
  than structurally zero.
- Seeds keep half the positions, bounding the worst-case cost on
  effectively-single-hop questions; that cost is now visible per stratum
  instead of averaged away.
- The displacement audit prices the interleave: if hop-held slots carry
  less gold than the seeds they evicted, the fixed quota is measurably
  wrong and score-aware admission has a target to beat.
- More moving parts in ranking (three queues instead of two), still
  deterministic and parameter-free.
- All prior hop measurements are superseded again; the archive keeps all
  three generations (additive; propagated two-population; propagated
  per-ring) as the scorer-evolution ablation.
