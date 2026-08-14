---
status: accepted
date: 2026-08-13
---

# 4. External Benchmarks Carry the Evidence; the Self-Benchmark Is a Diagnostic

## Context

LRAG can generate a benchmark from any corpus's own entity tables (the
self-benchmark): questions built by walking entity bridges, with gold
evidence known by construction. Its circularity is structural — questions
are generated from bridges the extractor already found, so every question
is one the index can represent. Extraction misses are invisible by
construction, and floor-vs-LRAG comparisons are biased toward LRAG.
Self-graded results cannot carry the project's headline claim.

Externally authored multi-hop datasets exist with gold supporting
evidence: HotpotQA, 2WikiMultihopQA, MuSiQue. Their standard distractor
setting (10–20 paragraphs per question) is too small to exercise
retrieval — over ten candidates, hop expansion is reranking and recall
saturates. Corpus-scale versions exist: BEIR's HotpotQA task (~5M
passages, published BM25/dense baselines) and pooled dev-split corpora for
MuSiQue/2Wiki. The literature also documents that a large share of
HotpotQA questions are answerable from one paragraph despite two being
annotated, while MuSiQue was constructed specifically to defeat such
shortcuts.

## Decision

1. **Evidence hierarchy.** Corpus-scale external results carry all
   headline claims: MuSiQue (pooled) is the multi-hop headline, BEIR
   HotpotQA provides scale and published-baseline comparison, 2Wiki sits
   between. Distractor-setting numbers are a sanity-check appendix only.
   The self-benchmark is a per-corpus diagnostic; its report embeds its
   own circularity caveat, and it is never used as proof — nor (v2) as
   reranker training data, which would fit the model to the index's blind
   spots.
2. **Calibration framing.** The dataset trio doubles as instrument
   calibration for the multi-hop diagnostic: HotpotQA is the known
   negative (documented shortcut-proneness), MuSiQue the known positive,
   2Wiki the discrimination test. Recovering the known properties
   validates the measurement; results are presented as calibration, never
   as concession.
3. **Hybrid answerability-based multi-hop definition.** `multihop_fraction`
   is defined by sufficiency, not annotation: a question is effectively
   single-hop when the floor's top-k contains the gold answer span
   (normalized substring match) AND at least one gold paragraph. Span-only
   overcounts single-hop via spurious answer-string matches; gold-only
   collapses back into annotation (which would read ~100% multi-hop on
   HotpotQA and contradict the known result). Yes/no and comparison
   questions are reported as a separate category — answer-span presence is
   meaningless for them. The per-dataset gap between annotated and
   effective multi-hop is itself a reported observation.

Alternatives rejected: distractor-setting evaluation as primary (measures
a setting where the target problem does not exist); annotation-based
multi-hop definition (contradicts the sufficiency literature); span-only
answerability (spurious matches); self-benchmark as evidence (circular).

## Consequences

- Headline claims rest on questions and gold evidence LRAG did not
  generate; extraction misses become observable in the miss breakdown.
- The BM25 floor must first reproduce the published BEIR baseline, which
  gates all downstream numbers (see ADR 0002).
- A correct diagnostic will say "hops don't help" on HotpotQA; the
  write-up must pair that number with the calibration framing or a
  skimming reader draws the wrong conclusion.
- The hybrid rule retains a residual annotation-dependence (the gold-
  paragraph conjunct) and the span match remains a proxy for sufficiency;
  both limits are disclosed wherever the metric is reported.
- Pooled MuSiQue/2Wiki corpora (~20k–120k paragraphs) are smaller than
  BEIR scale; reports disclose corpus size per dataset.
