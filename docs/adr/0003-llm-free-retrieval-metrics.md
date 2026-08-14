---
status: accepted
date: 2026-08-13
---

# 3. LLM-Free Retrieval Metrics

## Context

DESIGN.md frames the measurement bracket with a generation-quality ceiling:
"oracle evidence — the generator is handed the gold chunks." Running a
generator requires an LLM, and v1 is LLM-free end to end (no API keys, no
local model). The metrics must therefore be computable from retrieval
output and gold annotations alone. Additionally, comparing systems that
examine different numbers of candidates (BM25 top-k vs. hop expansion over
hundreds) invites an obvious fairness attack.

## Decision

All v1 metrics are retrieval-level, LLM-free:

- **BEIR HotpotQA**: nDCG@10 and recall@100 (directly comparable to
  published baselines) plus budget-matched recall@k.
- **Pooled corpora (MuSiQue, 2WikiMultihopQA)**: per-paragraph recall@k and
  all-gold@k for k in {2, 5, 10, 20}.
- **Self-benchmark bracket**: recall-based floor/hop rows; `oracle` is
  redefined as the fraction of questions whose full gold evidence set fits
  within k — the best any retriever could score at recall@k — not a
  generation ceiling.
- **Protocol**: comparisons are budget-matched (every system returns the
  same k) and every report includes candidates examined and wall-clock
  latency (median and p95, single-threaded) next to recall.
- **Hit rule**: in eval mode one source paragraph is one chunk (identity-
  preserving, no packing), and a hit means the retrieved chunk is a gold
  paragraph. The rule is printed in every report because chunking choices
  would otherwise silently inflate or deflate recall.

Alternative rejected: answer-generation metrics (EM/F1 with an LLM reading
retrieved evidence) — reintroduces cost, latency, and nondeterminism that
the project exists to avoid, and confounds retrieval quality with reader
quality.

## Consequences

- Every number is deterministic and reproducible on a laptop with no keys.
- Results are directly comparable to published retrieval-level baselines.
- We cannot claim anything about end-task answer quality; the report
  measures evidence delivery only, and the write-up must say so.
- The redefined oracle is weaker than a generation ceiling: it bounds
  retrieval, not the answer. This is the honest limit of an LLM-free
  harness.
