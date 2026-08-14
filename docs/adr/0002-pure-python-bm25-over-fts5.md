---
status: accepted
date: 2026-08-13
---

# 2. Pure-Python BM25 over SQLite FTS5

## Context

LRAG needs a lexical scoring layer: it is the seed ranker, the retrieval
floor that every measurement compares against, and a component of the final
deterministic scorer. The corpus store is SQLite, and the local SQLite build
ships FTS5 with a built-in `bm25()` ranking function (verified available:
SQLite 3.50.2, `ENABLE_FTS5`), so the zero-effort option exists.

Two constraints work against FTS5:

1. Every retrieval result must carry provenance down to per-stage, per-term
   score components (`explain` is a public tool). FTS5's `bm25()` returns a
   single opaque score; per-term contributions are not exposed.
2. The evaluation harness must reproduce the published BM25 baseline on
   BEIR HotpotQA before any downstream number is trusted. That requires
   control over tokenization, k1/b parameters, and the exact idf formula —
   FTS5 fixes its tokenizer and scoring internals.

At corpus scale (~5M passages) a naive row-per-posting table would hold on
the order of 10^8 rows with Python-loop scoring on top.

## Decision

We implement Okapi BM25 in Python. Postings are stored per term as packed
binary blobs (`(chunk_id uint32, tf uint16)` pairs) in SQLite; scoring
decodes only the query's terms, with a numpy fast path (the `eval` extra)
for corpus-scale runs and a stdlib fallback for small corpora. Parameters
(k1=1.5, b=0.75) and the idf formula live in config and are recorded in
every eval report.

Alternatives rejected:

- **SQLite FTS5**: no per-term score breakdown, no parameter control — see
  constraints above.
- **rank-bm25 (library)**: in-memory only, no persistence, no per-term
  explain API, and an extra dependency for ~100 lines of well-understood
  arithmetic.

## Consequences

- Per-term score components are available to `explain` and to the eval
  harness's miss diagnostics.
- Tokenization, parameters, and scoring are fully controllable, which makes
  the BEIR baseline-reproduction gate achievable and debuggable.
- We own correctness: the implementation must be validated against
  hand-computed fixtures and the published BEIR baseline (a failed
  reproduction blocks all downstream results by design).
- Packed-blob postings are a custom format: slightly more store code, and
  the postings table is not queryable with plain SQL.
