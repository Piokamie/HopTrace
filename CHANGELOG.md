# Change Log

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog* (https://keepachangelog.com/en/1.1.0/) and this project adheres to *Semantic Versioning*.

## [Unreleased]

### Added

• MINOR Deterministic ingest pipeline: structural chunker, rule-based mention extractor (+ optional spaCy NER via `hoptrace[ner]`), pluggable alias normalizer, single-SQLite-file corpora with atomic replace (`src/hoptrace/chunker.py`, `mentions.py`, `normalize.py`, `store.py`, `ingest.py`)
• MINOR Pure-Python Okapi BM25 with per-term explain and a numpy fast path for corpus scale; Lucene-style `english` analyzer (stopwords + dependency-free Porter stemmer) so the floor reproduces published baselines (`bm25.py`, `porter.py`, `tokenize.py`)
• MINOR Bounded hop expansion with expansion-time IDF specificity filtering, propagated path scoring (parent × bridge strength), per-ring interleaved ranking, and full per-stage provenance (`expand.py`, `score.py`, `retrieve.py`, `provenance.py`)
  - Experimental greedy submodular selection retained behind `RetrievalConfig.selection` (ADR 0008)
• MINOR External benchmark harness: BEIR HotpotQA (5.23M passages) with a hard BM25-reproduction gate (nDCG@10 0.6352 vs published 0.633), pooled MuSiQue/2WikiMultihopQA, stratified metrics, calibration diagnostic, miss taxonomy, displacement audit, beam/hub ablations (`src/hoptrace/eval/`, results in `docs/results.md`)
• MINOR Self-benchmark bracket: seeded question generation with a single-hop-proof multi-hop construction; floor/hop/oracle rows, sufficiency-based multihop_fraction, circularity caveat embedded in every payload (`bench.py`, `bracket.py`)
• MINOR MCP server over stdio with four tools — `ingest`, `retrieve`, `explain`, `bracket` — and mtime-invalidated corpus registry (`server.py`)
  - **API note**: `explain` takes `corpus_id` in addition to DESIGN.md's sketch (chunk ids are per-corpus)
• MINOR CLI: `hoptrace ingest | retrieve | explain | bracket | serve | eval` (`cli.py`)
• MINOR Demo corpus (`examples/office/`, 25 documents with engineered entity bridges), six designed 2-hop questions (`examples/QUESTIONS.md`), written walkthrough with real transcripts (`docs/walkthrough.md`), end-to-end proof tests (`tests/test_demo_corpus.py`)
• Decision log: ADRs 0001–0008 (`docs/adr/`), including the measured selection-policy trilemma that defines the v2 reranker's job
