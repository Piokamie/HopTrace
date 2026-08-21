# Change Log

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog* (https://keepachangelog.com/en/1.1.0/) and this project adheres to *Semantic Versioning*.

## [0.1.0] - 2026-08-21

### Added

• MINOR Path-aware learned reranker (v2, opt-in): a MiniLM-class cross-encoder rescores the deterministic candidate pool over (query, candidate, **route context** — the bridge entity plus a snippet of the hop-1 parent), so the model scores route quality rather than text relevance alone (`rerank.py`, ADR 0010, shipped form ADR 0012)
  - Admission stays deterministic: only `pool_order[:max(rerank_top_n, k)]` in interleave order is rescored (`Retriever.candidate_window`, the one definition the training builder also uses), so the ranker cannot reach outside the pool
  - ONNX Runtime inference behind `hoppath[rerank]` (no torch, no GPU); the int8 graph (`--rerank-precision int8`, the bundled default) is 1.6–1.9× faster than fp32 at ≤0.008 on every measured metric
  - The fine-tuned model ships with the repository: `models/hoppath-rerank-minilm-l6/` (int8 graph, tokenizer, manifest) is what `--rerank` uses by default; the fp32 graph is a sha256-pinned release download; the zero-shot base is `--rerank-model ms-marco-minilm-l6-v2`. Every registry file is checksum-pinned and verified on load
  - `RetrievalConfig.selection = "rerank"`, CLI `retrieve --rerank [--rerank-model] [--rerank-precision]` and `eval --selection rerank`, MCP `retrieve(rerank=true)`, `$HOPPATH_RERANK_MODEL`
  - Deterministic interleave remains the default; the full grid in `docs/results.md` reports it beside every reranked row
• MINOR Training pipeline (dev-only, `training/`): pool-derived training data taken from `Retriever.candidate_window` (inference's seed depth, order and serialization), hop-stratified negative sampling, fine-tune on GPU or CPU, ONNX export + int8 quantization, a training manifest emitted mechanically from the dataset manifest (retrieval settings, file hashes, `<sha>[-dirty]` trainer commit), `--from` to derive the exclusion-filtered set from one retrieval pass, and `bundle_artifact.py` to stage the in-repo files
• MINOR **Training-data wall enforced by the harness** (ADR 0011): models ship a manifest naming their training splits; `hoppath eval` refuses to score any split it lists, refuses a rerank model with no manifest, refuses any model that trained on the declared holdout (matched loosely, whatever the spelling) whatever is being evaluated, refuses a manifest whose file hashes do not describe the graphs beside it, and refuses a dirty-tree artifact unless `--allow-dirty-manifest`. Only the zero-shot base resolved from the registry is exempt — never a directory by name. 2WikiMultihopQA is held out of training entirely as the transfer test
• MINOR Evaluation-gold passages are excluded from training by default (`training/build_dataset.py`) — paragraphs recur across questions, so a split-level wall is not enough. Matching is whitespace-normalized on structured per-row keys across both HippoRAG evaluation sets; `--keep-eval-gold` opts out and models built that way are marked NOT PUBLISHABLE in every eval report. `training/audit_exposure.py` measures the exposure over every training source
• MINOR Corpus protocols made explicit and self-describing: `--dataset hipporag-musique|hipporag-2wiki` (the published 1,000-question samples and their exact corpora, sha256-pinned — the industry-comparable setting) and `--corpus-pool dev|all`; every report prints which protocol produced it, including the caveat that the dev-pooled setting is not comparable to published numbers. Cached pooled indexes are verified paragraph-by-paragraph on reuse
• MINOR Pool oracle in every hop report: the ceiling for any selection over the generated pool, at both the full-pool and top-N levels, so "achieved" is always printed against "achievable" (`harness.py`)
• MINOR `hoppath serve --http [--host --port]`: MCP streamable-http transport for clients that connect to a running process instead of spawning it; stdio remains the default. No authentication — localhost or behind a proxy
• MINOR `hoppath serve` now reports the package version to MCP clients (was an empty string)
• MINOR `eval --strata-file`: report metrics split by an arbitrary external partition (used for the training-exposure audit)
• MINOR Deterministic ingest pipeline: structural chunker, rule-based mention extractor (+ optional spaCy NER via `hoppath[ner]`), pluggable alias normalizer, single-SQLite-file corpora with atomic replace (`src/hoppath/chunker.py`, `mentions.py`, `normalize.py`, `store.py`, `ingest.py`)
• MINOR Pure-Python Okapi BM25 with per-term explain and a numpy fast path for corpus scale; Lucene-style `english` analyzer (stopwords + dependency-free Porter stemmer) so the floor reproduces published baselines (`bm25.py`, `porter.py`, `tokenize.py`)
• MINOR Bounded hop expansion with expansion-time IDF specificity filtering, propagated path scoring (parent × bridge strength), per-ring interleaved ranking, and full per-stage provenance (`expand.py`, `score.py`, `retrieve.py`, `provenance.py`)
  - Experimental greedy submodular selection retained behind `RetrievalConfig.selection` (ADR 0008)
• MINOR External benchmark harness: BEIR HotpotQA (5.23M passages) with a hard BM25-reproduction gate (nDCG@10 0.6352 vs published 0.633), pooled MuSiQue/2WikiMultihopQA, stratified metrics, calibration diagnostic, miss taxonomy, displacement audit, beam/hub ablations (`src/hoppath/eval/`, results in `docs/results.md`)
• MINOR Bracket `VERDICT` line + `verdict`/`verdict_code` over MCP: one sentence derived from the rows — single-hop ("plain BM25 covers it; hop retrieval has nothing to do here"), multi-hop ("keep hops on"), or hops-do-not-help — with an explicit "too few chunks for a stable reading" prefix under 200 chunks
• MINOR Self-benchmark bracket: seeded question generation with a single-hop-proof multi-hop construction; floor/hop/oracle rows, sufficiency-based multihop_fraction (`bench.py`, `bracket.py`)
• MINOR MCP server over stdio with four tools — `ingest`, `retrieve`, `explain`, `bracket` — and mtime-invalidated corpus registry (`server.py`)
• MINOR CLI: `hoppath ingest | retrieve | explain | bracket | serve | eval` (`cli.py`)
• MINOR Citations name files: every step of the `path` string carries the chunk's source file (`chunk#12 (people/marek-sosna.md: "…")`), `retrieve`/`explain`/`ingest` return `source_root` (the absolute directory that was ingested) beside the per-chunk relative `chunk.doc`, and the CLI result line shows the file — so an answer can point at the exact document, not a chunk id
• MINOR `matched_terms` on every evidence item (the query words the chunk actually contains) and on the CLI result line beside the raw BM25 — a lexical-only hit on a generic word is visible as such, since seed `score` is normalized to the top hit and cannot show it; an unresolved query mention now yields an explicit note ("no indexed entity matches "X"; results rest on word overlap alone")
• MINOR Agent skill (`skills/hoppath/SKILL.md`): how to drive the four MCP tools — ingest, bracket-before-promising-hops, retrieve + cite `path` verbatim, explain on doubt; copy into `.claude/skills/`
• MINOR Demo corpus (`examples/office/`, 25 documents with engineered entity bridges), seven designed multi-hop questions — including one two-bridge chain (Alicja Rud → Marek Sosna → Office B12 → the office file), which Marek's file no longer short-circuits (`examples/QUESTIONS.md`), written walkthrough with real transcripts (`docs/walkthrough.md`), end-to-end proof tests (`tests/test_demo_corpus.py`)
• Decision log: ADRs 0001–0013 (`docs/adr/`), including the selection-policy findings behind the reranker (0008), the path-aware design that answers it (0010), the training-data wall (0011), and the reranker's shipped form — bundled int8 default, pinned downloads (0012), and the pre-publication rename (0013)

### Changed

• Latency reporting now states its own conditions (`ONNX 4 intra-op threads, tokenizer single-threaded, warm` for rerank rows vs `single-threaded, warm`) instead of a hardcoded string — the rerank path is multi-threaded and the old label would have misreported it; the tokenizer's thread pool is pinned so the label is true
• `retrieve` output prints the learned score alongside the deterministic path score when reranking, since the learned one determines the order

### Fixed

• `tests/test_demo_corpus.py` set `HOPPATH_DATA_DIR` via `os.environ` without restoring it, leaking the override into every later test in the process (now a module-scoped `MonkeyPatch` with `undo`)
• Reranker training pools were seeded with 50 BM25 chunks (`retrieve(k=50)`) while inference seeds 20; the builder now takes `Retriever.candidate_window` so the model trains on the pool it scores, and the model was retrained
• `hoppath serve` no longer fails at import on an uninstalled source checkout (`version()` falls back to `__version__`)
• Over-long queries no longer raise from the tokenizer's `only_second` truncation; the query is pre-clipped to 128 tokens
• `--rerank-precision int8` against an artifact without an int8 graph fails before any download; downloads carry a timeout
• Missing-optional-dependency errors (`--rerank` without the extra, `ingest --ner` without spaCy) now print as `error: …` and exit 1 instead of raising a traceback
