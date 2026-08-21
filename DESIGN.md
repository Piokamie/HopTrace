# HopTrace — deterministic multi-hop retrieval with traces

HopTrace is a local, LLM-free retrieval engine exposed over MCP. The intended
flow: the user asks their assistant (Claude or any MCP client) a question,
the assistant calls HopTrace, HopTrace returns evidence chunks *with the path it
took to find them*, and the assistant generates the answer. No LLM calls
at ingest, none at retrieval.

## The problem

Standard RAG is one-shot similarity search: embed the query, take the top-k
nearest chunks, hope the evidence resembles the question. Three failure
modes are structural, not tunable:

1. **Multi-hop evidence is unreachable by construction.** "Where does
   Alicja Rud's manager sit?" — the chunk naming the manager's office
   shares no surface or embedding similarity with the query. It is only
   findable *from* the chunk that names Alicja Rud's manager. One-shot retrieval cannot
   take that step, at any k, with any embedding model.
2. **Paraphrase brittleness.** When the query phrases things differently
   than the document, nearest-neighbor recall degrades quietly — and
   nothing in the pipeline reports that it happened.
3. **No audit trail.** Top-k returns chunks with cosine scores. Nobody can
   say *why* a chunk was retrieved or *what was missed*, which makes
   failures undebuggable and client conversations vibes-based.

Common fixes place an LLM somewhere in the path: agentic retrieval loops
at query time (latency, cost, nondeterminism); graph-RAG systems such as
GraphRAG and LightRAG at ingest time (cost per corpus, stale graphs,
re-ingest bills). HopTrace's index is deterministic instead — rebuild is
free, extraction is versioned, and every edge points at the text that
produced it — which tests how much of what those calls buy is available
without them.

## Design principles

1. **Deterministic where the relation is deterministic.** Which entities a
   chunk mentions, and which chunks mention an entity, are facts — computed
   once at ingest, stored in tables, joined at query time. A "hop" is a
   join, not an inference.
2. **Learned only where selection is genuinely hard.** Hop expansion
   over-generates candidates. Ranking a few hundred candidates against a
   query is the only stage that needs a model: a small local one that
   ranks chunks it was given.
3. **Measure before believing.** Every corpus gets an evaluation bracket:
   a lexical-baseline floor, an oracle-evidence ceiling, and the fraction
   of questions that are multi-hop at all. If top-k already sits near the
   ceiling on a corpus, the report says so.

## Architecture

### Ingest (no LLM)

```
documents → chunker → mention extractor → tables
```

- **Chunker**: configurable target size with structural boundaries
  (headings, paragraphs) preferred over fixed windows.
- **Mention extractor**: rule-based surface-form extraction (capitalized
  spans, quoted terms, code identifiers, numbers-with-units) plus an
  optional off-the-shelf NER pass. No LLM. Extraction is deterministic and
  versioned, so re-ingest is cheap.
- **Tables** (SQLite; shipped schema in `store.py`):
  - `mentions(entity, chunk_id, span_start, span_end, surface)` — the
    inverted index, one row per occurrence (counts derived by SQL)
  - `cooccur(chunk_id, entity)` — a VIEW over mentions, not a copy
  - `aliases(surface, canonical)` — seeded from trivial normalization
    (case, punctuation, plural); pluggable for more (roadmap)

### Retrieval

```
query → query mentions → seed chunks → bounded hop expansion → scoring → top evidence + paths
```

1. **Seed**: extract mentions from the query; exact and normalized lookups
   into the inverted index; lexical scoring (BM25) over seeds.
2. **Hop expansion** (the core): for each seed chunk, collect its
   co-occurring entities; follow them back into the inverted index to reach
   second-ring chunks. Bounded by hop count (default 2) and beam width.
   Entity specificity is a first-class filter *at expansion time*, not just
   a scoring feature: entities are ranked by inverse document frequency
   before traversal, and hub entities above an empirically tuned
   document-frequency cap are never followed (at corpus scale real hubs
   sit near df ≈ 0.1% of chunks; IDF ranking does most of the work and the
   cap handles the tail). Without this, frequent entities — years, common
   first names, the corpus's own company name — connect everything to
   everything and fill the beam with junk. Hop-2 pool precision measures
   the filter's effect.
   Every reached chunk records its **hop path**:
   `query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"marek sosna" → chunk#12 (people/marek-sosna.md: "Marek Sosna leads the ingestion crew…")`
   — every step names the chunk and the file it came from.
3. **Scoring/selection**:
   - v1: two populations, two scales. Seed chunks are scored by query
     relevance (BM25, ratio-normalized; mention seeds by the stronger of
     BM25 and entity-match strength). Hop-reached chunks are scored by
     **path propagation**: `score = parent_score × bridge_strength`,
     where bridge strength is a property of the edge itself —
     bridge-entity specificity (IDF; link uniqueness falls out of the
     df-based term) and mention-count saturation. A hop-2 chunk is never
     scored against a query it cannot lexically resemble; its hop-1
     parent's relevance flows through the bridge. The final list
     interleaves per-ring queues — seeds hold every other slot, remaining
     slots round-robin across hop rings — each queue ordered by its own
     score; deterministic, explainable, and the best deterministic policy
     under gold-aware measurement. A greedy submodular admission mode
     (marginal-coverage, facility-location/MMR lineage) is implemented
     but experimental: its first validation showed that no gold-blind
     static coverage rule separates gold routes from junk routes — a
     bounded universe starves multi-hop targets, an unbounded one floods.
     That separation is the v2 ranker's job. Hop decay emerges from the
     product of bridge strengths, not a tuned constant.
   - v2 (shipped, opt-in): a **path-aware** cross-encoder rescoring the
     pool (ADR 0010, 0012). Its input is (query,
     candidate, *route context*) — the bridge entity of the last edge
     plus a snippet of the hop-1 parent — so what it learns is route
     quality, which no static rule supplies. Only the top-N candidates in
     deterministic interleave order are rescored
     (`Retriever.candidate_window`, shared with the training builder), so
     the ranker still cannot reach outside the pool and retrieval stays
     auditable. It is evaluated against the
     propagated deterministic baseline; the zero-shot control shows route
     context buying complete-evidence recall on its own (docs/results.md).
4. **Output**: chunks with provenance objects — matched entities, hop
   paths, per-stage scores. The generating model (and the user) can see
   exactly why each piece of evidence arrived.

### Learned ranking, deterministic hops

Hops are cheap, exact, and enumerable — learning them would be re-deriving
a join with gradient descent. Ranking is not: the relevance of a hop-2
chunk to the original question is contextual and fuzzy. ADR 0008 records
where that boundary sits — deterministic selection pushed to its edge
met a trilemma (unbounded coverage floods, query-anchored coverage
starves multi-hop targets, every interpolation needs a constant only the
eval can set), so route quality is statistical, and it is what the
ranker learns.

The unbounded horn assumes an entity universe minted from free text. A
curated controlled vocabulary — CMS taxonomies, category trees, author
registries — fixes that universe, which makes bounded-coverage selection
workable on such corpora. Coverage gaps surface as extraction misses.

Reranker training data comes from the external benchmarks, never from the
self-benchmark: labels derived from the same extraction that generated
the candidates would fit the model to the index's own blind spots.

## Metrology

Measurement comes from two places with very different evidential weight.

### External benchmarks (the headline numbers)

Retrieval quality is measured on externally authored multi-hop QA
datasets at corpus scale — hop expansion over ten distractor paragraphs is
reranking, not retrieval, and recall saturates there for almost any
method. The distractor setting is reported only as a secondary sanity
check. Each dataset has a distinct job:

- **MuSiQue** (the multi-hop headline): constructed to defeat single-hop
  shortcuts — the known-positive, where multi-hop retrieval should show
  a gain. Reported on HippoRAG's published 1,000-question sample and
  11,656-passage corpus (the externally comparable protocol); the pooled
  dev-split corpus is the development setting.
- **HotpotQA via BEIR** (scale + known-negative): the full ~5M-passage
  BEIR corpus, retrieval-level metrics, published BM25 and dense baselines
  to compare against directly. It is also the calibration negative for the
  multi-hop diagnostic: the literature documents that a large share of its
  questions are answerable from one paragraph despite two being annotated.
- **2WikiMultihopQA** (discrimination, and the reranker's transfer
  holdout): sits between the two and tests whether the diagnostic can
  tell them apart; reported on HippoRAG's published sample and corpus.
  Never enters reranker training.

**Protocol.** Comparisons are budget-matched: every system returns the
same k, and the table reports candidates examined and wall-clock latency
alongside recall, so a win that costs 50× the compute is visible as
such. Gold supporting facts are finer-grained than
chunks, so the hit rule is explicit and printed in every report: in eval
mode one source paragraph is one chunk (identity-preserving, no packing),
and a chunk is a hit iff it is a gold paragraph; chunk size cannot
silently inflate or deflate recall.

**Validation gate.** Before any HopTrace number is reported, HopTrace's own BM25
must reproduce the published BM25 baseline on BEIR HotpotQA to within a
point or two. If it doesn't, the tokenization, chunking, or scoring is
wrong and every downstream comparison is invalid.

### The multi-hop diagnostic is answerability-based

`multihop_fraction` is defined by sufficiency, not annotation: the
fraction of questions the floor already effectively answers — versus
questions where it can't. An annotation-based definition (gold evidence
spans two chunks; an entity bridge exists) would report ~100% multi-hop
on HotpotQA and flatly contradict the known result, because the
literature's finding is about sufficiency: one paragraph often suffices
despite two being annotated. Operationally (LLM-free, hybrid rule): a
question counts as effectively single-hop when the floor's top-k contains
the answer span AND at least one gold paragraph — span presence alone
overcounts (answer strings recur spuriously in non-gold text), gold
presence alone reduces to annotation; requiring both kills the spurious
matches at the cost of a little annotation-dependence. Yes/no and
comparison questions are tracked separately since answer-span presence is
meaningless for them.

This turns the benchmark suite into instrument calibration. The diagnostic
must independently recover HotpotQA's documented shortcut-proneness (low
effective multi-hop despite two-paragraph annotations), MuSiQue's genuine
multi-hop difficulty, and place 2Wiki between them. The per-dataset gap
between annotated and effective multi-hop is the diagnostic's main
output. Some corpora measure as not needing hops.

Two diagnostics ride along:

- **Miss breakdown**: extraction miss (a gold bridge entity was never
  extracted), alias miss (a query mention fails to normalize onto an
  indexed entity), hop-bound miss (gold reachable only beyond hop/beam
  limits), ranking miss (gold was in the candidate pool but fell below k).
  Each is distinguishable because each stage is inspectable — and external
  gold is what makes extraction misses observable at all.
- **Hop-2 pool precision**, with and without the specificity filter.

### Self-benchmark (per-corpus diagnostic only)

Every corpus still gets a generated benchmark: synthetic questions built
from its own entity tables — single-hop questions from one chunk's facts,
multi-hop questions by walking real entity bridges between chunks. Gold
evidence is known by construction and the benchmark regenerates on
re-ingest.

Its circularity is structural: questions are generated by walking bridges
the extractor already found, so extraction misses are invisible by
construction and floor-vs-HopTrace comparisons are biased toward
HopTrace. That makes it unusable as proof and useful as plumbing — an
estimate of the multi-hop fraction on *your* corpus (sufficiency-based,
as above), a floor-vs-hops sanity check, and a regression gate where
config or extractor changes re-run the bracket and degradations fail.
This caveat is printed in the report.

The bracket ends with a verdict derived from its own rows: "effectively
single-hop: plain BM25 — or any other single-stage retriever — covers
it; hop retrieval has nothing to do here", or "keep hops on", or "hops do
not recover them: use BM25 and read the miss breakdown" — prefixed with
a too-small-to-trust warning under a few hundred chunks.

## MCP interface

Four tools:

```
ingest(source, corpus_id, config?)             → {chunks, entities, table_stats}
retrieve(query, corpus_id, hops?, k?, rerank?) → {evidence: [{chunk, score, path, entities}]}
explain(chunk_id, corpus_id, query?)           → {why: path + scores + stage trace}
bracket(corpus_id, n_questions?)               → {floor, hoptrace_1hop, hoptrace_2hop, oracle, multihop_fraction, miss_breakdown, caveat}
```

The `retrieve` response is designed for the generating model to cite:
paths are human-readable strings that name files, so "according to
people/marek-sosna.md (chunk#12), reached via Alicja Rud → Marek Sosna"
is a one-liner for the assistant to surface; `source_root` in the
response turns the relative file into a link. Each item also lists the
query words it contains (`matched_terms`), so a lexical-only hit on a
generic word is visible as such.

## Known limits

1. **Coreference is out of scope, which caps multi-hop.** Real documents
   write "she reports to the CTO" and "his office is 4B"; bridging those
   requires judgment the ingest path refuses to exercise. Entity-string
   co-occurrence serves multi-hop questions whose bridges are *named on
   both ends*; the external benchmarks measure what that costs.
2. **Inflected languages break trivial aliasing.** Case, punctuation and
   plural normalization does not touch inflection — in Polish *Kowalski*,
   *Kowalskiego*, *Kowalskim* and *Kowalskiemu* are one entity indexed as
   four. The normalizer is a pluggable interface; a lemmatizing
   implementation (spaCy or Morfeusz) is the planned fix. Until then
   HopTrace is accurate on English and lossy on inflected languages.

## As built

Python; SQLite; MCP server over stdio or streamable-http; zero GPU; zero
API keys. v1 delivered:

- chunker + rule-based mention extractor (+ optional spaCy NER flag)
- mention/co-occurrence tables, trivial alias normalization behind a
  pluggable normalizer interface
- BM25 seeds, 2-hop bounded expansion with recorded paths and
  expansion-time specificity filtering
- deterministic scorer (BM25 + hop decay + path features)
- external benchmark harness at corpus scale (BEIR HotpotQA full corpus;
  pooled-corpus MuSiQue and 2WikiMultihopQA; distractor setting as sanity
  check only; datasets downloaded by script, never committed), with the
  budget-matched protocol, the BM25-reproduction validation gate, the
  answerability-based multi-hop diagnostic and its annotated-vs-effective
  gap, miss breakdown, and the hop-2 pool-precision ablation
- MCP server with the four tools
- bracket harness with self-benchmark generation (per-corpus diagnostic)
- one demo corpus with a written walkthrough of a multi-hop retrieval

## Shipped

**v1 — deterministic retrieval.** Ingest, bounded hop expansion,
propagated path scoring, per-ring interleave, provenance, the MCP server,
the eval harness with the BM25 reproduction gate, and the per-corpus
bracket. Architecture above.

**v2 — path-aware learned reranker.** MiniLM-class cross-encoder over
ONNX Runtime behind the `rerank` extra: no torch at inference, no GPU. Trained on MuSiQue and HotpotQA *train*
splits only, with the wall enforced mechanically — the artifact ships a
manifest and the eval harness refuses to score any split it lists, any
manifest that does not describe the graphs beside it, and any artifact
from a dirty trainer tree (ADR 0011). 2Wiki is held out of training
entirely and serves as the transfer test. The int8 graph, tokenizer and
manifest are committed under `models/`; the fp32 graph is a
checksum-pinned release download; `--rerank` defaults to the bundled
model. Opt-in (`selection="rerank"`, `--rerank`, MCP `rerank=true`); the
deterministic interleave remains the default until measurement justifies
switching, and the full grid in `docs/results.md` reports the deterministic
2hop row beside every reranked row.

## Roadmap

Grouped by what each item changes. Nothing below is built.

### Retrieval quality

Ordered by the pool oracle's remaining headroom: expansion coverage
dominates; ranking sits at 95%+ of the ceiling for the candidates it
sees (docs/results.md, pool oracle).

- **Wider rescoring budget**: `rerank_top_n` is the single largest
  measured lever on MuSiQue (~11 points of all-recall@20 sit outside the
  top 50; ~2 on 2Wiki), priced linearly in latency. Needs a sweep.
- **Marginal-completion selection over learned scores** (ADR 0010 point
  5, unmeasured): the learned score as the λ no
  static rule could set. Behind the budget sweep, since plain-sort
  already reaches 95%+ of the ceiling for scored candidates.
- **BM25 parameters, train vs product**: the reranker trains and is
  evaluated on pools seeded with k1=0.9/b=0.4 and is served over pools
  seeded with the product default 1.5/0.75. Either measure the product
  setting or make the eval setting the default; both are a re-measurement.
- **Alias resolution beyond normalization**: locality-sensitive bucketing
  of entity mentions (embedding-hashed, computed at ingest, frozen at
  query time) so paraphrased and abbreviated mentions land in the same
  bucket — extends the exact-match index without putting a model in the
  query path. `seed_alias` is a named, measured miss category.
- **Hop-positive reweighting**: training positives are ~16:1 seed-to-hop,
  and the fine-tune gained more on effectively-single-hop questions than
  on multi-hop ones. Upweighting hop positives targets that gap.
- **Best-route admission.** A chunk enters the pool with the first route
  that reaches it (earliest ring wins), and carries only that route. A
  generic query word can seed a document that bridges to the target at
  hop 1 through a weak entity, in which case the strong two-bridge route
  that also reaches it is never recorded: the chunk ranks as a weak hop-1
  candidate, and the reranker sees the weak route as its context.
  Observed on the demo corpus ("In which building does the manager of
  Alicja Rud sit?" → `building` seeds `budynek-a.md` → `budynek a` →
  `office-b12.md` at rank 10, while Alicja → Marek Sosna → Office B12
  exists). Keeping the highest-scoring route per chunk (or all routes for
  the reranker) is a retrieval change and needs the full eval regenerated.
- **Lemmatizing normalizer** for inflected languages (spaCy or Morfeusz
  behind the pluggable alias interface).
- **Sentence-initial multi-word names.** The trust rule drops an
  untrusted capitalized opener from a longer run, which is correct for
  "Yesterday Anna Kowalska" and wrong for a name that only ever opens
  sentences (indexed as its last token). Candidate fixes: collect
  multi-word runs in the corpus-wide trust pass, or treat a run of 2+
  capitalized non-stopword tokens as a name regardless of position.
  Either changes extraction, so it needs the full eval regenerated
  before it ships.

### Ingest

Ingest currently reads `.md`, `.markdown`, `.txt` and `.rst`; anything
else is skipped and listed. That covers documentation repositories and
exports, and excludes most of what organisations actually store.

- **Document formats**: PDF, DOCX, HTML as first-class inputs. Each is a
  text-extraction problem with its own failure mode — PDF loses reading
  order and hyphenates across line breaks, DOCX carries revision markup,
  HTML carries navigation chrome — and extraction quality feeds straight
  into entity extraction, so each format needs its own measurement.
  Extractors belong behind an
  optional extra (`hoptrace[documents]`), never in the core dependency
  set.
- **Structured-source ingest**: CMS backends expose curated metadata —
  taxonomies, categories, author registries — over JSON APIs. Ingest
  accepts these as first-class entities alongside extracted mentions:
  they are editor-vetted, carry zero extraction risk, and supply a fixed
  coverage universe for bounded selection (see the trilemma
  qualification above). This is the difference between indexing a CMS and
  indexing its rendered text.
- **Incremental ingest** (append without full rebuild), multi-corpus
  routing, entity-page summaries (all chunks for an entity, one call).

### Serving and distribution

MCP covers agentic harnesses. Most deployed consumers are not agentic —
website chatbots, CMS backends, plain services — and half the integrator
market composes retrieval through framework interfaces.

- **HTTP adapter (FastAPI), same verbs**: `POST /retrieve`, `POST
  /ingest` (auth-gated), `GET /explain/{chunk_id}`, `POST /bracket`.
  OpenAPI comes with the framework.
- **Response format**: `format=json` returns structured evidence with
  paths; `format=prompt` returns a ready-to-concatenate context block —
  chunks, citations and hop paths as plain text — so an integrator can do
  `context = curl(...); prompt = context + user_query;`. The JSON carries
  paths, so a chatbot can render sources under its answer ("according to
  [doc], via Alicja Rud → Marek Sosna").
- **No CMS modules**: expose HTTP and document the sidecar pattern (one
  curl example, one PHP-ish snippet).
- **Ops minimalism**: API-key header, `corpus_id` multi-tenancy (already
  in the tool schema), a Dockerfile. Rate limiting comes later.
- **Framework adapters** (`*-langchain`, `*-llamaindex`): thin shims
  exposing `retrieve` as a `BaseRetriever` (and the LlamaIndex
  equivalent), in **separate packages, never in core** — framework APIs
  break across minor versions, so adapter rot stays quarantined and
  separately versioned. Only `retrieve` is adapted: hop paths ride into
  document metadata, and the bracket, calibration and diagnostics have no
  framework slot. "No prompt chains,
  no orchestration" describes the engine; running as a component inside
  someone else's chain is an intended use. MCP adoption is eating the
  integration-directory role over time, but the CMS-agency market lives
  in these frameworks today.

## Non-goals

- Not a vector-DB replacement. Hybrid setups (HopTrace + dense retrieval as an
  additional seed source) are expected and supported; the bracket will
  tell you what each source contributes.
- No agentic orchestration, no query-time LLM calls, no prompt chains.
- Not a knowledge graph product. The tables are an index, not an ontology;
  nothing is extracted that requires judgment.
