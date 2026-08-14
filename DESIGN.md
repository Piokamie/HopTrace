# HopTrace — deterministic multi-hop retrieval with traces

> Formerly "LRAG". Renamed (ADR 0009): this engine retrieves — the
> "augmented generation" was always the external model's job, and the
> learned component is a v2 ranking layer, not the product's identity.
> HopTrace names what ships: hop retrieval, with the trace attached.

**Evidence delivery for LLMs: deterministic multi-hop retrieval, a learned
ranker only where ranking is genuinely hard, and per-corpus measurement that
tells you whether any of it was needed.**

HopTrace is a local, LLM-free retrieval engine exposed over MCP. The intended
flow: the user asks their assistant (Claude or any MCP client) a question,
the assistant calls HopTrace, HopTrace returns evidence chunks *with the path it
took to find them*, and the assistant generates the answer. No LLM calls at
ingest. No LLM calls at retrieval. The only model you pay for is the one
writing the answer.

## The problem

Standard RAG is one-shot similarity search: embed the query, take the top-k
nearest chunks, hope the evidence resembles the question. Three failure
modes are structural, not tunable:

1. **Multi-hop evidence is unreachable by construction.** "Where does
   Anna's manager sit?" — the chunk naming the manager's office shares no
   surface or embedding similarity with the query. It is only findable
   *from* the chunk that names Anna's manager. One-shot retrieval cannot
   take that step, at any k, with any embedding model.
2. **Paraphrase brittleness.** When the query phrases things differently
   than the document, nearest-neighbor recall degrades quietly — and
   nothing in the pipeline reports that it happened.
3. **No audit trail.** Top-k returns chunks with cosine scores. Nobody can
   say *why* a chunk was retrieved or *what was missed*, which makes
   failures undebuggable and client conversations vibes-based.

The industry's fixes each pay a heavy toll: agentic retrieval loops put an
LLM in the query path (latency, cost, nondeterminism); graph-RAG systems
put an LLM in the ingest path (cost per corpus, stale graphs, re-ingest
bills). HopTrace's bet is that most of what those LLM calls buy is available
deterministically.

## Design principles

1. **Deterministic where the relation is deterministic.** Which entities a
   chunk mentions, and which chunks mention an entity, are facts — computed
   once at ingest, stored in tables, joined at query time. A "hop" is a
   join, not an inference.
2. **Learned only where selection is genuinely hard.** Hop expansion
   over-generates candidates. Ranking a few hundred candidates against a
   query is the one place a model earns its keep — a small, local one,
   ranking chunks it was *given*, never retrieving on its own.
3. **Measure before believing.** Every corpus gets an evaluation bracket:
   a lexical-baseline floor, an oracle-evidence ceiling, and the fraction
   of questions that are multi-hop at all. If top-k already sits near the
   ceiling on a client's corpus, HopTrace's own report says so — the tool tells
   you when you don't need it.

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
  versioned — re-ingest is seconds, not dollars.
- **Tables** (SQLite; shipped schema in `store.py`):
  - `mentions(entity, chunk_id, span_start, span_end, surface)` — the
    inverted index, one row per occurrence (counts derived by SQL)
  - `cooccur(chunk_id, entity)` — a VIEW over mentions, not a copy
  - `aliases(surface, canonical)` — seeded from trivial normalization
    (case, punctuation, plural); pluggable for more (v2)

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
   sit near df ≈ 0.1% of chunks, so the cap must be set from data, not
   intuition — IDF ranking does most of the work and the cap handles the
   tail). Without this, frequent entities — years, common first names, the
   corpus's own company name — connect everything to everything and fill
   the beam with junk before any scorer can help. The filter's value is
   itself measured (see metrology: hop-2 pool precision).
   Every reached chunk records its **hop path**:
   `query:"Anna" → chunk#12 ("Anna reports to Kowalski") → entity:"Kowalski" → chunk#31 ("Kowalski sits in 4B")`.
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
     bounded universe starves multi-hop targets, an unbounded one floods
     (ADR 0008). That separation is precisely what the v2 learned ranker
     is for. Hop decay emerges from the product of bridge strengths
     instead of a tuned constant.
   - v2: a small learned reranker (cross-encoder scale, CPU-friendly)
     over the candidate pool. The ranker sees candidates and query only —
     it cannot reach outside the pool, so retrieval stays auditable. It
     must beat the propagated deterministic baseline, not a strawman that
     misprices hop-2 candidates by query similarity.
4. **Output**: chunks with provenance objects — matched entities, hop
   paths, per-stage scores. The generating model (and the user) can see
   exactly why each piece of evidence arrived.

### Why the ranker is learned and the hops are not

Hops are cheap, exact, and enumerable — learning them would be re-deriving
a join with gradient descent. Ranking is the opposite: relevance of a
hop-2 chunk to the original question is contextual and fuzzy. Spend the
model where the problem is actually statistical.

That claim is no longer intuition — the harness located its boundary
(ADR 0008). Deterministic selection was pushed to its edge: propagated
path scores, then greedy marginal-coverage admission over the candidate
pool. The gold-aware displacement audit measured the resulting trilemma:
an *unbounded* coverage universe floods, because candidates mint their own
novelty and junk routes outnumber gold routes by orders of magnitude; a
*query-anchored* universe starves multi-hop targets by construction,
because containing no query content is precisely what makes a target
multi-hop; and every interpolation between the two carries a constant that
can only be set by peeking at the eval. Route quality is therefore
genuinely statistical, not bookkeeping — a measured existence theorem for
the learned ranker, not a "ML goes here" sticker. It also fixes the
ranker's job description exactly: learn the one quantity no static rule
can supply (route quality), inside an objective shape the audit already
validated (marginal completion), behind a leakage line already drawn (the
proxy is never tuned on the audit's gold).

One qualification is corpus-dependent: the trilemma's unbounded horn
assumes the entity universe is minted from free text, where candidates
can print their own novelty. Sources that ship a curated controlled
vocabulary — CMS taxonomies, category trees, author registries — supply
an externally fixed coverage universe that candidates cannot inflate,
which reopens bounded-coverage selection as a deterministic option in
exactly those deployments. Whether the curated vocabulary actually covers
the bridges that matter is itself measurable (it surfaces as extraction
misses in the breakdown), so the choice between curated-universe
selection and the learned ranker is a per-corpus reading, not a belief.

One training note from the roadmap: reranker training data comes from the
external benchmarks (see metrology below), never from the self-benchmark —
training on labels derived from the same extraction that generated the
candidates would fit the model to the index's own blind spots. And hop-2
relevance labels only make sense once hop-1 selection is decent — the
training curriculum goes one hop at a time.

## Metrology (the part nobody ships)

Measurement comes from two places with very different evidential weight.

### External benchmarks (the headline numbers)

Retrieval quality is proven on externally authored multi-hop QA datasets,
at **corpus scale** — hop expansion over ten distractor paragraphs is
reranking, not retrieval, and recall saturates there for almost any
method. The distractor setting is reported only as a secondary sanity
check. Each dataset has a distinct job:

- **MuSiQue** (the multi-hop headline): constructed specifically to defeat
  single-hop shortcuts — the known-positive, where multi-hop machinery
  should earn its keep or the whole thesis fails. Evaluated over the
  pooled paragraph corpus of its dev split.
- **HotpotQA via BEIR** (scale + known-negative): the full ~5M-passage
  BEIR corpus, retrieval-level metrics, published BM25 and dense baselines
  to compare against directly. It is also the calibration negative for the
  multi-hop diagnostic: the literature documents that a large share of its
  questions are answerable from one paragraph despite two being annotated.
- **2WikiMultihopQA** (discrimination): sits between the two and tests
  whether the diagnostic can tell them apart.

**Protocol.** Comparisons are budget-matched: every system returns the
same k, and the table reports candidates examined and wall-clock latency
alongside recall — a win that costs 50× the compute is still a result,
but it is stated as one. Gold supporting facts are finer-grained than
chunks, so the hit rule is explicit and printed in every report: in eval
mode one source paragraph is one chunk (identity-preserving, no packing),
and a chunk is a hit iff it is a gold paragraph; chunk size cannot
silently inflate or deflate recall.

**Validation gate.** Before any HopTrace number is reported, HopTrace's own BM25
must reproduce the published BM25 baseline on BEIR HotpotQA to within a
point or two. If it doesn't, the tokenization, chunking, or scoring is
wrong and every downstream comparison is invalid. A floor you implemented
yourself isn't a floor until it's been checked against someone else's.

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
between annotated and effective multi-hop is itself a headline
observation — arguably the most interesting number the harness produces.
A tool that says "don't use me" on one dataset and "use me" on another is
far more credible than one that always says use me; when the report says a
corpus doesn't need hops, that is the diagnostic working correctly, and
the write-up presents it as calibration, never as concession.

Two diagnostics ride along:

- **Miss breakdown**: extraction miss (a gold bridge entity was never
  extracted), alias miss (a query mention fails to normalize onto an
  indexed entity), hop-bound miss (gold reachable only beyond hop/beam
  limits), ranking miss (gold was in the candidate pool but fell below k).
  Each is distinguishable because each stage is inspectable — and external
  gold is what makes extraction misses observable at all.
- **Hop-2 pool precision**, with and without the specificity filter — the
  hub-entity filter's value is measured, not asserted.

### Self-benchmark (per-corpus diagnostic only)

Every corpus still gets a generated benchmark: synthetic questions built
from its own entity tables — single-hop questions from one chunk's facts,
multi-hop questions by walking real entity bridges between chunks. Gold
evidence is known by construction and the benchmark regenerates on
re-ingest.

Its circularity is structural and must be stated: questions are generated
by walking bridges the extractor already found, so every question is one
the index can represent — extraction misses are invisible by construction,
and floor-vs-HopTrace comparisons are biased toward HopTrace. That makes the
self-benchmark useless as proof and useful as plumbing: an estimate of the
multi-hop fraction on *your* corpus (sufficiency-based, same definition as
above: a question is effectively multi-hop only when the floor's top-k
fails to cover it), a floor-vs-hops sanity check, and a
regression gate — config or extractor changes re-run the bracket, and
degradations fail loudly. The bracket report carries this caveat in its
own output.

If the bracket says "94% of questions on your corpus are single-hop and
BM25 recalls them at 0.97" — use BM25, keep your money. The eval harness
is the product as much as the retriever is.

## MCP interface

Four tools:

```
ingest(source, corpus_id, config?)      → {chunks, entities, table_stats}
retrieve(query, corpus_id, hops?, k?)   → {evidence: [{chunk, score, path, entities}]}
explain(chunk_id, corpus_id, query?)    → {why: path + scores + stage trace}
bracket(corpus_id, n_questions?)        → {floor, hoptrace@1hop, hoptrace@2hop, oracle, multihop_fraction, miss_breakdown}
```

The `retrieve` response is designed for the generating model to cite:
paths are human-readable strings, so "according to [chunk#31], reached via
Anna → Kowalski" is a one-liner for the assistant to surface.

## Known limits (stated, not discovered)

1. **Coreference is out of scope, and that caps multi-hop.** Real documents
   write "she reports to the CTO" and "his office is 4B"; bridging those
   requires judgment, and judgment is exactly what the ingest path refuses
   to do. Entity-string co-occurrence serves the multi-hop questions whose
   bridges are *named on both ends*. The external benchmarks measure how
   much that leaves on the table; the number is reported, not hidden.
2. **Inflected languages break trivial aliasing.** Case, punctuation and
   plural normalization does not touch inflection — in Polish, *Kowalski*,
   *Kowalskiego*, *Kowalskim* and *Kowalskiemu* are one entity and four
   surface forms that v1 indexes separately. The normalizer is a pluggable
   interface; a lemmatizing implementation (spaCy or Morfeusz) is the v2
   fix. Until then HopTrace is honest on English and lossy on inflected
   languages.

## V1 scope

Python; SQLite; stdio MCP server; zero GPU; zero API keys.

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

Build order is eval-first: the harness and the BM25 floor land before hop
expansion, so every stage of the build leaves a publishable measurement
behind — including the failure case, where "measured multi-hop on N
datasets, here is what one-shot retrieval actually misses" stands on its
own.

## V1.5 roadmap (serving adapter)

MCP covers agentic harnesses; most deployed consumers are not agentic —
website chatbots, CMS backends, plain services. V1.5 adds an HTTP adapter
so those can integrate without an MCP client:

- **FastAPI, same verbs**: `POST /retrieve`, `POST /ingest` (auth-gated,
  admin), `GET /explain/{chunk_id}`, `POST /bracket`. OpenAPI/Swagger
  comes free with the framework.
- **The adoption feature is the response format, not the endpoint**:
  `format=json` returns structured evidence with paths;
  `format=prompt` returns a ready-to-concatenate context block — chunks,
  citations, hop paths rendered as plain text — so the integrator who
  doesn't want to think does `context = curl(...); prompt = context +
  user_query;` and ships. Meeting that developer where they are is the
  product decision.
- **Provenance survives to the end user**: the JSON carries paths, so a
  chatbot can render sources under its answer — "according to [doc], via
  Anna → Kowalski". A bot that shows *why* it said something is a
  differentiator the client's customers actually see, and the
  milliseconds-scale hop latency is imperceptible in a chat flow.
- **No CMS modules**: expose clean HTTP; document the sidecar pattern
  (one curl example, one PHP-ish snippet) and let any CMS integrate
  itself. A module is maintenance surface; a pattern is documentation.
- **Ops minimalism**: API-key header, `corpus_id` multi-tenancy (already
  in the tool schema), a Dockerfile. Rate limiting and everything else
  waits for v2.
- **Structured-source ingest**: CMS backends expose curated metadata —
  taxonomies, categories, author registries — over JSON APIs. Ingest
  accepts these as first-class entities alongside extracted mentions:
  they are the highest-precision bridge entities available (editor-vetted,
  zero extraction risk), and they double as a fixed coverage universe for
  bounded selection (see the trilemma qualification above). This is the
  difference between indexing a CMS and merely indexing its rendered
  text.

## V2 roadmap

- **Learned reranker**: small cross-encoder trained on the external
  benchmarks — never on self-benchmark labels, which would fit the model
  to the index's own blind spots; one-hop first, then two-hop. CPU
  inference.
- **Lemmatizing normalizer** for inflected languages (spaCy or Morfeusz
  behind the pluggable alias interface).
- **Alias resolution beyond normalization**: locality-sensitive bucketing
  of entity mentions (embedding-hashed, computed at ingest, frozen at
  query time) so paraphrased and abbreviated mentions land in the same
  bucket — extends the exact-match index without putting a model in the
  query path.
- **Incremental ingest** (append without full rebuild), multi-corpus
  routing, entity-page summaries (all chunks for an entity, one call).

## V2.5 roadmap (framework adapters — distribution, not capability)

Half the integrator market composes retrieval through framework
interfaces (LangChain `BaseRetriever`, LlamaIndex equivalents) and
discovers components through those frameworks' integration directories;
to that market, an engine without a retriever class is invisible
regardless of quality. The fix is the architecture's existing pattern —
a third and fourth thin adapter over the same core:

- **Separate packages, never in core** (`*-langchain`, `*-llamaindex`):
  framework APIs churn with breaking changes across minor versions;
  adapters are thin, separately versioned, loosely pinned, so adapter
  rot stays quarantined from core releases.
- **Only `retrieve` gets adapted.** Hop paths ride into document
  metadata (provenance survives the interface); the bracket,
  calibration, and diagnostics have no framework slot and are not
  flattened into one. The adapter package README states it plainly:
  this is the door, the house is over there.
- **Non-goals clarification**: "no prompt chains, no orchestration"
  means the engine contains no chain — being a component inside someone
  else's chain is the intended use, not the sin.

Ships at the announce milestone alongside the HTTP adapter — both are
distribution surface. Longer-horizon note: MCP adoption is eating the
integration-directory role, so this gate weakens over time; in the
meantime the CMS-agency market lives in these frameworks and meeting
them at their interface costs ~a hundred lines total.

**Release gate**: dissolved by the rename (ADR 0009). The old name (LRAG)
promised a learned component and gated publication on the v2 reranker;
HopTrace promises hop retrieval with traces, which v1 delivers in full.
Publication timing is now a free choice, not a naming obligation.

## Non-goals

- Not a vector-DB replacement. Hybrid setups (HopTrace + dense retrieval as an
  additional seed source) are expected and supported; the bracket will
  tell you what each source contributes.
- No agentic orchestration, no query-time LLM calls, no prompt chains.
  HopTrace is a retrieval engine with a measurement harness, full stop.
- Not a knowledge graph product. The tables are an index, not an ontology;
  nothing is extracted that requires judgment.

## FAQ

**Why not GraphRAG/LightRAG?** They buy their entity graphs with LLM calls
at ingest — per-corpus cost, and the graph goes stale. HopTrace's index is
deterministic: rebuild is free, extraction is versioned, and every edge is
explainable by pointing at text.

**Why not just BM25?** BM25 is the floor, and HopTrace ships it as the floor.
What BM25 cannot do is take the second hop — and the bracket measures
exactly how much that costs on your corpus before you spend anything.

**Where does learning fit, if anywhere?** Selection, in v2. Candidates
are generated deterministically; a small model orders them. The learning
is spent on the only stage measured to be genuinely statistical (see the
trilemma above) — and the name no longer promises it, so v2 remains an
upgrade, not an IOU.
