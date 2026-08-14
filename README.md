# HopTrace

**Deterministic multi-hop retrieval with a measurement harness that tells
you whether you need it — exposed over MCP.** No LLM at ingest, none at
retrieval. The only model you pay for is the one writing the answer.

```
$ hoptrace retrieve "Where does the manager of Alicja Rud sit?" --corpus office
#2 chunk#12 [mention, hop 1] score 0.5231
   path: query:"Alicja Rud" → chunk#8 → entity:"marek sosna" → chunk#12 ("Marek Sosna leads the ingestion crew…")
```

The answer document shares **zero words** with that question — it is
reachable only by following the entity bridge, and the path line says
exactly how it was found. A hop is a join, not an inference: same query,
same path, every time. See the [walkthrough](docs/walkthrough.md) for
the full transcript including `explain`, where every BM25 term scores
0.0000 against the retrieved answer chunk.

## The measurement is the product

Most retrieval projects claim; this one measures, against evidence it
cannot grade itself:

- **The floor is a checked floor.** HopTrace's own BM25 reproduces the
  published baseline on BEIR HotpotQA (5.23M passages, 7,405 test
  queries): nDCG@10 **0.6352** vs published 0.633, recall@100 **0.7988**
  vs 0.796. Nothing downstream counts until that gate passes.
- **The multi-hop diagnostic is calibrated against known dataset
  properties.** Its answerability-based `multihop_fraction` independently
  recovers the literature's findings: MuSiQue (built to defeat shortcuts)
  measures **0.654** effectively multi-hop, 2WikiMultihopQA **0.478**,
  HotpotQA — documented as shortcut-prone — **0.377**, despite all three
  annotating ~100% of questions as multi-hop. An instrument that
  discriminates known cases is one you can point at your own corpus.
- **Where the diagnostic says multi-hop is real, hops pay.** In the
  effectively-multi-hop stratum, complete-evidence retrieval
  (all-gold@20) improves **3–5×** over the floor (2Wiki: 0.071 → 0.342;
  MuSiQue: 0.055 → 0.167). Where it says the corpus is shortcut-prone
  (HotpotQA), hops cost — **and the tool says so**. A retriever that
  recommends against itself where the data warrants is the credibility
  claim of this project.
- Full tables, stratified metrics, miss taxonomies, displacement audits,
  and ablations: [docs/results.md](docs/results.md). Every number is
  reproducible via `hoptrace eval`.

The per-corpus version of that honesty ships as a tool: `bracket`
generates a self-benchmark from *your* corpus and reports the floor, the
hop rows, and the multi-hop fraction — with its own circularity caveat
embedded in the payload. If it says your corpus is 94% single-hop, use
BM25 and keep your money.

## How it works

```
ingest:    documents → chunker → rule-based mention extractor → SQLite tables
retrieve:  query mentions → seed chunks → bounded hop expansion → interleaved ranking
           → evidence with recorded paths
```

Entity co-occurrence is a fact, computed once at ingest and joined at
query time. Expansion is IDF-filtered and beam-bounded; hop-reached
chunks are scored by **path propagation** (parent relevance × bridge
strength) — never by query similarity they cannot have. Everything is
deterministic and every stage is inspectable, which is what makes the
miss taxonomy (extraction / alias / hop-bound / ranking) computable.
Design details and the decision log: [DESIGN.md](DESIGN.md),
[docs/adr/](docs/adr/README.md).

The one genuinely statistical stage — ranking candidates against the
original question — is deliberately not learned in v1. The harness
located the boundary precisely (a measured trilemma; see DESIGN.md,
"Why the ranker is learned and the hops are not"), which is the v2
reranker's job description.

## Install & quickstart

Requires Python ≥ 3.10. Core has one dependency (the MCP SDK); the eval
harness needs `hoptrace[eval]` (numpy), the optional NER pass `hoptrace[ner]`
(spaCy).

```bash
uv sync            # or: pip install -e .
uv run hoptrace ingest examples/office --corpus office
uv run hoptrace retrieve "What equipment is in the room where the calibration team meets?" --corpus office
uv run hoptrace bracket --corpus office
```

Corpora are single SQLite files under `$HOPTRACE_DATA_DIR` (default
`./.hoptrace/`), replaced atomically on re-ingest.

### MCP server

```json
{
  "mcpServers": {
    "hoptrace": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/hoptrace", "hoptrace", "serve"],
      "env": { "HOPTRACE_DATA_DIR": "/path/to/hoptrace/.hoptrace" }
    }
  }
}
```

Set `HOPTRACE_DATA_DIR` explicitly for server use — the default (`./.hoptrace`)
is relative to the working directory, and MCP clients launch servers from
arbitrary directories.

Four tools: `ingest(source, corpus_id, config?)`,
`retrieve(query, corpus_id, hops?, k?)`,
`explain(chunk_id, corpus_id, query?)`, `bracket(corpus_id, n_questions?)`.
One deviation from DESIGN.md's sketch: `explain` takes `corpus_id`,
because chunk ids are per-corpus. Evidence items carry the citable path
string plus structured edges and per-stage score components.

## Known limits (stated, not discovered)

- **Coreference caps multi-hop.** "She reports to the CTO" cannot be
  bridged by entity strings; the external benchmarks measure what that
  leaves on the table, and the number is reported, not hidden.
- **Inflected languages break trivial aliasing** (Polish *Kowalski /
  Kowalskiego / Kowalskim* index as separate keys). The normalizer is a
  pluggable interface; lemmatization is the v2 fix.
- The self-benchmark is circular by construction and says so in its own
  report output.

## Reproducing the results

```bash
uv sync --extra eval
uv run hoptrace eval --dataset beir-hotpotqa --gate        # downloads + builds once (~1h)
uv run hoptrace eval --dataset musique --hops 2 --diagnostics
uv run hoptrace eval --dataset 2wiki --hops 2 --diagnostics
```

Datasets download to `$HOPTRACE_DATA_DIR/datasets/` with recorded checksums
and are never committed. Attributions: HotpotQA (CC BY-SA 4.0, via the
BEIR packaging), MuSiQue (CC BY 4.0), 2WikiMultihopQA (Apache 2.0);
published baselines from the Pyserini BEIR reproduction matrix and the
BEIR paper.

## License

MIT.
