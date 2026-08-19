# HopTrace

Deterministic multi-hop retrieval with a measurement harness that tells
you whether you need it, exposed over MCP. No LLM at ingest, none at
retrieval.

## The problem in three files

Three documents in a folder:

- `people/alicja-rud.md` — "Alicja Rud is a data engineer on the
  ingestion crew … who reports to Marek Sosna."
- `people/marek-sosna.md` — "Marek Sosna leads the ingestion crew … He
  occupies Office B12."
- `facilities/office-b12.md` — "Office B12 is a corner room on the
  second floor of Budynek A with a whiteboard wall and a view of the
  courtyard birches."

Ask *"What can the manager of Alicja Rud see from the window?"* A
retriever that matches the question against documents — keyword or
vector — finds the Alicja file. The answer is two files away: Alicja's
file names her manager, Marek's file names his office, the office file
has the view. Neither of the last two shares a word with the question,
and "a view of the courtyard birches" resembles nothing in it. The only
way to get there is to follow the names.

HopTrace does that, without a model. At ingest it builds an inverted
index from entities to the documents that mention them. At query time it
finds the documents that match the question, collects the entities they
mention, follows those entities back into the index to reach the next
ring of documents, and once more — then ranks the lot and returns each
result with the chain it was reached by:

```
$ hoptrace retrieve "What can the manager of Alicja Rud see from the window?" --corpus office -k 4
#1 chunk#8 (people/alicja-rud.md) [mention, hop 0] score 1.0000 (bm25 7.32, terms: Alicja, Rud)
#2 chunk#12 (people/marek-sosna.md) [mention, hop 1] score 0.5231 (bm25 0.00, terms: none)
#3 chunk#25 (vendors/mira-catering.md) [bm25_only, hop 0] score 0.4049 (bm25 2.96, terms: manager)
#4 chunk#5 (facilities/office-b12.md) [mention, hop 2] score 0.1723 (bm25 0.00, terms: none)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"marek sosna" → chunk#12 (people/marek-sosna.md) → entity:"office b12" → chunk#5 (facilities/office-b12.md: "Office B12 is a corner room on the second floor of Budynek…")
```

Result #4 shares zero words with the question (`terms: none`) and is two
bridges from anything that does. The `path` line names each file and the
entity that led to the next, so the assistant reading this over MCP can
answer "the courtyard birches, per `facilities/office-b12.md`, reached
via Alicja Rud → Marek Sosna → Office B12". Result #3 is a lexical hit on
the word *manager* alone, labelled as such. The
[walkthrough](docs/walkthrough.md) has the full transcript including
`explain`, where every BM25 term scores 0.0000 against a hop-reached
chunk.

Also in the box:

- **A learned reranker, opt-in.** A small cross-encoder rescores the
  candidates the deterministic stage found; it reorders them, it cannot
  add one. With it on, the numbers below beat a published LLM-built
  knowledge-graph system on MuSiQue and split 2Wiki with it, on its own
  1,000-question samples.
- **A benchmark that says whether you need any of this.** `bracket` runs
  on your own corpus and ends with a verdict: "effectively single-hop:
  plain BM25 covers it; hop retrieval has nothing to do here", or "keep
  hops on".
- **Every result names its file.** `chunk.doc` + `source_root` in the MCP
  response, and the file inside the `path` string.

## Measurements

BM25 reproduction on BEIR HotpotQA (5.23M passages, 7,405 test queries):
nDCG@10 0.6352 against a published 0.633, recall@100 0.7988 against
0.796. The eval harness gates on this.

On the 1,000-question samples, corpora and metrics published with
HippoRAG (Gutiérrez et al., NeurIPS 2024):

| | R@2 | R@5 |
|---|---|---|
| MuSiQue — HippoRAG, best variant (paper) | 41.0 | 52.1 |
| MuSiQue — HopTrace 2hop + rerank (measured here) | 53.7 | 66.0 |
| 2Wiki — HippoRAG, best variant (paper) | 71.5 | 89.5 |
| 2Wiki — HopTrace 2hop + rerank (measured here) | 73.6 | 86.3 |

2Wiki is held out of training entirely. The two BM25 floors agree to 0.1
points on MuSiQue but differ by 5.3 on 2Wiki, so part of that margin is
baseline rather than method.

The answerability-based `multihop_fraction` measures MuSiQue at 0.654
effectively multi-hop, 2WikiMultihopQA at 0.478 and HotpotQA at 0.377,
though all three annotate ~100% of questions as multi-hop. In the
effectively-multi-hop stratum, all-gold@20 goes from 0.071 to 0.342 on
2Wiki and 0.055 to 0.167 on MuSiQue (v1, development corpus); on
HotpotQA hops cost in aggregate.

Full tables, stratified metrics, miss taxonomies, displacement audits,
the training-leak audit and ablations are in
[docs/results.md](docs/results.md). Every row reproduces with
`hoptrace eval`: the fine-tuned reranker ships with the repository (int8
graph under `models/`) and as a checksum-pinned release download (fp32);
[training/](training/README.md) rebuilds it from scratch.

`bracket` runs the same diagnostic on your own corpus: floor, hop rows,
and multi-hop fraction.

## How it works

```
ingest:    documents → chunker → rule-based mention extractor → SQLite tables
retrieve:  query mentions → seed chunks → bounded hop expansion → interleaved ranking
           → evidence with recorded paths
```

Entity co-occurrence is computed once at ingest and joined at query time.
Expansion is IDF-filtered and beam-bounded; hop-reached chunks are scored
by path propagation (parent relevance × bridge strength) rather than by
query similarity they cannot have. Each stage is inspectable, so the miss
taxonomy (extraction / alias / hop-bound / ranking) is computable.
Design details and the decision log: [DESIGN.md](DESIGN.md),
[docs/adr/](docs/adr/README.md).

Ranking candidates against the question is the only stage left to a
learned model (DESIGN.md, "Learned ranking, deterministic hops"). An
**opt-in path-aware reranker** does it: a MiniLM-class cross-encoder scoring
(query, candidate, *route context*) over the pool the deterministic stage
generated. It reorders candidates and never reaches outside them, so
traces stay auditable. The zero-shot control isolates what route context
alone buys: on MuSiQue all-recall@5 rises 21.8 → 27.2 with the route
prefix and no fine-tuning, and a fine-tuned model *without* the prefix
does not reach that (docs/results.md, "Route context").

## Install & quickstart

Requires Python ≥ 3.10. Core has one direct dependency (the MCP SDK);
the eval harness needs `hoptrace[eval]` (numpy), the learned reranker
`hoptrace[rerank]` (ONNX Runtime + tokenizers — no torch, no GPU), the
optional NER pass `hoptrace[ner]` (spaCy).

```bash
uv sync            # or: pip install -e .
uv run hoptrace ingest examples/office --corpus office
uv run hoptrace retrieve "What equipment is in the room where the calibration team meets?" --corpus office
uv run hoptrace bracket --corpus office
```

Corpora are single SQLite files under `$HOPTRACE_DATA_DIR` (default
`./.hoptrace/`), replaced atomically on re-ingest.

Note that `uv sync` syncs the environment to exactly the base
dependencies — if you already installed extras, it **removes** them. Once
you are using `--extra eval` or `--extra rerank`, keep passing them:
`uv sync --extra eval --extra rerank`.

### On your own corpus

```bash
uv run hoptrace ingest /path/to/docs --corpus mydocs
uv run hoptrace bracket --corpus mydocs -n 200
uv run hoptrace retrieve "a question your documents answer" --corpus mydocs
```

`ingest` takes a file or a directory tree, walks it recursively, and
reads `.md`, `.markdown`, `.txt` and `.rst`. Anything else — PDFs, Word
documents, binaries, unreadable encodings — is skipped and listed on
stderr, so convert those to text first. Native PDF/DOCX/HTML ingest is on
the [roadmap](DESIGN.md#ingest), not built. Headings and paragraphs are
used as chunk boundaries where they exist. Re-ingesting the same
`--corpus` name replaces it atomically; extraction is
deterministic, so re-ingest is cheap and safe to repeat.

`bracket` is the decision step. It generates questions from your corpus's
own entity index and reports:

- `multihop_fraction` — the share of generated questions the BM25 floor
  cannot fully answer. Near zero means hop retrieval has little to do on
  this corpus and BM25 is the cheaper answer.
- `bm25-floor` vs `hoptrace@1hop` / `@2hop` — recall and all-gold at k.
  If the floor already matches the hop rows, the corpus is not
  entity-bridged in a way expansion can exploit.
- `miss_breakdown` — where lost gold went: `extraction` (the bridge
  entity was never extracted), `seed_alias` (query mention didn't
  resolve), `hop_bound` (beyond the beam), `ranking` (in the pool, below
  k). This tells you which knob matters: extraction misses want
  `--ner`; alias misses want normalization work; hop-bound misses want
  wider beams; ranking misses want the reranker.
- `VERDICT` — one sentence derived from the rows: *single-hop* ("plain
  BM25 … covers it; hop retrieval has nothing to do here"), *multi-hop*
  ("keep hops on"), or *hops do not help* ("use BM25 and read the miss
  breakdown"), prefixed with a warning when the corpus is too small for
  the fraction to be stable. `verdict_code` carries the same over MCP.

The bracket generates its questions from the same index it evaluates, so
it cannot see extraction misses and is biased toward HopTrace. It is a
plumbing check on your corpus, not evidence — its report says so, and the
cross-system numbers are in [docs/results.md](docs/results.md).

For calibration, `bracket` on two corpora — the demo with the default
`-n` as shown above, and 11,656 Wikipedia paragraphs with `-n 200`:

| | demo corpus (25 docs) | Wikipedia paragraphs (11,656) |
|---|---|---|
| questions generated | 55 | 200 |
| floor recall@8 / all-gold@8 | 0.955 / 0.909 | 0.728 / 0.490 |
| 1hop recall@8 / all-gold@8 | 1.000 / 1.000 | 0.835 / 0.735 |
| 2hop recall@8 / all-gold@8 | 1.000 / 1.000 | 0.793 / 0.650 |
| multihop_fraction | 0.091 | 0.510 |
| misses at 2hop | none | ranking 43, hop_bound 33 |

The demo corpus is small and its bridges were written on purpose, so it
saturates; treat it as an upper bound, not an expectation. Its
`multihop_fraction` also swings with `-n` (0.091 at the default, 0.333 at
`-n 20`) because 25 chunks support few distinct multi-hop questions —
a corpus this small cannot give the diagnostic a stable signal. Expect
the number to mean something from a few hundred chunks upward. The Wikipedia
column (the MuSiQue evaluation corpus, used here as an ordinary corpus)
is the more representative shape: hops lift complete-evidence retrieval
well above the floor, hop-1 beats hop-2, and the remaining misses split
between ranking and beam limits. Your corpus will differ; that is what
the command is for.

### Turning on the reranker

The learned reranker is opt-in and needs the extra installed:

```bash
uv sync --extra rerank
uv run hoptrace retrieve "your question" --corpus mydocs --rerank
```

With no `--rerank-model`, this uses the fine-tuned path-aware model that
ships with the repository — `models/hoptrace-rerank-minilm-l6/` holds the
int8 graph (23 MB), tokenizer and training manifest, so a clone runs it
offline. Its fp32 graph is a release download (91 MB, cached under
`$HOPTRACE_DATA_DIR/models/`, sha256-pinned): `--rerank-precision fp32`.
The headline tables report fp32; int8 is within 0.005 on every measured
metric at ~2× the speed. The zero-shot base the model was fine-tuned from
is `--rerank-model ms-marco-minilm-l6-v2` (fp32 only; asking it for int8
exits with `no model_int8.onnx in …: this artifact does not ship a int8
graph`). A directory built by [training/](training/README.md) works as
`--rerank-model /path/to/model-dir`, and `HOPTRACE_RERANK_MODEL` sets
the same for the MCP server.

Reranked output prints both scores, since the learned one sets the order:

```
#2 chunk#12 (people/marek-sosna.md) [mention, hop 1] rerank +1.044 (path 0.5231) (bm25 0.00, terms: none)
```

Expect ~0.5–1 s per query on CPU against ~10 ms deterministic. Without
the extra installed, `--rerank` exits with an install hint rather than a
traceback.

### Reading the diagnostics

Two notes in `retrieve` output are worth acting on:

- `unresolved mentions: X` with `seed_source: bm25_only` — a capitalized
  term in your query didn't match any indexed entity, so seeding fell
  back to BM25 alone and no hop started from it. Usually a partial name
  ("Sosna" when the index holds "marek sosna"), an inflected form, an
  abbreviation — or a name that is simply not in the corpus. The
  `terms:` list on each result line shows which query words the chunk
  actually contains; a top hit whose only match is a generic word
  (`terms: manager`) is a lexical coincidence, whatever its score — the
  `score` of the best seed is always 1.0 because seeds are normalized
  to the top BM25 hit. Run `hoptrace explain <chunk_id> --corpus X` to see which
  entities a chunk actually carries, and query with one of those. If the
  entity you expected is missing or truncated, see the sentence-initial
  limit below. Better alias handling is on the
  [roadmap](DESIGN.md#retrieval-quality).
- `[bm25_only, hop 0]` on every result — nothing was reached by hopping;
  the answer came from lexical matching alone, exactly as BM25 would.

Other knobs: `--hops` (0 disables expansion, 2 is default), `-k`
(results returned), `--ner` at ingest for spaCy-based entity extraction
on prose that isn't proper-noun-dense, `--analyzer simple` for no
stemming. Full flags: `hoptrace <subcommand> --help`.

### MCP server

`hoptrace serve` speaks MCP over stdio and exposes four tools:
`ingest(source, corpus_id, config?)`,
`retrieve(query, corpus_id, hops?, k?, rerank?)`,
`explain(chunk_id, corpus_id, query?)`, `bracket(corpus_id, n_questions?)`.

Ingest a corpus first, then point a client at the server:

```bash
uv run hoptrace ingest /path/to/docs --corpus mydocs
```

Claude Desktop (`claude_desktop_config.json`) or Claude Code
(`.mcp.json`):

```json
{
  "mcpServers": {
    "hoptrace": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/hoptrace", "hoptrace", "serve"],
      "env": {
        "HOPTRACE_DATA_DIR": "/abs/path/to/hoptrace/.hoptrace"
      }
    }
  }
}
```

Both paths must be absolute. `HOPTRACE_DATA_DIR` is not optional in
practice: it defaults to `./.hoptrace` relative to the working directory,
and clients launch servers from arbitrary directories — omit it and the
server will report an empty corpus list. `rerank=true` uses the bundled
fine-tuned model (requires the `rerank` extra); add
`"HOPTRACE_RERANK_MODEL": "/abs/path/to/model-dir"` to point it at
another artifact.

#### Network mode

The default above is stdio: the client launches `hoptrace serve` itself
and talks over the pipe. For clients that connect to an already-running
process — remote agents, a shared instance, containers — bind a socket
instead:

```bash
uv run hoptrace serve --http --port 8000
# hoptrace MCP server on http://127.0.0.1:8000/mcp (no auth)
```

Point a client at `http://host:8000/mcp` (MCP streamable-http transport).
There is **no authentication**: keep it on localhost, or put it behind a
reverse proxy or gateway that authenticates. `--host 0.0.0.0` exposes it
on all interfaces and should only be paired with one of those.

This is the MCP transport, not a REST API — the endpoints are JSON-RPC,
not `GET /retrieve`. A plain HTTP/REST adapter for non-MCP consumers is
on the [roadmap](DESIGN.md#serving-and-distribution).

To check the wiring without a client, use the network mode — a stdio
server exits as soon as its stdin closes, so there is nothing useful to
observe by running it by hand:

```bash
uv run hoptrace serve --http --port 8000 &
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'
```

The `ingest` tool's `config` object takes the CLI's ingest settings plus
the spaCy model name: `target_tokens`, `max_tokens`, `ner` (bool),
`spacy_model`, `analyzer` (`"english"` or `"simple"`). Unknown keys are
rejected with the list of accepted ones. Evidence items carry the citable path string,
the structured edges, and per-stage score components. The server ships instructions telling the
client to cite paths verbatim — "according to people/marek-sosna.md
(chunk#12), reached via Alicja Rud → Marek Sosna". Each evidence item names its file
(`chunk.doc`, relative to the response's `source_root`, the absolute
directory given to `ingest`), so a client can link to the document.

### For agents

[skills/hoptrace/SKILL.md](skills/hoptrace/SKILL.md) tells an agent how to
drive the four tools — ingest first, `bracket` to decide whether hops are
worth it, `retrieve` and cite the `path` string verbatim, `explain` when
an answer looks wrong. Copy it into your project's `.claude/skills/` (or
`~/.claude/skills/`) next to the server config above; the server also
ships the short version as MCP `instructions`.

## Known limits

- **Coreference caps multi-hop.** "She reports to the CTO" cannot be
  bridged by entity strings; the external benchmarks measure what that
  costs.
- **Inflected languages break trivial aliasing** (Polish *Kowalski /
  Kowalskiego / Kowalskim* index as separate keys). The normalizer is a
  pluggable interface; lemmatization is the planned fix.
- **A multi-word name that only ever starts sentences loses its first
  word.** Capitalization at a sentence start is ambiguous ("Yesterday"
  vs "Anna"), so a capitalized opener is trusted only if that word also
  appears capitalized mid-sentence somewhere in the corpus. A name like
  *Priya Nair* that appears exclusively as a sentence opener therefore
  indexes as `nair`. One mid-sentence occurrence anywhere in the corpus
  fixes it; `--ner` at ingest also avoids it. Check with
  `hoptrace explain <chunk_id> --corpus X`.
- The self-benchmark is circular by construction.

## Reproducing the results

```bash
uv sync --extra eval --extra rerank
uv run hoptrace eval --dataset beir-hotpotqa --gate        # downloads + builds once (~1h)
# the comparable protocol: HippoRAG's published 1,000-question samples + corpora
# (the table rows: bundled fine-tuned model, fp32 graph)
uv run hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-precision fp32 --diagnostics
uv run hoptrace eval --dataset hipporag-2wiki --hops 2 --selection rerank --rerank-precision fp32 --diagnostics
# the zero-shot control row
uv run hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-model ms-marco-minilm-l6-v2
# development setting (smaller haystack, not externally comparable)
uv run hoptrace eval --dataset musique --hops 2 --diagnostics
```

Datasets download to `$HOPTRACE_DATA_DIR/datasets/` with recorded checksums
and are never committed. Attributions: HotpotQA (CC BY-SA 4.0, via the
BEIR packaging), MuSiQue (CC BY 4.0), 2WikiMultihopQA (Apache 2.0);
published baselines from the Pyserini BEIR reproduction matrix and the
BEIR paper.

## License

MIT.
