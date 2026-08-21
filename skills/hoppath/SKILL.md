---
name: hoppath
description: Use HopPath's MCP tools (ingest, bracket, retrieve, explain) to answer questions from a local document corpus with cited evidence paths. Use when a HopPath server is connected and the user asks about their documents, wants multi-hop evidence, or asks whether their corpus needs hop retrieval at all.
---

# HopPath over MCP

HopPath is a deterministic, LLM-free multi-hop retriever. It returns
evidence chunks **with the path it took to reach them** — you answer from
the chunks and cite the paths. Four tools; same behaviour over stdio and
streamable-http.

## Workflow

1. **Is the corpus ingested?** `retrieve` on an unknown `corpus_id` returns
   an error listing the available corpora. If the user's documents are not
   there yet:
   `ingest(source="/abs/path/to/docs", corpus_id="<name>")` — walks a file
   or directory tree; reads `.md`, `.markdown`, `.txt`, `.rst`; skips and
   lists everything else. Re-ingest replaces the corpus atomically; it is
   deterministic, so repeating it is safe. Optional
   `config={"ner": true}` for prose that is not proper-noun-dense.
2. **Should hops be on at all?** Before promising multi-hop answers on a
   new corpus, `bracket(corpus_id, n_questions=200)` once and read:
   - `multihop_fraction` — share of generated questions the BM25 floor
     cannot fully answer. Near 0: this corpus is effectively single-hop;
     plain lexical search (`hops=0`) is the cheaper answer.
   - `floor` vs `hoppath_1hop` / `hoppath_2hop` — recall and all-gold
     at k. If the floor already matches, hops add nothing here.
   - `miss_breakdown` — `extraction` (bridge entity never extracted →
     re-ingest with `ner`), `seed_alias` (query wording didn't resolve →
     rephrase with names as the index spells them), `hop_bound`,
     `ranking`.
   - `verdict` / `verdict_code` (`single_hop` | `multi_hop` |
     `hops_do_not_help` | `unstable`) — the one-sentence reading of the
     rows; `single_hop` means answer with `hops=0` and say hop retrieval
     is not needed on this corpus.
   - `caveat` — the bracket is a self-benchmark; treat it as a plumbing
     check on this corpus, never as cross-system evidence.
3. **Answer a question:** `retrieve(query, corpus_id, hops=2, k=8)`.
   - `hops=0` is plain BM25; `2` is the default. `rerank=true` turns on
     the learned path-aware reranker (opt-in; ~0.5–1 s/query on CPU vs
     ~10 ms; needs the server's `rerank` extra).
   - Each `evidence` item has `chunk.text`, `chunk.doc` (source file),
     `chunk.title`, `hop`, `seed_source`, `path` (the citable string),
     `path_edges`, `score`, `entities`, `matched_terms` (the query words
     the chunk actually contains); the response carries `source_root`.
4. **Cite the path, verbatim, and name the file.** Say *"according to
   people/marek-sosna.md (chunk#12), reached via Alicja Rud → Marek Sosna"*. Every step of the
   `path` string names its source file; `chunk.doc` is that file relative
   to `source_root` (the absolute directory that was ingested), so you
   can point the user at the exact document. Hop-reached chunks
   (`hop` ≥ 1) often share no words with the question; the path is the
   justification, not lexical overlap.
5. **When an answer looks wrong or incomplete**, `explain(chunk_id,
   corpus_id, query=...)`: `query.in_pool` / `query.rank` say whether the
   chunk was ever reachable and where it ranked; `bm25_terms` shows the
   per-term lexical contribution; `entities` shows what the chunk
   actually carries so you can re-query with a name the index knows.

## Reading `retrieve` diagnostics

- `unresolved_mentions` non-empty and `notes` containing
  `seed_source: bm25_only` — a capitalized term in the query matched no
  indexed entity, so no hop started from it. Usually a partial or
  inflected name, or a name not in the corpus at all; check
  `explain(...).entities` on a seed chunk and re-query with that surface
  form. Before answering from such a result, look at `matched_terms`: a
  top hit whose only overlap is a generic word (`["manager"]`) is a
  lexical coincidence — say you found nothing about the named entity.
  `score` is relative (the best seed is always 1.0), so it cannot tell
  you this; `matched_terms` and `score.bm25` can.
- Every item `hop == 0` with `seed_source == "bm25_only"` — nothing was
  reached by hopping; the result is exactly what lexical search would
  give. Say so if the user expected a bridged answer.
- `pool_size` / `candidates_examined` — how much was considered; a tiny
  pool on a large corpus means the query seeded poorly.

## Limits to state

- Bridges must be **named on both ends**: "she reports to the CTO" cannot
  be followed. If the evidence chain needs coreference, say the path is
  not available and answer from what was retrieved.
- Inflected languages alias poorly (Polish cases index separately).
- The bracket cannot see extraction misses (it generates questions from
  the same index it evaluates).

## Server setup (for the user, not for you to run)

```json
{"mcpServers": {"hoppath": {"command": "uv",
  "args": ["run", "--directory", "/abs/path/to/hoppath", "hoppath", "serve"],
  "env": {"HOPPATH_DATA_DIR": "/abs/path/to/hoppath/.hoppath"}}}}
```

`HOPPATH_DATA_DIR` must be absolute — clients launch servers from
arbitrary directories and a relative default yields an empty corpus list.
`hoppath serve --http --port 8000` exposes the same tools at
`http://127.0.0.1:8000/mcp` (no auth: localhost or behind a proxy).
