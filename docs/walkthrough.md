# Walkthrough: a multi-hop retrieval, end to end

Every transcript below is real output from the CLI against the shipped
demo corpus (long `text:` lines elided with `…`, editorial notes marked
`# …`) ([examples/office](../examples/office)) — 25 short documents
about a fictional research institute, with entity bridges engineered so
that seven designed questions ([examples/QUESTIONS.md](../examples/QUESTIONS.md))
are genuinely multi-hop: their answer documents share no content words
with the questions (question 5 deliberately shares one stem — see
examples/QUESTIONS.md). All of this is pinned by
[tests/test_demo_corpus.py](../tests/test_demo_corpus.py).

## 1. Ingest

```
$ hoppath ingest examples/office --corpus office
corpus office: 25 documents, 25 chunks, 24 entities, 44 mentions (analyzer=english)
```

Deterministic, sub-second, no LLM. Re-running replaces the corpus
atomically.

## 2. The question the floor cannot answer

*Where does the manager of Alicja Rud sit?* The corpus never says.
One document says Alicja Rud reports to Marek Sosna; a different document
says Marek Sosna occupies Office B12 — and that second document contains
none of the question's words ("occupies", not "sits"; no "manager", no
"Alicja").

With hops disabled (the BM25 floor), the answer document does not appear
— the seed document is found (by its query mention), and the trail stops
there:

```
$ hoppath retrieve "Where does the manager of Alicja Rud sit?" --corpus office --hops 0
#1 chunk#8 (people/alicja-rud.md) [mention, hop 0] score 1.0000 (bm25 7.32, terms: Alicja, Rud)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md: "Alicja Rud is a data engineer on the ingestion crew at Ostr…")
# … #2, #3 are unrelated lexical matches; chunk#12 is absent
```

With hop expansion, it appears at rank 2 — and says how it got there:

```
$ hoppath retrieve "Where does the manager of Alicja Rud sit?" --corpus office --hops 2
#1 chunk#8 (people/alicja-rud.md) [mention, hop 0] score 1.0000 (bm25 7.32, terms: Alicja, Rud)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md: "Alicja Rud is a data engineer on the ingestion crew at Ostr…")
   text: Alicja Rud is a data engineer on the ingestion crew at Ostra Labs. …
#2 chunk#12 (people/marek-sosna.md) [mention, hop 1] score 0.5231 (bm25 0.00, terms: none)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"marek sosna" → chunk#12 (people/marek-sosna.md: "Marek Sosna leads the ingestion crew. Colleagues describe M…")
   text: Marek Sosna leads the ingestion crew. … He occupies Office B12. …
```

The path line is citable verbatim and names the file at every step:
*"according to people/marek-sosna.md (chunk#12), reached via Alicja Rud
→ Marek Sosna"*.

## 3. Why was that chunk retrieved? (`explain`)

```
$ hoppath explain 12 --corpus office --query "Where does the manager of Alicja Rud sit?"
chunk#12 (people/marek-sosna.md, title=Marek Sosna)
entities: marek sosna, office b12, tuesday
for query 'Where does the manager of Alicja Rud sit?': rank #2 (top-k)
path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"marek sosna" → chunk#12 (people/marek-sosna.md: "Marek Sosna leads the ingestion crew. Colleagues describe M…")
  bm25 term 'where': tf=0 df=0 score=0.0000
  bm25 term 'doe': tf=0 df=0 score=0.0000
  bm25 term 'manag': tf=0 df=1 score=0.0000
  bm25 term 'alicja': tf=0 df=1 score=0.0000
  bm25 term 'rud': tf=0 df=1 score=0.0000
  bm25 term 'sit': tf=0 df=1 score=0.0000
```

Every BM25 term contributes zero: the chunk shares no word with the
question and was retrieved through the recorded entity path alone.

## 4. Optional: the learned reranker (`--rerank`)

Everything above is deterministic. With the `rerank` extra installed, the
bundled path-aware cross-encoder (`models/hoppath-rerank-minilm-l6`, int8
graph) rescores the top candidates — it reorders the pool, it
never reaches outside it:

```
$ hoppath retrieve "Where does the manager of Alicja Rud sit?" --corpus office --hops 2 --rerank
#1 chunk#8 (people/alicja-rud.md) [mention, hop 0] rerank +4.586 (path 1.0000) (bm25 7.32, terms: Alicja, Rud)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md: "Alicja Rud is a data engineer on the ingestion crew at Ostr…")
#2 chunk#12 (people/marek-sosna.md) [mention, hop 1] rerank +0.535 (path 0.5231) (bm25 0.00, terms: none)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"marek sosna" → chunk#12 (people/marek-sosna.md: "Marek Sosna leads the ingestion crew. Colleagues describe M…")
#3 chunk#15 (people/tomasz-gil.md) [mention, hop 1] rerank -3.755 (path 0.3923) (bm25 0.00, terms: none)
   path: query:"Alicja Rud" → chunk#8 (people/alicja-rud.md) → entity:"ostra lab" → chunk#15 (people/tomasz-gil.md: "Tomasz Gil maintains the procurement ledger and reconciles…")
```

`path` is the deterministic propagated score, `rerank` the learned one
that set the order. They agree on the ranking here but not on
confidence: the deterministic scorer separates the answer (0.5231) from
an irrelevant hop through the "ostra lab" hub (0.3923) by 0.13, the
reranker by 4.3 logits. On a 25-chunk corpus the reranker's absolute
scores mean little (it is trained at corpus scale — on the two-bridge
question 7 it even demotes the hop-2 answer the deterministic ranking
surfaces); the measured gains are in [results.md](results.md).

Provenance is unchanged — same paths, same citation line (`text:` lines
elided above). Reranking is opt-in; the deterministic interleave stays
the default and both are reported in [results.md](results.md).
`--rerank-precision fp32` loads the fp32 graph (a release download),
`--rerank-model ms-marco-minilm-l6-v2` the zero-shot base, and any
directory built by `training/` works as `--rerank-model`
(`HOPPATH_RERANK_MODEL` does the same for the MCP server).

## 5. Does this corpus even need hops? (`bracket`)

```
$ hoppath bracket --corpus office -n 20
bracket over 15 generated questions (10 single-hop, 5 multi-hop), corpus 25 chunks, k=8, seed=0
  bm25-floor     recall@8: 0.8333   all-gold@8: 0.6667
  hoppath@1hop  recall@8: 1.0000   all-gold@8: 1.0000
  hoppath@2hop  recall@8: 1.0000   all-gold@8: 1.0000
  oracle (gold fits in k): 1.0000
  multihop_fraction (floor-insufficiency): 0.3333
  misses at 2hop: extraction=0, ranking=0, hop_bound=0, seed_alias=0
  note: corpus supported only 15 of 20 requested questions
  VERDICT: 25 chunks is too few for a stable reading (the fraction swings with -n); indicative only: 33% of generated questions need more than the BM25 floor, and hoppath@1hop lifts all-gold@8 from 0.67 to 1.00 (+0.33). Entity-bridged multi-hop is real on this corpus; keep hops on (hoppath@1hop).
  CAVEAT: self-benchmark: questions are generated from this corpus's own entity index, so extraction misses are invisible by construction and floor-vs-HopPath comparisons favor HopPath. Use this bracket to judge whether hop retrieval functions on YOUR corpus and how much of it is multi-hop — never as cross-system evidence.
```

Here ~33% of generated questions are beyond the floor and hop retrieval
covers them; the `VERDICT` line says so — and says when the corpus is too
small to trust the number. Where `multihop_fraction` comes back near zero
the verdict reads "effectively single-hop: plain BM25 … covers it", which
is the bracket recommending against its own product. The self-benchmark is a plumbing diagnostic; externally-validated
numbers are in [results.md](results.md).

## 6. The same four verbs over MCP

`hoppath serve` exposes `ingest`, `retrieve`, `explain`, and `bracket` as
MCP tools over stdio. Claude Desktop / Claude Code configuration:

```json
{
  "mcpServers": {
    "hoppath": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/hoppath", "hoppath", "serve"],
      "env": { "HOPPATH_DATA_DIR": "/path/to/hoppath/.hoppath" }
    }
  }
}
```

The `retrieve` tool returns the same evidence with both the structured
edges and the citable path strings; the server's instructions teach the
client to surface them ("according to people/marek-sosna.md (chunk#12), reached via Alicja Rud →
Marek Sosna").
