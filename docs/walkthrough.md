# Walkthrough: a multi-hop retrieval, end to end

Every transcript below is real output from the CLI against the shipped
demo corpus ([examples/office](../examples/office)) — 25 short documents
about a fictional research institute, with entity bridges engineered so
that six designed questions ([examples/QUESTIONS.md](../examples/QUESTIONS.md))
are genuinely two-hop: their answer documents share no content words with
the questions. All of this is pinned by
[tests/test_demo_corpus.py](../tests/test_demo_corpus.py).

## 1. Ingest

```
$ hoptrace ingest examples/office --corpus office
corpus office: 25 documents, 25 chunks, 24 entities, 45 mentions (analyzer=english)
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
$ hoptrace retrieve "Where does the manager of Alicja Rud sit?" --corpus office --hops 0
#1 chunk#8 [mention, hop 0] score 1.0000
   path: query:"Alicja Rud" → chunk#8 ("Alicja Rud is a data engineer on the ingestion crew at Ostr…")
   #2, #3, … are unrelated lexical matches; chunk#12 is absent
```

With hop expansion, it appears at rank 2 — and says how it got there:

```
$ hoptrace retrieve "Where does the manager of Alicja Rud sit?" --corpus office --hops 2
#1 chunk#8 [mention, hop 0] score 1.0000
   path: query:"Alicja Rud" → chunk#8 ("Alicja Rud is a data engineer on the ingestion crew at Ostr…")
   text: Alicja Rud is a data engineer on the ingestion crew at Ostra Labs. …
#2 chunk#12 [mention, hop 1] score 0.5231
   path: query:"Alicja Rud" → chunk#8 → entity:"marek sosna" → chunk#12 ("Marek Sosna leads the ingestion crew. Colleagues describe M…")
   text: Marek Sosna leads the ingestion crew. … He occupies Office B12 o…
```

That path line is the product: the generating model (or the user) can
cite it verbatim — *"according to chunk#12, reached via Alicja Rud →
Marek Sosna"*. A hop is a join, not an inference, so the same query
returns the same path every time.

## 3. Why was that chunk retrieved? (`explain`)

```
$ hoptrace explain 12 --corpus office --query "Where does the manager of Alicja Rud sit?"
chunk#12 (people/marek-sosna.md, title=Marek Sosna)
entities: budynek a, marek sosna, office b12, tuesday
for query 'Where does the manager of Alicja Rud sit?': rank #2 (top-k)
path: query:"Alicja Rud" → chunk#8 → entity:"marek sosna" → chunk#12 ("Marek Sosna leads the ingestion crew. Colleagues describe M…")
  bm25 term 'where': tf=0 df=0 score=0.0000
  bm25 term 'doe': tf=0 df=0 score=0.0000
  bm25 term 'manag': tf=0 df=1 score=0.0000
  bm25 term 'alicja': tf=0 df=1 score=0.0000
  bm25 term 'rud': tf=0 df=1 score=0.0000
  bm25 term 'sit': tf=0 df=1 score=0.0000
```

Every BM25 term contributes exactly zero — the chunk shares not one word
with the question. It was retrieved purely through the recorded entity
path. This is what "auditable retrieval" means concretely: the answer to
"why this chunk?" is a chain you can point at, not a cosine similarity.

## 4. Does this corpus even need hops? (`bracket`)

```
$ hoptrace bracket --corpus office -n 20
bracket over 15 generated questions (10 single-hop, 5 multi-hop), corpus 25 chunks, k=8, seed=0
  bm25-floor     recall@8: 0.8333   all-gold@8: 0.6667
  hoptrace@1hop  recall@8: 1.0000   all-gold@8: 1.0000
  hoptrace@2hop  recall@8: 1.0000   all-gold@8: 1.0000
  oracle (gold fits in k): 1.0000
  multihop_fraction (floor-insufficiency): 0.3333
  misses at 2hop: extraction=0, ranking=0, hop_bound=0, seed_alias=0
  note: corpus supported only 15 of 20 requested questions
  CAVEAT: self-benchmark: questions are generated from this corpus's own entity index, so extraction misses are invisible by construction and floor-vs-HopTrace comparisons favor HopTrace. Use this bracket to judge whether hop retrieval functions on YOUR corpus and how much of it is multi-hop — never as cross-system evidence.
```

The bracket answers the buying question per corpus: here, ~33% of
generated questions are beyond the floor, and hop retrieval covers all of
them. On a corpus where `multihop_fraction` comes back near zero, the
honest recommendation is printed by the tool itself: use BM25, keep your
money. Note the caveat is part of the report — the self-benchmark is a
plumbing diagnostic, never evidence. The externally-validated numbers
live in [results.md](results.md).

## 5. The same four verbs over MCP

`hoptrace serve` exposes `ingest`, `retrieve`, `explain`, and `bracket` as
MCP tools over stdio. Claude Desktop / Claude Code configuration:

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

The `retrieve` tool returns the same evidence with both the structured
edges and the citable path strings; the server's instructions teach the
client to surface them ("according to chunk#12, reached via Alicja Rud →
Marek Sosna").
