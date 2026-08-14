"""Self-benchmark generation: synthetic questions from a corpus's own
entity tables. Deterministic given a seed; no LLM, no wall clock.

Two question kinds:

- **Single-hop**: pick a chunk with >= 2 entities; the query is one
  entity's surface form plus content words from the same chunk. Gold =
  that chunk.
- **Multi-hop**: find a bridge entity shared by exactly two chunks (and
  nothing else shared between them); the query is built ENTIRELY from c1
  material — a c1-only entity surface plus c1 content words that do not
  occur in c2 — so c2 shares zero query lexemes by construction and is
  reachable only over the bridge. Gold = {c1, c2}.

Queries are keyword queries, not natural language — the benchmark
exercises retrieval plumbing, not language understanding.

The structural caveat (ADR 0004) is embedded in every bracket report:
questions are generated from bridges the extractor already found, so
extraction misses are invisible and floor-vs-HopTrace comparisons favor HopTrace.
Diagnostic, never evidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hoptrace.store import Store
from hoptrace.tokenize import LUCENE_STOPWORDS, analyze, tokenize

#: Bridges are useful when specific: df in [2, _BRIDGE_DF_MAX].
_BRIDGE_DF_MAX = 4
_BRIDGE_CANDIDATES = 5000
_CONTENT_WORDS_PER_QUERY = 2
_MAX_ATTEMPT_FACTOR = 30


@dataclass(frozen=True)
class BenchQuestion:
    qid: str
    text: str
    gold: frozenset[int]
    kind: str  # "single_hop" | "multi_hop"
    bridge: str | None = None


def generate_benchmark(
    store: Store,
    n_questions: int = 100,
    multihop_share: float = 0.5,
    seed: int = 0,
) -> list[BenchQuestion]:
    """Generate up to n_questions; returns what the corpus supports."""
    rng = random.Random(seed)
    n_multi = round(n_questions * multihop_share)
    n_single = n_questions - n_multi
    questions: list[BenchQuestion] = []
    questions.extend(_multi_hop(store, rng, n_multi))
    questions.extend(_single_hop(store, rng, n_single))
    return questions


def _content_words(
    text: str, entity_tokens: set[str], excluded: set[str], rng: random.Random
) -> list[str]:
    """Deterministic sample of content-bearing, non-entity tokens."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if (
            len(token) >= 3
            and token.isalpha()
            and token not in LUCENE_STOPWORDS
            and token not in entity_tokens
            and token not in excluded
        ):
            seen.setdefault(token, None)
    candidates = list(seen)
    if len(candidates) < _CONTENT_WORDS_PER_QUERY:
        return []
    return rng.sample(candidates, _CONTENT_WORDS_PER_QUERY)


def _entity_tokens(store: Store, chunk_id: int) -> set[str]:
    tokens: set[str] = set()
    for entity in store.entities_for_chunk(chunk_id):
        tokens.update(tokenize(entity))
        surface = store.entity_surface(entity, chunk_id)
        if surface:
            tokens.update(tokenize(surface))
    return tokens


def _single_hop(store: Store, rng: random.Random, n: int) -> list[BenchQuestion]:
    questions: list[BenchQuestion] = []
    n_chunks = store.n_chunks
    if n_chunks == 0 or n <= 0:
        return []
    attempts = 0
    while len(questions) < n and attempts < n * _MAX_ATTEMPT_FACTOR:
        attempts += 1
        chunk_id = rng.randint(1, n_chunks)
        entities = store.entities_for_chunk(chunk_id)
        if len(entities) < 2:
            continue
        entity = rng.choice(sorted(entities))
        surface = store.entity_surface(entity, chunk_id)
        if not surface:
            continue
        chunk = store.get_chunk(chunk_id)
        words = _content_words(chunk.text, _entity_tokens(store, chunk_id), set(), rng)
        if not words:
            continue
        questions.append(
            BenchQuestion(
                qid=f"single-{len(questions)}",
                text=f"{surface} {' '.join(words)}",
                gold=frozenset({chunk_id}),
                kind="single_hop",
            )
        )
    return questions


def _multi_hop(store: Store, rng: random.Random, n: int) -> list[BenchQuestion]:
    questions: list[BenchQuestion] = []
    if store.n_chunks == 0 or n <= 0:
        return []
    # The single-hop-proof check must run in the corpus's ANALYZED term
    # space: raw-token disjointness would still leave stem overlaps
    # ("sprawling" vs "sprawls") for the floor to exploit.
    analyzer = store.meta("analyzer") or "simple"
    bridges = store.entities_in_df_range(2, _BRIDGE_DF_MAX, _BRIDGE_CANDIDATES)
    rng.shuffle(bridges)
    for bridge in bridges:
        if len(questions) >= n:
            break
        chunk_ids = store.chunk_ids_for_entity(bridge)
        if len(chunk_ids) < 2:
            continue
        c1_id, c2_id = rng.sample(chunk_ids, 2)
        c1_entities = set(store.entities_for_chunk(c1_id))
        c2_entities = set(store.entities_for_chunk(c2_id))
        if c1_entities & c2_entities != {bridge}:
            continue  # a second shared entity would offer a non-bridge path
        c1_only = sorted(c1_entities - c2_entities - {bridge})
        if not c1_only:
            continue
        anchor = rng.choice(c1_only)
        surface = store.entity_surface(anchor, c1_id)
        if not surface:
            continue
        c1 = store.get_chunk(c1_id)
        c2 = store.get_chunk(c2_id)
        # words strictly from c1, absent from c2: c2 shares no query lexeme
        words = _content_words(
            c1.text,
            _entity_tokens(store, c1_id),
            excluded=set(tokenize(c2.text)),
            rng=rng,
        )
        if not words:
            continue
        text = f"{surface} {' '.join(words)}"
        if set(analyze(text, analyzer)) & set(analyze(c2.text, analyzer)):
            continue  # enforce the single-hop-proof property in analyzed terms
        questions.append(
            BenchQuestion(
                qid=f"multi-{len(questions)}",
                text=text,
                gold=frozenset({c1_id, c2_id}),
                kind="multi_hop",
                bridge=bridge,
            )
        )
    return questions
