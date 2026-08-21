"""Provenance objects: why each piece of evidence arrived.

Every retrieved chunk carries its hop path and per-stage score components;
the human-readable rendering follows the DESIGN.md format:

    query:"Anna" → chunk#12 ("Anna reports to Kowalski") →
    entity:"Kowalski" → chunk#31 ("Kowalski sits in 4B")
"""

from __future__ import annotations

from dataclasses import dataclass, field

_SNIPPET_CHARS = 60


@dataclass(frozen=True)
class HopEdge:
    """One traversal step: the entity followed and the chunk it reached."""

    entity: str
    chunk_id: int
    #: idf of the entity at traversal time, normalized to [0, 1].
    entity_specificity: float
    #: occurrences of the entity in the reached chunk.
    mention_count: int


@dataclass(frozen=True)
class HopPath:
    """From a query mention (or BM25 seeding) through zero or more hops."""

    #: query mention surface that seeded this path; None for BM25-only seeds
    query_mention: str | None
    edges: tuple[HopEdge, ...]

    @property
    def hop(self) -> int:
        return max(0, len(self.edges) - 1)

    @property
    def seed_source(self) -> str:
        return "mention" if self.query_mention is not None else "bm25_only"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-stage components of the ranking (ADR 0005).

    Seeds and hop-reached chunks live on different scales and are interleaved,
    so ``score`` is within-population and ``rank`` is the interleaved position.
    """

    population: str  # "seed" | "hop"
    hop: int
    #: raw BM25 vs the query; informational only for hop chunks
    bm25: float
    #: seeds: BM25 ratio-normalized to the best seed; hops: 0.0
    bm25_norm: float
    #: hops: strength of the last traversed edge; seeds: entity-match strength
    bridge_strength: float
    #: hops: the hop-1/hop-0 parent the score propagated from
    parent_chunk: int | None
    #: within-population score (seeds: query relevance; hops: propagated)
    score: float
    #: fraction of this chunk's coverage features novel at admission time
    novelty: float
    #: admission objective: score × novelty under submodular, learned score under rerank
    gain: float
    #: 0-based admission rank
    rank: int
    #: learned cross-encoder score when selection="rerank"; None otherwise
    rerank_score: float | None = None


@dataclass(frozen=True)
class Evidence:
    chunk_id: int
    text: str
    doc_path: str
    doc_title: str | None
    score: ScoreBreakdown
    path: HopPath
    #: entities of this chunk that occur on the path or in the query.
    matched_entities: tuple[str, ...]
    #: analyzed query terms present in the chunk — empty for a chunk reached by path alone
    matched_terms: tuple[str, ...] = ()
    #: (chunk_id, doc path) for every chunk on the path, so the citation can name files
    path_docs: tuple[tuple[int, str], ...] = ()

    def path_string(self, snippets: dict[int, str] | None = None) -> str:
        return render_path(self.path, snippets or {self.chunk_id: self.text}, dict(self.path_docs))


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    evidence: tuple[Evidence, ...]
    #: BM25-scored candidates plus expansion-reached chunks; approximate (the tallies can overlap)
    candidates_examined: int
    #: chunks in the candidate pool after expansion, before top-k
    pool_size: int
    #: hop per pool chunk, for the eval diagnostics
    pool_hops: dict[int, int] = field(default_factory=dict)
    #: seeds-only top-k (the list with no hop candidates admitted), for the displacement audit
    seed_top: tuple[int, ...] = ()
    #: whole pool in interleave order; prefix-consistent, so pool_order[:n] is a top-n selection
    pool_order: tuple[int, ...] = ()
    query_mentions: tuple[str, ...] = ()
    unresolved_mentions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def render_path(path: HopPath, snippets: dict[int, str], docs: dict[int, str] | None = None) -> str:
    """The citable one-liner: ``query:"Anna" → chunk#12 (people/anna.md: "…") →
    entity:"kowalski" → chunk#31 (rooms/4b.md: "…")``. Each chunk names its
    source file when known, so a citation can point the reader at a document."""
    docs = docs or {}
    parts: list[str] = []
    if path.query_mention is not None:
        parts.append(f'query:"{path.query_mention}"')
    else:
        parts.append("query:bm25")
    for i, edge in enumerate(path.edges):
        if i > 0:
            parts.append(f'entity:"{edge.entity}"')
        parts.append(
            _chunk_ref(edge.chunk_id, docs.get(edge.chunk_id), snippets.get(edge.chunk_id))
        )
    return " → ".join(parts)


def _chunk_ref(chunk_id: int, doc: str | None, text: str | None) -> str:
    snippet = _snippet(text or "")
    inner = ": ".join(part for part in (doc, f'"{snippet}"' if snippet else None) if part)
    return f"chunk#{chunk_id} ({inner})" if inner else f"chunk#{chunk_id}"


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _SNIPPET_CHARS:
        return flat
    return flat[: _SNIPPET_CHARS - 1].rstrip() + "…"
