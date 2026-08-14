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

    #: The query mention surface that seeded this path, or None when the
    #: seed came from BM25 alone.
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
    """Per-stage components of the deterministic ranking (ADR 0005).

    Seeds and hop-reached chunks live on different scales and are
    interleaved, so there is no single "final" number — ``score`` is the
    within-population score and ``rank`` is the interleaved position.
    """

    population: str  # "seed" | "hop"
    hop: int
    #: raw BM25 vs the query (informational for hop chunks — never their score)
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
    #: score × novelty — the greedy admission objective (ADR 0007)
    gain: float
    #: 0-based admission rank
    rank: int


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

    def path_string(self, snippets: dict[int, str] | None = None) -> str:
        return render_path(self.path, snippets or {self.chunk_id: self.text})


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    evidence: tuple[Evidence, ...]
    #: budget accounting: BM25-scored candidates plus distinct
    #: expansion-reached chunks — an approximation of distinct chunks
    #: examined (the two tallies can overlap slightly, and mention seeds
    #: scored out-of-band are not counted).
    candidates_examined: int
    #: chunks in the candidate pool after expansion, before top-k.
    pool_size: int
    #: the full pool, keyed for the eval diagnostics (miss breakdown, pool
    #: precision); hop per chunk.
    pool_hops: dict[int, int] = field(default_factory=dict)
    #: seeds-only ranking (top-k), for the displacement audit: what the
    #: list would have been had the interleave admitted no hop candidates.
    seed_top: tuple[int, ...] = ()
    query_mentions: tuple[str, ...] = ()
    unresolved_mentions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def render_path(path: HopPath, snippets: dict[int, str]) -> str:
    """The DESIGN.md one-liner for citing evidence."""
    parts: list[str] = []
    if path.query_mention is not None:
        parts.append(f'query:"{path.query_mention}"')
    else:
        parts.append("query:bm25")
    for i, edge in enumerate(path.edges):
        if i > 0:
            parts.append(f'entity:"{edge.entity}"')
        snippet = _snippet(snippets.get(edge.chunk_id, ""))
        parts.append(
            f'chunk#{edge.chunk_id} ("{snippet}")' if snippet else f"chunk#{edge.chunk_id}"
        )
    return " → ".join(parts)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _SNIPPET_CHARS:
        return flat
    return flat[: _SNIPPET_CHARS - 1].rstrip() + "…"
