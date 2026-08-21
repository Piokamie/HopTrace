"""Bounded hop expansion with recorded paths.

A hop is a join: chunk → co-occurring entities → chunks mentioning them.
Entity specificity filters at expansion time (IDF ranking + hub cutoff),
since hub entities would fill the beam before any scorer could help.
Ties break on (strength desc, id asc) / (idf desc, entity asc).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hoppath.config import RetrievalConfig
from hoppath.provenance import HopEdge, HopPath
from hoppath.store import Store


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    path: HopPath
    #: seeds: query relevance in [0, 1]; hops: parent_score × bridge_strength,
    #: not query similarity (ADR 0005)
    score: float
    hop: int


def bridge_strength(edge: HopEdge) -> float:
    """Strength of one traversed edge, in [0, 1): IDF-normalized entity
    specificity times a saturating mention count."""
    return edge.entity_specificity * edge.mention_count / (edge.mention_count + 1.0)


class Expander:
    """Reusable across queries on one corpus; df and posting-list caches are corpus-level."""

    def __init__(self, store: Store, cfg: RetrievalConfig) -> None:
        self._store = store
        self._cfg = cfg
        self._n = store.n_chunks
        self._max_idf = math.log(max(self._n, 2))
        self._df_cache: dict[str, int] = {}
        self._chunks_cache: dict[str, list[tuple[int, int]]] = {}

    def expand(
        self, seeds: list[Candidate], hops: int | None = None
    ) -> tuple[dict[int, Candidate], int]:
        """Expand seeds by up to ``hops`` (default cfg.hops).

        Returns (pool, touched): one candidate per chunk — the earliest ring
        that reached it wins, the strongest path within a ring — and the count
        of distinct chunks reached by expansion."""
        max_hops = self._cfg.hops if hops is None else hops
        pool: dict[int, Candidate] = {}
        for seed in seeds:
            self._admit(pool, seed)
        frontier = sorted(pool.values(), key=lambda c: (-c.score, c.chunk_id))
        # seeds are already counted in the caller's BM25 tally
        reached_ever: set[int] = set()

        for hop in range(1, max_hops + 1):
            reached: dict[int, Candidate] = {}
            for candidate in frontier:
                for entity, specificity in self._entities_to_follow(candidate):
                    for chunk_id, count in self._chunks_via(entity):
                        if chunk_id in pool:
                            continue
                        edge = HopEdge(
                            entity=entity,
                            chunk_id=chunk_id,
                            entity_specificity=specificity,
                            mention_count=count,
                        )
                        score = candidate.score * bridge_strength(edge)
                        current = reached.get(chunk_id)
                        if current is not None and current.score >= score:
                            continue
                        reached[chunk_id] = Candidate(
                            chunk_id=chunk_id,
                            path=HopPath(
                                candidate.path.query_mention, (*candidate.path.edges, edge)
                            ),
                            score=score,
                            hop=hop,
                        )
            reached_ever.update(reached)
            survivors = sorted(reached.values(), key=lambda c: (-c.score, c.chunk_id))[
                : self._cfg.frontier_chunks
            ]
            for candidate in survivors:
                self._admit(pool, candidate)
            frontier = survivors
        return pool, len(reached_ever)

    def _admit(self, pool: dict[int, Candidate], candidate: Candidate) -> None:
        current = pool.get(candidate.chunk_id)
        if current is None or candidate.score > current.score:
            pool[candidate.chunk_id] = candidate

    def _entities_to_follow(self, candidate: Candidate) -> list[tuple[str, float]]:
        """Co-occurring entities of the candidate's chunk, specificity-filtered
        and IDF-ranked; entities already on this path are skipped."""
        used = {edge.entity for edge in candidate.path.edges}
        entities = [e for e in self._store.entities_for_chunk(candidate.chunk_id) if e not in used]
        dfs = self._dfs(entities)
        hub_cap = self._cfg.hub_cap(self._n)
        ranked: list[tuple[str, float]] = []
        for entity in entities:
            df = dfs.get(entity, 0)
            if df <= 0:
                continue
            if self._cfg.specificity_filter and df > hub_cap:
                continue
            ranked.append((entity, self._specificity(df)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: self._cfg.beam_entities]

    def _chunks_via(self, entity: str) -> list[tuple[int, int]]:
        """Chunks mentioning the entity, ranked by mention count then id,
        beam-capped; one aggregated query per entity, cached."""
        cached = self._chunks_cache.get(entity)
        if cached is None:
            cached = self._store.entity_chunk_counts(entity, self._cfg.beam_chunks)
            self._chunks_cache[entity] = cached
        return cached

    def _specificity(self, df: int) -> float:
        """IDF normalized to [0, 1]: ln(N / df) / ln(N)."""
        return math.log(self._n / df) / self._max_idf if df else 0.0

    def _dfs(self, entities: list[str]) -> dict[str, int]:
        missing = [e for e in entities if e not in self._df_cache]
        if missing:
            self._df_cache.update(self._store.entity_dfs(missing))
            for entity in missing:
                self._df_cache.setdefault(entity, 0)
        return self._df_cache
