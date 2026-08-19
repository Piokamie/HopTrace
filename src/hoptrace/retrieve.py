"""Top-level retrieval: seed → bounded hop expansion → selection →
evidence with provenance.

Seeds come from query mentions (canonical, then alias lookup) plus BM25
top-N; seed scores are query relevance in [0, 1] and expansion propagates
them through bridge strengths. When no mention resolves, seeding is
BM25-only and the result notes it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from hoptrace.bm25 import Bm25, Bm25Vector
from hoptrace.config import RetrievalConfig
from hoptrace.expand import Candidate, Expander, bridge_strength
from hoptrace.mentions import QUERY_TRUST, MentionExtractor
from hoptrace.provenance import Evidence, HopEdge, HopPath, RetrievalResult, ScoreBreakdown
from hoptrace.rerank import PairScorer, default_scorer, rerank_select, window_size
from hoptrace.score import Selected, candidate_features, greedy_select, interleave
from hoptrace.store import ChunkRow, Store


@dataclass
class _CandidateStage:
    """Everything ``retrieve`` computes before selection."""

    notes: list[str]
    mentions: tuple[str, ...]
    resolved: list[tuple[str, str]]
    unresolved: tuple[str, ...]
    terms: list[str]
    bm25_raw: dict[int, float]
    bm25_candidates: int
    max_raw: float
    seed_top: tuple[int, ...]
    pool: dict[int, Candidate]
    touched: int
    pool_rows: dict[int, ChunkRow]
    pool_order: list[Candidate]


class Retriever:
    """Reusable per-corpus retriever (caches the BM25 scorer and the
    expander's corpus-level lookups)."""

    def __init__(
        self,
        store: Store,
        cfg: RetrievalConfig | None = None,
        extractor: MentionExtractor | None = None,
        scorer: PairScorer | None = None,
    ) -> None:
        self._store = store
        self._cfg = cfg or RetrievalConfig()
        self._extractor = extractor or MentionExtractor()
        self._scorer = scorer
        if self._cfg.selection == "rerank" and self._scorer is None:
            self._scorer = default_scorer(self._cfg)
        self._analyzer = store.meta("analyzer") or "simple"
        self._bm25: Bm25 | Bm25Vector
        try:
            self._bm25 = Bm25Vector(store, k1=self._cfg.bm25_k1, b=self._cfg.bm25_b)
        except RuntimeError:  # numpy not installed: pure-Python path
            self._bm25 = Bm25(store, k1=self._cfg.bm25_k1, b=self._cfg.bm25_b)
        self._expander = Expander(store, self._cfg)

    def retrieve(
        self, query: str, hops: int | None = None, k: int | None = None
    ) -> RetrievalResult:
        from hoptrace.tokenize import analyze

        cfg_hops = self._cfg.hops if hops is None else hops
        cfg_k = self._cfg.k if k is None else k
        stage = self._candidates(query, cfg_hops, cfg_k)
        notes, terms, bm25_raw, max_raw = stage.notes, stage.terms, stage.bm25_raw, stage.max_raw
        pool, pool_rows, pool_order = stage.pool, stage.pool_rows, stage.pool_order

        if self._cfg.selection == "submodular":
            query_terms = frozenset(terms)
            features = {
                cid: candidate_features(
                    pool[cid],
                    frozenset(analyze(pool_rows[cid].text, self._analyzer)),
                    query_terms,
                )
                for cid in pool
            }
            admitted = greedy_select(pool, features, cfg_k)
        elif self._cfg.selection == "rerank":
            scorer = self._scorer
            if scorer is None:  # unreachable: constructor resolves or raises
                raise RuntimeError("selection='rerank' requires a scorer")
            admitted = rerank_select(
                pool_order,
                query,
                {cid: row.text for cid, row in pool_rows.items()},
                scorer,
                cfg_k,
                self._cfg.rerank_top_n,
                self._cfg.rerank_path_context,
            )
        else:
            admitted = [
                Selected(candidate=c, novelty=1.0, gain=c.score) for c in pool_order[:cfg_k]
            ]

        hop_ids = [
            s.candidate.chunk_id
            for s in admitted
            if s.candidate.hop >= 1 and s.candidate.chunk_id not in bm25_raw
        ]
        if hop_ids:
            bm25_raw.update(self._bm25_for(terms, hop_ids))

        evidence: list[Evidence] = []
        query_entities = {entity for _, entity in stage.resolved}
        # query words (not stems) whose analyzed form occurs in the chunk
        query_words = [(w, set(analyze(w, self._analyzer))) for w in re.findall(r"\w+", query)]
        is_rerank = self._cfg.selection == "rerank"
        for rank, selected in enumerate(admitted):
            candidate = selected.candidate
            row = pool_rows[candidate.chunk_id]
            chunk_terms = set(analyze(row.text, self._analyzer))
            matched_terms = tuple(
                dict.fromkeys(w for w, stems in query_words if stems and stems <= chunk_terms)
            )
            last_edge = candidate.path.edges[-1]
            is_hop = candidate.hop >= 1
            parent = candidate.path.edges[-2].chunk_id if is_hop else None
            raw = bm25_raw.get(candidate.chunk_id, 0.0)
            breakdown = ScoreBreakdown(
                population="hop" if is_hop else "seed",
                hop=candidate.hop,
                bm25=raw,
                bm25_norm=(raw / max_raw if max_raw > 0 else 0.0) if not is_hop else 0.0,
                bridge_strength=bridge_strength(last_edge) if last_edge.entity else 0.0,
                parent_chunk=parent,
                score=candidate.score,
                novelty=selected.novelty,
                gain=selected.gain,
                rank=rank,
                rerank_score=selected.gain if is_rerank else None,
            )
            chunk_entities = set(self._store.entities_for_chunk(candidate.chunk_id))
            on_path = {edge.entity for edge in candidate.path.edges if edge.entity}
            evidence.append(
                Evidence(
                    chunk_id=candidate.chunk_id,
                    text=row.text,
                    doc_path=row.doc_path,
                    doc_title=row.doc_title,
                    score=breakdown,
                    path=candidate.path,
                    matched_entities=tuple(sorted(chunk_entities & (on_path | query_entities))),
                    matched_terms=matched_terms,
                    path_docs=tuple(
                        (edge.chunk_id, pool_rows[edge.chunk_id].doc_path)
                        for edge in candidate.path.edges
                        if edge.chunk_id in pool_rows
                    ),
                )
            )
        return RetrievalResult(
            query=query,
            evidence=tuple(evidence),
            candidates_examined=stage.bm25_candidates + stage.touched,
            pool_size=len(pool),
            pool_hops={cid: c.hop for cid, c in pool.items()},
            seed_top=stage.seed_top,
            pool_order=tuple(c.chunk_id for c in pool_order),
            query_mentions=stage.mentions,
            unresolved_mentions=stage.unresolved,
            notes=tuple(notes),
        )

    def candidate_window(
        self, query: str, hops: int | None = None, top_n: int | None = None
    ) -> tuple[list[Candidate], dict[int, str]]:
        """The candidates ``rerank_select`` would score for this configuration,
        plus texts for the whole pool (so hop parents are present); used by
        the training-data builder."""
        cfg_hops = self._cfg.hops if hops is None else hops
        cfg_top_n = self._cfg.rerank_top_n if top_n is None else top_n
        stage = self._candidates(query, cfg_hops, self._cfg.k)
        window = stage.pool_order[: window_size(cfg_top_n, self._cfg.k)]
        return window, {cid: row.text for cid, row in stage.pool_rows.items()}

    def _candidates(self, query: str, hops: int, k: int) -> _CandidateStage:
        """Everything before selection. ``k`` sets the BM25 seed depth, so a
        caller wanting inference's pool must pass inference's k."""
        from hoptrace.tokenize import analyze

        notes: list[str] = []
        mentions = self._extractor.extract(query, corpus_caps=QUERY_TRUST)
        resolved: list[tuple[str, str]] = []  # (surface, canonical-in-index)
        unresolved: list[str] = []
        for m in mentions:
            entity = self._resolve(m.entity, m.surface)
            if entity is None:
                unresolved.append(m.surface)
            else:
                resolved.append((m.surface, entity))
        if not resolved:
            notes.append(
                "seed_source: bm25_only (no query mention resolved)"
                if not unresolved
                else "seed_source: bm25_only — no indexed entity matches "
                + ", ".join(f'"{u}"' for u in unresolved)
                + "; results rest on word overlap alone, not on the named entities"
            )

        terms = analyze(query, self._analyzer)
        # depth follows k so the seeds-only list can always fill k
        seed_depth = max(self._cfg.seed_bm25_top_n, k)
        seed_pairs, bm25_candidates = self._bm25_top(terms, seed_depth)
        bm25_raw: dict[int, float] = dict(seed_pairs)

        mention_edges = self._mention_edges(resolved)
        missing = [cid for cid in mention_edges if cid not in bm25_raw]
        if missing:
            bm25_raw.update(self._bm25_for(terms, missing))

        max_raw = max(bm25_raw.values(), default=0.0)
        seeds: list[Candidate] = []
        for chunk_id in sorted(bm25_raw):
            bm25_norm = bm25_raw[chunk_id] / max_raw if max_raw > 0 else 0.0
            edge_info = mention_edges.get(chunk_id)
            if edge_info is not None:
                surface, edge = edge_info
                score = max(bm25_norm, bridge_strength(edge))
                path = HopPath(surface, (edge,))
            else:
                score = bm25_norm
                path = HopPath(None, (HopEdge("", chunk_id, 0.0, 0),))
            seeds.append(Candidate(chunk_id=chunk_id, path=path, score=score, hop=0))

        pool, touched = self._expander.expand(seeds, hops=hops)
        seed_top = tuple(
            c.chunk_id for c in sorted(seeds, key=lambda c: (-c.score, c.chunk_id))[:k]
        )
        pool_rows = {cid: self._store.get_chunk(cid) for cid in pool}
        # interleave is prefix-consistent: one full ordering serves selection and the eval ceilings
        pool_order = interleave(pool, len(pool))
        return _CandidateStage(
            notes=notes,
            mentions=tuple(m.surface for m in mentions),
            resolved=resolved,
            unresolved=tuple(unresolved),
            terms=terms,
            bm25_raw=bm25_raw,
            bm25_candidates=bm25_candidates,
            max_raw=max_raw,
            seed_top=seed_top,
            pool=pool,
            touched=touched,
            pool_rows=pool_rows,
            pool_order=pool_order,
        )

    def _resolve(self, canonical: str, surface: str) -> str | None:
        if self._store.entity_df(canonical) > 0:
            return canonical
        via_alias = self._store.alias_canonical(surface)
        if via_alias is not None and self._store.entity_df(via_alias) > 0:
            return via_alias
        return None

    def _mention_edges(self, resolved: Sequence[tuple[str, str]]) -> dict[int, tuple[str, HopEdge]]:
        """Best query-mention edge per seed chunk."""
        n = self._store.n_chunks
        max_idf = math.log(max(n, 2))
        edges: dict[int, tuple[str, HopEdge]] = {}
        for surface, entity in resolved:
            df = self._store.entity_df(entity)
            specificity = math.log(n / df) / max_idf if df else 0.0
            for chunk_id, count in self._store.entity_chunk_counts(entity, self._cfg.beam_chunks):
                edge = HopEdge(entity, chunk_id, specificity, count)
                current = edges.get(chunk_id)
                if current is None or bridge_strength(edge) > bridge_strength(current[1]):
                    edges[chunk_id] = (surface, edge)
        return edges

    def _bm25_top(self, terms: list[str], n: int) -> tuple[list[tuple[int, float]], int]:
        if isinstance(self._bm25, Bm25Vector):
            return self._bm25.top_k(terms, n)
        scores = self._bm25.scores(terms)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]
        return ranked, len(scores)

    def _bm25_for(self, terms: list[str], chunk_ids: list[int]) -> dict[int, float]:
        if isinstance(self._bm25, Bm25Vector):
            return self._bm25.scores_for(terms, chunk_ids)
        all_scores = self._bm25.scores(terms)
        return {cid: all_scores.get(cid, 0.0) for cid in chunk_ids}
