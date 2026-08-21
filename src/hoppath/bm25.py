"""Okapi BM25 (Lucene variant) over packed SQLite postings.

``Bm25`` accumulates in Python dicts (the MCP path, small corpora);
``Bm25Vector`` decodes posting blobs into numpy arrays (the eval path; numpy
comes from the ``eval`` extra). Document length is the analyzed term count
(``chunk_terms``), not the raw token count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoppath.store import Store

_NP_POSTING_DTYPE_FIELDS = [("id", "<u4"), ("tf", "<u2")]


@dataclass(frozen=True)
class TermScore:
    term: str
    df: int
    idf: float
    tf: int
    score: float


def idf(n_chunks: int, df: int) -> float:
    return math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))


class Bm25:
    """Pure-Python scorer for small corpora."""

    def __init__(self, store: Store, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = store
        self._k1 = k1
        self._b = b
        self._n = store.n_chunks
        self._avg = store.avg_term_len or 1.0

    def scores(self, terms: Sequence[str]) -> dict[int, float]:
        acc: dict[int, float] = {}
        for term in sorted(set(terms)):
            postings = self._store.postings(term)
            if not postings:
                continue
            w = idf(self._n, len(postings))
            for chunk_id, tf in postings:
                dl = self._store.term_count(chunk_id)
                acc[chunk_id] = acc.get(chunk_id, 0.0) + w * self._tf_part(tf, dl)
        return acc

    def top_k(self, terms: Sequence[str], k: int) -> list[tuple[int, float]]:
        acc = self.scores(terms)
        ranked = sorted(acc.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def explain(self, terms: Sequence[str], chunk_id: int) -> list[TermScore]:
        dl = self._store.term_count(chunk_id)
        result: list[TermScore] = []
        for term in dict.fromkeys(terms):
            df = self._store.term_df(term)
            w = idf(self._n, df) if df else 0.0
            tf = next((tf for cid, tf in self._store.postings(term) if cid == chunk_id), 0)
            score = w * self._tf_part(tf, dl) if tf else 0.0
            result.append(TermScore(term=term, df=df, idf=w, tf=tf, score=score))
        return result

    def _tf_part(self, tf: int, dl: int) -> float:
        return tf * (self._k1 + 1.0) / (tf + self._k1 * (1.0 - self._b + self._b * dl / self._avg))


class Bm25Vector:
    """Numpy scorer for corpus-scale evaluation.

    Requires contiguous chunk ids 1..N (guaranteed by ``StoreWriter``:
    autoincrement, no deletes) and numpy (the ``eval`` extra).
    """

    def __init__(self, store: Store, k1: float = 1.5, b: float = 0.75) -> None:
        np = _require_numpy()
        self._np = np
        self._store = store
        self._k1 = k1
        self._b = b
        self._n = store.n_chunks
        self._avg = store.avg_term_len or 1.0
        lengths = np.zeros(self._n, dtype=np.float32)
        seen = 0
        for chunk_id, n_terms in store.iter_term_counts():
            if not 1 <= chunk_id <= self._n:
                raise ValueError(f"non-contiguous chunk id {chunk_id} (n_chunks={self._n})")
            lengths[chunk_id - 1] = n_terms
            seen += 1
        if seen != self._n:
            raise ValueError(f"term counts cover {seen} of {self._n} chunks")
        self._norm = k1 * (1.0 - b + b * lengths / self._avg)
        self._dtype = np.dtype(_NP_POSTING_DTYPE_FIELDS)

    def scores_for(self, terms: Sequence[str], chunk_ids: Sequence[int]) -> dict[int, float]:
        """BM25 for exactly the given chunks; postings are chunk-id sorted, so
        each term is one binary search."""
        np = self._np
        wanted = np.array(sorted(set(chunk_ids)), dtype=np.int64)
        if wanted.size == 0:
            return {}
        acc = np.zeros(wanted.size, dtype=np.float64)
        for term in sorted(set(terms)):
            row = self._store.postings_blob(term)
            if row is None:
                continue
            df, blob = row
            arr = np.frombuffer(blob, dtype=self._dtype)
            ids = arr["id"].astype(np.int64)
            pos = np.searchsorted(ids, wanted)
            in_range = pos < ids.size
            hit = in_range.copy()
            hit[in_range] = ids[pos[in_range]] == wanted[in_range]
            if not hit.any():
                continue
            tf = arr["tf"][pos[hit]].astype(np.float64)
            w = idf(self._n, df)
            norm = self._norm[wanted[hit] - 1]
            acc[hit] += w * tf * (self._k1 + 1.0) / (tf + norm)
        return {int(cid): float(score) for cid, score in zip(wanted, acc, strict=True)}

    def top_k(self, terms: Sequence[str], k: int) -> tuple[list[tuple[int, float]], int]:
        """Returns (ranked results, candidates examined)."""
        np = self._np
        scores = np.zeros(self._n, dtype=np.float32)
        touched = np.zeros(self._n, dtype=bool)
        for term in sorted(set(terms)):
            row = self._store.postings_blob(term)
            if row is None:
                continue
            df, blob = row
            arr = np.frombuffer(blob, dtype=self._dtype)
            idx = arr["id"].astype(np.int64) - 1
            tf = arr["tf"].astype(np.float32)
            w = idf(self._n, df)
            scores[idx] += w * tf * (self._k1 + 1.0) / (tf + self._norm[idx])
            touched[idx] = True
        candidates = int(touched.sum())
        if candidates == 0:
            return [], 0
        k = min(k, candidates)
        # argpartition breaks k-th-score ties arbitrarily; re-rank the boundary
        # tie class for a deterministic (-score, chunk_id) order.
        rough = np.argpartition(-scores, k - 1)[:k]
        boundary = scores[rough].min()
        contenders = np.flatnonzero(scores >= boundary)
        order = np.lexsort((contenders, -scores[contenders]))
        chosen = contenders[order][:k]
        ranked = [(int(i) + 1, float(scores[i])) for i in chosen]
        return ranked, candidates


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "corpus-scale BM25 requires numpy: install with `pip install 'hoppath[eval]'`"
        ) from exc
    return numpy
