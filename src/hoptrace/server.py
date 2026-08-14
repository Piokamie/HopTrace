"""The MCP server: HopTrace's four public tools over stdio.

    ingest(source, corpus_id, config?)    -> {chunks, entities, table_stats}
    retrieve(query, corpus_id, hops?, k?) -> {evidence: [...]}
    explain(chunk_id, corpus_id, query?)  -> {why: ...}
    bracket(corpus_id, n_questions?)      -> {floor, ..., caveat}

Deviation from DESIGN.md documented in the README: ``explain`` takes
``corpus_id`` (chunk ids are per-corpus). Responses are plain JSON; hop
paths ship both as structured edges and as the human-readable one-liner
the generating model can cite verbatim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from hoptrace.bm25 import Bm25
from hoptrace.bracket import run_bracket
from hoptrace.config import (
    ChunkConfig,
    ExtractorConfig,
    RetrievalConfig,
    corpus_path,
    list_corpora,
)
from hoptrace.ingest import ingest_path
from hoptrace.provenance import Evidence
from hoptrace.retrieve import Retriever
from hoptrace.store import Store
from hoptrace.tokenize import ANALYZERS, analyze

mcp = MCPServer(
    "hoptrace",
    instructions=(
        "HopTrace is a deterministic multi-hop retrieval engine. Call `retrieve`"
        " to get evidence chunks with hop paths; cite paths verbatim (e.g."
        " 'according to chunk#31, reached via Anna → Kowalski'). Call"
        " `bracket` to measure whether this corpus needs hop retrieval at"
        " all — its report includes its own caveats."
    ),
)

_INGEST_CONFIG_KEYS = {"target_tokens", "max_tokens", "ner", "spacy_model", "analyzer"}

#: explain() re-ranks the whole candidate pool (bounded by beam caps, in
#: practice a few hundred) so below-default-k ranks stay visible.
_EXPLAIN_PROBE_K = 10_000


@dataclass
class _CachedCorpus:
    store: Store
    retriever: Retriever
    mtime_ns: int
    #: resolved path — HOPTRACE_DATA_DIR is re-read per call, so a cache entry
    #: must never survive a data-dir change on an mtime coincidence.
    path: Path


class CorpusRegistry:
    """Open corpora, invalidated when re-ingest atomically replaces the file."""

    def __init__(self) -> None:
        self._cache: dict[str, _CachedCorpus] = {}

    def get(self, corpus_id: str) -> _CachedCorpus:
        path = corpus_path(corpus_id)
        if not path.is_file():
            available = ", ".join(list_corpora()) or "(none)"
            raise ValueError(f"no corpus {corpus_id!r}; available: {available}")
        mtime_ns = path.stat().st_mtime_ns
        cached = self._cache.get(corpus_id)
        if cached is not None and cached.mtime_ns == mtime_ns and cached.path == path:
            return cached
        if cached is not None:
            cached.store.close()
        store = Store.open(path)
        cached = _CachedCorpus(
            store=store,
            retriever=Retriever(store, RetrievalConfig()),
            mtime_ns=mtime_ns,
            path=path,
        )
        self._cache[corpus_id] = cached
        return cached

    def invalidate(self, corpus_id: str) -> None:
        cached = self._cache.pop(corpus_id, None)
        if cached is not None:
            cached.store.close()


_registry = CorpusRegistry()


def ingest_impl(
    source: str, corpus_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = config or {}
    unknown = set(config) - _INGEST_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"unknown config keys {sorted(unknown)}; known: {sorted(_INGEST_CONFIG_KEYS)}"
        )
    analyzer = str(config.get("analyzer", "english"))
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}; expected one of {ANALYZERS}")
    chunk_cfg = ChunkConfig(
        target_tokens=int(config.get("target_tokens", ChunkConfig.target_tokens)),
        max_tokens=int(config.get("max_tokens", ChunkConfig.max_tokens)),
    )
    if not 0 < chunk_cfg.target_tokens <= chunk_cfg.max_tokens:
        raise ValueError(
            f"invalid chunk sizes: need 0 < target_tokens ({chunk_cfg.target_tokens})"
            f" <= max_tokens ({chunk_cfg.max_tokens})"
        )
    extractor_cfg = ExtractorConfig(
        ner=bool(config.get("ner", False)),
        spacy_model=str(config.get("spacy_model", ExtractorConfig.spacy_model)),
    )
    store, report = ingest_path(
        Path(source).expanduser(),
        corpus_id,
        chunk_cfg=chunk_cfg,
        extractor_cfg=extractor_cfg,
        analyzer=analyzer,
    )
    store.close()
    _registry.invalidate(corpus_id)
    return {
        "corpus_id": corpus_id,
        "documents": report.documents,
        "chunks": report.chunks,
        "entities": report.entities,
        "mentions": report.mentions,
        "table_stats": report.table_stats,
        "skipped": list(report.skipped),
        "analyzer": analyzer,
    }


def retrieve_impl(query: str, corpus_id: str, hops: int = 2, k: int = 8) -> dict[str, Any]:
    if hops not in (0, 1, 2):
        raise ValueError("hops must be 0, 1 or 2")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    cached = _registry.get(corpus_id)
    result = cached.retriever.retrieve(query, hops=hops, k=k)
    return {
        "query": result.query,
        "evidence": [_evidence_payload(e) for e in result.evidence],
        "query_mentions": list(result.query_mentions),
        "unresolved_mentions": list(result.unresolved_mentions),
        "pool_size": result.pool_size,
        "candidates_examined": result.candidates_examined,
        "notes": list(result.notes),
    }


def explain_impl(chunk_id: int, corpus_id: str, query: str | None = None) -> dict[str, Any]:
    cached = _registry.get(corpus_id)
    store = cached.store
    row = store.get_chunk(chunk_id)  # raises KeyError -> tool error for bad ids
    payload: dict[str, Any] = {
        "chunk": {
            "id": row.chunk_id,
            "text": row.text,
            "doc": row.doc_path,
            "title": row.doc_title,
            "heading_context": row.heading_context,
        },
        "entities": store.entities_for_chunk(chunk_id),
    }
    if query is None:
        return payload

    # Rank the whole pool so the chunk's position is visible even when it
    # fell below the default k.
    probe = cached.retriever.retrieve(query, k=_EXPLAIN_PROBE_K)
    cfg_k = RetrievalConfig().k
    hit = next((e for e in probe.evidence if e.chunk_id == chunk_id), None)
    analyzer = store.meta("analyzer") or "simple"
    terms = analyze(query, analyzer)
    bm25 = Bm25(store, k1=RetrievalConfig.bm25_k1, b=RetrievalConfig.bm25_b)
    payload["query"] = {
        "text": query,
        "in_pool": hit is not None,
        "in_top_k": hit is not None and hit.score.rank < cfg_k,
        "default_k": cfg_k,
        "rank": None if hit is None else hit.score.rank,
        "path": None if hit is None else hit.path_string(),
        "score": None if hit is None else asdict(hit.score),
        "bm25_terms": [asdict(t) for t in bm25.explain(terms, chunk_id)],
        "note": (
            None
            if hit is not None
            else "chunk not reached: no entity path from this query's seeds"
            " and no lexical overlap strong enough to seed it"
        ),
    }
    return payload


def bracket_impl(corpus_id: str, n_questions: int = 100) -> dict[str, Any]:
    if not 1 <= n_questions <= 10_000:
        raise ValueError("n_questions must be between 1 and 10000")
    cached = _registry.get(corpus_id)
    report = run_bracket(cached.store, RetrievalConfig(), n_questions=n_questions)
    payload = {
        "floor": asdict(report.rows[0]),
        "hoptrace_1hop": asdict(report.rows[1]),
        "hoptrace_2hop": asdict(report.rows[2]),
        "oracle": report.oracle,
        "multihop_fraction": report.multihop_fraction,
        "miss_breakdown": report.miss_breakdown,
        "n_questions": report.n_questions,
        "n_multi_hop": report.n_multi_hop,
        "k": report.k,
        "caveat": report.caveat,
        "notes": report.notes,
    }
    return payload


def _evidence_payload(evidence: Evidence) -> dict[str, Any]:
    return {
        "chunk": {
            "id": evidence.chunk_id,
            "text": evidence.text,
            "doc": evidence.doc_path,
            "title": evidence.doc_title,
        },
        "score": asdict(evidence.score),
        "hop": evidence.path.hop,
        "seed_source": evidence.path.seed_source,
        "path": evidence.path_string(),
        "path_edges": [asdict(edge) for edge in evidence.path.edges],
        "entities": list(evidence.matched_entities),
    }


@mcp.tool()
def ingest(source: str, corpus_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ingest a file or directory of text/markdown into a named corpus.

    Deterministic and LLM-free; re-ingesting replaces the corpus
    atomically. Optional config keys: target_tokens, max_tokens, ner,
    spacy_model, analyzer ("english" | "simple")."""
    return ingest_impl(source, corpus_id, config)


@mcp.tool()
def retrieve(query: str, corpus_id: str, hops: int = 2, k: int = 8) -> dict[str, Any]:
    """Retrieve evidence chunks with hop paths. Each evidence item carries
    a human-readable `path` you can cite verbatim, plus per-stage score
    components. hops=0 is plain BM25."""
    return retrieve_impl(query, corpus_id, hops, k)


@mcp.tool()
def explain(chunk_id: int, corpus_id: str, query: str | None = None) -> dict[str, Any]:
    """Explain a chunk: its entities, and — given a query — whether and
    why it was (or was not) retrieved: rank, hop path, score components,
    per-term BM25 breakdown."""
    return explain_impl(chunk_id, corpus_id, query)


@mcp.tool()
def bracket(corpus_id: str, n_questions: int = 100) -> dict[str, Any]:
    """Measure this corpus: BM25 floor vs HopTrace at hop 1/2 on a generated
    self-benchmark, the multi-hop fraction, and where misses happen. The
    report includes its own caveat — read it."""
    return bracket_impl(corpus_id, n_questions)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
