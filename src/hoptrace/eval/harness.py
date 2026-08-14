"""Budget-matched evaluation runner and metrics.

Every compared system returns the same k; every report carries candidates
examined and wall-clock latency next to recall, plus the hit rule and the
BM25 parameters used. Metrics are macro-averaged over queries.

Hit rule (printed in every report): eval indexing is identity-preserving —
one source paragraph is one chunk (indexed as "title text") — and a
retrieved chunk is a hit iff it is a gold paragraph.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from hoptrace.bm25 import Bm25, Bm25Vector
from hoptrace.config import RetrievalConfig
from hoptrace.eval.adapters import EvalQuestion
from hoptrace.eval.corpus_build import build_pooled_index, build_question_index, load_beir_answers
from hoptrace.eval.datasets import EVAL_B, EVAL_K1
from hoptrace.eval.diagnostics import (
    DisplacementAudit,
    MissBreakdown,
    QuestionOutcome,
    audit_displacement,
    classify_misses,
)
from hoptrace.retrieve import Retriever
from hoptrace.store import Store
from hoptrace.tokenize import analyze


def _check_analyzer(store: Store, analyzer: str) -> None:
    """A mismatched analyzer yields silently near-zero recall — the CLI
    guards its own path; library callers get the same guard here."""
    stored = store.meta("analyzer")
    if stored is not None and stored != analyzer:
        raise ValueError(
            f"store was built with analyzer={stored!r}, evaluation requested"
            f" {analyzer!r}; pass the stored analyzer or rebuild"
        )


HIT_RULE = (
    "hit rule: one source paragraph == one chunk (identity-preserving, indexed"
    " as 'title text'); a retrieved chunk is a hit iff it is a gold paragraph"
)


@dataclass
class LatencyStats:
    median_ms: float
    p95_ms: float

    @classmethod
    def from_seconds(cls, samples: Sequence[float]) -> LatencyStats:
        if not samples:
            return cls(median_ms=0.0, p95_ms=0.0)
        ordered = sorted(samples)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return cls(
            median_ms=statistics.median(ordered) * 1000.0,
            p95_ms=ordered[p95_index] * 1000.0,
        )


@dataclass
class StratumMetrics:
    n_queries: int
    recall_at: dict[int, float]
    all_gold_at: dict[int, float]


@dataclass
class FloorReport:
    dataset: str
    setting: str
    system: str
    n_queries: int
    n_chunks: int
    analyzer: str
    k1: float
    b: float
    #: recall@k macro-averaged: |gold in top-k| / |gold|
    recall_at: dict[int, float]
    #: all-gold@k macro-averaged: 1 if every gold paragraph is in top-k
    all_gold_at: dict[int, float]
    ndcg10: float | None
    candidates_mean: float
    candidates_median: float
    latency: LatencyStats
    notes: list[str] = field(default_factory=list)
    #: metrics split by effective-hop stratum — one aggregate number
    #: cannot summarize a corpus that is one-third single-hop.
    strata: dict[str, StratumMetrics] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_text(self) -> str:
        lines = [
            f"dataset: {self.dataset} ({self.setting}) — system: {self.system}",
            f"queries: {self.n_queries}   corpus chunks: {self.n_chunks:,}",
            f"analyzer: {self.analyzer}   BM25 k1={self.k1} b={self.b}",
            HIT_RULE,
        ]
        if self.ndcg10 is not None:
            lines.append(f"nDCG@10: {self.ndcg10:.4f}")
        for k in sorted(self.recall_at):
            lines.append(
                f"recall@{k}: {self.recall_at[k]:.4f}   all-gold@{k}: {self.all_gold_at[k]:.4f}"
            )
        if self.strata:
            for name in sorted(self.strata):
                metrics = self.strata[name]
                per_k = "   ".join(
                    f"r@{k}: {metrics.recall_at[k]:.4f} ag@{k}: {metrics.all_gold_at[k]:.4f}"
                    for k in sorted(metrics.recall_at)
                )
                lines.append(f"stratum {name} (n={metrics.n_queries}): {per_k}")
        lines.append(
            f"candidates examined: mean {self.candidates_mean:,.0f},"
            f" median {self.candidates_median:,.0f}"
        )
        lines.append(
            f"latency per query: median {self.latency.median_ms:.1f} ms,"
            f" p95 {self.latency.p95_ms:.1f} ms (single-threaded, warm)"
        )
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


class _StratifiedAccumulator:
    """Per-stratum recall/all-gold accumulation."""

    def __init__(self, ks: Sequence[int]) -> None:
        self._ks = tuple(ks)
        self._data: dict[str, dict[str, dict[int, list[float]]]] = {}

    def add(self, stratum: str, recall: dict[int, float], all_gold: dict[int, float]) -> None:
        bucket = self._data.setdefault(
            stratum,
            {"recall": {k: [] for k in self._ks}, "all_gold": {k: [] for k in self._ks}},
        )
        for k in self._ks:
            bucket["recall"][k].append(recall[k])
            bucket["all_gold"][k].append(all_gold[k])

    def result(self) -> dict[str, StratumMetrics]:
        return {
            stratum: StratumMetrics(
                n_queries=len(next(iter(bucket["recall"].values()), [])),
                recall_at={k: _mean(bucket["recall"][k]) for k in self._ks},
                all_gold_at={k: _mean(bucket["all_gold"][k]) for k in self._ks},
            )
            for stratum, bucket in self._data.items()
        }


def _recall_metrics(
    ranked: Sequence[int], gold: frozenset[int] | set[int], ks: Sequence[int]
) -> tuple[dict[int, float], dict[int, float]]:
    recall: dict[int, float] = {}
    all_gold: dict[int, float] = {}
    for k in ks:
        top = set(ranked[:k])
        found = len(top & gold)
        recall[k] = found / len(gold) if gold else 0.0
        all_gold[k] = 1.0 if gold and found == len(gold) else 0.0
    return recall, all_gold


def _ndcg_binary(ranked: Sequence[int], gold: frozenset[int] | set[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2.0) for i, chunk_id in enumerate(ranked[:k]) if chunk_id in gold)
    ideal = sum(1.0 / math.log2(i + 2.0) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_floor(
    store: Store,
    query_gold: Sequence[QueryGold],
    dataset: str,
    setting: str = "corpus-scale",
    analyzer: str = "english",
    k1: float = EVAL_K1,
    b: float = EVAL_B,
    ks: Sequence[int] = (2, 5, 10, 20),
    with_ndcg: bool = False,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
    extra_notes: Sequence[str] = (),
    strata: dict[str, str] | None = None,
) -> FloorReport:
    """BM25 floor over prebuilt (query, gold-chunk) pairs."""
    _check_analyzer(store, analyzer)
    scorer = Bm25Vector(store, k1=k1, b=b)
    max_k = max(ks)
    if limit is not None:
        query_gold = query_gold[:limit]

    recalls: dict[int, list[float]] = {k: [] for k in ks}
    all_golds: dict[int, list[float]] = {k: [] for k in ks}
    ndcgs: list[float] = []
    candidate_counts: list[int] = []
    latencies: list[float] = []
    stratified = _StratifiedAccumulator(ks) if strata is not None else None

    for i, qg in enumerate(query_gold, 1):
        terms = analyze(qg.text, analyzer)
        started = time.perf_counter()
        ranked_pairs, candidates = scorer.top_k(terms, max_k)
        latencies.append(time.perf_counter() - started)
        ranked = [chunk_id for chunk_id, _ in ranked_pairs]
        recall, all_gold = _recall_metrics(ranked, qg.gold, ks)
        for k in ks:
            recalls[k].append(recall[k])
            all_golds[k].append(all_gold[k])
        if stratified is not None and strata is not None:
            stratified.add(strata.get(qg.qid, "unknown"), recall, all_gold)
        if with_ndcg:
            ndcgs.append(_ndcg_binary(ranked, qg.gold, 10))
        candidate_counts.append(candidates)
        if progress and i % 500 == 0:
            progress(f"{i}/{len(query_gold)} queries")

    notes = list(extra_notes)
    if limit is not None:
        notes.append(f"limited to first {limit} queries — NOT the reportable number")
    return FloorReport(
        dataset=dataset,
        setting=setting,
        system="bm25-floor",
        n_queries=len(query_gold),
        n_chunks=store.n_chunks,
        analyzer=analyzer,
        k1=k1,
        b=b,
        recall_at={k: _mean(recalls[k]) for k in ks},
        all_gold_at={k: _mean(all_golds[k]) for k in ks},
        ndcg10=_mean(ndcgs) if with_ndcg else None,
        candidates_mean=_mean([float(c) for c in candidate_counts]),
        candidates_median=float(statistics.median(candidate_counts)) if candidate_counts else 0.0,
        latency=LatencyStats.from_seconds(latencies),
        notes=notes,
        strata=stratified.result() if stratified is not None else None,
    )


def evaluate_beir_floor(
    store: Store,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    dataset: str = "beir-hotpotqa",
    analyzer: str = "english",
    k1: float = EVAL_K1,
    b: float = EVAL_B,
    ks: Sequence[int] = (2, 5, 10, 20, 100),
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> FloorReport:
    """BM25 floor over a persistent BEIR index; nDCG@10 for the gate."""
    query_gold = beir_query_gold(store, queries, qrels)
    dropped = sum(1 for qid in queries if qid in qrels) - len(query_gold)
    notes = [f"{dropped} queries dropped: gold doc ids missing from index"] if dropped else []
    return evaluate_floor(
        store,
        query_gold,
        dataset,
        setting="corpus-scale",
        analyzer=analyzer,
        k1=k1,
        b=b,
        ks=ks,
        with_ndcg=True,
        limit=limit,
        progress=progress,
        extra_notes=notes,
    )


def evaluate_pooled_floor(
    questions: Sequence[EvalQuestion],
    dataset: str,
    analyzer: str = "english",
    k1: float = EVAL_K1,
    b: float = EVAL_B,
    ks: Sequence[int] = (2, 5, 10, 20),
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> FloorReport:
    """BM25 floor over the pooled dev-split corpus."""
    if limit is not None:
        questions = questions[:limit]
    pooled = build_pooled_index(questions, analyzer=analyzer, progress=progress)
    query_gold = pooled_query_gold(questions, pooled.gold_chunks)
    report = evaluate_floor(
        pooled.store,
        query_gold,
        dataset,
        setting="pooled",
        analyzer=analyzer,
        k1=k1,
        b=b,
        ks=ks,
        limit=None,  # already applied to questions above
        progress=progress,
    )
    pooled.store.close()
    if limit is not None:
        report.notes.append(f"limited to first {limit} questions — NOT the reportable number")
    return report


def evaluate_distractor_floor(
    questions: Sequence[EvalQuestion],
    dataset: str,
    analyzer: str = "english",
    k1: float = EVAL_K1,
    b: float = EVAL_B,
    ks: Sequence[int] = (2, 5),
    limit: int | None = None,
) -> FloorReport:
    """Sanity-check appendix: per-question corpora saturate for any method."""
    if limit is not None:
        questions = questions[:limit]
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    all_golds: dict[int, list[float]] = {k: [] for k in ks}
    latencies: list[float] = []
    candidate_counts: list[int] = []

    for question in questions:
        store, chunk_to_key = build_question_index(question, analyzer)
        gold = {cid for cid, key in chunk_to_key.items() if key in question.gold_keys}
        scorer = Bm25(store, k1=k1, b=b)
        terms = analyze(question.text, analyzer)
        started = time.perf_counter()
        ranked = [cid for cid, _ in scorer.top_k(terms, max(ks))]
        latencies.append(time.perf_counter() - started)
        candidate_counts.append(len(scorer.scores(terms)))
        recall, all_gold = _recall_metrics(ranked, gold, ks)
        for k in ks:
            recalls[k].append(recall[k])
            all_golds[k].append(all_gold[k])
        store.close()

    return FloorReport(
        dataset=dataset,
        setting="distractor (sanity check only — saturates by construction)",
        system="bm25-floor",
        n_queries=len(questions),
        n_chunks=0,
        analyzer=analyzer,
        k1=k1,
        b=b,
        recall_at={k: _mean(recalls[k]) for k in ks},
        all_gold_at={k: _mean(all_golds[k]) for k in ks},
        ndcg10=None,
        candidates_mean=_mean([float(c) for c in candidate_counts]),
        candidates_median=float(statistics.median(candidate_counts)) if candidate_counts else 0.0,
        latency=LatencyStats.from_seconds(latencies),
        notes=["per-question corpora of 10-20 paragraphs; reranking, not retrieval"],
    )


@dataclass(frozen=True)
class QueryGold:
    """One evaluable query: text, gold chunk ids, answer metadata."""

    qid: str
    text: str
    gold: frozenset[int]
    answer: str = ""
    qtype: str = "unknown"


def beir_query_gold(
    store: Store,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    queries_jsonl: Path | None = None,
) -> list[QueryGold]:
    doc_to_chunk = _doc_id_to_chunk_map(store, qrels)
    answers = load_beir_answers(queries_jsonl) if queries_jsonl else {}
    out: list[QueryGold] = []
    for qid in sorted(queries):
        if qid not in qrels:
            continue
        gold = frozenset(doc_to_chunk[d] for d in qrels[qid] if d in doc_to_chunk)
        if not gold:
            continue
        out.append(QueryGold(qid=qid, text=queries[qid], gold=gold, answer=answers.get(qid, "")))
    return out


def pooled_query_gold(
    questions: Sequence[EvalQuestion], gold_chunks: dict[str, frozenset[int]]
) -> list[QueryGold]:
    return [
        QueryGold(
            qid=q.qid,
            text=q.text,
            gold=gold_chunks[q.qid],
            answer=q.answer,
            qtype=q.qtype,
        )
        for q in questions
    ]


def evaluate_hoptrace(
    store: Store,
    query_gold: Sequence[QueryGold],
    dataset: str,
    setting: str,
    hops: int,
    cfg: RetrievalConfig | None = None,
    ks: Sequence[int] = (2, 5, 10, 20),
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
    misses: MissBreakdown | None = None,
    strata: dict[str, str] | None = None,
    displacement: DisplacementAudit | None = None,
) -> FloorReport:
    """Budget-matched HopTrace run at a given hop count over prebuilt gold."""
    base = cfg or RetrievalConfig()
    run_cfg = replace(base, hops=hops, k=max(ks))
    retriever = Retriever(store, run_cfg)
    if limit is not None:
        query_gold = query_gold[:limit]

    recalls: dict[int, list[float]] = {k: [] for k in ks}
    all_golds: dict[int, list[float]] = {k: [] for k in ks}
    candidate_counts: list[int] = []
    latencies: list[float] = []
    stratified = _StratifiedAccumulator(ks) if strata is not None else None

    for i, qg in enumerate(query_gold, 1):
        started = time.perf_counter()
        result = retriever.retrieve(qg.text)
        latencies.append(time.perf_counter() - started)
        ranked = [e.chunk_id for e in result.evidence]
        recall, all_gold = _recall_metrics(ranked, qg.gold, ks)
        for k in ks:
            recalls[k].append(recall[k])
            all_golds[k].append(all_gold[k])
        if stratified is not None and strata is not None:
            stratified.add(strata.get(qg.qid, "unknown"), recall, all_gold)
        candidate_counts.append(result.candidates_examined)
        if misses is not None:
            classify_misses(
                store, result, qg.gold, max(ks), misses, run_cfg.hub_cap(store.n_chunks)
            )
        if displacement is not None:
            audit_displacement(result, qg.gold, displacement.k, displacement)
        if progress and i % 200 == 0:
            progress(f"{i}/{len(query_gold)} queries (hops={hops})")

    notes = [
        f"retrieval config: hops={hops}, beam_entities={run_cfg.beam_entities},"
        f" beam_chunks={run_cfg.beam_chunks}, frontier_chunks={run_cfg.frontier_chunks},"
        f" hub_df_ratio={run_cfg.hub_df_ratio}, specificity_filter={run_cfg.specificity_filter},"
        " scoring=propagated+interleave (ADR 0005)"
    ]
    if limit is not None:
        notes.append(f"limited to first {limit} queries — NOT the reportable number")
    return FloorReport(
        dataset=dataset,
        setting=setting,
        system=f"hoptrace@{hops}hop",
        n_queries=len(query_gold),
        n_chunks=store.n_chunks,
        analyzer=store.meta("analyzer") or "simple",
        k1=run_cfg.bm25_k1,
        b=run_cfg.bm25_b,
        recall_at={k: _mean(recalls[k]) for k in ks},
        all_gold_at={k: _mean(all_golds[k]) for k in ks},
        ndcg10=None,
        candidates_mean=_mean([float(c) for c in candidate_counts]),
        candidates_median=float(statistics.median(candidate_counts)) if candidate_counts else 0.0,
        latency=LatencyStats.from_seconds(latencies),
        notes=notes,
        strata=stratified.result() if stratified is not None else None,
    )


def floor_outcomes(
    store: Store,
    query_gold: Sequence[QueryGold],
    analyzer: str = "english",
    k1: float = EVAL_K1,
    b: float = EVAL_B,
    answer_k: int = 10,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[QuestionOutcome]:
    """Floor top-k with chunk texts, for the calibration diagnostic."""
    _check_analyzer(store, analyzer)
    scorer = Bm25Vector(store, k1=k1, b=b)
    if limit is not None:
        query_gold = query_gold[:limit]
    outcomes: list[QuestionOutcome] = []
    for i, qg in enumerate(query_gold, 1):
        ranked_pairs, _ = scorer.top_k(analyze(qg.text, analyzer), answer_k)
        chunk_ids = tuple(cid for cid, _ in ranked_pairs)
        texts = tuple(store.get_chunk(cid).text for cid in chunk_ids)
        outcomes.append(
            QuestionOutcome(
                qid=qg.qid,
                qtype=qg.qtype,
                answer=qg.answer,
                gold_chunks=qg.gold,
                floor_top_chunks=chunk_ids,
                floor_top_texts=texts,
            )
        )
        if progress and i % 500 == 0:
            progress(f"{i}/{len(query_gold)} outcomes")
    return outcomes


def write_report(report: FloorReport, json_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report.to_json() + "\n", encoding="utf-8")


def _doc_id_to_chunk_map(store: Store, qrels: dict[str, set[str]]) -> dict[str, int]:
    """Reverse map for exactly the gold doc ids (small), not all 5M.

    BEIR indexing is identity-preserving, so each doc has exactly one chunk.
    """
    wanted = {doc_id for gold in qrels.values() for doc_id in gold}
    return {
        path: chunk_ids[0] for path, chunk_ids in store.chunks_by_doc_path(sorted(wanted)).items()
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
