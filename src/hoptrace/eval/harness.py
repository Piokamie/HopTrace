"""Budget-matched evaluation runner and metrics.

Every compared system returns the same k; every report carries candidates
examined, latency, the hit rule and the BM25 parameters next to recall.
Metrics are macro-averaged over queries.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

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
from hoptrace.rerank import ResolvedModel, resolve_model, sha256_of, window_size
from hoptrace.retrieve import Retriever
from hoptrace.store import Store
from hoptrace.tokenize import analyze


def _check_analyzer(store: Store, analyzer: str) -> None:
    """A mismatched analyzer yields silently near-zero recall."""
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

#: Transfer holdouts: excluded from training entirely (ADR 0011).
HOLDOUT_DATASETS = ("2wiki",)

#: eval dataset -> (manifest dataset name, split) the training wall guards.
EVAL_SPLITS: dict[str, tuple[str, str]] = {
    "beir-hotpotqa": ("hotpotqa", "test"),
    "musique": ("musique", "dev"),
    "2wiki": ("2wiki", "dev"),
    # HippoRAG samples from the validation sets, so these are dev.
    "hipporag-musique": ("musique", "dev"),
    "hipporag-2wiki": ("2wiki", "dev"),
}


class TrainingWallError(RuntimeError):
    """The model under evaluation was trained on the split being evaluated."""


def _norm_name(name: object) -> str:
    """Normalize a manifest dataset name: '2wiki', '2WikiMultihopQA', 'hipporag-2wiki' all match."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def check_training_wall(dataset: str, model: ResolvedModel, allow_dirty: bool = False) -> str:
    """Training-data wall: refuse a rerank model whose manifest lists the
    split under evaluation or any holdout dataset; a missing manifest is
    refused too. Returns a one-line provenance note for the report."""
    if model.is_zero_shot():
        return (
            f"rerank model {model.label}: zero-shot base (MS MARCO), no benchmark"
            " training; wall not applicable"
        )
    manifest_path = model.manifest
    if not manifest_path.is_file():
        raise TrainingWallError(
            f"no manifest.json inside {model.path}: cannot verify the training-data"
            " wall (ADR 0011); refusing to evaluate a rerank model of unknown provenance"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TrainingWallError(f"{manifest_path} is not a JSON object; refusing")
    if dataset not in EVAL_SPLITS:
        raise TrainingWallError(
            f"no training-wall mapping for dataset {dataset!r}: refusing rather than"
            f" evaluating unguarded (known: {sorted(EVAL_SPLITS)})"
        )
    eval_dataset, eval_split = EVAL_SPLITS[dataset]
    trained_on = manifest.get("trained_on")
    if not isinstance(trained_on, list) or not trained_on:
        raise TrainingWallError(
            f"{manifest_path} declares no trained_on sources: an artifact that cannot"
            " say what it learned from cannot be cleared; refusing"
        )
    holdout_keys = {_norm_name(h) for h in HOLDOUT_DATASETS}
    for source in trained_on:
        if not isinstance(source, dict) or "dataset" not in source or "split" not in source:
            raise TrainingWallError(
                f"{manifest_path}: every trained_on entry must be an object with"
                f" 'dataset' and 'split'; got {source!r}; refusing"
            )
        name = _norm_name(source["dataset"])
        # A holdout in training voids the transfer claim for every dataset; match loosely.
        if any(key in name for key in holdout_keys):
            raise TrainingWallError(
                f"training-data wall (ADR 0011): {source['dataset']!r} is a declared"
                f" holdout yet appears in trained_on of {manifest_path}"
                f" (split {source['split']!r}); refusing"
            )
        if name == _norm_name(eval_dataset) and source["split"] == eval_split:
            raise TrainingWallError(
                f"training-data wall (ADR 0011): {manifest_path} lists"
                f" {eval_dataset}/{eval_split} as training data — this evaluation would be"
                " circular; refusing"
            )
    _verify_manifest_files(model, manifest)
    commit = str(manifest.get("trainer_commit", "unknown"))
    dirty = bool(manifest.get("dirty")) or commit.endswith("-dirty")
    if dirty and not allow_dirty:
        raise TrainingWallError(
            f"{manifest_path} was written from a dirty trainer tree ({commit}): the"
            " artifact is not reproducible from any commit. Pass --allow-dirty-manifest"
            " to score it anyway (the report will say so)"
        )
    trained = ", ".join(f"{s['dataset']}/{s['split']}" for s in trained_on)
    # Kept-gold models pass the wall; the note has to flag them.
    exclusion = manifest.get("exclusion")
    if isinstance(exclusion, dict):
        provenance = (
            "eval-gold excluded from training"
            f" ({exclusion.get('eval_gold_passages_excluded', '?')} passages)"
        )
    else:
        provenance = "EXCLUSION NOT APPLIED — trained on evaluation gold passages; NOT PUBLISHABLE"
    note = (
        f"rerank model {manifest.get('model', model.path.name)} ({model.precision}):"
        f" base {manifest.get('base_model')}, trained on [{trained}], trainer {commit};"
        f" wall verified for {eval_dataset}/{eval_split}; {provenance}"
    )
    if dirty:
        note += "; DIRTY TRAINER TREE — not reproducible; NOT PUBLISHABLE"
    return note


def _verify_manifest_files(model: ResolvedModel, manifest: dict[str, Any]) -> None:
    """A clean manifest copied next to a different model.onnx must not clear the wall."""
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TrainingWallError(
            f"{model.manifest} records no file checksums ('files'); cannot bind the"
            " manifest to the artifact; refusing"
        )
    checked = 0
    for name, expected in files.items():
        path = model.path / str(name)
        if not path.is_file():
            continue
        actual = sha256_of(path)
        if actual != expected:
            raise TrainingWallError(
                f"{model.manifest} does not describe {path.name}: manifest sha256"
                f" {expected}, file {actual}; refusing"
            )
        checked += 1
    if model.graph.name not in files:
        raise TrainingWallError(
            f"{model.manifest} does not list the graph being scored ({model.graph.name}); refusing"
        )
    if checked == 0:
        raise TrainingWallError(f"{model.manifest}: none of the listed files are present; refusing")


@dataclass
class LatencyStats:
    median_ms: float
    p95_ms: float
    #: how the latency was measured; printed alongside it
    conditions: str = "single-threaded, warm"

    @classmethod
    def from_seconds(
        cls, samples: Sequence[float], conditions: str = "single-threaded, warm"
    ) -> LatencyStats:
        if not samples:
            return cls(median_ms=0.0, p95_ms=0.0, conditions=conditions)
        ordered = sorted(samples)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return cls(
            median_ms=statistics.median(ordered) * 1000.0,
            p95_ms=ordered[p95_index] * 1000.0,
            conditions=conditions,
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
    #: metrics per effective-hop stratum
    strata: dict[str, StratumMetrics] | None = None
    #: oracle ceiling per candidate set: "pool" = every candidate reached,
    #: "scored" = the top-N the reranker sees
    oracle: dict[str, StratumMetrics] | None = None

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
        if self.oracle:
            lines.append("pool oracle (ceiling for any selection over the generated pool):")
            for name in sorted(self.oracle):
                metrics = self.oracle[name]
                per_k = "   ".join(
                    f"r@{k}: {metrics.recall_at[k]:.4f} ag@{k}: {metrics.all_gold_at[k]:.4f}"
                    for k in sorted(metrics.recall_at)
                )
                lines.append(f"  {name:<8} {per_k}")
                achieved = "   ".join(
                    f"r@{k}: {_pct(self.recall_at[k], metrics.recall_at[k])}"
                    f" ag@{k}: {_pct(self.all_gold_at[k], metrics.all_gold_at[k])}"
                    for k in sorted(metrics.recall_at)
                )
                lines.append(f"  {'of ' + name:<8} {achieved}")
        lines.append(
            f"candidates examined: mean {self.candidates_mean:,.0f},"
            f" median {self.candidates_median:,.0f}"
        )
        lines.append(
            f"latency per query: median {self.latency.median_ms:.1f} ms,"
            f" p95 {self.latency.p95_ms:.1f} ms ({self.latency.conditions})"
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
        setting="pooled-dev",
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
    allow_dirty: bool = False,
) -> FloorReport:
    """Budget-matched HopTrace run at a given hop count over prebuilt gold."""
    base = cfg or RetrievalConfig()
    run_cfg = replace(base, hops=hops, k=max(ks))
    wall_notes: list[str] = []
    resolved: ResolvedModel | None = None
    if run_cfg.selection == "rerank":
        # resolve here so an $HOPTRACE_RERANK_MODEL override cannot inherit
        # the base model's exemption
        resolved = resolve_model(run_cfg.rerank_model, run_cfg.rerank_precision)
        wall_notes.append(check_training_wall(dataset, resolved, allow_dirty=allow_dirty))
    retriever = Retriever(store, run_cfg)
    if limit is not None:
        query_gold = query_gold[:limit]

    recalls: dict[int, list[float]] = {k: [] for k in ks}
    all_golds: dict[int, list[float]] = {k: [] for k in ks}
    candidate_counts: list[int] = []
    latencies: list[float] = []
    stratified = _StratifiedAccumulator(ks) if strata is not None else None
    oracle_acc = _StratifiedAccumulator(ks)

    for i, qg in enumerate(query_gold, 1):
        started = time.perf_counter()
        result = retriever.retrieve(qg.text)
        latencies.append(time.perf_counter() - started)
        ranked = [e.chunk_id for e in result.evidence]
        recall, all_gold = _recall_metrics(ranked, qg.gold, ks)
        for k in ks:
            recalls[k].append(recall[k])
            all_golds[k].append(all_gold[k])
        oracle_acc.add(
            "pool", *_recall_metrics(_oracle_order(result.pool_order, qg.gold), qg.gold, ks)
        )
        if run_cfg.selection == "rerank":
            # interleave and submodular range over the whole pool; only rerank has a scored window
            scored_n = window_size(run_cfg.rerank_top_n, max(ks))
            oracle_acc.add(
                "scored",
                *_recall_metrics(_oracle_order(result.pool_order[:scored_n], qg.gold), qg.gold, ks),
            )
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

    selection_note = {
        "interleave": "selection=interleave (ADR 0006/0008)",
        "submodular": "selection=submodular (experimental, ADR 0008)",
        "rerank": (
            f"selection=rerank top_n={run_cfg.rerank_top_n}"
            f" model={resolved.label if resolved is not None else '?'}"
            f"{'' if run_cfg.rerank_path_context else ' PATH-CONTEXT OFF (ablation)'}"
            " (ADR 0012)"
        ),
    }[run_cfg.selection]
    notes = [
        f"retrieval config: hops={hops}, beam_entities={run_cfg.beam_entities},"
        f" beam_chunks={run_cfg.beam_chunks}, frontier_chunks={run_cfg.frontier_chunks},"
        f" hub_df_ratio={run_cfg.hub_df_ratio}, specificity_filter={run_cfg.specificity_filter},"
        f" scoring=propagated (ADR 0005), {selection_note}",
        *wall_notes,
    ]
    if limit is not None:
        notes.append(f"limited to first {limit} queries — NOT the reportable number")
    is_rerank = run_cfg.selection == "rerank"
    return FloorReport(
        dataset=dataset,
        setting=setting,
        system=f"hoptrace@{hops}hop{'+rerank' if is_rerank else ''}",
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
        latency=LatencyStats.from_seconds(
            latencies,
            f"ONNX {run_cfg.rerank_threads} intra-op threads, tokenizer single-threaded, warm"
            if is_rerank
            else "single-threaded, warm",
        ),
        notes=notes,
        strata=stratified.result() if stratified is not None else None,
        oracle=oracle_acc.result(),
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
    """Reverse map for the gold doc ids only; BEIR indexing is identity-preserving,
    one chunk per doc."""
    wanted = {doc_id for gold in qrels.values() for doc_id in gold}
    return {
        path: chunk_ids[0] for path, chunk_ids in store.chunks_by_doc_path(sorted(wanted)).items()
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _oracle_order(candidates: Sequence[int], gold: frozenset[int] | set[int]) -> list[int]:
    """Best-case ordering of a candidate list: its gold members first."""
    hits = [cid for cid in candidates if cid in gold]
    return hits + [cid for cid in candidates if cid not in gold]


def _pct(achieved: float, ceiling: float) -> str:
    """Fraction of the ceiling reached; undefined when the ceiling is 0."""
    return "n/a" if ceiling <= 0 else f"{100.0 * achieved / ceiling:5.1f}%"
