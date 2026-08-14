"""The per-corpus bracket: floor, HopTrace at hop 1 and 2, oracle, and the
sufficiency-based multi-hop fraction — measured on the corpus's own
self-benchmark.

Diagnostic, never evidence (ADR 0004): the caveat is part of the report
payload itself, not just documentation. Externally-validated numbers live
in docs/results.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from hoptrace.bench import BenchQuestion, generate_benchmark
from hoptrace.bm25 import Bm25
from hoptrace.config import RetrievalConfig
from hoptrace.eval.diagnostics import MISS_KINDS, MissBreakdown, classify_misses
from hoptrace.retrieve import Retriever
from hoptrace.store import Store
from hoptrace.tokenize import analyze

CAVEAT = (
    "self-benchmark: questions are generated from this corpus's own entity"
    " index, so extraction misses are invisible by construction and"
    " floor-vs-HopTrace comparisons favor HopTrace. Use this bracket to judge"
    " whether hop retrieval functions on YOUR corpus and how much of it is"
    " multi-hop — never as cross-system evidence."
)


@dataclass
class SystemRow:
    system: str
    recall: float
    all_gold: float


@dataclass
class BracketReport:
    corpus_chunks: int
    n_questions: int
    n_single_hop: int
    n_multi_hop: int
    k: int
    seed: int
    rows: list[SystemRow]
    #: fraction of questions whose full gold set fits within k — the best
    #: any retriever could score at all-gold@k (LLM-free ceiling, ADR 0003)
    oracle: float
    #: sufficiency-based: fraction of questions the floor's top-k fails to
    #: fully cover
    multihop_fraction: float
    miss_breakdown: dict[str, int]
    caveat: str = CAVEAT
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_text(self) -> str:
        lines = [
            f"bracket over {self.n_questions} generated questions"
            f" ({self.n_single_hop} single-hop, {self.n_multi_hop} multi-hop),"
            f" corpus {self.corpus_chunks:,} chunks, k={self.k}, seed={self.seed}",
        ]
        for row in self.rows:
            lines.append(
                f"  {row.system:<14} recall@{self.k}: {row.recall:.4f}"
                f"   all-gold@{self.k}: {row.all_gold:.4f}"
            )
        lines.append(f"  oracle (gold fits in k): {self.oracle:.4f}")
        lines.append(f"  multihop_fraction (floor-insufficiency): {self.multihop_fraction:.4f}")
        misses = ", ".join(f"{kind}={self.miss_breakdown[kind]}" for kind in MISS_KINDS)
        lines.append(f"  misses at 2hop: {misses}")
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(f"  CAVEAT: {self.caveat}")
        return "\n".join(lines)


def run_bracket(
    store: Store,
    cfg: RetrievalConfig | None = None,
    n_questions: int = 100,
    seed: int = 0,
) -> BracketReport:
    cfg = cfg or RetrievalConfig()
    questions = generate_benchmark(store, n_questions=n_questions, seed=seed)
    if not questions:
        raise ValueError("the corpus supports no benchmark questions (too few entities or bridges)")
    k = cfg.k
    analyzer = store.meta("analyzer") or "simple"

    floor = Bm25(store, k1=cfg.bm25_k1, b=cfg.bm25_b)
    floor_tops = {
        q.qid: [cid for cid, _ in floor.top_k(analyze(q.text, analyzer), k)] for q in questions
    }

    rows = [_row_floor(questions, floor_tops, k)]
    misses = MissBreakdown(dataset="self-benchmark", hops=2, k=k)
    retriever = Retriever(store, cfg)
    for hops in (1, 2):
        recalls: list[float] = []
        all_golds: list[float] = []
        for question in questions:
            result = retriever.retrieve(question.text, hops=hops, k=k)
            top = [e.chunk_id for e in result.evidence]
            found = len(set(top) & question.gold)
            recalls.append(found / len(question.gold))
            all_golds.append(1.0 if found == len(question.gold) else 0.0)
            if hops == 2:
                classify_misses(
                    store, result, question.gold, k, misses, cfg.hub_cap(store.n_chunks)
                )
        rows.append(
            SystemRow(
                system=f"hoptrace@{hops}hop",
                recall=sum(recalls) / len(recalls),
                all_gold=sum(all_golds) / len(all_golds),
            )
        )

    n_multi = sum(1 for q in questions if q.kind == "multi_hop")
    insufficient = sum(1 for q in questions if len(set(floor_tops[q.qid]) & q.gold) < len(q.gold))
    notes = []
    if len(questions) < n_questions:
        notes.append(f"corpus supported only {len(questions)} of {n_questions} requested questions")
    return BracketReport(
        corpus_chunks=store.n_chunks,
        n_questions=len(questions),
        n_single_hop=len(questions) - n_multi,
        n_multi_hop=n_multi,
        k=k,
        seed=seed,
        rows=rows,
        oracle=sum(1 for q in questions if len(q.gold) <= k) / len(questions),
        multihop_fraction=insufficient / len(questions),
        miss_breakdown=dict(misses.misses),
        notes=notes,
    )


def _row_floor(
    questions: list[BenchQuestion], floor_tops: dict[str, list[int]], k: int
) -> SystemRow:
    recalls: list[float] = []
    all_golds: list[float] = []
    for question in questions:
        found = len(set(floor_tops[question.qid]) & question.gold)
        recalls.append(found / len(question.gold))
        all_golds.append(1.0 if found == len(question.gold) else 0.0)
    return SystemRow(
        system="bm25-floor",
        recall=sum(recalls) / len(recalls),
        all_gold=sum(all_golds) / len(all_golds),
    )
