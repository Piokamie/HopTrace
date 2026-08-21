"""Per-corpus bracket: BM25 floor, HopPath at 1 and 2 hops, oracle, and
the multi-hop fraction, measured on the corpus's own self-benchmark.
Diagnostic only (ADR 0004); externally validated numbers live in
docs/results.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from hoppath.bench import BenchQuestion, generate_benchmark
from hoppath.bm25 import Bm25
from hoppath.config import RetrievalConfig
from hoppath.eval.diagnostics import MISS_KINDS, MissBreakdown, classify_misses
from hoppath.retrieve import Retriever
from hoppath.store import Store
from hoppath.tokenize import analyze

CAVEAT = (
    "self-benchmark: questions are generated from this corpus's own entity"
    " index, so extraction misses are invisible by construction and"
    " floor-vs-HopPath comparisons favor HopPath. Use this bracket to judge"
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
    #: fraction of questions whose gold set fits within k: the all-gold@k ceiling
    oracle: float
    #: fraction of questions the floor's top-k fails to fully cover
    multihop_fraction: float
    miss_breakdown: dict[str, int]
    caveat: str = CAVEAT
    notes: list[str] = field(default_factory=list)
    #: one of VERDICTS, derived from the rows; ``verdict`` is the sentence
    verdict_code: str = ""
    verdict: str = ""

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
        lines.append(f"  VERDICT: {self.verdict}")
        lines.append(f"  CAVEAT: {self.caveat}")
        return "\n".join(lines)


#: Thresholds behind the verdict. Small and arbitrary on purpose: the bracket
#: is a plumbing check, so the verdict only states what the rows already show.
MIN_MULTIHOP_FRACTION = 0.10
MIN_HOP_GAIN = 0.05
MIN_CHUNKS_FOR_STABLE_VERDICT = 200

VERDICTS = ("single_hop", "multi_hop", "hops_do_not_help", "unstable")


def verdict_for(
    rows: list[SystemRow], multihop_fraction: float, k: int, corpus_chunks: int
) -> tuple[str, str]:
    """(code, sentence). Decided from the floor-vs-hop rows and the
    multi-hop fraction; says so when the corpus is too small to trust."""
    floor = rows[0]
    best = max(rows[1:], key=lambda r: (r.all_gold, r.recall))
    gain = best.all_gold - floor.all_gold
    pct = round(100 * multihop_fraction)
    if corpus_chunks < MIN_CHUNKS_FOR_STABLE_VERDICT:
        size = (
            f"{corpus_chunks} chunks is too few for a stable reading (the fraction swings"
            f" with -n); indicative only: "
        )
    else:
        size = ""
    if multihop_fraction < MIN_MULTIHOP_FRACTION and gain < MIN_HOP_GAIN:
        return "unstable" if size else "single_hop", (
            f"{size}{pct}% of generated questions need more than the BM25 floor and hops"
            f" add {gain:+.2f} all-gold@{k}. This corpus is effectively single-hop:"
            " plain BM25 — or any other single-stage retriever — covers it; hop"
            " retrieval has nothing to do here."
        )
    if gain < MIN_HOP_GAIN:
        return "unstable" if size else "hops_do_not_help", (
            f"{size}{pct}% of generated questions need more than the floor, but hops do"
            f" not recover them ({best.system} all-gold@{k} {best.all_gold:.2f} vs floor"
            f" {floor.all_gold:.2f}). Use BM25 and read the miss breakdown:"
            " the bridges are not entity-shaped, or not extracted."
        )
    return "unstable" if size else "multi_hop", (
        f"{size}{pct}% of generated questions need more than the BM25 floor, and"
        f" {best.system} lifts all-gold@{k} from {floor.all_gold:.2f} to {best.all_gold:.2f}"
        f" ({gain:+.2f}). Entity-bridged multi-hop is real on this corpus; keep hops on"
        f" ({best.system})."
    )


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
                system=f"hoppath@{hops}hop",
                recall=sum(recalls) / len(recalls),
                all_gold=sum(all_golds) / len(all_golds),
            )
        )

    n_multi = sum(1 for q in questions if q.kind == "multi_hop")
    insufficient = sum(1 for q in questions if len(set(floor_tops[q.qid]) & q.gold) < len(q.gold))
    notes = []
    if len(questions) < n_questions:
        notes.append(f"corpus supported only {len(questions)} of {n_questions} requested questions")
    multihop_fraction = insufficient / len(questions)
    code, sentence = verdict_for(rows, multihop_fraction, k, store.n_chunks)
    return BracketReport(
        corpus_chunks=store.n_chunks,
        n_questions=len(questions),
        n_single_hop=len(questions) - n_multi,
        n_multi_hop=n_multi,
        k=k,
        seed=seed,
        rows=rows,
        oracle=sum(1 for q in questions if len(q.gold) <= k) / len(questions),
        multihop_fraction=multihop_fraction,
        miss_breakdown=dict(misses.misses),
        notes=notes,
        verdict_code=code,
        verdict=sentence,
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
