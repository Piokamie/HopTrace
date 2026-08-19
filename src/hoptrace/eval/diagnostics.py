"""Instrument calibration and failure attribution, all LLM-free.

Effective multi-hop (hybrid rule, ADR 0004): a question is effectively
single-hop when the floor's top-``answer_k`` contains the gold answer span
and at least one gold paragraph; yes/no and comparison questions are
excluded. Miss breakdown attributes each missed gold chunk to a stage:
``extraction`` (no entities on the chunk), ``ranking`` (in the pool, below
k), ``hop_bound`` (outside the pool but sharing a followable entity with a
pool chunk), ``seed_alias`` (no entity path). Pool precision compares the
hop-2 pool with the specificity filter on and off.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hoptrace.provenance import RetrievalResult
from hoptrace.store import Store

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_YES_NO = frozenset({"yes", "no"})

MISS_KINDS = ("extraction", "ranking", "hop_bound", "seed_alias")


def normalize_span(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class QuestionOutcome:
    """Floor outcome for one question, the calibration unit."""

    qid: str
    qtype: str
    answer: str
    gold_chunks: frozenset[int]
    floor_top_chunks: tuple[int, ...]
    floor_top_texts: tuple[str, ...]


@dataclass
class CalibrationReport:
    dataset: str
    answer_k: int
    n_questions: int
    #: yes/no + comparison questions, excluded from the span rule.
    n_excluded: int
    #: questions with no answer string in the dataset
    n_no_answer: int
    n_effective_single_hop: int
    annotated_multihop_fraction: float
    effective_multihop_fraction: float

    @property
    def gap(self) -> float:
        return self.annotated_multihop_fraction - self.effective_multihop_fraction

    def to_text(self) -> str:
        return (
            f"calibration [{self.dataset}] (answer_k={self.answer_k},"
            f" n={self.n_questions}, excluded yes-no/comparison={self.n_excluded},"
            f" no-answer={self.n_no_answer}):\n"
            f"  annotated multi-hop: {self.annotated_multihop_fraction:.3f}\n"
            f"  effective multi-hop: {self.effective_multihop_fraction:.3f}"
            " (hybrid rule: answer span AND >=1 gold in floor top-k)\n"
            f"  gap (annotated - effective): {self.gap:+.3f}"
        )


#: ``no_answer`` is its own stratum so answer-less exports cannot inflate ``effective_multi``.
STRATA = ("effective_single", "effective_multi", "excluded", "no_answer")

#: Floor top-k for the hybrid rule; a protocol constant, not a tunable.
ANSWER_K = 10

#: k at which miss breakdowns and displacement audits are computed.
DIAGNOSTIC_K = 20


def question_stratum(outcome: QuestionOutcome, answer_k: int = ANSWER_K) -> str:
    """Effectively single-hop iff the floor's top-k contains the answer
    span and at least one gold paragraph."""
    if not outcome.answer.strip():
        return "no_answer"
    if outcome.qtype == "comparison" or normalize_span(outcome.answer) in _YES_NO:
        return "excluded"
    top = outcome.floor_top_chunks[:answer_k]
    texts = outcome.floor_top_texts[: len(top)]
    answer = normalize_span(outcome.answer)
    span_hit = bool(answer) and any(answer in normalize_span(t) for t in texts)
    gold_hit = any(cid in outcome.gold_chunks for cid in top)
    return "effective_single" if span_hit and gold_hit else "effective_multi"


def stratify(outcomes: Sequence[QuestionOutcome], answer_k: int) -> dict[str, str]:
    """qid -> stratum, for splitting every downstream metric."""
    return {o.qid: question_stratum(o, answer_k) for o in outcomes}


def calibrate(
    dataset: str, outcomes: Sequence[QuestionOutcome], answer_k: int = ANSWER_K
) -> CalibrationReport:
    strata = [question_stratum(o, answer_k) for o in outcomes]
    n_excluded = strata.count("excluded")
    n_no_answer = strata.count("no_answer")
    n_single = strata.count("effective_single")
    n_eligible = len(outcomes) - n_excluded - n_no_answer
    n_annotated_multi = sum(1 for o in outcomes if len(o.gold_chunks) > 1)
    return CalibrationReport(
        dataset=dataset,
        answer_k=answer_k,
        n_questions=len(outcomes),
        n_excluded=n_excluded,
        n_no_answer=n_no_answer,
        n_effective_single_hop=n_single,
        annotated_multihop_fraction=(n_annotated_multi / len(outcomes) if outcomes else 0.0),
        effective_multihop_fraction=((n_eligible - n_single) / n_eligible if n_eligible else 0.0),
    )


@dataclass
class MissBreakdown:
    dataset: str
    hops: int
    k: int
    n_gold: int = 0
    n_found: int = 0
    misses: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MISS_KINDS, 0))

    def to_text(self) -> str:
        lines = [
            f"miss breakdown [{self.dataset}] at hops={self.hops}, k={self.k}:"
            f" gold={self.n_gold}, found={self.n_found}"
        ]
        total_missed = max(1, self.n_gold - self.n_found)
        for kind in MISS_KINDS:
            count = self.misses[kind]
            lines.append(f"  {kind}: {count} ({count / total_missed:.1%} of misses)")
        return "\n".join(lines)


def classify_misses(
    store: Store,
    result: RetrievalResult,
    gold_chunks: frozenset[int],
    k: int,
    breakdown: MissBreakdown,
    hub_cap: float,
) -> None:
    """Attribute each missed gold chunk to a pipeline stage (in-place)."""
    top = {e.chunk_id for e in result.evidence[:k]}
    pool = set(result.pool_hops)
    breakdown.n_gold += len(gold_chunks)
    breakdown.n_found += len(gold_chunks & top)

    pool_entities: set[str] | None = None
    for gold in sorted(gold_chunks - top):
        gold_entities = set(store.entities_for_chunk(gold))
        if not gold_entities:
            breakdown.misses["extraction"] += 1
        elif gold in pool:
            breakdown.misses["ranking"] += 1
        else:
            if pool_entities is None:
                pool_entities = set()
                for chunk_id in pool:
                    pool_entities.update(store.entities_for_chunk(chunk_id))
            followable = {
                e for e in gold_entities & pool_entities if 0 < store.entity_df(e) <= hub_cap
            }
            if followable:
                breakdown.misses["hop_bound"] += 1
            else:
                breakdown.misses["seed_alias"] += 1


@dataclass
class DisplacementAudit:
    """Gold held by hop-derived top-k slots vs gold in the seeds they evicted;
    net gold is the interleave's measured value."""

    dataset: str
    hops: int
    k: int
    n_queries: int = 0
    hop_slots: int = 0
    hop_gold: int = 0
    displaced_seeds: int = 0
    displaced_gold: int = 0

    @property
    def net_gold(self) -> int:
        return self.hop_gold - self.displaced_gold

    def to_text(self) -> str:
        hop_rate = self.hop_gold / self.hop_slots if self.hop_slots else 0.0
        disp_rate = self.displaced_gold / self.displaced_seeds if self.displaced_seeds else 0.0
        return (
            f"displacement audit [{self.dataset}] at hops={self.hops}, k={self.k}"
            f" (n={self.n_queries}):\n"
            f"  hop-held slots: {self.hop_slots} — gold in {self.hop_gold} ({hop_rate:.1%})\n"
            f"  seeds evicted:  {self.displaced_seeds} — gold in {self.displaced_gold}"
            f" ({disp_rate:.1%})\n"
            f"  net gold from interleave: {self.net_gold:+d}"
        )


def audit_displacement(
    result: RetrievalResult,
    gold_chunks: frozenset[int],
    k: int,
    audit: DisplacementAudit,
) -> None:
    """Compare the interleaved top-k against the seeds-only top-k (in-place)."""
    top = result.evidence[:k]
    top_ids = {e.chunk_id for e in top}
    hop_ids = [e.chunk_id for e in top if e.score.population == "hop"]
    displaced = [cid for cid in result.seed_top[:k] if cid not in top_ids]
    audit.n_queries += 1
    audit.hop_slots += len(hop_ids)
    audit.hop_gold += sum(1 for cid in hop_ids if cid in gold_chunks)
    audit.displaced_seeds += len(displaced)
    audit.displaced_gold += sum(1 for cid in displaced if cid in gold_chunks)


@dataclass
class PoolPrecision:
    dataset: str
    hops: int
    #: mean |gold ∩ pool| / |pool| and mean gold recall of the pool itself
    precision_on: float
    precision_off: float
    pool_recall_on: float
    pool_recall_off: float
    mean_pool_on: float
    mean_pool_off: float
    n_queries: int

    def to_text(self) -> str:
        return (
            f"hop-{self.hops} pool precision [{self.dataset}] (n={self.n_queries}):\n"
            f"  specificity filter ON : precision {self.precision_on:.4f},"
            f" pool recall {self.pool_recall_on:.4f}, mean pool {self.mean_pool_on:,.0f}\n"
            f"  specificity filter OFF: precision {self.precision_off:.4f},"
            f" pool recall {self.pool_recall_off:.4f}, mean pool {self.mean_pool_off:,.0f}"
        )


def pool_precision(
    dataset: str,
    hops: int,
    run_on: Callable[[str], RetrievalResult],
    run_off: Callable[[str], RetrievalResult],
    queries: Sequence[tuple[str, frozenset[int]]],
) -> PoolPrecision:
    """Compare candidate pools with the specificity filter on vs off."""
    stats = {"on": [0.0, 0.0, 0.0], "off": [0.0, 0.0, 0.0]}  # precision, recall, size
    for text, gold in queries:
        for label, runner in (("on", run_on), ("off", run_off)):
            result = runner(text)
            pool = set(result.pool_hops)
            hit = len(pool & gold)
            stats[label][0] += hit / len(pool) if pool else 0.0
            stats[label][1] += hit / len(gold) if gold else 0.0
            stats[label][2] += len(pool)
    n = max(1, len(queries))
    return PoolPrecision(
        dataset=dataset,
        hops=hops,
        precision_on=stats["on"][0] / n,
        precision_off=stats["off"][0] / n,
        pool_recall_on=stats["on"][1] / n,
        pool_recall_off=stats["off"][1] / n,
        mean_pool_on=stats["on"][2] / n,
        mean_pool_off=stats["off"][2] / n,
        n_queries=len(queries),
    )
