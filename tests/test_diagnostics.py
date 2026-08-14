from pathlib import Path

import pytest

from hoptrace.config import ChunkConfig, RetrievalConfig
from hoptrace.eval.diagnostics import (
    DisplacementAudit,
    MissBreakdown,
    QuestionOutcome,
    audit_displacement,
    calibrate,
    classify_misses,
    normalize_span,
    pool_precision,
    question_stratum,
    stratify,
)
from hoptrace.ingest import SourceDocument, ingest_documents
from hoptrace.retrieve import Retriever
from hoptrace.store import Store

pytest.importorskip("numpy")

FIXTURES = Path(__file__).parent / "fixtures"


def outcome(
    qid: str,
    answer: str,
    gold: frozenset[int],
    top: tuple[int, ...],
    texts: tuple[str, ...],
    qtype: str = "bridge",
) -> QuestionOutcome:
    return QuestionOutcome(
        qid=qid,
        qtype=qtype,
        answer=answer,
        gold_chunks=gold,
        floor_top_chunks=top,
        floor_top_texts=texts,
    )


def test_normalize_span() -> None:
    assert normalize_span("Jan  Kowalski!") == "jan kowalski"
    assert normalize_span("café") == normalize_span("café")


def test_hybrid_rule() -> None:
    outcomes = [
        # span present AND gold present -> effectively single-hop
        outcome("q1", "Kowalski", frozenset({1}), (1, 2), ("Jan Kowalski leads.", "other")),
        # span present but NO gold in top -> stays multi-hop (spurious span)
        outcome("q2", "Kowalski", frozenset({9}), (1, 2), ("Jan Kowalski leads.", "other")),
        # gold present but span absent -> stays multi-hop
        outcome("q3", "Room 204", frozenset({1}), (1,), ("Jan Kowalski leads.",)),
    ]
    report = calibrate("test", outcomes, answer_k=10)
    assert report.n_effective_single_hop == 1
    assert report.effective_multihop_fraction == pytest.approx(2 / 3)


def test_yes_no_and_comparison_excluded() -> None:
    outcomes = [
        outcome("q1", "yes", frozenset({1}), (1,), ("yes it is",)),
        outcome("q2", "Paris", frozenset({1}), (1,), ("Paris",), qtype="comparison"),
        outcome("q3", "Paris", frozenset({1}), (1,), ("In Paris.",)),
    ]
    report = calibrate("test", outcomes, answer_k=10)
    assert report.n_excluded == 2
    assert report.n_effective_single_hop == 1
    # effective fraction computed over eligible questions only
    assert report.effective_multihop_fraction == 0.0


def test_annotated_fraction_and_gap() -> None:
    outcomes = [
        outcome("q1", "A", frozenset({1, 2}), (1, 2), ("has A here", "x")),
        outcome("q2", "B", frozenset({3, 4}), (9,), ("nothing",)),
    ]
    report = calibrate("test", outcomes, answer_k=10)
    assert report.annotated_multihop_fraction == 1.0
    assert report.effective_multihop_fraction == 0.5
    assert report.gap == pytest.approx(0.5)


def test_no_answer_stratum() -> None:
    """Answer-less questions must never silently count as effective
    multi-hop — that would saturate the calibration metric."""
    outcomes = [
        outcome("q1", "", frozenset({1}), (9,), ("nothing",)),  # no answer
        outcome("q2", "Paris", frozenset({1}), (1,), ("In Paris.",)),  # single
        outcome("q3", "Rome", frozenset({2}), (9,), ("nothing",)),  # multi
    ]
    assert question_stratum(outcomes[0], 10) == "no_answer"
    report = calibrate("test", outcomes, answer_k=10)
    assert report.n_no_answer == 1
    # eligible = 2 (q2, q3); q3 is the only effective multi-hop
    assert report.effective_multihop_fraction == pytest.approx(0.5)
    assert "no-answer=1" in report.to_text()


def test_stratify_matches_calibrate() -> None:
    outcomes = [
        outcome("q1", "Kowalski", frozenset({1}), (1, 2), ("Jan Kowalski leads.", "x")),
        outcome("q2", "Room 204", frozenset({9}), (1,), ("nothing here",)),
        outcome("q3", "yes", frozenset({1}), (1,), ("yes",)),
    ]
    strata = stratify(outcomes, answer_k=10)
    assert strata == {
        "q1": "effective_single",
        "q2": "effective_multi",
        "q3": "excluded",
    }
    assert question_stratum(outcomes[0], 10) == "effective_single"


def test_displacement_audit() -> None:
    from hoptrace.provenance import (
        Evidence,
        HopEdge,
        HopPath,
        RetrievalResult,
        ScoreBreakdown,
    )

    def ev(chunk_id: int, population: str, rank: int) -> Evidence:
        hop = 1 if population == "hop" else 0
        return Evidence(
            chunk_id=chunk_id,
            text="",
            doc_path="p",
            doc_title=None,
            score=ScoreBreakdown(
                population=population,
                hop=hop,
                bm25=0.0,
                bm25_norm=0.0,
                bridge_strength=0.5 if hop else 0.0,
                parent_chunk=1 if hop else None,
                score=0.5,
                novelty=1.0,
                gain=0.5,
                rank=rank,
            ),
            path=HopPath("q", tuple(HopEdge("e", chunk_id, 0.5, 1) for _ in range(hop + 1))),
            matched_entities=(),
        )

    # interleaved top-4: seeds 1, 3; hops 10 (gold), 11 (junk)
    # seeds-only top-4 would have been 1, 3, 4 (gold), 5
    result = RetrievalResult(
        query="q",
        evidence=(ev(1, "seed", 0), ev(10, "hop", 1), ev(3, "seed", 2), ev(11, "hop", 3)),
        candidates_examined=100,
        pool_size=10,
        seed_top=(1, 3, 4, 5),
    )
    audit = DisplacementAudit(dataset="test", hops=2, k=4)
    audit_displacement(result, frozenset({10, 4}), k=4, audit=audit)
    assert audit.hop_slots == 2
    assert audit.hop_gold == 1
    assert audit.displaced_seeds == 2  # 4 and 5 evicted
    assert audit.displaced_gold == 1  # chunk 4 was gold
    assert audit.net_gold == 0
    assert "net gold" in audit.to_text()


# --- miss classification over a real small corpus ---

DOCS = [
    SourceDocument("p0", "Anna Nowak reports to Kowalski on the platform team.", "p0"),
    SourceDocument("p1", "Kowalski occupies Room 4B near the atrium.", "p1"),
    SourceDocument("p2", "A plain paragraph with no names or identifiers here.", "p2"),
    SourceDocument("p3", "Zofia Lis chairs the safety committee quarterly.", "p3"),
]


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(
        DOCS, None, chunk_cfg=ChunkConfig(identity=True), analyzer="english"
    )
    return store


def test_classify_misses_kinds(store: Store) -> None:
    cfg = RetrievalConfig(hops=2, k=2, hub_df_ratio=0.7, hub_df_floor=1, bm25_k1=0.9, bm25_b=0.4)
    retriever = Retriever(store, cfg)
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    breakdown = MissBreakdown(dataset="test", hops=2, k=2)

    # chunk 3 has NO entities -> extraction miss; chunk 4 (Zofia) has
    # entities but no path from this query -> seed_alias miss
    gold = frozenset({2, 3, 4})
    classify_misses(store, result, gold, k=2, breakdown=breakdown, hub_cap=3.0)
    assert breakdown.n_gold == 3
    assert breakdown.misses["extraction"] >= 1
    assert breakdown.misses["seed_alias"] >= 1
    total = breakdown.n_found + sum(breakdown.misses.values())
    assert total == breakdown.n_gold


def test_ranking_miss(store: Store) -> None:
    cfg = RetrievalConfig(hops=2, k=1, hub_df_ratio=0.7, hub_df_floor=1, bm25_k1=0.9, bm25_b=0.4)
    retriever = Retriever(store, cfg)
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    breakdown = MissBreakdown(dataset="test", hops=2, k=1)
    # gold = the hop-2 chunk; with k=1 it is in the pool but ranked out
    gold = frozenset({2})
    if result.evidence[0].chunk_id != 2:
        classify_misses(store, result, gold, k=1, breakdown=breakdown, hub_cap=3.0)
        assert breakdown.misses["ranking"] == 1


def test_pool_precision_ablation(store: Store) -> None:
    base = RetrievalConfig(hops=2, hub_df_ratio=0.7, hub_df_floor=1, bm25_k1=0.9, bm25_b=0.4)
    on = Retriever(store, base)
    from dataclasses import replace

    off = Retriever(store, replace(base, specificity_filter=False))
    queries = [("Where does the manager of Anna Nowak sit?", frozenset({1, 2}))]
    result = pool_precision("test", 2, on.retrieve, off.retrieve, queries)
    assert result.n_queries == 1
    assert 0.0 <= result.precision_on <= 1.0
    assert result.mean_pool_off >= result.mean_pool_on
    assert "specificity filter ON" in result.to_text()
