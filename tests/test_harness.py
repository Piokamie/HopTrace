from pathlib import Path

import pytest

from hoptrace.eval.adapters import (
    load_beir_qrels,
    load_beir_queries,
    load_hotpot_format,
    load_musique,
)
from hoptrace.eval.corpus_build import build_beir_index
from hoptrace.eval.harness import (
    HIT_RULE,
    LatencyStats,
    evaluate_beir_floor,
    evaluate_distractor_floor,
    evaluate_pooled_floor,
)
from hoptrace.store import Store

FIXTURES = Path(__file__).parent / "fixtures"

pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def beir_store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    target = tmp_path_factory.mktemp("beir") / "index.sqlite"
    build_beir_index(FIXTURES / "beir" / "corpus.jsonl", target, analyzer="english")
    return Store.open(target)


def test_beir_floor_finds_gold(beir_store: Store) -> None:
    queries = load_beir_queries(FIXTURES / "beir" / "queries.jsonl")
    qrels = load_beir_qrels(FIXTURES / "beir" / "qrels" / "test.tsv")
    report = evaluate_beir_floor(beir_store, queries, qrels, ks=(2, 5))
    assert report.n_queries == 2
    assert report.n_chunks == 6
    # each query reaches one of its two golds lexically, so the floor recalls half
    assert report.recall_at[5] == pytest.approx(0.5)
    assert report.all_gold_at[5] == 0.0
    assert report.ndcg10 is not None and report.ndcg10 > 0
    assert report.candidates_mean > 0
    assert report.latency.median_ms >= 0


def test_beir_floor_limit_notes(beir_store: Store) -> None:
    queries = load_beir_queries(FIXTURES / "beir" / "queries.jsonl")
    qrels = load_beir_qrels(FIXTURES / "beir" / "qrels" / "test.tsv")
    report = evaluate_beir_floor(beir_store, queries, qrels, ks=(2,), limit=1)
    assert report.n_queries == 1
    assert any("NOT the reportable number" in note for note in report.notes)


def test_report_text_carries_protocol(beir_store: Store) -> None:
    queries = load_beir_queries(FIXTURES / "beir" / "queries.jsonl")
    qrels = load_beir_qrels(FIXTURES / "beir" / "qrels" / "test.tsv")
    report = evaluate_beir_floor(beir_store, queries, qrels, ks=(2,), k1=0.9, b=0.4)
    text = report.to_text()
    assert HIT_RULE in text
    assert "k1=0.9" in text
    assert "latency" in text
    assert "candidates examined" in text


def test_pooled_floor_musique() -> None:
    questions = load_musique(FIXTURES / "musique_dev.jsonl")
    report = evaluate_pooled_floor(questions, dataset="musique", ks=(2, 5))
    assert report.setting == "pooled-dev"
    assert report.n_queries == 2
    assert report.n_chunks == 6
    assert 0.0 <= report.recall_at[5] <= 1.0
    assert report.ndcg10 is None


def test_distractor_floor_2wiki() -> None:
    questions = load_hotpot_format(FIXTURES / "2wiki_dev.json")
    report = evaluate_distractor_floor(questions, dataset="2wiki", ks=(2,))
    assert "sanity check" in report.setting
    assert report.n_queries == 2
    # 2-3 paragraphs per question: recall@2 saturates by construction
    assert report.recall_at[2] >= 0.5


def test_latency_stats() -> None:
    stats = LatencyStats.from_seconds([0.001, 0.002, 0.003, 0.004, 0.100])
    assert stats.median_ms == pytest.approx(3.0)
    assert stats.p95_ms == pytest.approx(100.0)
    assert LatencyStats.from_seconds([]) == LatencyStats(0.0, 0.0)


def test_json_roundtrip(beir_store: Store) -> None:
    import json

    queries = load_beir_queries(FIXTURES / "beir" / "queries.jsonl")
    qrels = load_beir_qrels(FIXTURES / "beir" / "qrels" / "test.tsv")
    report = evaluate_beir_floor(beir_store, queries, qrels, ks=(2,))
    payload = json.loads(report.to_json())
    assert payload["dataset"] == "beir-hotpotqa"
    assert payload["system"] == "bm25-floor"
