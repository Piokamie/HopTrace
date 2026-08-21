import json

import pytest
from test_bench import DOCS

from hoppath.bracket import CAVEAT, run_bracket
from hoppath.config import ChunkConfig, RetrievalConfig
from hoppath.ingest import SourceDocument, ingest_documents
from hoppath.store import Store


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(
        DOCS, None, chunk_cfg=ChunkConfig(identity=True), analyzer="english"
    )
    return store


@pytest.fixture(scope="module")
def report(store: Store):
    return run_bracket(store, RetrievalConfig(k=4), n_questions=12, seed=0)


def test_rows_and_counts(report) -> None:
    assert [row.system for row in report.rows] == ["bm25-floor", "hoppath@1hop", "hoppath@2hop"]
    assert report.n_questions == report.n_single_hop + report.n_multi_hop
    assert report.n_multi_hop > 0


def test_hops_beat_floor_on_generated_bridges(report) -> None:
    floor, hop1, hop2 = report.rows
    # generated multi-hop questions are single-hop-proof, so the floor cannot cover them fully
    assert hop1.all_gold > floor.all_gold
    assert hop2.all_gold >= hop1.all_gold * 0.99


def test_oracle_and_multihop_fraction(report) -> None:
    assert report.oracle == 1.0  # gold sets of size <= 2, k=4
    # floor-insufficiency must at least cover the constructed multi-hop share
    assert report.multihop_fraction >= report.n_multi_hop / report.n_questions


def test_caveat_embedded_everywhere(report) -> None:
    assert report.caveat == CAVEAT
    assert "never as cross-system evidence" in report.to_text()
    payload = json.loads(report.to_json())
    assert "extraction misses are invisible" in payload["caveat"]


def test_report_text_shape(report) -> None:
    text = report.to_text()
    assert "bm25-floor" in text
    assert "hoppath@2hop" in text
    assert "multihop_fraction" in text
    assert "misses at 2hop" in text


def test_deterministic(store: Store) -> None:
    a = run_bracket(store, RetrievalConfig(k=4), n_questions=8, seed=3)
    b = run_bracket(store, RetrievalConfig(k=4), n_questions=8, seed=3)
    assert a.to_json() == b.to_json()


def test_empty_corpus_raises() -> None:
    store, _ = ingest_documents(
        [SourceDocument("p0", "word", "p0")], None, chunk_cfg=ChunkConfig(identity=True)
    )
    with pytest.raises(ValueError, match="no benchmark questions"):
        run_bracket(store, n_questions=5)


def test_verdict_is_derived_from_the_rows() -> None:
    from hoppath.bracket import SystemRow, verdict_for

    floor = SystemRow("bm25-floor", 0.95, 0.92)
    flat = [floor, SystemRow("hoppath@1hop", 0.96, 0.93), SystemRow("hoppath@2hop", 0.95, 0.92)]
    code, text = verdict_for(flat, 0.05, 8, 5_000)
    assert code == "single_hop" and "plain BM25" in text
    lifted = [floor, SystemRow("hoppath@1hop", 0.99, 0.99), SystemRow("hoppath@2hop", 0.98, 0.97)]
    code, text = verdict_for(lifted, 0.40, 8, 5_000)
    assert code == "multi_hop" and "keep hops on (hoppath@1hop)" in text
    stuck = [floor, SystemRow("hoppath@1hop", 0.95, 0.93), SystemRow("hoppath@2hop", 0.95, 0.92)]
    code, text = verdict_for(stuck, 0.40, 8, 5_000)
    assert code == "hops_do_not_help" and "Use BM25" in text
    code, text = verdict_for(lifted, 0.40, 8, 25)
    assert code == "unstable" and "too few" in text


def test_report_carries_verdict(report) -> None:
    assert report.verdict_code in {"single_hop", "multi_hop", "hops_do_not_help", "unstable"}
    assert "VERDICT:" in report.to_text()
    assert report.verdict in report.to_text()
