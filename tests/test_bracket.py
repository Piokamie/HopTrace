import json

import pytest
from test_bench import DOCS

from hoptrace.bracket import CAVEAT, run_bracket
from hoptrace.config import ChunkConfig, RetrievalConfig
from hoptrace.ingest import SourceDocument, ingest_documents
from hoptrace.store import Store


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
    assert [row.system for row in report.rows] == ["bm25-floor", "hoptrace@1hop", "hoptrace@2hop"]
    assert report.n_questions == report.n_single_hop + report.n_multi_hop
    assert report.n_multi_hop > 0


def test_hops_beat_floor_on_generated_bridges(report) -> None:
    floor, hop1, hop2 = report.rows
    # multi-hop questions are single-hop-proof by construction, so the
    # floor cannot fully cover them; hop retrieval must do strictly better
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
    assert "hoptrace@2hop" in text
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
