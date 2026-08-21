import pytest

from hoppath.bench import generate_benchmark
from hoppath.config import ChunkConfig
from hoppath.ingest import SourceDocument, ingest_documents
from hoppath.store import Store
from hoppath.tokenize import tokenize

# Bridge-rich fixture: several exclusive entity bridges (each shared by
# exactly two chunks), enough entities per chunk for single-hop questions.
DOCS = [
    SourceDocument("p0", "Anna Nowak coordinates deliveries with Piotr Zajac weekly.", "p0"),
    SourceDocument("p1", "Piotr Zajac maintains the Krakow warehouse inventory ledger.", "p1"),
    SourceDocument("p2", "Marta Lis approves invoices from Delta Logistics monthly.", "p2"),
    SourceDocument("p3", "Delta Logistics operates a fleet of electric vans nightly.", "p3"),
    SourceDocument("p4", "Tomasz Wrona schedules maintenance for Hala Nine equipment.", "p4"),
    SourceDocument("p5", "Hala Nine houses the calibration rigs and spare turbines.", "p5"),
    SourceDocument("p6", "Ewa Konik audits the Rybnik depot safety records quarterly.", "p6"),
    SourceDocument("p7", "The Rybnik depot stores refrigerated produce overnight.", "p7"),
    SourceDocument("p8", "Unrelated musings about weather patterns and cloud shapes.", "p8"),
]


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(
        DOCS, None, chunk_cfg=ChunkConfig(identity=True), analyzer="english"
    )
    return store


def test_reproducible_given_seed(store: Store) -> None:
    a = generate_benchmark(store, n_questions=10, seed=0)
    b = generate_benchmark(store, n_questions=10, seed=0)
    assert a == b


def test_different_seed_differs(store: Store) -> None:
    a = generate_benchmark(store, n_questions=10, seed=0)
    b = generate_benchmark(store, n_questions=10, seed=7)
    assert a != b


def test_question_mix_and_shapes(store: Store) -> None:
    questions = generate_benchmark(store, n_questions=10, seed=0)
    assert questions
    kinds = {q.kind for q in questions}
    assert "multi_hop" in kinds
    assert "single_hop" in kinds
    for q in questions:
        assert q.text.strip()
        if q.kind == "single_hop":
            assert len(q.gold) == 1
            assert q.bridge is None
        else:
            assert len(q.gold) == 2
            assert q.bridge is not None


def test_multi_hop_is_single_hop_proof(store: Store) -> None:
    """The second gold chunk shares no query lexeme; it is reachable only over the bridge."""
    questions = [
        q for q in generate_benchmark(store, n_questions=12, seed=0) if q.kind == "multi_hop"
    ]
    assert questions
    for q in questions:
        query_tokens = set(tokenize(q.text))
        c1_id, c2_id = sorted(q.gold)
        overlaps = [
            gold_id
            for gold_id in (c1_id, c2_id)
            if query_tokens & set(tokenize(store.get_chunk(gold_id).text))
        ]
        # exactly one gold chunk (c1) overlaps the query; the other shares nothing
        assert len(overlaps) == 1


def test_multi_hop_bridge_is_sole_shared_entity(store: Store) -> None:
    for q in generate_benchmark(store, n_questions=12, seed=0):
        if q.kind != "multi_hop":
            continue
        c1_id, c2_id = sorted(q.gold)
        shared = set(store.entities_for_chunk(c1_id)) & set(store.entities_for_chunk(c2_id))
        assert shared == {q.bridge}


def test_caps_to_what_corpus_supports(store: Store) -> None:
    questions = generate_benchmark(store, n_questions=500, seed=0)
    assert 0 < len(questions) <= 500
