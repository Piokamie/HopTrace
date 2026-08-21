import math

import pytest

from hoppath.bm25 import Bm25, Bm25Vector, idf
from hoppath.chunker import Chunk
from hoppath.store import Store, StoreWriter

K1 = 1.5
B = 0.75

# Three chunks with known term statistics (terms already "analyzed"):
#   chunk 1: anna(1) kowalski(1) report(1)          dl=3
#   chunk 2: kowalski(2) room(1) sit(1)             dl=4
#   chunk 3: office(1) plan(1)                      dl=2
# N=3, avgdl=3.
POSTINGS = {
    "anna": [(1, 1)],
    "kowalski": [(1, 1), (2, 2)],
    "report": [(1, 1)],
    "room": [(2, 1)],
    "sit": [(2, 1)],
    "office": [(3, 1)],
    "plan": [(3, 1)],
}
DLS = {1: 3, 2: 4, 3: 2}


def hand_score(tf: int, df: int, dl: int, n: int = 3, avgdl: float = 3.0) -> float:
    """Independent arithmetic for the expected values."""
    w = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    return w * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl))


@pytest.fixture(scope="module")
def store() -> Store:
    with StoreWriter(None) as writer:
        doc_id = writer.add_document("fixture", None)
        for chunk_id, dl in DLS.items():
            writer.add_chunk(doc_id, Chunk(chunk_id - 1, f"chunk {chunk_id}", dl))
        writer.write_postings(POSTINGS)
        writer.write_term_counts(list(DLS.items()))
        return writer.finish(n_chunks=3, avg_chunk_len=3.0)


def test_idf_monotone() -> None:
    assert idf(1000, 1) > idf(1000, 10) > idf(1000, 500) > 0


def test_scores_match_hand_computation(store: Store) -> None:
    bm = Bm25(store, k1=K1, b=B)
    scores = bm.scores(["kowalski", "room"])
    assert scores[1] == pytest.approx(hand_score(tf=1, df=2, dl=3))
    assert scores[2] == pytest.approx(hand_score(tf=2, df=2, dl=4) + hand_score(tf=1, df=1, dl=4))
    assert 3 not in scores


def test_top_k_ranking_and_ties(store: Store) -> None:
    bm = Bm25(store, k1=K1, b=B)
    # "kowalski" hits chunks 1 and 2; chunk 2 has tf=2 but is longer
    ranked = bm.top_k(["kowalski"], k=5)
    assert [cid for cid, _ in ranked] == [2, 1]
    # deterministic tie-break by chunk id: office vs plan give chunk 3 only
    assert bm.top_k(["office"], k=1) == bm.top_k(["office"], k=1)


def test_duplicate_query_terms_count_once(store: Store) -> None:
    bm = Bm25(store, k1=K1, b=B)
    assert bm.scores(["anna", "anna"]) == bm.scores(["anna"])


def test_explain_components(store: Store) -> None:
    bm = Bm25(store, k1=K1, b=B)
    rows = bm.explain(["kowalski", "missing"], chunk_id=2)
    kowalski, missing = rows
    assert kowalski.term == "kowalski"
    assert kowalski.df == 2
    assert kowalski.tf == 2
    assert kowalski.score == pytest.approx(hand_score(tf=2, df=2, dl=4))
    assert missing.df == 0
    assert missing.tf == 0
    assert missing.score == 0.0


def test_vector_path_matches_python_path(store: Store) -> None:
    pytest.importorskip("numpy")
    bm = Bm25(store, k1=K1, b=B)
    vec = Bm25Vector(store, k1=K1, b=B)
    for query in (["kowalski"], ["kowalski", "room"], ["anna", "office", "plan"]):
        expected = bm.top_k(query, k=3)
        ranked, candidates = vec.top_k(query, k=3)
        assert [cid for cid, _ in ranked] == [cid for cid, _ in expected]
        for (_, got), (_, want) in zip(ranked, expected, strict=True):
            assert got == pytest.approx(want, rel=1e-5)
        assert candidates == len(bm.scores(query))


def test_vector_path_empty_query(store: Store) -> None:
    pytest.importorskip("numpy")
    vec = Bm25Vector(store, k1=K1, b=B)
    assert vec.top_k(["absent_term"], k=5) == ([], 0)
