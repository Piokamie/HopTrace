from hoptrace.expand import Candidate
from hoptrace.provenance import HopEdge, HopPath
from hoptrace.score import Feature, candidate_features, greedy_select, interleave


def seed(chunk_id: int, score: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        path=HopPath(None, (HopEdge("", chunk_id, 0.0, 0),)),
        score=score,
        hop=0,
    )


def hop(chunk_id: int, score: float, bridge: str, hop_n: int = 1) -> Candidate:
    edges = (
        HopEdge("query_ent", 1, 0.9, 1),
        *(HopEdge(bridge, chunk_id, 0.5, 1) for _ in range(hop_n)),
    )
    return Candidate(chunk_id=chunk_id, path=HopPath("q", edges), score=score, hop=hop_n)


def as_pool(*candidates: Candidate) -> dict[int, Candidate]:
    return {c.chunk_id: c for c in candidates}


def feats(**by_id: list[tuple[str, str]]) -> dict[int, frozenset[Feature]]:
    return {int(key[1:]): frozenset(value) for key, value in by_id.items()}


def test_first_slot_goes_to_best_relevance() -> None:
    pool = as_pool(seed(1, 0.9), seed(2, 0.5))
    features = feats(c1=[("term", "a")], c2=[("term", "b")])
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 2)]
    assert ranked == [1, 2]


def test_redundant_seed_loses_to_complementary_hop() -> None:
    # seed 2 duplicates seed 1's coverage; the hop chunk covers a new
    # bridge aspect and wins slot 2 despite lower raw score
    pool = as_pool(seed(1, 0.9), seed(2, 0.8), hop(10, 0.4, "bridge_x"))
    features = feats(
        c1=[("term", "anna"), ("term", "report")],
        c2=[("term", "anna"), ("term", "report")],
        c10=[("ent", "query_ent"), ("ent", "bridge_x")],
    )
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 3)]
    assert ranked[0] == 1
    assert ranked[1] == 10  # complementary hop beats redundant seed
    assert ranked[2] == 2  # redundancy fills last, by relevance fallback


def test_ring2_wins_on_merit_no_reservation_needed() -> None:
    # ring-2 chunk covers the only uncovered aspect: admitted on merit
    pool = as_pool(seed(1, 0.9), hop(10, 0.4, "b1"), hop(20, 0.1, "b2", hop_n=2))
    features = feats(
        c1=[("term", "a")],
        c10=[("ent", "query_ent"), ("ent", "b1")],
        c20=[("ent", "b2")],
    )
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 3)]
    assert set(ranked) == {1, 10, 20}


def test_duplicating_ring2_loses_on_merit() -> None:
    # a ring-2 chunk covering nothing new ranks below a fresher seed
    pool = as_pool(seed(1, 0.9), seed(2, 0.5), hop(20, 0.2, "b1", hop_n=2))
    features = feats(
        c1=[("ent", "b1"), ("term", "a")],
        c2=[("term", "b")],
        c20=[("ent", "b1")],
    )
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 3)]
    assert ranked == [1, 2, 20]


def test_novelty_and_gain_recorded() -> None:
    pool = as_pool(seed(1, 0.8), seed(2, 0.6))
    features = feats(c1=[("term", "a"), ("term", "b")], c2=[("term", "a"), ("term", "c")])
    selected = greedy_select(pool, features, 2)
    assert selected[0].novelty == 1.0
    assert selected[0].gain == 0.8
    assert selected[1].novelty == 0.5  # "a" already covered
    assert selected[1].gain == 0.3


def test_zero_gain_fallback_by_relevance() -> None:
    pool = as_pool(seed(1, 0.9), seed(2, 0.7), seed(3, 0.8))
    features = feats(c1=[("term", "a")], c2=[("term", "a")], c3=[("term", "a")])
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 3)]
    assert ranked == [1, 3, 2]  # exhausted coverage -> relevance order


def test_deterministic_ties_by_chunk_id() -> None:
    pool = as_pool(seed(9, 0.5), seed(4, 0.5))
    features = feats(c9=[("term", "x")], c4=[("term", "y")])
    ranked = [s.candidate.chunk_id for s in greedy_select(pool, features, 2)]
    assert ranked == [4, 9]


def test_candidate_features_composition() -> None:
    candidate = hop(10, 0.4, "kowalski")
    features = candidate_features(
        candidate,
        chunk_terms=frozenset({"room", "4b", "anna"}),
        query_terms=frozenset({"anna", "sit"}),
    )
    assert features == frozenset({("term", "anna"), ("ent", "query_ent"), ("ent", "kowalski")})


def test_empty_pool() -> None:
    assert greedy_select({}, {}, 5) == []
    assert interleave({}, 5) == []


# --- interleave (default selection) ---


def hop_ring(chunk_id: int, score: float, hop_n: int) -> Candidate:
    edges = tuple(HopEdge(f"e{i}", chunk_id, 0.5, 1) for i in range(hop_n + 1))
    return Candidate(chunk_id=chunk_id, path=HopPath("q", edges), score=score, hop=hop_n)


def test_interleave_seeds_every_other_slot() -> None:
    pool = as_pool(seed(1, 0.9), seed(2, 0.5), hop_ring(10, 0.8, 1), hop_ring(11, 0.2, 1))
    assert [c.chunk_id for c in interleave(pool, 4)] == [1, 10, 2, 11]


def test_interleave_ring2_round_robin() -> None:
    pool = as_pool(
        seed(1, 0.9),
        seed(2, 0.8),
        hop_ring(10, 0.6, 1),
        hop_ring(11, 0.5, 1),
        hop_ring(20, 0.09, 2),
        hop_ring(21, 0.05, 2),
    )
    # slots: seed, ring1, seed, ring2, (seeds done) ring1, ring2
    assert [c.chunk_id for c in interleave(pool, 6)] == [1, 10, 2, 20, 11, 21]


def test_interleave_exhausted_population_fills() -> None:
    pool = as_pool(seed(1, 0.9), hop_ring(10, 0.8, 1), hop_ring(11, 0.5, 1))
    assert [c.chunk_id for c in interleave(pool, 3)] == [1, 10, 11]
