from dataclasses import replace

import pytest

from hoptrace.config import ChunkConfig, RetrievalConfig
from hoptrace.expand import Candidate, Expander, bridge_strength
from hoptrace.ingest import SourceDocument, ingest_documents
from hoptrace.provenance import HopEdge, HopPath
from hoptrace.store import Store

# The canonical 2-hop shape: Anna -> Kowalski -> Room 204, plus a hub
# entity ("Meridian") present in every chunk to exercise the specificity
# filter, and unrelated chunks to make expansion selective.
DOCS = [
    SourceDocument("p0", "Anna Nowak reports to Kowalski at Meridian.", "p0"),
    SourceDocument("p1", "Kowalski sits in Room 204 at Meridian.", "p1"),
    SourceDocument("p2", "Room 204 overlooks the Meridian courtyard garden.", "p2"),
    SourceDocument("p3", "The Meridian cafeteria serves lunch at noon daily.", "p3"),
    SourceDocument("p4", "Meridian was founded in Krakow decades ago.", "p4"),
    SourceDocument("p5", "Krakow has an old town square with cafes.", "p5"),
]


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(DOCS, None, chunk_cfg=ChunkConfig(identity=True))
    return store


def seed_for(store: Store, entity: str, chunk_id: int) -> Candidate:
    edge = HopEdge(entity, chunk_id, 0.9, 1)
    return Candidate(
        chunk_id=chunk_id,
        path=HopPath(entity, (edge,)),
        score=bridge_strength(edge),
        hop=0,
    )


# hub_df_ratio=0.7: on 6 chunks the cap is 4.2, so "meridian" (df 5-6) is a
# hub and "kowalski"/"room 204" (df 2) are not.
CFG = RetrievalConfig(hops=2, hub_df_ratio=0.7, hub_df_floor=1)


def test_two_hop_reaches_bridge_target(store: Store) -> None:
    pool, touched = Expander(store, CFG).expand([seed_for(store, "anna nowak", 1)])
    # hop 1 via "kowalski" reaches chunk 2; hop 2 via "room 204" reaches chunk 3
    assert 2 in pool
    assert 3 in pool
    assert pool[2].hop == 1
    assert pool[3].hop == 2
    assert [e.entity for e in pool[3].path.edges[1:]] == ["kowalski", "room 204"]
    # touched counts distinct expansion-reached chunks (seeds excluded)
    assert touched == len(pool) - 1


def test_one_hop_bound(store: Store) -> None:
    cfg = replace(CFG, hops=1)
    pool, _ = Expander(store, cfg).expand([seed_for(store, "anna nowak", 1)])
    assert 2 in pool
    assert 3 not in pool  # requires hop 2


def test_hub_entity_not_followed(store: Store) -> None:
    # seed p4 (chunk 5): its entities are "meridian" (hub, df 5-6) and
    # "krakow" (df 2). With the filter on, only krakow is followed.
    pool, _ = Expander(store, CFG).expand([seed_for(store, "meridian", 5)])
    assert 6 in pool  # via krakow
    assert 2 not in pool  # only reachable via the hub


def bm25_seed(chunk_id: int) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        path=HopPath(None, (HopEdge("", chunk_id, 0.0, 0),)),
        score=0.5,
        hop=0,
    )


def test_specificity_filter_off_lets_hubs_through(store: Store) -> None:
    # seed BM25-style so no entity is "already used" and the hub itself is
    # followable when the filter is off
    cfg = replace(CFG, specificity_filter=False)
    pool, _ = Expander(store, cfg).expand([bm25_seed(5)])
    # with the filter off the hub connects (nearly) everything
    assert 2 in pool
    # with it on, the hub is never followed
    pool_on, _ = Expander(store, CFG).expand([bm25_seed(5)])
    assert 2 not in pool_on


def test_determinism(store: Store) -> None:
    runs = []
    for _ in range(2):
        pool, touched = Expander(store, CFG).expand([seed_for(store, "anna nowak", 1)])
        runs.append(
            (
                touched,
                sorted(
                    (cid, c.score, c.hop, tuple(e.entity for e in c.path.edges))
                    for cid, c in pool.items()
                ),
            )
        )
    assert runs[0] == runs[1]


def test_earliest_ring_wins(store: Store) -> None:
    # chunk 3 is reachable at hop 1 (from p1) and hop 2 (from p0); the
    # earliest ring keeps it — later rings never revisit pool members
    seeds = [seed_for(store, "anna nowak", 1), seed_for(store, "kowalski", 2)]
    pool, _ = Expander(store, CFG).expand(seeds)
    assert len({c.chunk_id for c in pool.values()}) == len(pool)
    assert pool[3].hop == 1


def test_entities_never_refollowed(store: Store) -> None:
    pool, _ = Expander(store, CFG).expand([seed_for(store, "anna nowak", 1)])
    for candidate in pool.values():
        entities = [e.entity for e in candidate.path.edges if e.entity]
        assert len(entities) == len(set(entities))


def test_frontier_cap(store: Store) -> None:
    cfg = replace(CFG, frontier_chunks=1)
    pool, _ = Expander(store, cfg).expand([seed_for(store, "anna nowak", 1)])
    hop1 = [c for c in pool.values() if c.hop == 1]
    assert len(hop1) <= 1
