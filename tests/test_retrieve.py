import pytest

from hoptrace.config import ChunkConfig, RetrievalConfig
from hoptrace.ingest import SourceDocument, ingest_documents
from hoptrace.retrieve import Retriever
from hoptrace.store import Store

# The DESIGN.md worked example, as documents: the answer chunk ("Kowalski
# sits in 4B") shares no lexical overlap with the query "Where does Anna's
# manager sit?" beyond the bridge entity.
DOCS = [
    SourceDocument("p0", "Anna Nowak reports to Kowalski on the platform team.", "p0"),
    # deliberately zero lexical overlap with the test query ("occupies",
    # not "sits") — reachable only through the kowalski bridge
    SourceDocument("p1", "Kowalski occupies Room 4B near the atrium.", "p1"),
    SourceDocument("p2", "The platform team ships the ingestion service.", "p2"),
    SourceDocument("p3", "The atrium cafe closes at five.", "p3"),
    SourceDocument("p4", "Budget planning happens every quarter at the company.", "p4"),
]

CFG = RetrievalConfig(hops=2, k=4, hub_df_ratio=0.7, hub_df_floor=1)


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(
        DOCS, None, chunk_cfg=ChunkConfig(identity=True), analyzer="english"
    )
    return store


@pytest.fixture(scope="module")
def retriever(store: Store) -> Retriever:
    return Retriever(store, CFG)


def test_two_hop_question_reaches_answer_chunk(retriever: Retriever) -> None:
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    ids = [e.chunk_id for e in result.evidence]
    assert 2 in ids  # "Kowalski sits in Room 4B" — only reachable via hop
    answer = next(e for e in result.evidence if e.chunk_id == 2)
    assert answer.score.hop >= 1
    assert "kowalski" in [edge.entity for edge in answer.path.edges]


def test_hops_zero_misses_answer_chunk(retriever: Retriever) -> None:
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?", hops=0)
    # zero lexical overlap with p1 means the floor cannot surface it at all
    assert all(e.chunk_id != 2 for e in result.evidence)


def test_path_string_cites_bridge(retriever: Retriever) -> None:
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    answer = next(e for e in result.evidence if e.chunk_id == 2)
    rendered = answer.path_string()
    assert 'entity:"kowalski"' in rendered
    assert "chunk#2" in rendered


def test_provenance_components_present(retriever: Retriever) -> None:
    result = retriever.retrieve("Anna Nowak platform team")
    assert result.evidence
    for rank, e in enumerate(result.evidence):
        assert e.score.score >= 0.0
        assert e.score.hop == e.path.hop
        assert e.score.population == ("hop" if e.path.hop >= 1 else "seed")
        assert e.score.rank == rank
        if e.score.population == "hop":
            assert e.score.parent_chunk is not None
            assert e.score.bridge_strength > 0.0
        assert e.path.seed_source in ("mention", "bm25_only")
    assert result.candidates_examined >= result.pool_size >= len(result.evidence)


def test_no_mentions_falls_back_to_bm25(retriever: Retriever) -> None:
    result = retriever.retrieve("budget planning quarter")
    assert any("bm25_only" in note for note in result.notes)
    assert any(e.chunk_id == 5 for e in result.evidence)
    assert result.query_mentions == ()


def test_unresolved_mentions_reported(retriever: Retriever) -> None:
    result = retriever.retrieve("Where does Zbigniew Nieznany work?")
    assert "Zbigniew Nieznany" in result.unresolved_mentions


def test_k_and_hops_overrides(retriever: Retriever) -> None:
    result = retriever.retrieve("Anna Nowak", k=1)
    assert len(result.evidence) == 1


def test_determinism(retriever: Retriever) -> None:
    a = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    b = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    assert [(e.chunk_id, e.score.score) for e in a.evidence] == [
        (e.chunk_id, e.score.score) for e in b.evidence
    ]


def test_submodular_admission_surfaces_answer_early(retriever: Retriever) -> None:
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    # the answer chunk covers the kowalski bridge aspect nothing else has:
    # greedy admission must place it in the top 2
    top2 = [e.chunk_id for e in result.evidence[:2]]
    assert 2 in top2
    answer = next(e for e in result.evidence if e.chunk_id == 2)
    assert answer.score.population == "hop"
    assert answer.score.novelty > 0.0
    assert answer.score.gain > 0.0
