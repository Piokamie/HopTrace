"""End-to-end proof over the shipped demo corpus: the designed questions
from examples/QUESTIONS.md are answered at hops=2 via their documented
bridges and missed by the floor (hops=0)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from hoptrace.bracket import run_bracket
from hoptrace.config import RetrievalConfig, corpus_path
from hoptrace.ingest import ingest_path
from hoptrace.retrieve import Retriever
from hoptrace.store import Store

OFFICE = Path(__file__).parent.parent / "examples" / "office"

# (question, answer doc, bridge entity) — rows 1-4, 6 and 7 of QUESTIONS.md;
# row 5 is the documented weakly-lexical case, tested separately.
DESIGNED = [
    (
        "Where does the manager of Alicja Rud sit?",
        "people/marek-sosna.md",
        "marek sosna",
    ),
    (
        "What equipment is in the room where the calibration team meets?",
        "facilities/hala-d.md",
        "hala d",
    ),
    (
        "Which initiative involves the person who maintains the procurement ledger?",
        "projects/vega.md",
        "tomasz gil",
    ),
    (
        "Who approves invoices sent by the vendor servicing the elevators?",
        "people/beata-lis.md",
        "koleo serwis",
    ),
    (
        "What colour of pass is needed for the floor where the vault is?",
        "security/badges.md",
        "level 3",
    ),
    (
        "What can the manager of Alicja Rud see from the window?",
        "facilities/office-b12.md",
        "office b12",
    ),
]


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    patcher = pytest.MonkeyPatch()
    patcher.setenv("HOPTRACE_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    store, report = ingest_path(OFFICE, "office", analyzer="english")
    assert report.documents == 25  # QUESTIONS.md lives OUTSIDE the corpus dir
    assert corpus_path("office").is_file()
    yield store
    patcher.undo()


@pytest.fixture(scope="module")
def retriever(store: Store) -> Retriever:
    return Retriever(store, RetrievalConfig(hops=2, k=8))


def answer_chunks(store: Store, doc: str) -> set[int]:
    return set(store.chunks_by_doc_path([doc]).get(doc, []))


@pytest.mark.parametrize(("question", "answer_doc", "bridge"), DESIGNED)
def test_designed_questions_answered_at_two_hops(
    store: Store, retriever: Retriever, question: str, answer_doc: str, bridge: str
) -> None:
    gold = answer_chunks(store, answer_doc)
    assert gold, f"missing answer doc {answer_doc}"

    result = retriever.retrieve(question, hops=2)
    found = [e for e in result.evidence if e.chunk_id in gold]
    assert found, f"{question!r} did not surface {answer_doc} at hops=2"
    evidence = found[0]
    assert evidence.path.hop >= 1
    path_entities = [edge.entity for edge in evidence.path.edges]
    assert bridge in path_entities, f"path {path_entities} does not cite bridge {bridge!r}"


@pytest.mark.parametrize(("question", "answer_doc", "bridge"), DESIGNED)
def test_designed_questions_missed_by_floor(
    store: Store, retriever: Retriever, question: str, answer_doc: str, bridge: str
) -> None:
    gold = answer_chunks(store, answer_doc)
    result = retriever.retrieve(question, hops=0)
    assert all(e.chunk_id not in gold for e in result.evidence), (
        f"{question!r} reached {answer_doc} lexically — the question is not"
        " single-hop-proof; fix the corpus or the question"
    )


def test_two_bridge_question_is_reached_at_hop_two(store: Store, retriever: Retriever) -> None:
    """Question 7: Alicja -> Marek Sosna -> Office B12; the answer file is two
    bridges from the seed and the recorded path shows both, in order."""
    gold = answer_chunks(store, "facilities/office-b12.md")
    result = retriever.retrieve("What can the manager of Alicja Rud see from the window?", hops=2)
    hit = next(e for e in result.evidence if e.chunk_id in gold)
    assert hit.path.hop == 2
    assert [edge.entity for edge in hit.path.edges][1:] == ["marek sosna", "office b12"]
    assert hit.matched_terms == ()
    assert [doc for _, doc in hit.path_docs] == [
        "people/alicja-rud.md",
        "people/marek-sosna.md",
        "facilities/office-b12.md",
    ]


def test_weakly_lexical_case_documented(store: Store, retriever: Retriever) -> None:
    """Question 5 shares one stem ('constructed') with its answer doc: the
    floor may rank it, but only the hop result explains it via the bridge."""
    question = "When was the building that hosts the archive constructed?"
    gold = answer_chunks(store, "facilities/budynek-c.md")
    result = retriever.retrieve(question, hops=2)
    found = [e for e in result.evidence if e.chunk_id in gold]
    assert found
    if found[0].path.hop >= 1:
        assert "budynek c" in [edge.entity for edge in found[0].path.edges]


def test_bracket_runs_clean(store: Store) -> None:
    report = run_bracket(store, RetrievalConfig(k=8), n_questions=20, seed=0)
    assert report.n_questions > 0
    assert report.rows[2].all_gold >= report.rows[0].all_gold
    assert "never as cross-system evidence" in report.caveat
