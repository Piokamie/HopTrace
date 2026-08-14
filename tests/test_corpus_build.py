from pathlib import Path

import pytest

from hoptrace.eval.adapters import load_hotpot_format, load_musique
from hoptrace.eval.corpus_build import (
    build_beir_index,
    build_pooled_index,
    build_question_index,
    flat_text,
)
from hoptrace.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


def test_flat_text() -> None:
    assert flat_text("Title", "body") == "Title body"
    assert flat_text("", "body") == "body"


def test_build_beir_index(tmp_path: Path) -> None:
    target = tmp_path / "beir.sqlite"
    stats = build_beir_index(FIXTURES / "beir" / "corpus.jsonl", target, analyzer="english")
    assert stats.passages == 6
    assert stats.entities > 0

    store = Store.open(target)
    assert store.n_chunks == 6
    assert store.meta("analyzer") == "english"
    # identity mapping: doc path is the BEIR id, one chunk per doc
    mapping = store.chunks_by_doc_path(["d1", "d4"])
    assert mapping == {"d1": [1], "d4": [4]}
    # flat indexing: title participates in the chunk text
    assert store.get_chunk(4).text.startswith("Room 204")
    # the title is an entity of its chunk (span sentinel -1)
    assert 2 in store.chunk_ids_for_entity("jan kowalski")
    # stemmed postings exist under the english analyzer
    assert store.term_df("meridian") > 0
    store.close()


def test_build_beir_index_limit(tmp_path: Path) -> None:
    target = tmp_path / "beir.sqlite"
    stats = build_beir_index(
        FIXTURES / "beir" / "corpus.jsonl", target, analyzer="english", limit=2
    )
    assert stats.passages == 2
    store = Store.open(target)
    assert store.n_chunks == 2
    store.close()


def test_pooled_index_dedup_and_gold() -> None:
    questions = load_musique(FIXTURES / "musique_dev.jsonl")
    pooled = build_pooled_index(questions, analyzer="english")
    # 6 paragraphs across questions, all unique in the fixture
    assert pooled.store.n_chunks == 6
    for question in questions:
        gold = pooled.gold_chunks[question.qid]
        assert len(gold) == 2
        for chunk_id in gold:
            title, _text = pooled.chunk_keys[chunk_id]
            assert title in {p.title for p in question.paragraphs if p.is_gold}
    pooled.store.close()


def test_pooled_index_dedups_repeated_paragraphs() -> None:
    questions = load_musique(FIXTURES / "musique_dev.jsonl")
    doubled = [*questions, *questions]
    pooled = build_pooled_index(doubled, analyzer="english")
    assert pooled.store.n_chunks == 6  # no duplicates from pooling twice
    pooled.store.close()


def test_question_index() -> None:
    questions = load_hotpot_format(FIXTURES / "2wiki_dev.json")
    store, chunk_to_key = build_question_index(questions[0], analyzer="english")
    assert store.n_chunks == 3
    assert set(chunk_to_key.values()) == {"Jan Kowalski", "Room 204", "Krakow"}
    store.close()


def test_deterministic_build(tmp_path: Path) -> None:
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    build_beir_index(FIXTURES / "beir" / "corpus.jsonl", a)
    build_beir_index(FIXTURES / "beir" / "corpus.jsonl", b)
    sa, sb = Store.open(a), Store.open(b)
    assert sa.table_stats() == sb.table_stats()
    assert sa.postings("meridian") == sb.postings("meridian")
    assert sa.entities_for_chunk(2) == sb.entities_for_chunk(2)
    sa.close()
    sb.close()


def test_load_beir_answers() -> None:
    from hoptrace.eval.corpus_build import load_beir_answers

    answers = load_beir_answers(FIXTURES / "beir" / "queries.jsonl")
    assert answers == {"q1": "Jan Kowalski", "q2": "Room 204"}


@pytest.mark.parametrize("analyzer", ["english", "simple"])
def test_analyzer_recorded(tmp_path: Path, analyzer: str) -> None:
    target = tmp_path / "x.sqlite"
    build_beir_index(FIXTURES / "beir" / "corpus.jsonl", target, analyzer=analyzer)
    store = Store.open(target)
    assert store.meta("analyzer") == analyzer
    store.close()
