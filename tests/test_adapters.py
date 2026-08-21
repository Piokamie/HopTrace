import json
from pathlib import Path

import pytest

from hoppath.eval.adapters import (
    SchemaError,
    iter_beir_corpus,
    load_beir_qrels,
    load_beir_queries,
    load_hotpot_format,
    load_musique,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_beir_corpus() -> None:
    rows = list(iter_beir_corpus(FIXTURES / "beir" / "corpus.jsonl"))
    assert len(rows) == 6
    assert rows[0] == (
        "d1",
        "Anna Nowak",
        "Anna Nowak is an engineer who reports to Jan Kowalski at Meridian Labs.",
    )


def test_beir_queries_and_qrels() -> None:
    queries = load_beir_queries(FIXTURES / "beir" / "queries.jsonl")
    assert queries["q1"] == "Who does Anna Nowak report to?"
    qrels = load_beir_qrels(FIXTURES / "beir" / "qrels" / "test.tsv")
    assert qrels == {"q1": {"d1", "d2"}, "q2": {"d2", "d4"}}


def test_musique() -> None:
    questions = load_musique(FIXTURES / "musique_dev.jsonl")
    assert len(questions) == 2
    q = questions[0]
    assert q.qid == "2hop__1001_1002"
    assert q.qtype == "2hop"
    assert q.answer == "Jan Kowalski"
    assert len(q.paragraphs) == 3
    assert q.gold_keys == {"0-Anna Nowak", "1-Jan Kowalski"}


def test_2wiki_hotpot_format() -> None:
    questions = load_hotpot_format(FIXTURES / "2wiki_dev.json")
    assert len(questions) == 2
    bridge, comparison = questions
    assert bridge.qtype == "bridge"
    assert bridge.gold_keys == {"Jan Kowalski", "Room 204"}
    # sentences are joined into one paragraph text
    assert "joined Meridian Labs" in bridge.paragraphs[0].text
    assert comparison.qtype == "comparison"
    assert comparison.answer == "yes"


def test_schema_drift_fails_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q?"}) + "\n")
    with pytest.raises((SchemaError, KeyError)):
        load_musique(bad)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(SchemaError, match="top-level list"):
        load_hotpot_format(bad_json)

    bad_qrels = tmp_path / "bad.tsv"
    bad_qrels.write_text("no header here\nq1 d1 1\n")
    with pytest.raises(SchemaError, match="header"):
        load_beir_qrels(bad_qrels)
