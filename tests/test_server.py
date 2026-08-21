import asyncio
from pathlib import Path

import pytest

import hoppath.server as server

DOCS = {
    "people.md": "# People\n\nAnna Nowak reports to Kowalski. We asked Anna about the atrium.",
    "rooms.md": "# Rooms\n\nKowalski occupies Room 4B near the atrium fountain.",
    "cafe.md": "# Cafe\n\nThe atrium fountain cafe closes at five daily.",
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    # the registry caches per corpus id across tests; isolate it
    monkeypatch.setattr(server, "_registry", server.CorpusRegistry())
    src = tmp_path / "docs"
    src.mkdir()
    for name, text in DOCS.items():
        (src / name).write_text(text)
    return src


def test_all_four_tools_registered() -> None:
    tools = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert tools == {"ingest", "retrieve", "explain", "bracket"}


def test_ingest_retrieve_flow(data_dir: Path) -> None:
    report = server.ingest_impl(str(data_dir), "office")
    assert report["documents"] == 3
    assert report["chunks"] == 3
    assert report["analyzer"] == "english"
    assert report["table_stats"]["chunks"] == 3

    payload = server.retrieve_impl("Where does the manager of Anna Nowak sit?", "office")
    assert payload["evidence"]
    item = next(e for e in payload["evidence"] if "Room 4B" in e["chunk"]["text"])
    assert item["hop"] >= 1
    assert "kowalski" in item["path"]
    # the file is one field away and inside the citation string
    assert item["chunk"]["doc"].endswith(".md")
    assert f"({item['chunk']['doc']}:" in item["path"]
    assert payload["source_root"] == str(data_dir.resolve())
    assert report["source_root"] == str(data_dir.resolve())
    explained = server.explain_impl(item["chunk"]["id"], "office")
    assert explained["source_root"] == str(data_dir.resolve())
    # the hop-reached answer shares no query words; the seed does
    assert item["matched_terms"] == []
    seed = next(e for e in payload["evidence"] if e["hop"] == 0)
    assert "Anna" in seed["matched_terms"] or "Nowak" in seed["matched_terms"]
    assert item["path_edges"][-1]["entity"] == "kowalski"
    assert isinstance(item["score"]["score"], float)


def test_unresolved_mentions_are_called_out(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    payload = server.retrieve_impl("where is Zofia Borowska manager", "office")
    assert payload["unresolved_mentions"] == ["Zofia Borowska"]
    assert all(e["seed_source"] == "bm25_only" for e in payload["evidence"])
    assert any('no indexed entity matches "Zofia Borowska"' in n for n in payload["notes"])


def test_retrieve_hops_zero_is_floor(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    payload = server.retrieve_impl("Where does the manager of Anna Nowak sit?", "office", hops=0)
    assert all("Room 4B" not in e["chunk"]["text"] for e in payload["evidence"])


def test_explain_reached_and_unreached(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    query = "Where does the manager of Anna Nowak sit?"

    # find the answer chunk id via retrieve
    payload = server.retrieve_impl(query, "office")
    answer = next(e for e in payload["evidence"] if "Room 4B" in e["chunk"]["text"])
    explained = server.explain_impl(answer["chunk"]["id"], "office", query)
    assert explained["query"]["in_pool"]
    assert explained["query"]["path"] and "kowalski" in explained["query"]["path"]
    assert explained["entities"]

    # a chunk with no path from this query
    cafe = next(
        e
        for e in server.retrieve_impl("fountain cafe daily", "office")["evidence"]
        if "closes at five" in e["chunk"]["text"]
    )
    explained = server.explain_impl(cafe["chunk"]["id"], "office")
    assert "query" not in explained


def test_bracket_tool(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    payload = server.bracket_impl("office", n_questions=6)
    assert payload["floor"]["system"] == "bm25-floor"
    assert payload["hoppath_2hop"]["system"] == "hoppath@2hop"
    assert 0.0 <= payload["multihop_fraction"] <= 1.0
    assert "never as cross-system evidence" in payload["caveat"]


def test_reingest_invalidates_registry(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    before = server.retrieve_impl("Anna Nowak", "office")
    assert before["evidence"]

    (data_dir / "people.md").write_text("# People\n\nBarbara Con runs the archive room.")
    server.ingest_impl(str(data_dir), "office")
    after = server.retrieve_impl("Barbara Con archive", "office")
    assert any("Barbara Con" in e["chunk"]["text"] for e in after["evidence"])


def test_unknown_corpus_lists_available(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    with pytest.raises(ValueError, match="available: office"):
        server.retrieve_impl("anything", "missing")


def test_bad_arguments_rejected(data_dir: Path) -> None:
    server.ingest_impl(str(data_dir), "office")
    with pytest.raises(ValueError, match="hops"):
        server.retrieve_impl("q", "office", hops=5)
    with pytest.raises(ValueError, match="unknown config keys"):
        server.ingest_impl(str(data_dir), "office", {"chunk_size": 100})
    with pytest.raises(ValueError, match="analyzer"):
        server.ingest_impl(str(data_dir), "office", {"analyzer": "german"})
