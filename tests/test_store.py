from pathlib import Path

import pytest

from hoptrace.chunker import Chunk
from hoptrace.mentions import Mention
from hoptrace.store import MAX_TF, Store, StoreWriter, pack_postings, unpack_postings


def test_pack_roundtrip() -> None:
    pairs = [(1, 3), (42, 1), (2**32 - 1, MAX_TF)]
    assert unpack_postings(pack_postings(pairs)) == pairs


def test_pack_clamps_tf() -> None:
    assert unpack_postings(pack_postings([(1, MAX_TF + 5)])) == [(1, MAX_TF)]


def _build(target: Path | None) -> Store:
    with StoreWriter(target) as writer:
        doc_id = writer.add_document("guide.md", "Office Guide")
        c1 = writer.add_chunk(doc_id, Chunk(0, "Anna reports to Kowalski", 4, "People"))
        c2 = writer.add_chunk(doc_id, Chunk(1, "Kowalski sits in Room 204", 5, "Rooms"))
        writer.add_mentions(
            c1,
            [
                Mention("anna", "Anna", 0, 4),
                Mention("kowalski", "Kowalski", 16, 24),
            ],
        )
        writer.add_mentions(c2, [Mention("kowalski", "Kowalski", 0, 8)])
        writer.add_aliases([("Anna", "anna"), ("Kowalski", "kowalski")])
        writer.write_postings({"kowalski": [(c1, 1), (c2, 1)], "anna": [(c1, 1)]})
        writer.write_term_counts([(c1, 4), (c2, 5)])
        writer.set_meta("extractor_version", "1")
        return writer.finish(n_chunks=2, avg_chunk_len=4.5)


def test_memory_store_roundtrip() -> None:
    store = _build(None)
    assert store.n_chunks == 2
    assert store.avg_chunk_len == 4.5
    assert store.avg_term_len == 4.5
    assert store.term_count(1) == 4
    assert store.term_count(99) == 0
    assert list(store.iter_term_counts()) == [(1, 4), (2, 5)]
    assert store.postings_blob("anna") is not None
    assert store.meta("schema_version") == "1"
    assert store.meta("extractor_version") == "1"

    chunk = store.get_chunk(1)
    assert chunk.text == "Anna reports to Kowalski"
    assert chunk.heading_context == "People"
    assert chunk.doc_title == "Office Guide"

    assert store.chunk_ids_for_entity("kowalski") == [1, 2]
    assert store.entities_for_chunk(1) == ["anna", "kowalski"]
    assert store.entity_df("kowalski") == 2
    assert store.entity_df("nobody") == 0
    assert store.mention_count("kowalski", 1) == 1
    assert store.postings("kowalski") == [(1, 1), (2, 1)]
    assert store.postings("missing") == []
    assert store.term_df("anna") == 1
    assert store.alias_canonical("Anna") == "anna"
    assert store.alias_canonical("Nobody") is None
    assert store.table_stats()["entities"] == 2


def test_file_store_atomic_build(tmp_path: Path) -> None:
    target = tmp_path / "office.sqlite"
    store = _build(target)
    assert store.n_chunks == 2
    assert target.is_file()
    assert not target.with_name(target.name + ".tmp").exists()
    store.close()

    reopened = Store.open(target)
    assert reopened.get_chunk(2).text == "Kowalski sits in Room 204"
    reopened.close()


def test_rebuild_replaces_previous_corpus(tmp_path: Path) -> None:
    target = tmp_path / "office.sqlite"
    _build(target).close()
    with StoreWriter(target) as writer:
        doc_id = writer.add_document("other.md", None)
        writer.add_chunk(doc_id, Chunk(0, "Fresh content", 2, None))
        store = writer.finish(n_chunks=1, avg_chunk_len=2.0)
    assert store.n_chunks == 1
    assert store.get_chunk(1).text == "Fresh content"
    store.close()


def test_abort_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "office.sqlite"
    with StoreWriter(target) as writer:
        writer.add_document("guide.md", None)
        # exiting without finish() aborts
    assert not target.exists()
    assert not target.with_name(target.name + ".tmp").exists()


def test_readonly_store_rejects_writes(tmp_path: Path) -> None:
    target = tmp_path / "office.sqlite"
    _build(target).close()
    store = Store.open(target)
    with pytest.raises(Exception, match=r"readonly|attempt to write"):
        store._conn.execute("INSERT INTO meta(key, value) VALUES ('x', 'y')")
    store.close()


def test_open_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Store.open(tmp_path / "absent.sqlite")
