from pathlib import Path

import pytest

from hoppath.config import ChunkConfig
from hoppath.ingest import SourceDocument, ingest_documents, ingest_path

DOCS = [
    SourceDocument(
        path="people.md",
        text="# People\n\nAnna reports to Kowalski. We asked Anna about Atlas.",
        title="People",
    ),
    # "Kowalski" here is sentence-initial; only the corpus-wide two-pass
    # extraction (evidence from people.md) keeps it as an entity.
    SourceDocument(
        path="rooms.md",
        text="# Rooms\n\nKowalski sits in Room 204.",
        title="Rooms",
    ),
]


def test_ingest_documents_in_memory() -> None:
    store, report = ingest_documents(DOCS, target=None)
    assert report.documents == 2
    assert report.chunks == 2
    assert report.entities >= 3  # anna, kowalski, atlas, room 204
    assert report.mentions >= 4
    assert report.table_stats["chunks"] == 2

    # hop path exists: kowalski bridges both chunks
    assert len(store.chunk_ids_for_entity("kowalski")) == 2
    # postings consistent with df
    assert store.term_df("kowalski") == len(store.postings("kowalski")) == 2
    assert store.meta("extractor_version") == "1"
    assert store.meta("config") is not None
    store.close()


def test_aliases_populated() -> None:
    store, _ = ingest_documents(DOCS, target=None)
    assert store.alias_canonical("Kowalski") == "kowalski"
    store.close()


def test_empty_documents_skipped() -> None:
    docs = [*DOCS, SourceDocument(path="empty.md", text="   \n\n  ")]
    _, report = ingest_documents(docs, target=None)
    assert report.documents == 2


def test_ingest_path_walks_and_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nAnna reports to Kowalski.")
    sub = src / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("Kowalski sits in Room 204.")
    (src / "binary.bin").write_bytes(b"\x00\x01\x02")
    (src / "broken.md").write_bytes(b"\xff\xfe invalid \xff")

    store, report = ingest_path(src, "office")
    assert report.documents == 2
    assert sorted(Path(p).name for p in report.skipped) == ["binary.bin", "broken.md"]
    assert (tmp_path / "data" / "office.sqlite").is_file()
    store.close()


def test_ingest_path_no_text_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "binary.bin").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="no ingestable"):
        ingest_path(src, "office")


def test_identity_mode_preserves_paragraph_mapping() -> None:
    paragraphs = ["Anna reports to Kowalski.", "Kowalski sits in Room 204."]
    docs = [SourceDocument(path=f"p{i}", text=p, title=f"p{i}") for i, p in enumerate(paragraphs)]
    store, report = ingest_documents(docs, target=None, chunk_cfg=ChunkConfig(identity=True))
    assert report.chunks == len(paragraphs)
    for chunk_id, expected in zip((1, 2), paragraphs, strict=True):
        assert store.get_chunk(chunk_id).text == expected
    store.close()
