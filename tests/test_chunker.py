from hoppath.chunker import chunk_document
from hoppath.config import ChunkConfig

DOC = """# Office Guide

## People

Anna reports to Kowalski. She works on the Atlas project.

Kowalski leads the platform team.

## Rooms

Kowalski sits in Room 204.
"""


def test_heading_boundaries_never_crossed() -> None:
    chunks = chunk_document(DOC, ChunkConfig(target_tokens=1000))
    headings = [c.heading_context for c in chunks]
    assert headings == ["People", "Rooms"]
    assert "Anna reports" in chunks[0].text
    assert "Room 204" in chunks[1].text


def test_packing_respects_target() -> None:
    # tiny target: each paragraph becomes its own chunk
    chunks = chunk_document(DOC, ChunkConfig(target_tokens=5))
    assert len(chunks) == 3
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_oversized_paragraph_splits_at_sentences() -> None:
    sentences = " ".join(f"Sentence number {i} has five tokens." for i in range(40))
    chunks = chunk_document(sentences, ChunkConfig(target_tokens=30, max_tokens=60))
    assert len(chunks) > 1
    # no chunk grossly exceeds target, and no sentence was cut
    assert all(c.text.rstrip().endswith(".") for c in chunks)
    assert all(c.n_tokens <= 40 for c in chunks)


def test_identity_mode_one_chunk_verbatim() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\n# Not a heading context"
    chunks = chunk_document(text, ChunkConfig(identity=True))
    assert len(chunks) == 1
    assert chunks[0].text == text.strip()
    assert chunks[0].heading_context is None


def test_empty_inputs() -> None:
    assert chunk_document("") == []
    assert chunk_document("\n\n\n") == []
    assert chunk_document("", ChunkConfig(identity=True)) == []


def test_token_counts_match_tokenizer() -> None:
    from hoppath.tokenize import tokenize

    chunks = chunk_document(DOC)
    for chunk in chunks:
        assert chunk.n_tokens == len(tokenize(chunk.text))
