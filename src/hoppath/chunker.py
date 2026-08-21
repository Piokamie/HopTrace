"""Structural chunking on heading and paragraph boundaries.

Paragraphs are packed greedily up to the target size; a pack never crosses
a heading. A paragraph over ``max_tokens`` is split at sentence boundaries,
and a single over-long sentence stands alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hoppath.config import ChunkConfig
from hoppath.tokenize import tokenize

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    n_tokens: int
    heading_context: str | None = None


def chunk_document(text: str, cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    if cfg.identity:
        stripped = text.strip()
        if not stripped:
            return []
        return [Chunk(ordinal=0, text=stripped, n_tokens=len(tokenize(stripped)))]

    pieces: list[tuple[str | None, str, int]] = []
    for heading, paragraph in _paragraphs(text):
        n = len(tokenize(paragraph))
        if n > cfg.max_tokens:
            pieces.extend(
                (heading, part, len(tokenize(part)))
                for part in _split_long_paragraph(paragraph, cfg.target_tokens)
            )
        else:
            pieces.append((heading, paragraph, n))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_heading: str | None = None

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunk_text = "\n\n".join(buf)
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=chunk_text,
                    n_tokens=len(tokenize(chunk_text)),
                    heading_context=buf_heading,
                )
            )
            buf = []
            buf_tokens = 0

    for heading, piece, n in pieces:
        if buf and (heading != buf_heading or buf_tokens + n > cfg.target_tokens):
            flush()
        buf_heading = heading
        buf.append(piece)
        buf_tokens += n
    flush()
    return chunks


def _paragraphs(text: str) -> list[tuple[str | None, str]]:
    """Split into (heading_context, paragraph_text) pairs."""
    result: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if lines:
            result.append((heading, "\n".join(lines)))
            lines = []

    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            heading = heading_match.group(2)
        elif line.strip():
            lines.append(line.rstrip())
        else:
            flush()
    flush()
    return result


def _split_long_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    """Greedily pack sentences up to the target; never cut inside a sentence."""
    parts: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sentence in _SENTENCE_RE.split(paragraph):
        n = len(tokenize(sentence))
        if buf and buf_tokens + n > target_tokens:
            parts.append(" ".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(sentence)
        buf_tokens += n
    if buf:
        parts.append(" ".join(buf))
    return parts
