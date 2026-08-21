"""Ingest pipeline: documents → chunks → mentions → tables. No LLM.

Postings accumulate in memory and are flushed once at the end; corpus-scale
builders drive ``StoreWriter`` directly, shard by shard.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hoppath.chunker import Chunk, chunk_document
from hoppath.config import ChunkConfig, ExtractorConfig, corpus_path
from hoppath.mentions import EXTRACTOR_VERSION, Mention, MentionExtractor, collect_non_initial_caps
from hoppath.normalize import Normalizer, TrivialNormalizer
from hoppath.store import Store, StoreWriter
from hoppath.tokenize import analyze

_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".rst"})


@dataclass(frozen=True)
class SourceDocument:
    path: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    entities: int
    mentions: int
    table_stats: dict[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()


def ingest_documents(
    docs: Iterable[SourceDocument],
    target: Path | None,
    chunk_cfg: ChunkConfig | None = None,
    extractor_cfg: ExtractorConfig | None = None,
    normalizer: Normalizer | None = None,
    analyzer: str = "simple",
    title_mentions: bool = False,
    source_root: str | None = None,
) -> tuple[Store, IngestReport]:
    """Build a corpus from documents; ``target=None`` builds in memory.

    Two passes: chunk everything and collect corpus-wide non-sentence-initial
    capitalization, then extract mentions with that evidence, so a
    sentence-initial name is kept if it appears mid-sentence anywhere.

    ``title_mentions`` indexes each document's title as an entity of its
    first chunk (span ``(-1, -1)``) and trusts title words corpus-wide, for
    encyclopedic corpora where the title is the entity a paragraph is about.
    """
    chunk_cfg = chunk_cfg or ChunkConfig()
    extractor_cfg = extractor_cfg or ExtractorConfig()
    extractor = MentionExtractor(extractor_cfg, normalizer)
    active_normalizer = normalizer or TrivialNormalizer()

    chunked: list[tuple[SourceDocument, list[Chunk]]] = []
    corpus_caps: set[str] = set()
    for doc in docs:
        chunks = chunk_document(doc.text, chunk_cfg)
        if not chunks:
            continue
        chunked.append((doc, chunks))
        for chunk in chunks:
            corpus_caps.update(collect_non_initial_caps(chunk.text))
        if title_mentions and doc.title:
            corpus_caps.update(t.casefold() for t in doc.title.split())

    postings: dict[str, list[tuple[int, int]]] = {}
    entities: set[str] = set()
    n_documents = 0
    n_chunks = 0
    n_mentions = 0
    total_tokens = 0

    term_counts: list[tuple[int, int]] = []
    with StoreWriter(target) as writer:
        writer.set_meta("extractor_version", EXTRACTOR_VERSION)
        writer.set_meta("analyzer", analyzer)
        if source_root is not None:
            writer.set_meta("source_root", source_root)
        writer.set_meta_json(
            "config",
            {"chunk": vars(chunk_cfg), "extractor": vars(extractor_cfg), "analyzer": analyzer},
        )
        for doc, chunks in chunked:
            n_documents += 1
            doc_id = writer.add_document(doc.path, doc.title)
            for chunk in chunks:
                chunk_id = writer.add_chunk(doc_id, chunk)
                n_chunks += 1
                total_tokens += chunk.n_tokens
                terms = analyze(chunk.text, analyzer)
                term_counts.append((chunk_id, len(terms)))
                for term, tf in Counter(terms).items():
                    postings.setdefault(term, []).append((chunk_id, tf))
                mentions = extractor.extract(chunk.text, corpus_caps)
                if title_mentions and doc.title and chunk.ordinal == 0:
                    title_entity = active_normalizer.canonical(doc.title)
                    if len(title_entity) >= 2:
                        mentions = [
                            *mentions,
                            Mention(entity=title_entity, surface=doc.title, start=-1, end=-1),
                        ]
                writer.add_mentions(chunk_id, mentions)
                writer.add_aliases((m.surface, m.entity) for m in mentions)
                n_mentions += len(mentions)
                entities.update(m.entity for m in mentions)
        writer.write_postings(postings)
        writer.write_term_counts(term_counts)
        avg_len = total_tokens / n_chunks if n_chunks else 0.0
        store = writer.finish(n_chunks=n_chunks, avg_chunk_len=avg_len)

    report = IngestReport(
        documents=n_documents,
        chunks=n_chunks,
        entities=len(entities),
        mentions=n_mentions,
        table_stats=store.table_stats(),
    )
    return store, report


def ingest_path(
    source: Path,
    corpus_id: str,
    chunk_cfg: ChunkConfig | None = None,
    extractor_cfg: ExtractorConfig | None = None,
    normalizer: Normalizer | None = None,
    analyzer: str = "simple",
) -> tuple[Store, IngestReport]:
    """Ingest a file or a directory tree of text files into a named corpus."""
    docs: list[SourceDocument] = []
    skipped: list[str] = []
    files = [source] if source.is_file() else sorted(source.rglob("*"))
    for file in files:
        if not file.is_file():
            continue
        if source.is_dir() and file.suffix.lower() not in _TEXT_SUFFIXES:
            skipped.append(str(file))
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped.append(str(file))
            continue
        rel = str(file.relative_to(source) if source.is_dir() else file.name)
        docs.append(SourceDocument(path=rel, text=text, title=_title_of(text) or rel))
    if not docs:
        raise ValueError(f"no ingestable text files under {source}")

    store, report = ingest_documents(
        docs,
        corpus_path(corpus_id),
        chunk_cfg,
        extractor_cfg,
        normalizer,
        analyzer=analyzer,
        source_root=str(source.resolve()),
    )
    return store, IngestReport(
        documents=report.documents,
        chunks=report.chunks,
        entities=report.entities,
        mentions=report.mentions,
        table_stats=report.table_stats,
        skipped=tuple(skipped),
    )


def _title_of(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return None
    return None
