"""Corpus builders for the eval harness.

``build_beir_index`` streams a multi-million-passage BEIR corpus.jsonl with
packed posting accumulators and bulk inserts; ``ingest_documents`` would
materialize it all in RAM. The pooled and explicit builders delegate to
``ingest_documents``. All index ``title + " " + text`` (the Anserini "flat"
condition) with identity chunking, one chunk per source paragraph, and record
the title as an entity of its chunk: on encyclopedic corpora the title is the
name other paragraphs bridge with.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hoptrace.config import ChunkConfig, ExtractorConfig
from hoptrace.eval.adapters import EvalQuestion, Paragraph, iter_beir_corpus
from hoptrace.ingest import SourceDocument, ingest_documents
from hoptrace.mentions import EXTRACTOR_VERSION, MentionExtractor
from hoptrace.normalize import TrivialNormalizer
from hoptrace.store import MAX_TF, Store, StoreWriter
from hoptrace.tokenize import analyze, tokenize

_POSTING = struct.Struct("<IH")
_BATCH = 20_000


@dataclass(frozen=True)
class BuildStats:
    passages: int
    entities: int
    mentions: int
    elapsed_s: float


def flat_text(title: str, text: str) -> str:
    return f"{title} {text}".strip() if title else text


def build_beir_index(
    corpus_jsonl: Path,
    target: Path,
    analyzer: str = "english",
    with_mentions: bool = True,
    limit: int | None = None,
    progress_every: int = 200_000,
) -> BuildStats:
    """Stream corpus.jsonl into a persistent flat index at ``target``."""
    started = time.perf_counter()
    normalizer = TrivialNormalizer()
    extractor = MentionExtractor(ExtractorConfig(), normalizer)

    # Pass A: title words as the trust set for the sentence-initial caps rule
    # (cheaper than the full sweep ingest does).
    title_words: set[str] = set()
    for n_passages, (_, title, _text) in enumerate(iter_beir_corpus(corpus_jsonl), 1):
        if title:
            title_words.update(w.casefold() for w in title.split())
        if limit is not None and n_passages >= limit:
            break

    postings: dict[str, bytearray] = {}
    n_mentions = 0

    with StoreWriter(target, cache_mb=512) as writer:
        writer.set_meta("extractor_version", EXTRACTOR_VERSION)
        writer.set_meta("analyzer", analyzer)
        writer.set_meta("flavor", "beir-flat")

        doc_rows: list[tuple[int, str, str | None]] = []
        chunk_rows: list[tuple[int, int, int, str, int, str | None]] = []
        mention_rows: list[tuple[str, int, int, int, str]] = []
        term_counts: list[tuple[int, int]] = []

        def flush() -> None:
            writer.add_documents_bulk(doc_rows)
            writer.add_chunks_bulk(chunk_rows)
            writer.add_mention_rows(mention_rows)
            writer.write_term_counts(term_counts)
            doc_rows.clear()
            chunk_rows.clear()
            mention_rows.clear()
            term_counts.clear()

        chunk_id = 0
        total_tokens = 0
        for doc_id_str, title, text in iter_beir_corpus(corpus_jsonl):
            chunk_id += 1
            indexed = flat_text(title, text)
            n_raw = len(tokenize(indexed))
            total_tokens += n_raw
            doc_rows.append((chunk_id, doc_id_str, title or None))
            chunk_rows.append((chunk_id, chunk_id, 0, indexed, n_raw, None))

            terms = analyze(indexed, analyzer)
            term_counts.append((chunk_id, len(terms)))
            for term, tf in Counter(terms).items():
                postings.setdefault(term, bytearray()).extend(
                    _POSTING.pack(chunk_id, min(tf, MAX_TF))
                )

            if with_mentions:
                mentions = extractor.extract(indexed, title_words)
                for m in mentions:
                    mention_rows.append((m.entity, chunk_id, m.start, m.end, m.surface))
                n_mentions += len(mentions)
                if title:
                    title_entity = normalizer.canonical(title)
                    if len(title_entity) >= 2:
                        mention_rows.append((title_entity, chunk_id, -1, -1, title))
                        n_mentions += 1

            if len(chunk_rows) >= _BATCH:
                flush()
            if progress_every and chunk_id % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[hoptrace] indexed {chunk_id:,} passages"
                    f" ({chunk_id / max(elapsed, 1e-9):,.0f}/s)",
                    file=sys.stderr,
                    flush=True,
                )
            if limit is not None and chunk_id >= limit:
                break
        flush()

        writer.write_packed_postings(
            (term, len(buf) // _POSTING.size, bytes(buf)) for term, buf in postings.items()
        )
        avg_len = total_tokens / chunk_id if chunk_id else 0.0
        store = writer.finish(n_chunks=chunk_id, avg_chunk_len=avg_len)
        # counted in SQL: an in-RAM entity set would be gigabytes at Wikipedia scale
        n_entities = store.table_stats()["entities"]
        store.close()

    return BuildStats(
        passages=chunk_id,
        entities=n_entities,
        mentions=n_mentions,
        elapsed_s=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class PooledIndex:
    store: Store
    #: chunk_id -> pooled paragraph key (title, text)
    chunk_keys: dict[int, tuple[str, str]]
    #: per question: qid -> gold chunk ids
    gold_chunks: dict[str, frozenset[int]]


def _check_reuse_analyzer(store: Store, analyzer: str, target: Path) -> None:
    """A cached index built with another analyzer silently yields near-zero recall."""
    stored = store.meta("analyzer")
    if stored is not None and stored != analyzer:
        store.close()
        raise ValueError(
            f"cached index {target} was built with analyzer={stored!r}, run requested"
            f" {analyzer!r}: delete it or pass the stored analyzer"
        )


def _verify_reuse_mapping(
    store: Store, docs: Sequence[SourceDocument], path_to_chunk: dict[str, int], target: Path
) -> None:
    """Chunk-count equality alone would accept a same-sized cache built from
    a differently ordered question set and map gold onto the wrong chunks."""
    for doc in docs:
        chunk = store.get_chunk(path_to_chunk[doc.path])
        if chunk.text != doc.text:
            store.close()
            raise ValueError(
                f"cached index {target} does not match this corpus: {doc.path} holds"
                f" a different paragraph (built from another question set or --limit?)."
                " Delete the file to rebuild"
            )


def paragraph_pool_key(paragraph: Paragraph) -> tuple[str, str]:
    return (paragraph.title, paragraph.text)


def build_pooled_index(
    questions: Sequence[EvalQuestion],
    target: Path | None = None,
    analyzer: str = "english",
    progress: Callable[[str], None] | None = None,
    reuse: bool = False,
) -> PooledIndex:
    """Union of all questions' paragraphs, deduped by (title, text).

    ``reuse`` opens an existing ``target`` instead of re-ingesting (the
    all-splits pool is ~100k paragraphs).
    """
    docs: list[SourceDocument] = []
    key_to_ordinal: dict[tuple[str, str], int] = {}
    for question in questions:
        for paragraph in question.paragraphs:
            key = paragraph_pool_key(paragraph)
            if key not in key_to_ordinal:
                key_to_ordinal[key] = len(docs)
                docs.append(
                    SourceDocument(
                        path=f"p{len(docs)}",
                        text=flat_text(paragraph.title, paragraph.text),
                        title=paragraph.title,
                    )
                )
    if progress:
        progress(f"pooled corpus: {len(docs)} unique paragraphs")

    reused = reuse and target is not None and target.is_file()
    if reused and target is not None:
        store = Store.open(target)
        _check_reuse_analyzer(store, analyzer, target)
        n_chunks = store.n_chunks
        if progress:
            progress(f"reusing pooled index at {target} ({n_chunks:,} chunks)")
    else:
        store, report = ingest_documents(
            docs,
            target,
            chunk_cfg=ChunkConfig(identity=True),
            analyzer=analyzer,
            title_mentions=True,
        )
        n_chunks = report.chunks
    # ingest drops chunk-less documents, which would shift the ordinals
    if n_chunks != len(docs):
        store.close()
        raise ValueError(
            f"identity mapping broken: {len(docs)} paragraphs produced"
            f" {n_chunks} chunks (empty paragraph, or a cached index built"
            " from a different question set?)"
        )
    path_to_chunk = {
        path: ids[0] for path, ids in store.chunks_by_doc_path([d.path for d in docs]).items()
    }
    if reused and target is not None:
        _verify_reuse_mapping(store, docs, path_to_chunk, target)
    ordinal_chunk = {ordinal: path_to_chunk[f"p{ordinal}"] for ordinal in range(len(docs))}
    chunk_keys = {ordinal_chunk[ordinal]: key for key, ordinal in key_to_ordinal.items()}
    gold_chunks = {
        question.qid: frozenset(
            ordinal_chunk[key_to_ordinal[paragraph_pool_key(p)]]
            for p in question.paragraphs
            if p.is_gold
        )
        for question in questions
    }
    return PooledIndex(store=store, chunk_keys=chunk_keys, gold_chunks=gold_chunks)


@dataclass
class ExplicitIndex:
    store: Store
    #: (title, text) -> chunk id
    key_to_chunk: dict[tuple[str, str], int]


def build_explicit_index(
    entries: Sequence[tuple[str, str]],
    target: Path | None = None,
    analyzer: str = "english",
    progress: Callable[[str], None] | None = None,
    reuse: bool = False,
) -> ExplicitIndex:
    """Index a published corpus given as (title, text) entries."""
    seen: dict[tuple[str, str], int] = {}
    docs: list[SourceDocument] = []
    for title, text in entries:
        key = (title, text)
        if key in seen:
            continue
        seen[key] = len(docs)
        docs.append(SourceDocument(path=f"p{len(docs)}", text=flat_text(title, text), title=title))
    if len(seen) != len(entries) and progress:
        progress(f"corpus: {len(entries) - len(seen)} duplicate (title, text) entries collapsed")
    if progress:
        progress(f"corpus: {len(docs)} passages")

    reused = reuse and target is not None and target.is_file()
    if reused and target is not None:
        store = Store.open(target)
        _check_reuse_analyzer(store, analyzer, target)
        n_chunks = store.n_chunks
        if progress:
            progress(f"reusing index at {target} ({n_chunks:,} chunks)")
    else:
        store, report = ingest_documents(
            docs,
            target,
            chunk_cfg=ChunkConfig(identity=True),
            analyzer=analyzer,
            title_mentions=True,
        )
        n_chunks = report.chunks
    if n_chunks != len(docs):
        store.close()
        raise ValueError(
            f"identity mapping broken: {len(docs)} passages produced {n_chunks} chunks"
            " (empty passage, or a cached index built from a different corpus?)"
        )
    path_to_chunk = {
        path: ids[0] for path, ids in store.chunks_by_doc_path([d.path for d in docs]).items()
    }
    if reused and target is not None:
        _verify_reuse_mapping(store, docs, path_to_chunk, target)
    return ExplicitIndex(
        store=store,
        key_to_chunk={key: path_to_chunk[f"p{ordinal}"] for key, ordinal in seen.items()},
    )


def build_question_index(
    question: EvalQuestion, analyzer: str = "english"
) -> tuple[Store, dict[int, str]]:
    """Per-question in-memory index for the distractor sanity check."""
    docs = [
        SourceDocument(
            path=paragraph.key,
            text=flat_text(paragraph.title, paragraph.text),
            title=paragraph.title,
        )
        for paragraph in question.paragraphs
    ]
    store, report = ingest_documents(
        docs,
        None,
        chunk_cfg=ChunkConfig(identity=True),
        analyzer=analyzer,
        title_mentions=True,
    )
    if report.chunks != len(docs):
        store.close()
        raise ValueError(
            f"identity mapping broken: {len(docs)} paragraphs produced"
            f" {report.chunks} chunks (empty paragraph in question {question.qid})"
        )
    mapping = store.chunks_by_doc_path([p.key for p in question.paragraphs])
    chunk_to_key = {mapping[p.key][0]: p.key for p in question.paragraphs}
    return store, chunk_to_key


def load_beir_answers(queries_jsonl: Path) -> dict[str, str]:
    """Answers from BEIR queries metadata, where present."""
    answers: dict[str, str] = {}
    with queries_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = (row.get("metadata") or {}).get("answer")
            if answer:
                answers[str(row["_id"])] = str(answer)
    return answers
