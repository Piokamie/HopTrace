"""SQLite persistence: one corpus per database file.

``StoreWriter`` builds ``<target>.tmp`` and atomically replaces the target
on ``finish()``, so readers never see a half-built corpus; ``Store`` opens
read-only. Postings are one row per term holding a packed blob of
``(chunk_id uint32, tf uint16)`` pairs (a row per posting would be ~10^8
rows at corpus scale).
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from hoppath.chunker import Chunk
from hoppath.mentions import Mention

SCHEMA_VERSION = "1"

#: SQLite bound-parameter limit headroom for IN (...) batch lookups.
_SQL_PARAM_BATCH = 900

_POSTING = struct.Struct("<IH")
MAX_TF = 0xFFFF

_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents(
    doc_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    title TEXT
);
CREATE TABLE chunks(
    chunk_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    n_tokens INTEGER NOT NULL,
    heading_context TEXT
);
CREATE TABLE mentions(
    entity TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id),
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    surface TEXT NOT NULL,
    PRIMARY KEY(entity, chunk_id, span_start)
) WITHOUT ROWID;
CREATE VIEW cooccur AS SELECT DISTINCT chunk_id, entity FROM mentions;
CREATE TABLE aliases(
    surface TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX idx_aliases_canonical ON aliases(canonical);
CREATE TABLE postings(
    term TEXT PRIMARY KEY,
    df INTEGER NOT NULL,
    blob BLOB NOT NULL
);
-- BM25 document length (analyzed terms); differs from chunks.n_tokens, the
-- raw tokenizer count used for chunk sizing.
CREATE TABLE chunk_terms(
    chunk_id INTEGER PRIMARY KEY,
    n_terms INTEGER NOT NULL
);
CREATE TABLE corpus_stats(
    n_chunks INTEGER NOT NULL,
    avg_chunk_len REAL NOT NULL,
    avg_term_len REAL NOT NULL
);
"""

# Created at finish(): maintaining these during a bulk load thrashes the page cache.
_MENTION_INDEXES = """
CREATE INDEX idx_mentions_entity ON mentions(entity);
CREATE INDEX idx_mentions_chunk ON mentions(chunk_id);
"""

# Materialized at finish(): hop expansion reads df too often for a COUNT(DISTINCT) per lookup.
_ENTITY_STATS = """
CREATE TABLE entity_stats AS
    SELECT entity, COUNT(DISTINCT chunk_id) AS df FROM mentions GROUP BY entity;
CREATE UNIQUE INDEX idx_entity_stats ON entity_stats(entity);
"""


def pack_postings(pairs: Sequence[tuple[int, int]]) -> bytes:
    return b"".join(_POSTING.pack(chunk_id, min(tf, MAX_TF)) for chunk_id, tf in pairs)


def unpack_postings(blob: bytes) -> list[tuple[int, int]]:
    return [(int(chunk_id), int(tf)) for chunk_id, tf in _POSTING.iter_unpack(blob)]


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    doc_id: int
    ordinal: int
    text: str
    n_tokens: int
    heading_context: str | None
    doc_path: str
    doc_title: str | None


class StoreWriter:
    """Builds a corpus database; use as a context manager or call finish()."""

    def __init__(self, target: Path | None, cache_mb: int = 64) -> None:
        self._target = target
        self._finished = False
        if target is None:
            self._tmp: Path | None = None
            self._conn = sqlite3.connect(":memory:")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._tmp = target.with_name(target.name + ".tmp")
            self._tmp.unlink(missing_ok=True)
            self._conn = sqlite3.connect(self._tmp)
            # Durability is pointless (tmp is discarded on failure); the default
            # ~2 MB page cache collapses corpus-scale builds of the mentions b-tree.
            self._conn.execute("PRAGMA journal_mode = OFF")
            self._conn.execute("PRAGMA synchronous = OFF")
            self._conn.execute(f"PRAGMA cache_size = {-1024 * cache_mb}")
        self._conn.executescript(_SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._finished:
            self.abort()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def set_meta_json(self, key: str, value: object) -> None:
        self.set_meta(key, json.dumps(value, sort_keys=True))

    def add_document(self, path: str, title: str | None = None) -> int:
        cur = self._conn.execute("INSERT INTO documents(path, title) VALUES (?, ?)", (path, title))
        return int(cur.lastrowid or 0)

    def add_chunk(self, doc_id: int, chunk: Chunk) -> int:
        cur = self._conn.execute(
            "INSERT INTO chunks(doc_id, ordinal, text, n_tokens, heading_context)"
            " VALUES (?, ?, ?, ?, ?)",
            (doc_id, chunk.ordinal, chunk.text, chunk.n_tokens, chunk.heading_context),
        )
        return int(cur.lastrowid or 0)

    def add_mentions(self, chunk_id: int, mentions: Sequence[Mention]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO mentions(entity, chunk_id, span_start, span_end, surface)"
            " VALUES (?, ?, ?, ?, ?)",
            [(m.entity, chunk_id, m.start, m.end, m.surface) for m in mentions],
        )

    def add_aliases(self, pairs: Iterable[tuple[str, str]]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO aliases(surface, canonical) VALUES (?, ?)", list(pairs)
        )

    # Bulk paths with caller-assigned ids, for corpus-scale builders.

    def add_documents_bulk(self, rows: Iterable[tuple[int, str, str | None]]) -> None:
        """(doc_id, path, title) rows with explicit ids."""
        self._conn.executemany("INSERT INTO documents(doc_id, path, title) VALUES (?, ?, ?)", rows)

    def add_chunks_bulk(self, rows: Iterable[tuple[int, int, int, str, int, str | None]]) -> None:
        """(chunk_id, doc_id, ordinal, text, n_tokens, heading_context) rows."""
        self._conn.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, ordinal, text, n_tokens, heading_context)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def add_mention_rows(self, rows: Iterable[tuple[str, int, int, int, str]]) -> None:
        """(entity, chunk_id, span_start, span_end, surface) rows."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO mentions(entity, chunk_id, span_start, span_end, surface)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def write_postings(self, postings: Mapping[str, Sequence[tuple[int, int]]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO postings(term, df, blob) VALUES (?, ?, ?)",
            [(term, len(pairs), pack_postings(sorted(pairs))) for term, pairs in postings.items()],
        )

    def write_packed_postings(self, rows: Iterable[tuple[str, int, bytes]]) -> None:
        """Bulk path for corpus-scale builders that pack blobs themselves."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO postings(term, df, blob) VALUES (?, ?, ?)", rows
        )

    def write_term_counts(self, counts: Iterable[tuple[int, int]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunk_terms(chunk_id, n_terms) VALUES (?, ?)", counts
        )

    def finish(self, n_chunks: int, avg_chunk_len: float) -> Store:
        self._conn.executescript(_MENTION_INDEXES)
        self._conn.executescript(_ENTITY_STATS)
        row = self._conn.execute("SELECT AVG(n_terms) FROM chunk_terms").fetchone()
        avg_term_len = float(row[0]) if row and row[0] is not None else 0.0
        self._conn.execute(
            "INSERT INTO corpus_stats(n_chunks, avg_chunk_len, avg_term_len) VALUES (?, ?, ?)",
            (n_chunks, avg_chunk_len, avg_term_len),
        )
        self._conn.commit()
        self._finished = True
        if self._target is None:
            return Store(self._conn)
        self._conn.close()
        assert self._tmp is not None
        os.replace(self._tmp, self._target)
        return Store.open(self._target)

    def abort(self) -> None:
        self._finished = True
        self._conn.close()
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)


def ensure_entity_stats(path: Path) -> bool:
    """Materialize entity_stats on an older corpus that lacks it; True if created."""
    conn = sqlite3.connect(path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entity_stats'"
        ).fetchone()
        if exists:
            return False
        conn.executescript(_ENTITY_STATS)
        conn.commit()
        return True
    finally:
        conn.close()


class Store:
    """Read-only view of a built corpus."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, path: Path) -> Store:
        if not path.is_file():
            raise FileNotFoundError(f"no corpus at {path}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    def meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    @property
    def n_chunks(self) -> int:
        row = self._conn.execute("SELECT n_chunks FROM corpus_stats").fetchone()
        return int(row[0])

    @property
    def avg_chunk_len(self) -> float:
        row = self._conn.execute("SELECT avg_chunk_len FROM corpus_stats").fetchone()
        return float(row[0])

    @property
    def avg_term_len(self) -> float:
        """Average analyzed term count per chunk (BM25 document length)."""
        row = self._conn.execute("SELECT avg_term_len FROM corpus_stats").fetchone()
        return float(row[0])

    def term_count(self, chunk_id: int) -> int:
        row = self._conn.execute(
            "SELECT n_terms FROM chunk_terms WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return 0 if row is None else int(row[0])

    def iter_term_counts(self) -> Iterator[tuple[int, int]]:
        """All (chunk_id, n_terms) rows ordered by chunk_id."""
        cur = self._conn.execute("SELECT chunk_id, n_terms FROM chunk_terms ORDER BY chunk_id")
        for row in cur:
            yield int(row[0]), int(row[1])

    def get_chunk(self, chunk_id: int) -> ChunkRow:
        row = self._conn.execute(
            "SELECT c.chunk_id, c.doc_id, c.ordinal, c.text, c.n_tokens, c.heading_context,"
            " d.path, d.title"
            " FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no chunk {chunk_id}")
        return ChunkRow(
            chunk_id=int(row[0]),
            doc_id=int(row[1]),
            ordinal=int(row[2]),
            text=str(row[3]),
            n_tokens=int(row[4]),
            heading_context=None if row[5] is None else str(row[5]),
            doc_path=str(row[6]),
            doc_title=None if row[7] is None else str(row[7]),
        )

    def chunk_ids_for_entity(self, entity: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT DISTINCT chunk_id FROM mentions WHERE entity = ? ORDER BY chunk_id",
            (entity,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def entities_for_chunk(self, chunk_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT entity FROM mentions WHERE chunk_id = ? ORDER BY entity",
            (chunk_id,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def entity_df(self, entity: str) -> int:
        row = self._conn.execute(
            "SELECT df FROM entity_stats WHERE entity = ?", (entity,)
        ).fetchone()
        return 0 if row is None else int(row[0])

    def entity_dfs(self, entities: Sequence[str]) -> dict[str, int]:
        """Batch df lookup; absent entities are omitted (df 0)."""
        result: dict[str, int] = {}
        batch_size = _SQL_PARAM_BATCH
        ordered = sorted(set(entities))
        for i in range(0, len(ordered), batch_size):
            batch = ordered[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT entity, df FROM entity_stats WHERE entity IN ({placeholders})",
                batch,
            ).fetchall()
            for entity, df in rows:
                result[str(entity)] = int(df)
        return result

    def entity_surface(self, entity: str, chunk_id: int) -> str | None:
        """First as-written surface form of an entity in a chunk."""
        row = self._conn.execute(
            "SELECT surface FROM mentions WHERE entity = ? AND chunk_id = ?"
            " ORDER BY span_start LIMIT 1",
            (entity, chunk_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def entities_in_df_range(self, lo: int, hi: int, limit: int) -> list[str]:
        """Entities with lo <= df <= hi, deterministic order."""
        rows = self._conn.execute(
            "SELECT entity FROM entity_stats WHERE df BETWEEN ? AND ? ORDER BY entity LIMIT ?",
            (lo, hi, limit),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def entity_chunk_counts(self, entity: str, limit: int) -> list[tuple[int, int]]:
        """(chunk_id, mention_count) for an entity, by count desc then chunk_id."""
        rows = self._conn.execute(
            "SELECT chunk_id, COUNT(*) AS n FROM mentions WHERE entity = ?"
            " GROUP BY chunk_id ORDER BY n DESC, chunk_id LIMIT ?",
            (entity, limit),
        ).fetchall()
        return [(int(r[0]), int(r[1])) for r in rows]

    def mention_count(self, entity: str, chunk_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM mentions WHERE entity = ? AND chunk_id = ?",
            (entity, chunk_id),
        ).fetchone()
        return int(row[0])

    def postings(self, term: str) -> list[tuple[int, int]]:
        row = self._conn.execute("SELECT blob FROM postings WHERE term = ?", (term,)).fetchone()
        return [] if row is None else unpack_postings(row[0])

    def postings_blob(self, term: str) -> tuple[int, bytes] | None:
        """(df, packed blob) for numpy decoding."""
        row = self._conn.execute("SELECT df, blob FROM postings WHERE term = ?", (term,)).fetchone()
        return None if row is None else (int(row[0]), bytes(row[1]))

    def term_df(self, term: str) -> int:
        row = self._conn.execute("SELECT df FROM postings WHERE term = ?", (term,)).fetchone()
        return 0 if row is None else int(row[0])

    def chunks_by_doc_path(self, paths: Sequence[str]) -> dict[str, list[int]]:
        """Map document paths to their chunk ids (batch lookup)."""
        mapping: dict[str, list[int]] = {}
        batch_size = _SQL_PARAM_BATCH
        ordered = sorted(set(paths))
        for i in range(0, len(ordered), batch_size):
            batch = ordered[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                "SELECT d.path, c.chunk_id FROM documents d"
                " JOIN chunks c ON c.doc_id = d.doc_id"
                f" WHERE d.path IN ({placeholders}) ORDER BY c.chunk_id",
                batch,
            ).fetchall()
            for path, chunk_id in rows:
                mapping.setdefault(str(path), []).append(int(chunk_id))
        return mapping

    def alias_canonical(self, surface: str) -> str | None:
        row = self._conn.execute(
            "SELECT canonical FROM aliases WHERE surface = ?", (surface,)
        ).fetchone()
        return None if row is None else str(row[0])

    def table_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for table in ("documents", "chunks", "mentions", "aliases", "postings"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[table] = int(row[0])
        row = self._conn.execute("SELECT COUNT(DISTINCT entity) FROM mentions").fetchone()
        stats["entities"] = int(row[0])
        return stats
