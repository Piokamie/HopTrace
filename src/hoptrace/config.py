"""Configuration dataclasses and data-directory resolution.

Every tunable that could differ between environments or experiments lives
here, never as a magic number at a call site. Retrieval defaults are the
values recorded in eval reports.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_CORPUS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 200
    max_tokens: int = 400
    #: Eval mode: one document becomes exactly one chunk, no packing or
    #: splitting, so chunk ids map 1:1 onto source paragraphs (the hit rule).
    identity: bool = False


@dataclass(frozen=True)
class ExtractorConfig:
    ner: bool = False
    spacy_model: str = "en_core_web_sm"


@dataclass(frozen=True)
class RetrievalConfig:
    hops: int = 2
    k: int = 8
    #: Number of BM25-ranked chunks admitted as seeds alongside mention hits.
    seed_bm25_top_n: int = 20
    #: Per source chunk, how many co-occurring entities are followed (by IDF).
    beam_entities: int = 8
    #: Per hop, how many reached chunks are kept per followed entity.
    beam_chunks: int = 16
    #: Per hop, at most this many newly reached chunks (ranked by path
    #: strength) survive into the pool and the next frontier — bounds the
    #: candidate pool to "a few hundred" total.
    frontier_chunks: int = 64
    #: Entities mentioned in more than this fraction of chunks are never
    #: followed. Tuned empirically; see the hub ablation in the eval report.
    hub_df_ratio: float = 0.001
    #: Absolute floor for the hub cutoff: on small corpora the ratio alone
    #: computes a cap of ~1 chunk and filters every bridge entity.
    hub_df_floor: int = 10
    #: Ablation switch for the expansion-time specificity filter.
    specificity_filter: bool = True
    #: Selection policy: "interleave" (measured best, default) or
    #: "submodular" (experimental; failed first gold-aware validation —
    #: ADR 0008).
    selection: str = "interleave"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def hub_cap(self, n_chunks: int) -> float:
        """Document-frequency cutoff above which entities are never followed."""
        return max(float(self.hub_df_floor), self.hub_df_ratio * n_chunks)


def data_dir() -> Path:
    """Local data root for corpora and downloaded datasets."""
    return Path(os.environ.get("HOPTRACE_DATA_DIR", ".hoptrace"))


def corpus_path(corpus_id: str) -> Path:
    if not _CORPUS_ID_RE.match(corpus_id):
        raise ValueError(f"invalid corpus id {corpus_id!r}: use letters, digits, '.', '_' or '-'")
    return data_dir() / f"{corpus_id}.sqlite"


def list_corpora() -> list[str]:
    root = data_dir()
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.sqlite"))
