"""Configuration dataclasses and data-directory resolution.

Retrieval defaults are the values recorded in eval reports.
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
    #: Eval mode: one document becomes one chunk, so chunk ids map 1:1 onto paragraphs.
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
    #: Per hop, newly reached chunks (by path strength) kept for the pool and next frontier.
    frontier_chunks: int = 64
    #: Entities mentioned in more than this fraction of chunks are not followed
    #: (tuned; see the hub ablation in the eval report).
    hub_df_ratio: float = 0.001
    #: Floor for the hub cutoff; on small corpora the ratio alone filters every bridge entity.
    hub_df_floor: int = 10
    #: Ablation switch for the expansion-time specificity filter.
    specificity_filter: bool = True
    #: "interleave" (default), "submodular" (experimental; ADR 0008) or
    #: "rerank" (cross-encoder over the pool; ADR 0010).
    selection: str = "interleave"
    #: rerank: pool candidates rescored per query, in interleave order.
    rerank_top_n: int = 50
    #: rerank: pairs per ONNX inference batch.
    rerank_batch: int = 16
    #: rerank: ONNX intra-op threads; fixed so scores do not vary with core count.
    rerank_threads: int = 4
    #: rerank: model directory or registry name; None resolves
    #: $HOPTRACE_RERANK_MODEL, then the bundled model.
    rerank_model: str | None = None
    #: rerank: "fp32" or "int8" (1.6-1.9x faster at <=0.008 metric cost); None
    #: takes the artifact's default.
    rerank_precision: str | None = None
    #: rerank ablation: False drops the route-context prefix so the model scores bare text.
    rerank_path_context: bool = True
    #: Product BM25 parameters. The eval harness and training pipeline use
    #: k1=0.9, b=0.4 (published baseline); the skew is noted in DESIGN.md.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def hub_cap(self, n_chunks: int) -> float:
        """Document-frequency cutoff above which entities are never followed."""
        return max(float(self.hub_df_floor), self.hub_df_ratio * n_chunks)


def data_dir() -> Path:
    """Local data root for corpora and downloaded datasets."""
    return Path(os.environ.get("HOPTRACE_DATA_DIR", ".hoptrace"))


def models_dir() -> Path:
    """Local root for downloaded rerank model artifacts."""
    return data_dir() / "models"


def corpus_path(corpus_id: str) -> Path:
    if not _CORPUS_ID_RE.match(corpus_id):
        raise ValueError(f"invalid corpus id {corpus_id!r}: use letters, digits, '.', '_' or '-'")
    return data_dir() / f"{corpus_id}.sqlite"


def list_corpora() -> list[str]:
    root = data_dir()
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.sqlite"))
