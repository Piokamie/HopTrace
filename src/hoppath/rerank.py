"""Learned rescoring of the deterministic candidate pool (ADR 0012).

A hop candidate's text is prefixed with the bridge entity of its last edge
and a snippet of its parent chunk, so the cross-encoder scores route
quality, not just text relevance. Only the top-N pool candidates in
interleave order are rescored. ONNX inference needs the ``rerank`` extra;
the scorer is injectable. Models come from a registry with pinned checksums.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from hoppath.config import models_dir
from hoppath.score import Selected

if TYPE_CHECKING:
    from hoppath.config import RetrievalConfig
    from hoppath.expand import Candidate

#: Parent-chunk snippet length in the route context (chars).
PARENT_SNIPPET_CHARS = 160

#: Tokenizer truncation length; ``only_second`` cuts text_b, not the query.
MAX_LENGTH = 256

#: Query budget within MAX_LENGTH; ``only_second`` raises if text_a alone overflows.
MAX_QUERY_TOKENS = 128

PRECISIONS = ("fp32", "int8")

#: Download timeout (s); a stalled connection would otherwise hang the MCP call.
DOWNLOAD_TIMEOUT = 60


class PairScorer(Protocol):
    """Scores (text_a, text_b) pairs; higher means more relevant."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def serialize_pair(
    query: str, candidate: Candidate, texts: dict[int, str], path_context: bool = True
) -> tuple[str, str]:
    """Cross-encoder input pair; shared by inference and the training builder.
    ``path_context=False`` drops the route prefix (eval ablation)."""
    text_b = texts[candidate.chunk_id]
    if path_context and candidate.hop >= 1:
        last = candidate.path.edges[-1]
        # KeyError here is an admission bug: earliest-ring-wins admits parents before children.
        parent = texts[candidate.path.edges[-2].chunk_id]
        text_b = f"via {last.entity} | {_snippet(parent)} || {text_b}"
    return query, text_b


def rerank_select(
    pool_order: Sequence[Candidate],
    query: str,
    texts: dict[int, str],
    scorer: PairScorer,
    k: int,
    top_n: int,
    path_context: bool = True,
) -> list[Selected]:
    """Rescore the first ``top_n`` pool candidates; top-k by learned score,
    ties on chunk id. ``gain`` carries the learned score. At least ``k`` are
    rescored so a small ``top_n`` cannot shorten the result."""
    ordered = list(pool_order[: window_size(top_n, k)])
    scores = scorer.score([serialize_pair(query, c, texts, path_context) for c in ordered])
    ranked = sorted(zip(ordered, scores, strict=True), key=lambda t: (-t[1], t[0].chunk_id))
    return [Selected(candidate=c, novelty=1.0, gain=s) for c, s in ranked[:k]]


def window_size(top_n: int, k: int) -> int:
    """Pool candidates the ranker scores; shared with the eval oracle and training builder."""
    return max(top_n, k)


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    #: pinned; verified after download and on every load of a registry model
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    files: tuple[ModelFile, ...]
    #: the untrained base — the only model exempt from the ADR 0011 wall
    zero_shot: bool = False
    #: graph loaded when the caller does not choose one
    default_precision: str = "fp32"

    def file(self, name: str) -> ModelFile:
        for f in self.files:
            if f.name == name:
                return f
        raise KeyError(name)


_HUB = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main"
_RELEASE = "https://github.com/Piokamie/HopPath/releases/download/v0.1.0"

#: Unpinned entries are refused, never downloaded.
_UNPINNED = "unpinned"

MODELS: dict[str, ModelSpec] = {
    "hoppath-rerank-minilm-l6": ModelSpec(
        name="hoppath-rerank-minilm-l6",
        default_precision="int8",
        files=(
            ModelFile(
                "model_int8.onnx",
                f"{_RELEASE}/model_int8.onnx",
                "3dba0f4e51c710f6f1a8ef556b212225a5f34c3568a066b26d0a30d0e2b79742",
            ),
            ModelFile(
                "model.onnx",
                f"{_RELEASE}/model.onnx",
                "404abde448e18ee9f7840a627ccaccd499ce355b8c23abb53cef0ba5c7e415ca",
            ),
            ModelFile(
                "tokenizer.json",
                f"{_RELEASE}/tokenizer.json",
                "769987267933b9a47f55b5415f934fd3fe55315fbf2c6b84f1ba097fef28edb4",
            ),
            ModelFile(
                "manifest.json",
                f"{_RELEASE}/manifest.json",
                "93201ad18856be3b864246dd16d2dd9af57d30c23ad12636e065056b20616e55",
            ),
        ),
    ),
    "ms-marco-minilm-l6-v2": ModelSpec(
        name="ms-marco-minilm-l6-v2",
        zero_shot=True,
        files=(
            ModelFile(
                "model.onnx",
                f"{_HUB}/onnx/model.onnx",
                "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
            ),
            ModelFile(
                "tokenizer.json",
                f"{_HUB}/tokenizer.json",
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            ),
        ),
    ),
}

DEFAULT_RERANK_MODEL = "hoppath-rerank-minilm-l6"
ZERO_SHOT_MODEL = "ms-marco-minilm-l6-v2"

#: Registry files committed under ``<repo>/models/``; absent in a wheel, where
#: they are downloaded to ``models_dir()`` instead.
BUNDLED_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def graph_file(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"rerank precision must be one of {PRECISIONS}, got {precision!r}")
    return "model_int8.onnx" if precision == "int8" else "model.onnx"


def effective_model_spec(spec: str | None) -> str:
    """Config value after the ``$HOPPATH_RERANK_MODEL`` and default fallbacks;
    the training wall must check this, not the raw field."""
    return spec or os.environ.get("HOPPATH_RERANK_MODEL") or DEFAULT_RERANK_MODEL


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    #: effective graph: the caller's choice or the artifact's default
    precision: str
    #: registry entry, or None for a caller-supplied directory
    spec: ModelSpec | None

    @property
    def graph(self) -> Path:
        return self.path / graph_file(self.precision)

    @property
    def manifest(self) -> Path:
        return self.path / "manifest.json"

    def is_zero_shot(self) -> bool:
        """True only for the registry base model without a manifest; a
        caller-supplied directory is never exempt from the training wall."""
        return self.spec is not None and self.spec.zero_shot and not self.manifest.exists()

    @property
    def label(self) -> str:
        return f"{self.path.name} ({self.precision})"


def resolve_model(spec: str | None, precision: str | None = None) -> ResolvedModel:
    """Locate a model directory holding the requested graph.

    A caller-supplied directory is used as is. A registry name resolves to
    the bundled repository copy when complete, else to ``models_dir()/<name>``
    with missing files copied from the bundle or downloaded, always against
    the pinned sha256. ``spec=None`` falls back to ``$HOPPATH_RERANK_MODEL``
    then the bundled model; ``precision=None`` takes the artifact's default."""
    spec = effective_model_spec(spec)
    if precision is not None:
        graph_file(precision)  # validate early, before any download
    as_path = Path(spec).expanduser()
    if as_path.is_dir():
        return _resolve_directory(as_path, precision)
    if spec not in MODELS:
        raise RuntimeError(
            f"unknown rerank model {spec!r}: pass a local model directory or one"
            f" of {sorted(MODELS)}"
        )
    return _resolve_registry(MODELS[spec], precision)


def _resolve_directory(model_dir: Path, precision: str | None) -> ResolvedModel:
    if not (model_dir / "tokenizer.json").is_file():
        raise RuntimeError(f"rerank model directory {model_dir} lacks tokenizer.json")
    if precision is None:
        precision = "fp32" if (model_dir / "model.onnx").is_file() else "int8"
    resolved = ResolvedModel(model_dir, precision, None)
    _require_graph(resolved)
    return resolved


def _resolve_registry(model: ModelSpec, precision: str | None) -> ResolvedModel:
    precision = precision or model.default_precision
    needed = ["tokenizer.json", graph_file(precision)]
    if any(f.name == "manifest.json" for f in model.files):
        needed.append("manifest.json")
    for name in needed:
        try:
            model.file(name)
        except KeyError:
            raise RuntimeError(
                f"no {name} in registry model {model.name}: this artifact does not"
                f" ship a {precision} graph"
            ) from None

    bundled = BUNDLED_MODELS_DIR / model.name
    if all((bundled / name).is_file() for name in needed):
        for name in needed:
            _verify(bundled / name, model.file(name).sha256, f"{model.name}/{name}")
        return ResolvedModel(bundled, precision, model)

    target = models_dir() / model.name
    target.mkdir(parents=True, exist_ok=True)
    for name in needed:
        f = model.file(name)
        label = f"{model.name}/{name}"
        if (target / name).is_file():
            _verify(target / name, f.sha256, label)
        elif (bundled / name).is_file():
            _verify(bundled / name, f.sha256, label)
            shutil.copyfile(bundled / name, target / name)
        else:
            _download(target / name, f, label)
    return ResolvedModel(target, precision, model)


def _require_graph(resolved: ResolvedModel) -> None:
    if not resolved.graph.is_file():
        raise RuntimeError(
            f"no {resolved.graph.name} in {resolved.path}: this artifact does not"
            f" ship a {resolved.precision} graph"
        )


class OnnxCrossEncoder:
    """Batched CPU inference over (text_a, text_b) pairs."""

    def __init__(
        self,
        model: ResolvedModel | Path,
        batch_size: int = 16,
        max_length: int = MAX_LENGTH,
        threads: int = 4,
        precision: str | None = None,
    ):
        ort, tokenizer_cls = _require_runtime()
        if isinstance(model, Path):
            model = _resolve_directory(model, precision)
        elif precision is not None and precision != model.precision:
            model = ResolvedModel(model.path, precision, model.spec)
        _require_graph(model)
        # keep tokenizers' Rayon pool from using every core
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model.graph), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        tokenizer = tokenizer_cls.from_file(str(model.path / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        tokenizer.enable_padding()
        self._tokenizer = tokenizer
        # only_second truncation refuses single sequences, so the query gets its own tokenizer
        query_tokenizer = tokenizer_cls.from_file(str(model.path / "tokenizer.json"))
        query_tokenizer.enable_truncation(max_length=MAX_QUERY_TOKENS, strategy="longest_first")
        self._query_tokenizer = query_tokenizer
        self._batch_size = batch_size
        self.model = model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import numpy as np  # onnxruntime dependency, present with the extra

        pairs = [(self._clip_query(a), b) for a, b in pairs]
        # length-sorted so batches pad to similar lengths
        order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][1]), i))
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), self._batch_size):
            indices = order[start : start + self._batch_size]
            encodings = self._tokenizer.encode_batch([pairs[i] for i in indices])
            feeds: dict[str, Any] = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            logits = np.asarray(self._session.run(None, feeds)[0]).reshape(-1)
            for i, value in zip(indices, logits, strict=True):
                scores[i] = float(value)
        return scores

    def _clip_query(self, text: str) -> str:
        """Pre-truncate the query; ``only_second`` raises when text_a alone overflows."""
        encoding = self._query_tokenizer.encode(text, add_special_tokens=False)
        if not encoding.overflowing:
            return text
        end = encoding.offsets[-1][1]
        return text[:end]


def default_scorer(cfg: RetrievalConfig) -> PairScorer:
    """Resolve the configured ONNX cross-encoder; requires the ``rerank`` extra."""
    _require_runtime()
    resolved = resolve_model(cfg.rerank_model, cfg.rerank_precision)
    return OnnxCrossEncoder(resolved, batch_size=cfg.rerank_batch, threads=cfg.rerank_threads)


def _require_runtime() -> tuple[Any, Any]:
    try:
        import onnxruntime
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "reranking requires the rerank extra: install with `pip install 'hoppath[rerank]'`"
        ) from exc
    return onnxruntime, Tokenizer


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    if expected == _UNPINNED:
        raise RuntimeError(
            f"{label} has no pinned checksum in this build of hoppath; the file at"
            f" {path} cannot be trusted. Pass --rerank-model <directory> to use it explicitly"
        )
    actual = sha256_of(path)
    if actual != expected:
        raise RuntimeError(
            f"checksum mismatch for {label} at {path}: expected {expected}, got {actual}."
            " Delete the file to re-download it, or pass --rerank-model <directory>"
            " to use a local artifact deliberately"
        )


def _download(target: Path, spec: ModelFile, label: str) -> None:
    if spec.sha256 == _UNPINNED:
        raise RuntimeError(
            f"{label} has no pinned checksum in this build of hoppath and will not be"
            f" downloaded from {spec.url}. Fetch it manually and pass --rerank-model"
            f" <directory>"
        )
    print(f"[hoppath] downloading {label} from {spec.url}", file=sys.stderr)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(spec.url, timeout=DOWNLOAD_TIMEOUT) as response,
            tmp.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1 << 20)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"download failed for {label} from {spec.url}: {exc}."
            f" Fetch it manually and place it at {target}"
        ) from exc
    actual = sha256_of(tmp)
    if actual != spec.sha256:
        tmp.rename(target.with_suffix(target.suffix + ".corrupt"))
        raise RuntimeError(
            f"checksum mismatch for {label} downloaded from {spec.url}: expected"
            f" {spec.sha256}, got {actual}. The file was set aside as {target.name}.corrupt"
        )
    tmp.rename(target)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= PARENT_SNIPPET_CHARS:
        return flat
    return flat[: PARENT_SNIPPET_CHARS - 1].rstrip() + "…"
