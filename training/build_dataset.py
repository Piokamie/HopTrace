"""Build the reranker training set from the retriever's own pools.

Each train-split question contributes its ``Retriever.candidate_window``
(the window rerank inference scores), serialized with ``serialize_pair``
and labelled by pool-gold membership; queries with no positive in the
window are dropped. 2WikiMultihopQA is the transfer holdout (ADR 0011)
and is refused. Evaluation-gold passages are excluded by default, as
candidate and as route-context parent. ``dataset_manifest.json`` records
sources, checksums, retrieval settings and drop counts.

Run (needs the eval extra):

    uv run --extra eval python training/build_dataset.py --dataset all

Exclusion ablation pair from one retrieval pass:

    uv run --extra eval python training/build_dataset.py --keep-eval-gold --out DIR_FULL
    uv run --extra eval python training/build_dataset.py --from DIR_FULL --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from hoppath.config import RetrievalConfig, data_dir
from hoppath.eval.adapters import load_beir_qrels, load_beir_queries, load_musique
from hoppath.eval.corpus_build import build_pooled_index
from hoppath.eval.datasets import EVAL_B, EVAL_K1, ensure_dataset
from hoppath.eval.harness import QueryGold, beir_query_gold, pooled_query_gold
from hoppath.rerank import serialize_pair
from hoppath.retrieve import Retriever
from hoppath.store import Store

#: Rescoring budget at inference (``RetrievalConfig.rerank_top_n``).
TOP_N = 50
HOPS = 2
#: Inference's k sets the BM25 seed depth (``max(seed_bm25_top_n, k)``); training must match.
K_INFER = 20
HOLDOUT = ("2wiki",)

RETRIEVAL_CFG = RetrievalConfig(
    hops=HOPS, k=K_INFER, rerank_top_n=TOP_N, bm25_k1=EVAL_K1, bm25_b=EVAL_B
)

#: Gold passages of these sets are excluded from training; paragraphs recur
#: across questions, so a split-level wall is not enough.
EVAL_GOLD_SOURCES = ("hipporag-musique", "hipporag-2wiki")

#: ``cand_key``/``parent_key`` let exclusion and audits match on structure;
#: ``text_b`` cannot be re-split (prose can contain the delimiters).
ROW_FIELDS = ("qid", "dataset", "text_a", "text_b", "label", "hop", "cand_key", "parent_key")


def norm_text(text: str) -> str:
    return " ".join(text.split())


def key_of(text: str) -> str:
    """Whitespace-normalized passage identity; corpora join sentences differently."""
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()[:24]


def eval_gold_keys() -> set[str]:
    """Keys of every gold passage in every evaluation set."""
    from hoppath.eval.adapters import load_hotpot_format, load_musique_json
    from hoppath.eval.corpus_build import flat_text

    gold: set[str] = set()
    for name in EVAL_GOLD_SOURCES:
        path = ensure_dataset(name)
        questions = load_musique_json(path) if "musique" in name else load_hotpot_format(path)
        for question in questions:
            for paragraph in question.paragraphs:
                if paragraph.is_gold:
                    gold.add(key_of(flat_text(paragraph.title, paragraph.text)))
    return gold


def check_not_holdout(dataset: str) -> None:
    if any(h in dataset.lower() for h in HOLDOUT):
        raise ValueError(
            f"{dataset!r} is the transfer holdout (ADR 0011): it never enters training"
        )


def split_of(source_file: Path) -> str:
    """Split from the source filename; refuses anything but train."""
    name = source_file.name.lower()
    if "train" not in name:
        raise ValueError(f"{source_file} does not look like a train split; refusing")
    return "train"


def question_rows(retriever: Retriever, qg: QueryGold, dataset: str) -> list[dict[str, Any]]:
    """Training rows for one question, in inference order and serialization."""
    window, texts = retriever.candidate_window(qg.text, hops=HOPS, top_n=TOP_N)
    rows: list[dict[str, Any]] = []
    for candidate in window:
        text_a, text_b = serialize_pair(qg.text, candidate, texts)
        parent = texts[candidate.path.edges[-2].chunk_id] if candidate.hop >= 1 else None
        rows.append(
            {
                "qid": qg.qid,
                "dataset": dataset,
                "text_a": text_a,
                "text_b": text_b,
                "label": 1 if candidate.chunk_id in qg.gold else 0,
                "hop": candidate.hop,
                "cand_key": key_of(texts[candidate.chunk_id]),
                "parent_key": key_of(parent) if parent is not None else None,
            }
        )
    return rows


def is_eval_gold(row: dict[str, Any], gold_keys: set[str]) -> bool:
    """True if the candidate or its route-context parent is an evaluation gold passage."""
    if row["cand_key"] in gold_keys:
        return True
    return row["parent_key"] is not None and row["parent_key"] in gold_keys


class _SplitWriter:
    """Applies exclusion, drops positive-free queries, writes rows, and
    keeps the counts the manifest reports."""

    def __init__(self, out_path: Path, gold_keys: set[str] | None) -> None:
        self._out = out_path.open("w", encoding="utf-8")
        self._gold = gold_keys
        self.kept = self.dropped = self.positives = self.rows = self.excluded_rows = 0
        self.gold_present: set[str] = set()

    def add_query(self, rows: list[dict[str, Any]]) -> None:
        if self._gold is not None:
            for r in rows:
                for key in (r["cand_key"], r["parent_key"]):
                    if key in self._gold:
                        self.gold_present.add(key)
            before = len(rows)
            rows = [r for r in rows if not is_eval_gold(r, self._gold)]
            self.excluded_rows += before - len(rows)
        n_pos = sum(r["label"] for r in rows)
        if n_pos == 0:
            self.dropped += 1
            return
        self.kept += 1
        self.positives += n_pos
        self.rows += len(rows)
        for row in rows:
            self._out.write(json.dumps({k: row[k] for k in ROW_FIELDS}, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._out.close()

    def stats(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "dropped_no_positive": self.dropped,
            "rows": self.rows,
            "positives": self.positives,
            "excluded_eval_gold_rows": self.excluded_rows,
            # 0 for a source known to overlap means the text matching missed
            "eval_gold_passages_present": len(self.gold_present)
            if self._gold is not None
            else None,
        }


def build_split(
    dataset: str,
    store: Store,
    query_gold: list[QueryGold],
    out_path: Path,
    source_file: Path,
    limit: int | None,
    gold_keys: set[str] | None = None,
) -> dict[str, Any]:
    check_not_holdout(dataset)
    split = split_of(source_file)
    retriever = Retriever(store, RETRIEVAL_CFG)
    writer = _SplitWriter(out_path, gold_keys)
    started = time.monotonic()
    try:
        for i, qg in enumerate(query_gold, 1):
            writer.add_query(question_rows(retriever, qg, dataset))
            if i % 500 == 0:
                rate = i / (time.monotonic() - started)
                log(f"{dataset}: {i}/{len(query_gold)} queries ({rate:.1f}/s)")
    finally:
        writer.close()
    return {
        "dataset": dataset,
        "split": split,
        "source": str(source_file),
        "source_sha256": _sha256(source_file),
        "questions": len(query_gold),
        "limit": limit,
        **writer.stats(),
        "output": out_path.name,
    }


def build_musique(out_dir: Path, limit: int | None, gold_keys: set[str] | None) -> dict[str, Any]:
    source = ensure_dataset("musique-train")
    questions = load_musique(source)
    if limit is not None:
        questions = questions[:limit]
    pooled = build_pooled_index(
        questions, target=out_dir / "musique-train-pool.sqlite", progress=log
    )
    try:
        query_gold = pooled_query_gold(questions, pooled.gold_chunks)
        return build_split(
            "musique",
            pooled.store,
            query_gold,
            out_dir / "musique-train.jsonl",
            source,
            limit,
            gold_keys,
        )
    finally:
        pooled.store.close()


def build_hotpotqa(out_dir: Path, limit: int | None, gold_keys: set[str] | None) -> dict[str, Any]:
    dataset_dir = ensure_dataset("beir-hotpotqa")
    index_path = data_dir() / "eval" / "beir-hotpotqa.sqlite"
    if not index_path.is_file():
        raise SystemExit(
            f"no BEIR index at {index_path}: build it first via"
            " `hoppath eval --dataset beir-hotpotqa` (hours-scale)"
        )
    qrels_path = dataset_dir / "qrels" / "train.tsv"
    queries = load_beir_queries(dataset_dir / "queries.jsonl")
    qrels = load_beir_qrels(qrels_path)
    store = Store.open(index_path)
    try:
        query_gold = beir_query_gold(store, queries, qrels)
        if limit is not None:
            query_gold = query_gold[:limit]
        return build_split(
            "hotpotqa",
            store,
            query_gold,
            out_dir / "hotpotqa-train.jsonl",
            qrels_path,
            limit,
            gold_keys,
        )
    finally:
        store.close()


def _queries(path: Path) -> Iterator[list[dict[str, Any]]]:
    """Rows grouped by consecutive qid (the builder writes them contiguously)."""
    group: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if group and row["qid"] != group[0]["qid"]:
                yield group
                group = []
            group.append(row)
    if group:
        yield group


def derive_filtered(from_dir: Path, out_dir: Path, gold_keys: set[str]) -> dict[str, Any]:
    """Exclusion-filtered set from an unfiltered build; row-wise on the keys, no retrieval."""
    manifest = json.loads((from_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("exclusion_ablation") is not None:
        raise ValueError(f"{from_dir} is already exclusion-filtered; --from needs a full build")
    sources: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        src_path = from_dir / source["output"]
        writer = _SplitWriter(out_dir / source["output"], gold_keys)
        try:
            for i, rows in enumerate(_queries(src_path), 1):
                writer.add_query(rows)
                if i % 5000 == 0:
                    log(f"{source['dataset']}: {i} queries filtered")
        finally:
            writer.close()
        derived = {k: v for k, v in source.items() if k not in writer.stats()}
        derived.update(writer.stats())
        sources.append(derived)
    for name in ("musique-train-pool.sqlite",):
        if (from_dir / name).is_file() and not (out_dir / name).exists():
            shutil.copyfile(from_dir / name, out_dir / name)
    return {**manifest, "sources": sources, "derived_from": str(from_dir)}


def write_manifest(
    out_dir: Path,
    sources: list[dict[str, Any]],
    gold_keys: set[str] | None,
    base: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "builder": "training/build_dataset.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "retrieval": {
            "hops": HOPS,
            "k": K_INFER,
            "seed_depth": max(RETRIEVAL_CFG.seed_bm25_top_n, K_INFER),
            "top_n": TOP_N,
            "order": "interleave",
            "bm25_k1": EVAL_K1,
            "bm25_b": EVAL_B,
            "window": "Retriever.candidate_window (ADR 0012)",
        },
        "serialize_format": "via {entity} | {parent_snippet} || {text}",
        "holdout": list(HOLDOUT),
        "sources": sources,
        "exclusion_ablation": None
        if gold_keys is None
        else {
            "reason": "evaluation gold passages removed from training (as candidate and"
            " as route-context parent, whitespace-normalized match), so training"
            " exposure cannot contribute to reported gains",
            "eval_gold_passages_excluded": len(gold_keys),
            "sources": list(EVAL_GOLD_SOURCES),
        },
    }
    if base is not None:
        manifest["retrieval"] = base["retrieval"]
        manifest["derived_from"] = base["derived_from"]
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("musique", "hotpotqa", "all"), default="all")
    parser.add_argument(
        "--limit-hotpot",
        type=int,
        default=30_000,
        help="cap HotpotQA train queries (85k total; the cap is recorded in the manifest)",
    )
    parser.add_argument("--limit-musique", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: $HOPPATH_DATA_DIR/training"
    )
    parser.add_argument(
        "--keep-eval-gold",
        action="store_true",
        help="do NOT exclude evaluation gold passages from training. Only for"
        " measuring what the exclusion costs — the shipped artifact excludes them,"
        " and numbers from a kept-gold model are not publishable",
    )
    parser.add_argument(
        "--from",
        dest="from_dir",
        type=Path,
        default=None,
        help="derive the exclusion-filtered set from an unfiltered build (no retrieval)",
    )
    args = parser.parse_args(argv)
    if args.from_dir is not None and args.keep_eval_gold:
        parser.error("--from derives the filtered set; --keep-eval-gold makes no sense with it")

    out_dir = args.out or (data_dir() / "training")
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_keys: set[str] | None = None
    if not args.keep_eval_gold:
        gold_keys = eval_gold_keys()
        log(f"excluding {len(gold_keys):,} evaluation gold passages from training")

    if args.from_dir is not None:
        assert gold_keys is not None
        derived = derive_filtered(args.from_dir, out_dir, gold_keys)
        sources = derived["sources"]
        manifest_path = write_manifest(out_dir, sources, gold_keys, base=derived)
    else:
        sources = []
        if args.dataset in ("musique", "all"):
            sources.append(build_musique(out_dir, args.limit_musique, gold_keys))
        if args.dataset in ("hotpotqa", "all"):
            sources.append(build_hotpotqa(out_dir, args.limit_hotpot, gold_keys))
        manifest_path = write_manifest(out_dir, sources, gold_keys)

    log(f"manifest: {manifest_path}")
    for source in sources:
        log(
            f"{source['dataset']}: {source['rows']} rows from {source['kept']} queries"
            f" ({source['positives']} positives, {source['dropped_no_positive']} dropped"
            f" gold-free, {source['excluded_eval_gold_rows']} eval-gold rows excluded,"
            f" {source['eval_gold_passages_present']} distinct eval-gold passages present)"
        )
    return 0


def log(message: str) -> None:
    print(f"[build] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_of(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Every row across the given jsonl files (audit helper)."""
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
