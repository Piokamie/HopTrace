"""Fine-tune the path-aware cross-encoder and export it for HopTrace.

Reads the jsonl rows ``build_dataset.py`` produced, fine-tunes
``cross-encoder/ms-marco-MiniLM-L6-v2`` with pointwise BCE, exports ONNX
(fp32 + dynamic-int8), and derives the model ``manifest.json`` from the
dataset manifest; the eval harness refuses to score any split it lists.

Sampling: the pool is ~3% positive and hop >= 1 positives are ~17x rarer
than seed positives, so every positive is kept and negatives are
subsampled per query with a seed/hop quota. A held-back slice of train
queries drives early stopping.

    uv run --group train --extra eval python training/train_reranker.py

Torch is a dev-only dependency group; the artifact runs on the ``rerank``
extra alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_LENGTH = 256  # == hoptrace.rerank.MAX_LENGTH; the exported graph is length-agnostic
HOLDOUT = ("2wiki",)


@dataclass
class Row:
    qid: str
    dataset: str
    text_a: str
    text_b: str
    label: int
    hop: int


def load_rows(paths: list[Path]) -> dict[str, list[Row]]:
    by_query: dict[str, list[Row]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                by_query[f"{r['dataset']}:{r['qid']}"].append(
                    Row(r["qid"], r["dataset"], r["text_a"], r["text_b"], r["label"], r["hop"])
                )
    return by_query


def sample_query(rows: list[Row], rng: random.Random, neg_seed: int, neg_hop: int) -> list[Row]:
    """All positives + a bounded, hop-stratified negative sample."""
    positives = [r for r in rows if r.label == 1]
    seed_neg = [r for r in rows if r.label == 0 and r.hop == 0]
    hop_neg = [r for r in rows if r.label == 0 and r.hop >= 1]
    rng.shuffle(seed_neg)
    rng.shuffle(hop_neg)
    return positives + seed_neg[:neg_seed] + hop_neg[:neg_hop]


def split_queries(
    keys: list[str], holdback: float, rng: random.Random
) -> tuple[list[str], list[str]]:
    keys = sorted(keys)
    rng.shuffle(keys)
    n_val = int(len(keys) * holdback)
    return keys[n_val:], keys[:n_val]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="model output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--neg-seed", type=int, default=6, help="seed negatives per query")
    parser.add_argument("--neg-hop", type=int, default=6, help="hop negatives per query")
    parser.add_argument("--holdback", type=float, default=0.03, help="train-slice validation")
    parser.add_argument("--max-queries", type=int, default=None, help="subsample queries (dev)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2000, help="steps between validations")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from hoptrace.config import data_dir

    train_dir = args.data_dir or (data_dir() / "training")
    out_dir = args.out or (data_dir() / "models" / "hoptrace-rerank-minilm-l6")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((train_dir / "dataset_manifest.json").read_text())
    for source in dataset_manifest["sources"]:
        if any(h in source["dataset"] for h in HOLDOUT):
            raise SystemExit(f"refusing to train: {source['dataset']} is the holdout (ADR 0011)")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"device={device} base={BASE_MODEL}")

    paths = [train_dir / s["output"] for s in dataset_manifest["sources"]]
    by_query = load_rows(paths)
    keys = list(by_query)
    if args.max_queries is not None:
        keys = sorted(keys)
        rng.shuffle(keys)
        keys = keys[: args.max_queries]
    train_keys, val_keys = split_queries(keys, args.holdback, rng)
    log(f"queries: {len(train_keys)} train / {len(val_keys)} held-back (train-split slice)")

    def build_examples(qkeys: list[str], sample: bool) -> list[Row]:
        out: list[Row] = []
        for key in qkeys:
            rows = by_query[key]
            out.extend(sample_query(rows, rng, args.neg_seed, args.neg_hop) if sample else rows)
        return out

    train_rows = build_examples(train_keys, sample=True)
    n_pos = sum(r.label for r in train_rows)
    hop_pos = sum(1 for r in train_rows if r.label == 1 and r.hop >= 1)
    log(f"train rows: {len(train_rows)} ({n_pos} pos, {hop_pos} hop-pos)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * ((len(train_rows) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / max(total_steps, 1))
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def encode(rows: list[Row]) -> dict[str, torch.Tensor]:
        enc = tokenizer(
            [r.text_a for r in rows],
            [r.text_b for r in rows],
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def validate() -> dict[str, float]:
        """Per-query top1/MRR on the held-back slice, plus top1 on hop-positive queries."""
        model.eval()
        top1 = mrr = 0.0
        hop_top1 = hop_n = 0
        for key in val_keys:
            rows = by_query[key]
            scores: list[float] = []
            for start in range(0, len(rows), 64):
                batch = rows[start : start + 64]
                logits = model(**encode(batch)).logits.reshape(-1)
                scores.extend(logits.float().cpu().tolist())
            order = sorted(range(len(rows)), key=lambda i: -scores[i])
            first_gold = next((rank for rank, i in enumerate(order) if rows[i].label == 1), None)
            if first_gold is None:
                continue
            top1 += first_gold == 0
            mrr += 1.0 / (first_gold + 1)
            if any(r.label == 1 and r.hop >= 1 for r in rows):
                hop_n += 1
                hop_top1 += rows[order[0]].label == 1 and rows[order[0]].hop >= 1
        model.train()
        n = max(len(val_keys), 1)
        return {
            "top1": top1 / n,
            "mrr": mrr / n,
            "hop_gold_top1": hop_top1 / max(hop_n, 1),
            "hop_queries": hop_n,
        }

    log(f"zero-shot held-back: {validate()}")
    history: list[dict[str, Any]] = []
    best_mrr = -1.0
    step = 0
    started = time.monotonic()
    model.train()
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        running = 0.0
        for start in range(0, len(train_rows), args.batch_size):
            batch = train_rows[start : start + args.batch_size]
            labels = torch.tensor([float(r.label) for r in batch], device=device)
            logits = model(**encode(batch)).logits.reshape(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                elapsed = time.monotonic() - started
                log(
                    f"epoch {epoch} step {step}/{total_steps} loss {running / 100:.4f}"
                    f" ({step / elapsed:.1f} steps/s)"
                )
                running = 0.0
            if step % args.eval_every == 0 or step == total_steps:
                metrics = validate()
                metrics.update(step=step, epoch=epoch)
                history.append(metrics)
                log(f"validate: {metrics}")
                if metrics["mrr"] > best_mrr:
                    best_mrr = metrics["mrr"]
                    model.save_pretrained(out_dir / "hf")
                    tokenizer.save_pretrained(out_dir / "hf")
                    log(f"saved best (mrr {best_mrr:.4f}) -> {out_dir / 'hf'}")

    export_onnx(out_dir)
    write_manifest(
        out_dir, dataset_manifest, args, history, best_mrr, len(train_keys), len(val_keys)
    )
    log(f"done: {out_dir}")
    return 0


def export_onnx(out_dir: Path) -> None:
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_dir = out_dir / "hf"
    model = AutoModelForSequenceClassification.from_pretrained(hf_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    enc = tokenizer(["q"], ["d"], return_tensors="pt", padding="max_length", max_length=16)
    inputs = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    dyn = {0: "batch", 1: "seq"}
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": dyn,
            "attention_mask": dyn,
            "token_type_ids": dyn,
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    quantize_dynamic(str(onnx_path), str(out_dir / "model_int8.onnx"), weight_type=QuantType.QInt8)
    # the runtime loads a bare tokenizer.json (tokenizers lib), not the HF dir
    (out_dir / "tokenizer.json").write_bytes((hf_dir / "tokenizer.json").read_bytes())
    log(f"exported {onnx_path} + int8; fp32 {onnx_path.stat().st_size / 1e6:.0f} MB")


def write_manifest(
    out_dir: Path,
    dataset_manifest: dict[str, Any],
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_mrr: float,
    n_train: int,
    n_val: int,
) -> None:
    trained_on = [
        {
            "dataset": s["dataset"],
            "split": s["split"],
            "source_sha256": s["source_sha256"],
            "questions": s["kept"],
            "limit": s["limit"],
        }
        for s in dataset_manifest["sources"]
    ]
    commit = trainer_commit()
    if commit.endswith("-dirty"):
        log(
            "WARNING: trainer tree is dirty — this artifact is not reproducible from any"
            " commit; the eval harness will refuse it without --allow-dirty-manifest"
        )
    manifest = {
        "model": out_dir.name,
        "base_model": BASE_MODEL,
        "trained_on": trained_on,
        "holdout": list(HOLDOUT),
        "serialize_format": dataset_manifest["serialize_format"],
        "retrieval": dataset_manifest["retrieval"],
        "exclusion": dataset_manifest.get("exclusion_ablation"),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "neg_seed": args.neg_seed,
            "neg_hop": args.neg_hop,
            "holdback_fraction": args.holdback,
            "train_queries": n_train,
            "heldback_queries": n_val,
            "seed": args.seed,
            "max_queries": args.max_queries,
            "best_heldback_mrr": best_mrr,
            "history": history,
        },
        "files": {
            name: _sha256(out_dir / name)
            for name in ("model.onnx", "model_int8.onnx", "tokenizer.json")
            if (out_dir / name).is_file()
        },
        "trainer_commit": commit,
        "dirty": commit.endswith("-dirty"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def trainer_commit() -> str:
    """``<sha>`` or ``<sha>-dirty``."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=40", "--match=NOTHING"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log(message: str) -> None:
    print(f"[train] {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
