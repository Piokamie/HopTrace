"""Measure training-set exposure of the HippoRAG eval corpus: shared
questions, eval-corpus passages the model scored, and eval gold passages
it scored (and scored as label=1). Matches on the builder's
``cand_key``/``parent_key``.

    uv run --extra eval python training/audit_exposure.py [--train-dir DIR] [--eval NAME]

Writes ``$HOPPATH_DATA_DIR/eval/<eval>-exposure.json`` (the per-question
buckets ``hoppath eval --strata-file`` consumes) and a text report beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hoppath.config import data_dir
from hoppath.eval.adapters import load_corpus_entries, load_hotpot_format, load_musique_json
from hoppath.eval.corpus_build import flat_text
from hoppath.eval.datasets import ensure_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import key_of, rows_of


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=data_dir() / "training")
    parser.add_argument(
        "--eval", choices=("hipporag-musique", "hipporag-2wiki"), default="hipporag-musique"
    )
    args = parser.parse_args(argv)

    manifest = json.loads((args.train_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    sources = [(s["dataset"], args.train_dir / s["output"]) for s in manifest["sources"]]

    seen: set[str] = set()
    seen_positive: set[str] = set()
    seen_by_source: dict[str, set[str]] = {name: set() for name, _ in sources}
    train_qids: set[str] = set()
    rows = 0
    for name, path in sources:
        for r in rows_of([path]):
            rows += 1
            train_qids.add(r["qid"])
            keys = [r["cand_key"]] + ([r["parent_key"]] if r["parent_key"] else [])
            seen.update(keys)
            seen_by_source[name].update(keys)
            if r["label"] == 1:
                seen_positive.add(r["cand_key"])
    print(
        f"training rows {rows:,} from {len(train_qids):,} train questions ({len(sources)} sources)"
    )
    print(
        f"distinct passages the model saw     : {len(seen):,} (candidate or route-context parent)"
    )
    print(f"  ...of those, scored as POSITIVE   : {len(seen_positive):,}")

    corpus_keys = {
        key_of(flat_text(title, text))
        for title, text in load_corpus_entries(ensure_dataset(f"{args.eval}-corpus"))
    }
    print(f"\n{args.eval} corpus passages : {len(corpus_keys):,}")
    print(
        f"  also seen during training         : {len(corpus_keys & seen):,}"
        f" ({100 * len(corpus_keys & seen) / len(corpus_keys):.1f}%)"
    )
    for name, keys in seen_by_source.items():
        print(f"    via {name:<10}: {len(corpus_keys & keys):,}")
    print(
        f"  scored as POSITIVE in training    : {len(corpus_keys & seen_positive):,}"
        f" ({100 * len(corpus_keys & seen_positive) / len(corpus_keys):.1f}%)"
    )

    questions_path = ensure_dataset(args.eval)
    questions = (
        load_musique_json(questions_path)
        if "musique" in args.eval
        else load_hotpot_format(questions_path)
    )
    eval_qids = {q.qid for q in questions}
    print(f"\neval questions                      : {len(questions):,}")
    print(f"  also a TRAINING question          : {len(eval_qids & train_qids):,}")

    exposure: dict[str, list[str]] = {"none": [], "seen": [], "seen_positive": []}
    gold_total = gold_seen = gold_pos = 0
    for q in questions:
        golds = [key_of(flat_text(p.title, p.text)) for p in q.paragraphs if p.is_gold]
        gold_total += len(golds)
        n_seen = sum(g in seen for g in golds)
        n_pos = sum(g in seen_positive for g in golds)
        gold_seen += n_seen
        gold_pos += n_pos
        bucket = "seen_positive" if n_pos else ("seen" if n_seen else "none")
        exposure[bucket].append(q.qid)

    print(f"\ngold passages of eval questions      : {gold_total:,}")
    print(
        f"  seen during training              : {gold_seen:,} ({100 * gold_seen / gold_total:.1f}%)"
    )
    print(
        f"  scored as POSITIVE in training    : {gold_pos:,} ({100 * gold_pos / gold_total:.1f}%)"
    )
    print("\nper-question exposure buckets:")
    for name, qids in exposure.items():
        print(f"  {name:<14} {len(qids):>5} questions")

    out_dir = data_dir() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.eval}-exposure.json"
    out.write_text(json.dumps({k: sorted(v) for k, v in exposure.items()}, indent=2))
    report = out.with_name(f"{args.eval}-exposure-report.txt")
    report.write_text(
        f"training set: {args.train_dir} (sources: {', '.join(n for n, _ in sources)})\n"
        f"training rows {rows} from {len(train_qids)} train questions\n"
        f"distinct passages seen in training: {len(seen)} (positives: {len(seen_positive)})\n"
        f"eval corpus passages: {len(corpus_keys)};"
        f" also seen in training: {len(corpus_keys & seen)}"
        f" ({100 * len(corpus_keys & seen) / len(corpus_keys):.1f}%)"
        + "".join(f"; via {n}: {len(corpus_keys & k)}" for n, k in seen_by_source.items())
        + "\n"
        f"eval questions: {len(questions)};"
        f" also training questions: {len(eval_qids & train_qids)}\n"
        f"eval gold passages: {gold_total};"
        f" seen in training: {gold_seen} ({100 * gold_seen / gold_total:.1f}%);"
        f" scored as POSITIVE: {gold_pos} ({100 * gold_pos / gold_total:.1f}%)\n"
        + "".join(f"bucket {k}: {len(v)} questions\n" for k, v in exposure.items())
    )
    print(f"\nwrote {out} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
