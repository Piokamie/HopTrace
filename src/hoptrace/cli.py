"""Command-line interface.

Product subcommands: ``ingest``, ``retrieve``, ``explain``, ``bracket``,
``serve`` (the MCP server over stdio). Developer subcommand: ``eval``
(the external benchmark harness behind docs/results.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from hoptrace.config import RetrievalConfig, data_dir
from hoptrace.eval.adapters import (
    EvalQuestion,
    load_beir_qrels,
    load_beir_queries,
    load_corpus_entries,
    load_hotpot_format,
    load_musique,
    load_musique_json,
)
from hoptrace.eval.corpus_build import (
    build_beir_index,
    build_explicit_index,
    build_pooled_index,
)
from hoptrace.eval.datasets import (
    BEIR_HOTPOTQA_BASELINES,
    EVAL_B,
    EVAL_K1,
    GATE_TOLERANCE,
    ensure_dataset,
)
from hoptrace.eval.diagnostics import (
    ANSWER_K,
    DIAGNOSTIC_K,
    DisplacementAudit,
    MissBreakdown,
    PoolPrecision,
    calibrate,
    pool_precision,
    stratify,
)
from hoptrace.eval.harness import (
    FloorReport,
    QueryGold,
    beir_query_gold,
    evaluate_distractor_floor,
    evaluate_floor,
    evaluate_hoptrace,
    floor_outcomes,
    pooled_query_gold,
    write_report,
)
from hoptrace.rerank import PRECISIONS
from hoptrace.retrieve import Retriever
from hoptrace.store import Store, ensure_entity_stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hoptrace", description="HopTrace retrieval engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="ingest text/markdown files into a corpus")
    ing.add_argument("source", type=Path, help="file or directory")
    ing.add_argument("--corpus", required=True, dest="corpus_id")
    ing.add_argument("--target-tokens", type=int, default=None)
    ing.add_argument("--max-tokens", type=int, default=None)
    ing.add_argument("--ner", action="store_true", help="add the optional spaCy NER pass")
    ing.add_argument("--analyzer", default="english", choices=("english", "simple"))

    ret = sub.add_parser("retrieve", help="retrieve evidence with hop paths")
    ret.add_argument("query")
    ret.add_argument("--corpus", required=True, dest="corpus_id")
    ret.add_argument("--hops", type=int, default=2, choices=(0, 1, 2))
    ret.add_argument("-k", type=int, default=8)
    ret.add_argument(
        "--rerank",
        action="store_true",
        help="rescore top candidates with the learned cross-encoder (requires the rerank extra)",
    )
    ret.add_argument(
        "--rerank-model",
        default=None,
        help="local model directory or registry name (default: $HOPTRACE_RERANK_MODEL,"
        " else the bundled fine-tuned model; ms-marco-minilm-l6-v2 is the zero-shot base)",
    )
    ret.add_argument(
        "--rerank-precision",
        choices=PRECISIONS,
        default=None,
        help="graph to load (default: what the artifact ships — int8 for the bundled"
        " model, fp32 for the base)",
    )
    ret.add_argument("--json", action="store_true", dest="as_json")

    ex = sub.add_parser("explain", help="explain why a chunk was (not) retrieved")
    ex.add_argument("chunk_id", type=int)
    ex.add_argument("--corpus", required=True, dest="corpus_id")
    ex.add_argument("--query", default=None)
    ex.add_argument("--json", action="store_true", dest="as_json")

    br = sub.add_parser("bracket", help="measure this corpus (self-benchmark diagnostic)")
    br.add_argument("--corpus", required=True, dest="corpus_id")
    br.add_argument("-n", "--n-questions", type=int, default=100, dest="n_questions")
    br.add_argument("--seed", type=int, default=0)
    br.add_argument("--json", action="store_true", dest="as_json")

    srv = sub.add_parser("serve", help="run the MCP server")
    srv.add_argument(
        "--http",
        action="store_true",
        help="bind a socket (streamable-http) instead of stdio, for clients that"
        " connect to a running process. No authentication — localhost or behind a proxy.",
    )
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)

    ev = sub.add_parser("eval", help="run the external benchmark harness (developer)")
    ev.add_argument(
        "--dataset",
        required=True,
        choices=("beir-hotpotqa", "musique", "2wiki", "hipporag-musique", "hipporag-2wiki"),
        help="hipporag-* use the published HippoRAG protocol: their 1,000-question"
        " validation sample over their exact candidate corpus (the comparable setting)",
    )
    ev.add_argument("--hops", type=int, default=0, choices=(0, 1, 2), help="0 = BM25 floor")
    ev.add_argument(
        "--setting",
        choices=("corpus", "distractor"),
        default="corpus",
        help="corpus-scale (reportable) or per-question distractor (sanity appendix)",
    )
    ev.add_argument(
        "--corpus-pool",
        choices=("dev", "all"),
        default="dev",
        help="musique/2wiki haystack: 'dev' pools dev paragraphs only (development"
        " setting); 'all' pools train+dev paragraphs with dev-only queries (the"
        " IRCoT-comparable protocol). Ignored for beir-hotpotqa (true corpus-scale).",
    )
    ev.add_argument("--limit", type=int, default=None, help="quick partial run (not reportable)")
    ev.add_argument("--k1", type=float, default=EVAL_K1)
    ev.add_argument("--b", type=float, default=EVAL_B)
    ev.add_argument("--analyzer", default="english", choices=("english", "simple"))
    ev.add_argument("--hub-df-ratio", type=float, default=None, help="override hub cutoff")
    ev.add_argument(
        "--selection",
        choices=("interleave", "submodular", "rerank"),
        default=None,
        help="selection policy override (submodular is experimental;"
        " rerank requires the rerank extra)",
    )
    ev.add_argument(
        "--rerank-model",
        default=None,
        help="rerank model: local directory or registry name (default:"
        " $HOPTRACE_RERANK_MODEL, else the bundled fine-tuned model;"
        " ms-marco-minilm-l6-v2 is the zero-shot control). Fine-tuned models must"
        " carry a manifest; the training-data wall is checked",
    )
    ev.add_argument(
        "--rerank-precision",
        choices=PRECISIONS,
        default=None,
        help="graph to load (default: what the artifact ships); the report names it",
    )
    ev.add_argument(
        "--allow-dirty-manifest",
        action="store_true",
        help="score a model whose manifest records a dirty trainer tree (not"
        " reproducible; the report says so)",
    )
    ev.add_argument("--rerank-top-n", type=int, default=None, help="rescoring budget (default 50)")
    ev.add_argument(
        "--no-path-context",
        action="store_true",
        help="rerank ablation: score bare candidate text without the route prefix",
    )
    ev.add_argument("--beam-entities", type=int, default=None, help="beam sweep override")
    ev.add_argument("--beam-chunks", type=int, default=None, help="beam sweep override")
    ev.add_argument("--frontier-chunks", type=int, default=None, help="beam sweep override")
    ev.add_argument(
        "--no-specificity-filter",
        action="store_true",
        help="ablation: disable the expansion-time specificity filter",
    )
    ev.add_argument(
        "--diagnostics",
        action="store_true",
        help="calibration (effective vs annotated multihop) + miss breakdown",
    )
    ev.add_argument(
        "--strata-file",
        type=Path,
        default=None,
        help="JSON {stratum_name: [qid, ...]} — report metrics split by an arbitrary"
        " external partition (e.g. training-exposure buckets) instead of the"
        " calibration strata",
    )
    ev.add_argument(
        "--pool-ablation",
        type=int,
        default=None,
        metavar="N",
        help="hop-2 pool precision with/without specificity filter over first N queries",
    )
    ev.add_argument("--json", type=Path, default=None, dest="json_path")
    ev.add_argument("--rebuild", action="store_true", help="rebuild the persistent BEIR index")
    ev.add_argument(
        "--gate",
        action="store_true",
        help="check the BM25 floor against the pinned published baseline",
    )

    args = parser.parse_args(argv)
    # rerank-only flags under another policy would bypass the training-data wall
    if args.command == "eval":
        if args.selection != "rerank":
            for flag, value in (
                ("--rerank-model", args.rerank_model),
                ("--rerank-precision", args.rerank_precision),
                ("--rerank-top-n", args.rerank_top_n),
                ("--no-path-context", args.no_path_context),
                ("--allow-dirty-manifest", args.allow_dirty_manifest),
            ):
                if value is not None and value is not False:
                    parser.error(f"{flag} requires --selection rerank")
        elif args.hops == 0:
            parser.error("--selection rerank needs --hops 1 or 2 (hops 0 is the BM25 floor)")
        if args.dataset.startswith("hipporag-"):
            if args.corpus_pool != "dev":
                parser.error("--corpus-pool does not apply to hipporag-* (fixed published corpus)")
            if args.setting == "distractor":
                parser.error("--setting distractor does not apply to hipporag-* datasets")
    if args.command == "retrieve" and not args.rerank:
        for flag, value in (
            ("--rerank-model", args.rerank_model),
            ("--rerank-precision", args.rerank_precision),
        ):
            if value is not None:
                parser.error(f"{flag} requires --rerank")

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "ingest": _cmd_ingest,
        "retrieve": _cmd_retrieve,
        "explain": _cmd_explain,
        "bracket": _cmd_bracket,
        "serve": _cmd_serve,
        "eval": _cmd_eval,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _progress(message: str) -> None:
    print(f"[hoptrace] {message}", file=sys.stderr, flush=True)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from hoptrace.server import ingest_impl

    config: dict[str, object] = {"analyzer": args.analyzer, "ner": args.ner}
    if args.target_tokens is not None:
        config["target_tokens"] = args.target_tokens
    if args.max_tokens is not None:
        config["max_tokens"] = args.max_tokens
    report = ingest_impl(str(args.source), args.corpus_id, config)
    print(
        f"corpus {report['corpus_id']}: {report['documents']} documents,"
        f" {report['chunks']} chunks, {report['entities']} entities,"
        f" {report['mentions']} mentions (analyzer={report['analyzer']})"
    )
    for skipped in report["skipped"]:
        print(f"skipped: {skipped}", file=sys.stderr)
    return 0


def _cmd_retrieve(args: argparse.Namespace) -> int:
    from hoptrace.server import retrieve_impl

    payload = retrieve_impl(
        args.query,
        args.corpus_id,
        hops=args.hops,
        k=args.k,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        rerank_precision=args.rerank_precision,
    )
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0
    for item in payload["evidence"]:
        score = item["score"]
        rank = score["rank"]
        # under rerank the order is by learned score, so lead with it
        shown = (
            f"score {score['score']:.4f}"
            if score.get("rerank_score") is None
            else f"rerank {score['rerank_score']:+.3f} (path {score['score']:.4f})"
        )
        terms = ", ".join(item["matched_terms"]) or "none"
        print(
            f"#{rank + 1} chunk#{item['chunk']['id']} ({item['chunk']['doc']})"
            f" [{item['seed_source']}, hop {item['hop']}] {shown}"
            f" (bm25 {score['bm25']:.2f}, terms: {terms})"
        )
        print(f"   path: {item['path']}")
        text = " ".join(item["chunk"]["text"].split())
        print(f"   text: {text[:160]}{'…' if len(text) > 160 else ''}")
    if payload["unresolved_mentions"]:
        print(f"unresolved mentions: {', '.join(payload['unresolved_mentions'])}")
    for note in payload["notes"]:
        print(f"note: {note}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from hoptrace.server import explain_impl

    payload = explain_impl(args.chunk_id, args.corpus_id, args.query)
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0
    chunk = payload["chunk"]
    print(f"chunk#{chunk['id']} ({chunk['doc']}, title={chunk['title']})")
    print(f"entities: {', '.join(payload['entities']) or '(none)'}")
    query_info = payload.get("query")
    if query_info:
        if query_info["in_pool"]:
            place = "top-k" if query_info["in_top_k"] else "pool, below default k"
            print(f"for query {query_info['text']!r}: rank #{query_info['rank'] + 1} ({place})")
            print(f"path: {query_info['path']}")
        else:
            print(f"for query {query_info['text']!r}: NOT reached — {query_info['note']}")
        for term in query_info["bm25_terms"]:
            print(
                f"  bm25 term {term['term']!r}: tf={term['tf']} df={term['df']}"
                f" score={term['score']:.4f}"
            )
    return 0


def _cmd_bracket(args: argparse.Namespace) -> int:
    from hoptrace.bracket import run_bracket
    from hoptrace.config import corpus_path
    from hoptrace.store import Store

    path = corpus_path(args.corpus_id)
    if not path.is_file():
        from hoptrace.config import list_corpora

        print(
            f"error: no corpus {args.corpus_id!r}; available:"
            f" {', '.join(list_corpora()) or '(none)'}",
            file=sys.stderr,
        )
        return 1
    store = Store.open(path)
    try:
        report = run_bracket(store, n_questions=args.n_questions, seed=args.seed)
    finally:
        store.close()
    print(report.to_json() if args.as_json else report.to_text())
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from hoptrace.server import main as serve_main

    if args.http:
        print(
            f"hoptrace MCP server on http://{args.host}:{args.port}/mcp (no auth)",
            file=sys.stderr,
        )
    serve_main("streamable-http" if args.http else "stdio", args.host, args.port)
    return 0


def _retrieval_cfg(args: argparse.Namespace) -> RetrievalConfig:
    cfg = RetrievalConfig(bm25_k1=args.k1, bm25_b=args.b)
    if args.hub_df_ratio is not None:
        cfg = replace(cfg, hub_df_ratio=args.hub_df_ratio)
    if args.no_specificity_filter:
        cfg = replace(cfg, specificity_filter=False)
    if args.beam_entities is not None:
        cfg = replace(cfg, beam_entities=args.beam_entities)
    if args.beam_chunks is not None:
        cfg = replace(cfg, beam_chunks=args.beam_chunks)
    if args.frontier_chunks is not None:
        cfg = replace(cfg, frontier_chunks=args.frontier_chunks)
    if args.selection is not None:
        cfg = replace(cfg, selection=args.selection)
    if args.rerank_model is not None:
        cfg = replace(cfg, rerank_model=args.rerank_model)
    if args.no_path_context:
        cfg = replace(cfg, rerank_path_context=False)
    if args.rerank_precision is not None:
        cfg = replace(cfg, rerank_precision=args.rerank_precision)
    if args.rerank_top_n is not None:
        cfg = replace(cfg, rerank_top_n=args.rerank_top_n)
    return cfg


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.setting == "distractor":
        if args.dataset == "beir-hotpotqa":
            print("beir-hotpotqa has no distractor setting; use musique/2wiki", file=sys.stderr)
            return 2
        if args.hops != 0 or args.diagnostics or args.pool_ablation:
            print("distractor is a floor-only sanity appendix", file=sys.stderr)
            return 2
        questions = _load_questions(args.dataset)
        report = evaluate_distractor_floor(
            questions,
            dataset=args.dataset,
            analyzer=args.analyzer,
            k1=args.k1,
            b=args.b,
            limit=args.limit,
        )
        print(report.to_text())
        write_report(report, args.json_path)
        return 0

    store, query_gold, cleanup = _open_corpus(args)
    try:
        cfg = _retrieval_cfg(args)
        is_beir = args.dataset == "beir-hotpotqa"
        if is_beir:
            setting, protocol_note = "corpus-scale", None
        elif args.dataset.startswith("hipporag-"):
            setting = "hipporag-protocol"
            protocol_note = (
                "corpus protocol: HippoRAG published sample — 1,000 validation questions"
                " over their exact candidate corpus (Gutiérrez et al., NeurIPS 2024)."
                " Comparable to published R@2/R@5 and all-recall numbers."
            )
        elif args.corpus_pool == "all":
            setting = "pooled-all-splits"
            protocol_note = (
                "corpus protocol: haystack pools train+dev paragraphs; queries and gold"
                " are dev-only (IRCoT-comparable). Train paragraphs are corpus mass, not"
                " training labels."
            )
        else:
            setting = "pooled-dev"
            protocol_note = (
                "corpus protocol: haystack pools dev paragraphs only — a development"
                " setting, NOT comparable to published open-domain numbers (which pool"
                " across splits). Use --corpus-pool all for those."
            )
        misses: MissBreakdown | None = None
        displacement: DisplacementAudit | None = None
        strata: dict[str, str] | None = None
        calibration_text: str | None = None

        if args.diagnostics:
            # strata first: they change what "better" means for the numbers below
            outcomes = floor_outcomes(
                store,
                query_gold,
                analyzer=args.analyzer,
                k1=args.k1,
                b=args.b,
                limit=args.limit,
                progress=_progress,
            )
            strata = stratify(outcomes, answer_k=ANSWER_K)
            calibration_text = calibrate(args.dataset, outcomes, answer_k=ANSWER_K).to_text()

        if args.strata_file is not None:
            partition = json.loads(args.strata_file.read_text(encoding="utf-8"))
            strata = {qid: name for name, qids in partition.items() for qid in qids}

        if args.hops == 0:
            report = evaluate_floor(
                store,
                query_gold,
                args.dataset,
                setting=setting,
                analyzer=args.analyzer,
                k1=args.k1,
                b=args.b,
                ks=(2, 5, 10, 20, 100) if is_beir else (2, 5, 10, 20),
                with_ndcg=is_beir,
                limit=args.limit,
                progress=_progress,
                strata=strata,
            )
        else:
            if args.diagnostics:
                misses = MissBreakdown(dataset=args.dataset, hops=args.hops, k=DIAGNOSTIC_K)
                displacement = DisplacementAudit(
                    dataset=args.dataset, hops=args.hops, k=DIAGNOSTIC_K
                )
            report = evaluate_hoptrace(
                store,
                query_gold,
                args.dataset,
                setting=setting,
                hops=args.hops,
                cfg=cfg,
                ks=(2, 5, 10, 20),
                limit=args.limit,
                progress=_progress,
                misses=misses,
                strata=strata,
                displacement=displacement,
                allow_dirty=args.allow_dirty_manifest,
            )

        if protocol_note is not None:
            report.notes.append(protocol_note)
        print(report.to_text())
        write_report(report, args.json_path)
        for extra in (misses, displacement):
            if extra is not None:
                print()
                print(extra.to_text())
        if calibration_text is not None:
            print()
            print(calibration_text)

        if args.pool_ablation:
            print()
            print(_run_pool_ablation(args, store, query_gold, cfg).to_text())

        if args.gate:
            return _check_gate(args, report)
        return 0
    finally:
        cleanup()


def _load_questions(dataset: str) -> list[EvalQuestion]:
    data_path = ensure_dataset(dataset)
    if dataset.startswith("musique"):
        return load_musique(data_path)
    return load_hotpot_format(data_path)


def _open_corpus(
    args: argparse.Namespace,
) -> tuple[Store, list[QueryGold], Callable[[], None]]:
    if args.dataset.startswith("hipporag-"):
        return _open_hipporag_corpus(args)

    if args.dataset == "beir-hotpotqa":
        dataset_dir = ensure_dataset("beir-hotpotqa")
        index_path = data_dir() / "eval" / "beir-hotpotqa.sqlite"
        if args.rebuild or not index_path.is_file():
            _progress(f"building BEIR index at {index_path} (one-time, hours-scale)")
            stats = build_beir_index(
                dataset_dir / "corpus.jsonl", index_path, analyzer=args.analyzer
            )
            _progress(
                f"index built: {stats.passages:,} passages, {stats.entities:,} entities,"
                f" {stats.mentions:,} mentions in {stats.elapsed_s / 3600:.2f} h"
            )
        if ensure_entity_stats(index_path):
            _progress("materialized entity_stats on the existing index (one-time migration)")
        store = Store.open(index_path)
        stored_analyzer = store.meta("analyzer")
        if stored_analyzer != args.analyzer:
            store.close()
            raise SystemExit(
                f"index was built with analyzer={stored_analyzer!r}; rerun with"
                f" --analyzer {stored_analyzer} or --rebuild"
            )
        queries = load_beir_queries(dataset_dir / "queries.jsonl")
        qrels = load_beir_qrels(dataset_dir / "qrels" / "test.tsv")
        query_gold = beir_query_gold(store, queries, qrels, dataset_dir / "queries.jsonl")
        return store, query_gold, store.close

    questions = _load_questions(args.dataset)
    if args.limit is not None:
        questions = questions[: args.limit]

    if args.corpus_pool == "all":
        # IRCoT-style protocol: the haystack pools train+dev paragraphs; queries and gold
        # stay dev-only
        pool_questions = _load_questions(f"{args.dataset}-train") + questions
        eval_qids = {q.qid for q in questions}
        train_qids = {q.qid for q in pool_questions[: -len(questions)]}
        if eval_qids & train_qids:
            raise ValueError(
                f"{len(eval_qids & train_qids)} qids appear in both the train and dev"
                " splits: pooled gold would be ambiguous"
            )
        # a --limit run pools fewer paragraphs; keep its cache apart from the full index
        suffix = f"-limit{args.limit}" if args.limit is not None else ""
        index_path = data_dir() / "eval" / f"{args.dataset}-pool-all{suffix}.sqlite"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        pooled = build_pooled_index(
            pool_questions,
            target=index_path,
            analyzer=args.analyzer,
            progress=_progress,
            reuse=True,
        )
    else:
        pool_questions = questions
        pooled = build_pooled_index(questions, analyzer=args.analyzer, progress=_progress)

    query_gold = pooled_query_gold(questions, pooled.gold_chunks)
    return pooled.store, query_gold, pooled.store.close


def _open_hipporag_corpus(
    args: argparse.Namespace,
) -> tuple[Store, list[QueryGold], Callable[[], None]]:
    """HippoRAG protocol: their 1,000-question sample over their candidate
    corpus; gold is mapped by (title, text) and a miss is a hard error."""
    questions_path = ensure_dataset(args.dataset)
    corpus_path_ = ensure_dataset(f"{args.dataset}-corpus")
    questions = (
        load_musique_json(questions_path)
        if args.dataset == "hipporag-musique"
        else load_hotpot_format(questions_path)
    )
    if args.limit is not None:
        questions = questions[: args.limit]
    index_path = data_dir() / "eval" / f"{args.dataset}.sqlite"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_explicit_index(
        load_corpus_entries(corpus_path_),
        target=index_path,
        analyzer=args.analyzer,
        progress=_progress,
        reuse=True,
    )
    query_gold: list[QueryGold] = []
    for question in questions:
        gold: set[int] = set()
        for paragraph in question.paragraphs:
            if not paragraph.is_gold:
                continue
            key = (paragraph.title, paragraph.text)
            if key not in index.key_to_chunk:
                index.store.close()
                raise ValueError(
                    f"{args.dataset}: gold paragraph {paragraph.title!r} of question"
                    f" {question.qid} is not in the published corpus — the sample and"
                    " corpus files disagree"
                )
            gold.add(index.key_to_chunk[key])
        query_gold.append(
            QueryGold(
                qid=question.qid,
                text=question.text,
                gold=frozenset(gold),
                answer=question.answer,
                qtype=question.qtype,
            )
        )
    return index.store, query_gold, index.store.close


def _run_pool_ablation(
    args: argparse.Namespace,
    store: Store,
    query_gold: list[QueryGold],
    cfg: RetrievalConfig,
) -> PoolPrecision:
    hops = args.hops or 2
    sample = query_gold[: args.pool_ablation]
    # pool precision is selection-independent; avoid loading a rerank model
    cfg = replace(cfg, hops=hops, selection="interleave")
    on = Retriever(store, replace(cfg, specificity_filter=True))
    off = Retriever(store, replace(cfg, specificity_filter=False))
    result = pool_precision(
        dataset=args.dataset,
        hops=hops,
        run_on=on.retrieve,
        run_off=off.retrieve,
        queries=[(qg.text, qg.gold) for qg in sample],
    )
    return result


def _check_gate(args: argparse.Namespace, report: FloorReport) -> int:
    if args.dataset != "beir-hotpotqa" or report.ndcg10 is None:
        print("gate: only defined for beir-hotpotqa nDCG@10 (floor run)", file=sys.stderr)
        return 2
    if args.limit is not None:
        print("gate: refusing to gate a --limit partial run", file=sys.stderr)
        return 2
    target = float(BEIR_HOTPOTQA_BASELINES["bm25_flat_ndcg10"])
    delta = report.ndcg10 - target
    ok = abs(delta) <= GATE_TOLERANCE
    verdict = "PASS" if ok else "FAIL"
    print(
        f"gate [{verdict}]: nDCG@10 {report.ndcg10:.4f} vs published flat {target:.3f}"
        f" (delta {delta:+.4f}, tolerance ±{GATE_TOLERANCE})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
