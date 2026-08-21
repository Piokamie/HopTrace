"""Rerank selection above the model, with an injected scorer (no onnxruntime needed)."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from hoppath.config import ChunkConfig, RetrievalConfig, models_dir
from hoppath.expand import Candidate
from hoppath.ingest import SourceDocument, ingest_documents
from hoppath.provenance import HopEdge, HopPath
from hoppath.rerank import PARENT_SNIPPET_CHARS, rerank_select, serialize_pair
from hoppath.retrieve import Retriever
from hoppath.score import interleave
from hoppath.store import Store


class RecordingScorer:
    def __init__(self, fn: Callable[[str, str], float]) -> None:
        self._fn = fn
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        batch = list(pairs)
        self.calls.append(batch)
        return [self._fn(a, b) for a, b in batch]


def _seed(chunk_id: int, score: float = 0.5) -> Candidate:
    path = HopPath("anna", (HopEdge("anna", chunk_id, 0.8, 2),))
    return Candidate(chunk_id=chunk_id, path=path, score=score, hop=0)


def _ordered(pool: dict[int, Candidate]) -> list[Candidate]:
    """Pools reach rerank_select already in interleave order."""
    return interleave(pool, len(pool))


def _hop(chunk_id: int, parent_id: int, score: float = 0.3) -> Candidate:
    edges = (
        HopEdge("anna", parent_id, 0.8, 2),
        HopEdge("kowalski", chunk_id, 0.7, 1),
    )
    return Candidate(chunk_id=chunk_id, path=HopPath("anna", edges), score=score, hop=1)


def test_serialize_seed_passes_bare_text() -> None:
    texts = {1: "Anna reports to Kowalski."}
    assert serialize_pair("who?", _seed(1), texts) == ("who?", "Anna reports to Kowalski.")


def test_serialize_hop_prefixes_route_context() -> None:
    texts = {1: "Anna reports to Kowalski.", 2: "Kowalski sits in 4B."}
    text_a, text_b = serialize_pair("who?", _hop(2, parent_id=1), texts)
    assert text_a == "who?"
    assert text_b == "via kowalski | Anna reports to Kowalski. || Kowalski sits in 4B."


def test_serialize_hop_snippets_long_parent() -> None:
    texts = {1: "word\n" * 200, 2: "target"}
    _, text_b = serialize_pair("q", _hop(2, parent_id=1), texts)
    prefix, _, _ = text_b.partition(" || ")
    snippet = prefix.removeprefix("via kowalski | ")
    assert len(snippet) <= PARENT_SNIPPET_CHARS
    assert snippet.endswith("…")
    assert "\n" not in snippet


def test_serialize_path_context_off_is_bare_text() -> None:
    texts = {1: "Anna reports to Kowalski.", 2: "Kowalski sits in 4B."}
    assert serialize_pair("q", _hop(2, parent_id=1), texts, path_context=False) == (
        "q",
        "Kowalski sits in 4B.",
    )
    # and the parent need not even be present
    assert serialize_pair("q", _hop(2, parent_id=99), {2: "x"}, path_context=False) == ("q", "x")


def test_serialize_hop_missing_parent_is_a_bug() -> None:
    with pytest.raises(KeyError):
        serialize_pair("q", _hop(2, parent_id=99), {2: "target"})


def test_rerank_select_scores_exactly_the_interleave_prefix() -> None:
    """Exactly the first top_n of the pool, in interleave order, reach the scorer."""
    pool = {c.chunk_id: c for c in [_seed(1), _seed(2), _seed(3), _hop(4, 1), _hop(5, 1)]}
    texts = {cid: f"text {cid}" for cid in pool}
    scorer = RecordingScorer(lambda a, b: 0.0)
    rerank_select(_ordered(pool), "q", texts, scorer, k=2, top_n=3)
    assert len(scorer.calls) == 1
    expected = [serialize_pair("q", c, texts) for c in interleave(pool, 3)]
    assert scorer.calls[0] == expected
    assert [c.chunk_id for c in interleave(pool, 3)] == [1, 4, 2]


def test_rerank_select_orders_by_learned_score() -> None:
    pool = {c.chunk_id: c for c in [_seed(1, 0.9), _seed(2, 0.8), _hop(3, 1, 0.1)]}
    texts = {cid: f"text {cid}" for cid in pool}
    # the learned score inverts the deterministic order
    scorer = RecordingScorer(lambda a, b: 1.0 if b.endswith("text 3") else 0.0)
    selected = rerank_select(_ordered(pool), "q", texts, scorer, k=2, top_n=3)
    assert [s.candidate.chunk_id for s in selected] == [3, 1]
    assert selected[0].gain == 1.0


def test_rerank_never_returns_a_candidate_outside_the_pool() -> None:
    """The learned ranker reorders the pool and never reaches outside it."""
    pool = {c.chunk_id: c for c in [_seed(1, 0.9), _seed(2, 0.1), _hop(3, 1, 0.5), _hop(4, 1, 0.2)]}
    texts = {cid: f"text {cid}" for cid in pool}
    scorer = RecordingScorer(lambda a, b: -99.0)
    selected = rerank_select(_ordered(pool), "q", texts, scorer, k=10, top_n=50)
    assert {s.candidate.chunk_id for s in selected} <= set(pool)
    assert len(selected) == len(pool)  # k beyond the pool yields the pool, not padding


def test_rerank_honours_k_above_the_rescoring_budget() -> None:
    """rerank_top_n bounds how many are scored, not how many are returned."""
    pool = {c.chunk_id: _seed(c.chunk_id, 1.0 / c.chunk_id) for c in map(_seed, range(1, 21))}
    texts = {cid: f"text {cid}" for cid in pool}
    scorer = RecordingScorer(lambda a, b: 0.0)
    selected = rerank_select(_ordered(pool), "q", texts, scorer, k=15, top_n=5)
    assert len(selected) == 15


def test_rerank_select_ties_break_on_chunk_id() -> None:
    pool = {c.chunk_id: c for c in [_seed(9, 0.2), _seed(4, 0.9), _seed(7, 0.5)]}
    texts = {cid: "same" for cid in pool}
    scorer = RecordingScorer(lambda a, b: 1.0)
    selected = rerank_select(_ordered(pool), "q", texts, scorer, k=3, top_n=3)
    assert [s.candidate.chunk_id for s in selected] == [4, 7, 9]


DOCS = [
    SourceDocument("p0", "Anna Nowak reports to Kowalski on the platform team.", "p0"),
    SourceDocument("p1", "Kowalski occupies Room 4B near the atrium.", "p1"),
    SourceDocument("p2", "The platform team ships the ingestion service.", "p2"),
    SourceDocument("p3", "The atrium cafe closes at five.", "p3"),
    SourceDocument("p4", "Budget planning happens every quarter at the company.", "p4"),
]


@pytest.fixture(scope="module")
def store() -> Store:
    store, _ = ingest_documents(
        DOCS, None, chunk_cfg=ChunkConfig(identity=True), analyzer="english"
    )
    return store


def _rerank_cfg() -> RetrievalConfig:
    return RetrievalConfig(
        hops=2, k=4, hub_df_ratio=0.7, hub_df_floor=1, selection="rerank", rerank_top_n=8
    )


def test_retriever_end_to_end_with_injected_scorer(store: Store) -> None:
    scorer = RecordingScorer(lambda a, b: 1.0 if "4B" in b else 0.0)
    retriever = Retriever(store, _rerank_cfg(), scorer=scorer)
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    assert "4B" in result.evidence[0].text
    assert result.evidence[0].score.rerank_score == 1.0
    assert all(e.score.rerank_score is not None for e in result.evidence)
    # the hop candidate reached the scorer with its route context attached
    scored_texts = [b for batch in scorer.calls for _, b in batch]
    assert any(b.startswith("via kowalski | ") and "4B" in b for b in scored_texts)


def test_retriever_rerank_stays_inside_the_pool(store: Store) -> None:
    """Only pool members come back, and the scorer sees exactly the top-N window."""
    scorer = RecordingScorer(lambda a, b: -99.0)
    cfg = _rerank_cfg()
    retriever = Retriever(store, cfg, scorer=scorer)
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    returned = [e.chunk_id for e in result.evidence]
    assert set(returned) <= set(result.pool_order)
    assert len(returned) == min(cfg.k, len(result.pool_order))
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == min(len(result.pool_order), max(cfg.rerank_top_n, cfg.k))


def test_retriever_rerank_is_deterministic(store: Store) -> None:
    def run() -> list[int]:
        scorer = RecordingScorer(lambda a, b: float(len(b) % 7))
        retriever = Retriever(store, _rerank_cfg(), scorer=scorer)
        result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
        return [e.chunk_id for e in result.evidence]

    assert run() == run()


def test_interleave_leaves_rerank_score_unset(store: Store) -> None:
    retriever = Retriever(store, RetrievalConfig(hops=2, k=4, hub_df_ratio=0.7, hub_df_floor=1))
    result = retriever.retrieve("Where does the manager of Anna Nowak sit?")
    assert all(e.score.rerank_score is None for e in result.evidence)


def test_rerank_without_scorer_raises_at_construction(store: Store) -> None:
    # missing extra or unknown model: either way it fails at construction, not mid-query
    cfg = replace(_rerank_cfg(), rerank_model="no-such-model")
    with pytest.raises(RuntimeError, match="rerank"):
        Retriever(store, cfg)


def _load_builder():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "training" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_refuses_2wiki_holdout() -> None:
    builder = _load_builder()
    with pytest.raises(ValueError, match="transfer holdout"):
        builder.check_not_holdout("2wiki")
    with pytest.raises(ValueError, match="transfer holdout"):
        builder.check_not_holdout("2wiki-custom")
    builder.check_not_holdout("musique")  # does not raise


def test_builder_rows_are_exactly_what_inference_scores(store: Store) -> None:
    """Builder rows match what a rerank Retriever with inference's k hands the scorer."""
    from hoppath.eval.harness import QueryGold

    builder = _load_builder()
    chunk_ids = [ids[0] for ids in store.chunks_by_doc_path([d.path for d in DOCS]).values()]
    answer_chunk = next(cid for cid in chunk_ids if "4B" in store.get_chunk(cid).text)
    cfg = replace(builder.RETRIEVAL_CFG, hub_df_ratio=0.7, hub_df_floor=1)
    query = "Where does the manager of Anna Nowak sit?"
    qg = QueryGold(qid="t1", text=query, gold=frozenset({answer_chunk}))

    rows = builder.question_rows(Retriever(store, cfg), qg, "musique")
    scorer = RecordingScorer(lambda a, b: 0.0)
    Retriever(store, replace(cfg, selection="rerank"), scorer=scorer).retrieve(query)

    assert 0 < len(rows) <= builder.TOP_N
    assert [(r["text_a"], r["text_b"]) for r in rows] == scorer.calls[0]
    assert {r["label"] for r in rows} == {0, 1}
    hop_rows = [r for r in rows if r["hop"] >= 1]
    assert hop_rows and all(r["text_b"].startswith("via ") for r in hop_rows)
    assert all(r["parent_key"] for r in hop_rows) and all(r["cand_key"] for r in rows)
    positive = next(r for r in rows if r["label"] == 1)
    assert "4B" in positive["text_b"]


def test_builder_uses_inference_seed_depth() -> None:
    """The builder's config seeds as deeply as inference (k=20), no deeper."""
    builder = _load_builder()
    cfg = builder.RETRIEVAL_CFG
    assert max(cfg.seed_bm25_top_n, cfg.k) == max(RetrievalConfig().seed_bm25_top_n, 20)
    assert cfg.rerank_top_n == builder.TOP_N


def test_builder_split_is_read_from_the_file_name(tmp_path) -> None:
    builder = _load_builder()
    assert builder.split_of(tmp_path / "musique_ans_v1.0_train.jsonl") == "train"
    assert builder.split_of(tmp_path / "train.tsv") == "train"
    with pytest.raises(ValueError, match="does not look like a train split"):
        builder.split_of(tmp_path / "musique_ans_v1.0_dev.jsonl")


def test_builder_exclusion_is_whitespace_insensitive_and_structural() -> None:
    builder = _load_builder()
    gold = {builder.key_of("Title  Some   text\nacross lines")}
    row = {"cand_key": builder.key_of("Title Some text across lines"), "parent_key": None}
    assert builder.is_eval_gold(row, gold)
    row = {
        "cand_key": builder.key_of("other"),
        "parent_key": builder.key_of("Title Some text across lines"),
    }
    assert builder.is_eval_gold(row, gold)
    assert not builder.is_eval_gold({"cand_key": "x", "parent_key": None}, gold)


def test_builder_derive_filtered_matches_a_fresh_filtered_build(tmp_path) -> None:
    """--from reproduces a fresh exclusion-on build: same rows, drops and counts."""
    import json

    builder = _load_builder()
    gold_keys = {builder.key_of("gold passage")}
    rows = [
        # q1: positive survives, one row excluded by candidate
        {
            "qid": "q1",
            "dataset": "musique",
            "text_a": "q",
            "text_b": "gold passage",
            "label": 0,
            "hop": 0,
            "cand_key": builder.key_of("gold passage"),
            "parent_key": None,
        },
        {
            "qid": "q1",
            "dataset": "musique",
            "text_a": "q",
            "text_b": "answer",
            "label": 1,
            "hop": 0,
            "cand_key": builder.key_of("answer"),
            "parent_key": None,
        },
        # q2: its only positive quotes gold as parent → excluded → query dropped
        {
            "qid": "q2",
            "dataset": "musique",
            "text_a": "q",
            "text_b": "via e | gold passage || x",
            "label": 1,
            "hop": 1,
            "cand_key": builder.key_of("x"),
            "parent_key": builder.key_of("gold passage"),
        },
        {
            "qid": "q2",
            "dataset": "musique",
            "text_a": "q",
            "text_b": "neg",
            "label": 0,
            "hop": 0,
            "cand_key": builder.key_of("neg"),
            "parent_key": None,
        },
    ]
    full = tmp_path / "full"
    full.mkdir()
    (full / "musique-train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (full / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "retrieval": {"k": 20},
                "holdout": ["2wiki"],
                "serialize_format": "x",
                "exclusion_ablation": None,
                "sources": [
                    {
                        "dataset": "musique",
                        "split": "train",
                        "source": "s",
                        "source_sha256": "0",
                        "questions": 2,
                        "limit": None,
                        "kept": 2,
                        "dropped_no_positive": 0,
                        "rows": 4,
                        "positives": 2,
                        "excluded_eval_gold_rows": 0,
                        "eval_gold_passages_present": None,
                        "output": "musique-train.jsonl",
                    }
                ],
            }
        )
    )
    out = tmp_path / "filtered"
    out.mkdir()
    derived = builder.derive_filtered(full, out, gold_keys)
    source = derived["sources"][0]
    assert source["kept"] == 1 and source["dropped_no_positive"] == 1
    assert source["rows"] == 1 and source["positives"] == 1
    assert source["excluded_eval_gold_rows"] == 2 and source["eval_gold_passages_present"] == 1
    assert derived["derived_from"] == str(full)
    written = [json.loads(line) for line in (out / "musique-train.jsonl").read_text().splitlines()]
    assert [r["text_b"] for r in written] == ["answer"]
    # a fresh filtered writer over the same rows agrees exactly
    fresh = builder._SplitWriter(tmp_path / "fresh.jsonl", gold_keys)
    fresh.add_query(rows[:2])
    fresh.add_query(rows[2:])
    fresh.close()
    assert fresh.stats() == {k: source[k] for k in fresh.stats()}
    assert (tmp_path / "fresh.jsonl").read_text() == (out / "musique-train.jsonl").read_text()
    builder.write_manifest(out, derived["sources"], gold_keys, base=derived)
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["derived_from"] == str(full)
    assert manifest["exclusion_ablation"]["eval_gold_passages_excluded"] == 1
    with pytest.raises(ValueError, match="already exclusion-filtered"):
        builder.derive_filtered(out, tmp_path / "again", gold_keys)


def _stub_artifact(model_dir, trained_on, holdout=("2wiki",), **extra):
    """Fine-tuned-artifact stub: graphs, tokenizer, and a manifest whose hashes match them."""
    import json

    from hoppath.rerank import ResolvedModel, sha256_of

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"stub-fp32")
    (model_dir / "model_int8.onnx").write_bytes(b"stub-int8")
    (model_dir / "tokenizer.json").write_bytes(b"{}")
    manifest = {
        "model": "test-model",
        "base_model": "base",
        "trained_on": trained_on,
        "holdout": list(holdout),
        "trainer_commit": "abc123",
        "files": {
            name: sha256_of(model_dir / name)
            for name in ("model.onnx", "model_int8.onnx", "tokenizer.json")
        },
        **extra,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    return ResolvedModel(model_dir, "fp32", None)


TRAIN_ONLY = [{"dataset": "musique", "split": "train"}, {"dataset": "hotpotqa", "split": "train"}]


def test_training_wall_passes_for_untrained_split(tmp_path) -> None:
    from hoppath.eval.harness import check_training_wall

    model = _stub_artifact(tmp_path, TRAIN_ONLY)
    note = check_training_wall("musique", model)
    assert "wall verified for musique/dev" in note
    assert "(fp32)" in note and "abc123" in note
    assert "wall verified" in check_training_wall("beir-hotpotqa", model)
    assert "wall verified" in check_training_wall("2wiki", model)


def test_training_wall_refuses_evaluated_split(tmp_path) -> None:
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, [{"dataset": "musique", "split": "dev"}])
    with pytest.raises(TrainingWallError, match="musique/dev"):
        check_training_wall("musique", model)


@pytest.mark.parametrize("name", ["2wiki", "2WikiMultihopQA", "hipporag-2wiki", "2wiki-train"])
def test_training_wall_refuses_holdout_in_training(tmp_path, name: str) -> None:
    """Any spelling of the holdout trips the wall; the manifest is untrusted data."""
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, [{"dataset": name, "split": "train"}])
    with pytest.raises(TrainingWallError, match="declared holdout"):
        check_training_wall("2wiki", model)


def test_training_wall_refuses_holdout_even_for_other_datasets(tmp_path) -> None:
    """A holdout in training is refused whatever dataset is evaluated."""
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, [{"dataset": "2wiki", "split": "train"}])
    with pytest.raises(TrainingWallError, match="declared holdout"):
        check_training_wall("musique", model)


def test_training_wall_refuses_empty_or_malformed_trained_on(tmp_path) -> None:
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    with pytest.raises(TrainingWallError, match="no trained_on"):
        check_training_wall("musique", _stub_artifact(tmp_path / "a", []))
    with pytest.raises(TrainingWallError, match="must be an object"):
        check_training_wall("musique", _stub_artifact(tmp_path / "b", ["musique/train"]))
    with pytest.raises(TrainingWallError, match="must be an object"):
        check_training_wall("musique", _stub_artifact(tmp_path / "c", [{"dataset": "musique"}]))


def test_training_wall_refuses_unmapped_dataset(tmp_path) -> None:
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, TRAIN_ONLY)
    with pytest.raises(TrainingWallError, match="no training-wall mapping"):
        check_training_wall("some-new-benchmark", model)


def test_training_wall_flags_unexcluded_model(tmp_path) -> None:
    """Kept-gold models pass the wall; the report has to flag them."""
    from hoppath.eval.harness import check_training_wall

    assert "NOT PUBLISHABLE" in check_training_wall("musique", _stub_artifact(tmp_path, TRAIN_ONLY))


def test_training_wall_reports_exclusion_when_applied(tmp_path) -> None:
    from hoppath.eval.harness import check_training_wall

    model = _stub_artifact(tmp_path, TRAIN_ONLY, exclusion={"eval_gold_passages_excluded": 3795})
    note = check_training_wall("musique", model)
    assert "eval-gold excluded" in note and "3795" in note
    assert "NOT PUBLISHABLE" not in note


def test_training_wall_refuses_missing_manifest(tmp_path) -> None:
    from hoppath.eval.harness import TrainingWallError, check_training_wall
    from hoppath.rerank import ResolvedModel

    (tmp_path / "model.onnx").write_bytes(b"stub")
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(TrainingWallError, match=r"no manifest\.json"):
        check_training_wall("musique", ResolvedModel(tmp_path, "fp32", None))


def test_training_wall_binds_manifest_to_files(tmp_path) -> None:
    """A manifest beside a graph it does not describe fails the wall."""
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, TRAIN_ONLY)
    (tmp_path / "model.onnx").write_bytes(b"a different graph")
    with pytest.raises(TrainingWallError, match=r"does not describe model\.onnx"):
        check_training_wall("musique", model)


def test_training_wall_requires_file_hashes_and_the_scored_graph(tmp_path) -> None:
    import json

    from hoppath.eval.harness import TrainingWallError, check_training_wall
    from hoppath.rerank import ResolvedModel

    model = _stub_artifact(tmp_path, TRAIN_ONLY)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"].pop("model_int8.onnx")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TrainingWallError, match="does not list the graph"):
        check_training_wall("musique", ResolvedModel(tmp_path, "int8", None))
    manifest.pop("files")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TrainingWallError, match="no file checksums"):
        check_training_wall("musique", model)


def test_training_wall_refuses_dirty_trainer_unless_allowed(tmp_path) -> None:
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    model = _stub_artifact(tmp_path, TRAIN_ONLY, trainer_commit="abc123-dirty")
    with pytest.raises(TrainingWallError, match="dirty trainer tree"):
        check_training_wall("musique", model)
    note = check_training_wall("musique", model, allow_dirty=True)
    assert "DIRTY TRAINER TREE" in note and "NOT PUBLISHABLE" in note


def test_zero_shot_exemption_only_for_the_registry_base(tmp_path, monkeypatch) -> None:
    """Exemption follows the resolved registry entry, not the directory name;
    a registry location with a manifest is fine-tuned."""
    from hoppath import rerank
    from hoppath.eval.harness import TrainingWallError, check_training_wall

    bundled = tmp_path / "models"
    base_dir = bundled / rerank.ZERO_SHOT_MODEL
    base_dir.mkdir(parents=True)
    (base_dir / "model.onnx").write_bytes(b"base-graph")
    (base_dir / "tokenizer.json").write_bytes(b"{}")
    spec = rerank.ModelSpec(
        name=rerank.ZERO_SHOT_MODEL,
        zero_shot=True,
        files=(
            rerank.ModelFile(
                "model.onnx", "https://x/model.onnx", rerank.sha256_of(base_dir / "model.onnx")
            ),
            rerank.ModelFile(
                "tokenizer.json", "https://x/t.json", rerank.sha256_of(base_dir / "tokenizer.json")
            ),
        ),
    )
    monkeypatch.setattr(rerank, "BUNDLED_MODELS_DIR", bundled)
    monkeypatch.setitem(rerank.MODELS, rerank.ZERO_SHOT_MODEL, spec)

    resolved = rerank.resolve_model(rerank.ZERO_SHOT_MODEL)
    assert resolved.is_zero_shot()
    assert "zero-shot base" in check_training_wall("musique", resolved)

    # same files under a directory *named* like the base, passed as a path
    monkeypatch.chdir(bundled)
    as_dir = rerank.resolve_model(rerank.ZERO_SHOT_MODEL)  # cwd dir wins: it is a path now
    assert as_dir.spec is None and not as_dir.is_zero_shot()
    with pytest.raises(TrainingWallError, match=r"no manifest\.json"):
        check_training_wall("musique", as_dir)

    # a registry location that grew a manifest is fine-tuned: wall applies
    monkeypatch.chdir(tmp_path)
    _stub_artifact(base_dir, [{"dataset": "musique", "split": "dev"}])
    resolved = rerank.ResolvedModel(base_dir, "fp32", spec)
    assert not resolved.is_zero_shot()
    with pytest.raises(TrainingWallError, match="musique/dev"):
        check_training_wall("musique", resolved)


def test_evaluate_hoppath_runs_wall_before_scoring(store: Store, tmp_path) -> None:
    """A trained-on-dev manifest stops the run before any query is scored."""
    from hoppath.eval.harness import QueryGold, TrainingWallError, evaluate_hoppath

    _stub_artifact(tmp_path, [{"dataset": "musique", "split": "dev"}])
    cfg = replace(_rerank_cfg(), rerank_model=str(tmp_path))
    gold = [QueryGold(qid="q", text="Where does Kowalski sit?", gold=frozenset({1}))]
    with pytest.raises(TrainingWallError):
        evaluate_hoppath(store, gold, "musique", "pooled-dev", hops=2, cfg=cfg, ks=(2,))


def test_evaluate_hoppath_env_override_hits_the_wall(
    store: Store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rerank_model=None resolves $HOPPATH_RERANK_MODEL and checks it."""
    from hoppath.eval.harness import QueryGold, TrainingWallError, evaluate_hoppath

    _stub_artifact(tmp_path, [{"dataset": "musique", "split": "dev"}])
    monkeypatch.setenv("HOPPATH_RERANK_MODEL", str(tmp_path))
    gold = [QueryGold(qid="q", text="Where does Kowalski sit?", gold=frozenset({1}))]
    with pytest.raises(TrainingWallError, match="musique/dev"):
        evaluate_hoppath(store, gold, "musique", "pooled-dev", hops=2, cfg=_rerank_cfg(), ks=(2,))


def test_evaluate_hoppath_labels_rerank_system(
    store: Store, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """System label/notes for a fine-tuned model; default_scorer patched so no onnx is needed."""
    import hoppath.retrieve as retrieve_mod
    from hoppath.eval.harness import QueryGold, evaluate_hoppath

    monkeypatch.delenv("HOPPATH_RERANK_MODEL", raising=False)
    _stub_artifact(tmp_path, TRAIN_ONLY, exclusion={"eval_gold_passages_excluded": 1})
    monkeypatch.setattr(
        retrieve_mod, "default_scorer", lambda cfg: RecordingScorer(lambda a, b: float(len(b)))
    )
    gold = [QueryGold(qid="q", text="Where does Kowalski sit?", gold=frozenset({1}))]
    cfg = replace(_rerank_cfg(), rerank_model=str(tmp_path), rerank_precision="int8")
    report = evaluate_hoppath(store, gold, "musique", "pooled-dev", hops=2, cfg=cfg, ks=(2,))
    assert report.system == "hoppath@2hop+rerank"
    assert any("wall verified for musique/dev" in n and "(int8)" in n for n in report.notes)
    assert any("model=" in n and "(int8)" in n for n in report.notes)
    assert "ONNX 4 intra-op threads" in report.latency.conditions


def test_models_dir_under_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPPATH_DATA_DIR", "/tmp/ht-test")
    assert str(models_dir()) == "/tmp/ht-test/models"


def test_resolve_model_rejects_unknown_name() -> None:
    from hoppath.rerank import resolve_model

    with pytest.raises(RuntimeError, match="unknown rerank model"):
        resolve_model("no-such-model")


def test_resolve_model_rejects_incomplete_directory(tmp_path) -> None:
    from hoppath.rerank import resolve_model

    (tmp_path / "model.onnx").write_bytes(b"stub")
    with pytest.raises(RuntimeError, match=r"tokenizer\.json"):
        resolve_model(str(tmp_path))


def test_resolve_model_precision_follows_the_artifact(tmp_path) -> None:
    from hoppath.rerank import resolve_model

    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    (tmp_path / "model_int8.onnx").write_bytes(b"stub")
    assert resolve_model(str(tmp_path)).precision == "int8"  # only graph present
    (tmp_path / "model.onnx").write_bytes(b"stub")
    assert resolve_model(str(tmp_path)).precision == "fp32"  # fp32 preferred when both
    assert resolve_model(str(tmp_path), "int8").precision == "int8"


def test_resolve_model_missing_graph_is_a_clear_error(tmp_path) -> None:
    from hoppath.rerank import resolve_model

    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    (tmp_path / "model.onnx").write_bytes(b"stub")
    with pytest.raises(RuntimeError, match="does not ship a int8 graph"):
        resolve_model(str(tmp_path), "int8")
    with pytest.raises(ValueError, match="precision must be one of"):
        resolve_model(str(tmp_path), "INT8")


def test_registry_refuses_unpinned_downloads(tmp_path, monkeypatch) -> None:
    """An unpinned registry entry is neither fetched nor trusted from a same-named data dir."""
    from hoppath import rerank

    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(rerank, "BUNDLED_MODELS_DIR", tmp_path / "no-bundle")
    spec = rerank.ModelSpec(
        name="pending",
        default_precision="int8",
        files=(
            rerank.ModelFile("model_int8.onnx", "https://x/m", rerank._UNPINNED),
            rerank.ModelFile("tokenizer.json", "https://x/t", rerank._UNPINNED),
        ),
    )
    monkeypatch.setitem(rerank.MODELS, "pending", spec)
    with pytest.raises(RuntimeError, match="no pinned checksum"):
        rerank.resolve_model("pending")
    stale = tmp_path / "data" / "models" / "pending"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "model_int8.onnx").write_bytes(b"old")
    (stale / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="no pinned checksum"):
        rerank.resolve_model("pending")


def test_registry_prefers_bundled_copy_and_verifies_it(tmp_path, monkeypatch) -> None:
    from hoppath import rerank

    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    bundled = tmp_path / "models" / "bundled"
    bundled.mkdir(parents=True)
    (bundled / "model_int8.onnx").write_bytes(b"int8")
    (bundled / "tokenizer.json").write_bytes(b"{}")
    spec = rerank.ModelSpec(
        name="bundled",
        default_precision="int8",
        files=(
            rerank.ModelFile(
                "model_int8.onnx", "https://x/m", rerank.sha256_of(bundled / "model_int8.onnx")
            ),
            rerank.ModelFile("model.onnx", "https://x/f", "0" * 64),
            rerank.ModelFile(
                "tokenizer.json", "https://x/t", rerank.sha256_of(bundled / "tokenizer.json")
            ),
        ),
    )
    monkeypatch.setattr(rerank, "BUNDLED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setitem(rerank.MODELS, "bundled", spec)
    resolved = rerank.resolve_model("bundled")
    assert resolved.path == bundled and resolved.precision == "int8"
    assert not (tmp_path / "data").exists()  # nothing copied, nothing downloaded
    (bundled / "model_int8.onnx").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        rerank.resolve_model("bundled")


def test_registry_copies_bundled_files_when_a_download_is_needed(tmp_path, monkeypatch) -> None:
    """fp32 requested, only int8 bundled: shared files copied, the unreachable download
    names its URL, nothing half-written stays."""
    from hoppath import rerank

    monkeypatch.setenv("HOPPATH_DATA_DIR", str(tmp_path / "data"))
    bundled = tmp_path / "models" / "bundled"
    bundled.mkdir(parents=True)
    (bundled / "model_int8.onnx").write_bytes(b"int8")
    (bundled / "tokenizer.json").write_bytes(b"{}")
    spec = rerank.ModelSpec(
        name="bundled",
        default_precision="int8",
        files=(
            rerank.ModelFile(
                "model_int8.onnx", "https://x/m", rerank.sha256_of(bundled / "model_int8.onnx")
            ),
            rerank.ModelFile("model.onnx", "http://127.0.0.1:9/model.onnx", "0" * 64),
            rerank.ModelFile(
                "tokenizer.json", "https://x/t", rerank.sha256_of(bundled / "tokenizer.json")
            ),
        ),
    )
    monkeypatch.setattr(rerank, "BUNDLED_MODELS_DIR", tmp_path / "models")
    monkeypatch.setitem(rerank.MODELS, "bundled", spec)
    with pytest.raises(RuntimeError, match="download failed"):
        rerank.resolve_model("bundled", "fp32")
    target = tmp_path / "data" / "models" / "bundled"
    assert (target / "tokenizer.json").is_file()  # copied from the bundle
    assert not (target / "model.onnx.part").exists()


def test_onnx_cross_encoder_real_model() -> None:
    """Integration: needs the rerank extra and a downloaded base model; usually skips."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from hoppath.rerank import ZERO_SHOT_MODEL, OnnxCrossEncoder

    model_dir = models_dir() / ZERO_SHOT_MODEL
    if not (model_dir / "model.onnx").is_file():
        pytest.skip("rerank base model not downloaded")
    scorer = OnnxCrossEncoder(model_dir, batch_size=2)
    pairs = [
        ("where does kowalski sit", "Kowalski occupies Room 4B near the atrium."),
        ("where does kowalski sit", "The atrium cafe closes at five."),
    ]
    first = scorer.score(pairs)
    assert first[0] > first[1]  # relevant beats irrelevant
    assert scorer.score(pairs) == first  # deterministic
    # an absurdly long query is clipped, not a tokenizer exception
    long_query = "kowalski " * 400
    assert len(scorer.score([(long_query, "Kowalski occupies Room 4B.")])) == 1
