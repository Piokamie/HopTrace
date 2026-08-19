# Reranker training (dev-only)

Nothing here ships in the package. Inference needs only the `rerank`
extra (ONNX Runtime + tokenizers); this directory holds the pipeline
that produces the model artifact, kept in-repo for reproducibility. It
is outside the package's mypy scope (`[tool.mypy] files = ["src"]`) and
runs under the `train` dependency group.

## Training-data wall (ADR 0011)

- Sources: MuSiQue-Ans **train** and HotpotQA (BEIR) **train** splits
  only; dev/test splits are the published evaluation baseline. The split
  is read off the source file name.
- **2WikiMultihopQA never enters training** — it is the transfer test.
  `build_dataset.py` refuses it.
- Every artifact carries a manifest naming its training sources, the
  retrieval settings its pools were built with, and the sha256 of every
  file; the eval harness refuses to evaluate any (dataset, split) the
  manifest lists, refuses a manifest that does not describe the graphs
  beside it, and refuses an artifact trained from a dirty tree unless
  told `--allow-dirty-manifest`.

## 1. Build the training set

```
uv run --extra eval python training/build_dataset.py --dataset all
```

For every train question the deterministic retriever produces **exactly
the candidate window rerank inference scores** — `Retriever.candidate_window`
with inference's k (20 → 20 BM25 seeds), hops=2, interleave order, top 50,
eval BM25 parameters (k1 0.9, b 0.4). Each candidate is serialized with
the same `hoptrace.rerank.serialize_pair` the scorer uses and labeled by
pool-gold membership. Non-gold pool members are the hard negatives.
Queries with no positive in the window are dropped and counted. The
manifest records `retrieval.seed_depth`; a different seed depth is a
differently shaped pool.

**Evaluation gold passages are excluded by default.** Wikipedia
paragraphs recur across questions, so a passage that is gold for an
evaluation question can otherwise appear in training as a distractor for
some unrelated train question. The builder drops any row where an
evaluation-gold passage (from either HippoRAG evaluation set) is the
scored candidate or is quoted in the route-context prefix, matching on
whitespace-normalized text via per-row `cand_key`/`parent_key`, and
records the count — plus how many distinct eval-gold passages occurred
in each source at all — in the manifest. Pass `--keep-eval-gold` only to
measure what the exclusion costs; the eval harness marks any model built
that way `NOT PUBLISHABLE` in its report.

The exclusion ablation pair comes from one retrieval pass:

```
uv run --extra eval python training/build_dataset.py --keep-eval-gold --out $HOPTRACE_DATA_DIR/training-full
uv run --extra eval python training/build_dataset.py --from $HOPTRACE_DATA_DIR/training-full --out $HOPTRACE_DATA_DIR/training
```

`--from` filters row-wise on the structured keys and writes the same
rows a fresh exclusion-on build would, with `derived_from` recorded.

Writes to `$HOPTRACE_DATA_DIR/training/`: `musique-train.jsonl`,
`hotpotqa-train.jsonl`, `dataset_manifest.json`, plus the reusable
`musique-train-pool.sqlite` pooled index.

Costs (Apple Silicon, single process): MuSiQue-train downloads 241 MB
and pools 84,559 unique paragraphs; HotpotQA reuses the persistent
5.2M-passage BEIR index (build it first with `hoptrace eval --dataset
beir-hotpotqa` if missing) and is capped at 30k of ~85k train queries by
default (`--limit-hotpot`; the cap is recorded in the manifest). Expect a
few hours total.

The product's `hoptrace retrieve` seeds with BM25 k1=1.5, b=0.75 while
training and eval use 0.9/0.4 (the published-baseline configuration): the
shipped reranker is trained on eval-seeded pools and served over
product-seeded ones (tracked in DESIGN.md's roadmap).

## 2. Fine-tune and export

```
uv sync --group train --extra eval
uv run --group train --extra eval python training/train_reranker.py
```

Fine-tunes `cross-encoder/ms-marco-MiniLM-L6-v2` as a pointwise binary
scorer (BCE) on torch (MPS when available). Sampling per query: every
positive plus at most `--neg-seed` seed negatives and `--neg-hop` hop
negatives (defaults 6/6); the pool is ~3% positive and hop positives
are roughly 17× rarer than seed positives, hence the hop quota. A
`--holdback` slice (3%) of train queries drives checkpoint selection by
per-query MRR.

Output (`$HOPTRACE_DATA_DIR/models/hoptrace-rerank-minilm-l6/`):
`model.onnx` (fp32, ~91 MB), `model_int8.onnx` (dynamic int8, ~23 MB),
`tokenizer.json`, `hf/` (checkpoint), and `manifest.json` — derived from
`dataset_manifest.json`, recording `trained_on`, `holdout`, source
checksums, retrieval settings, training hyperparameters, the validation
history, output file checksums, and the trainer commit (`<sha>` or
`<sha>-dirty`; train from a clean, committed tree). The eval harness
reads this manifest and refuses to run when the split under evaluation
appears in `trained_on` (`hoptrace eval … --selection rerank --rerank-model <dir>`).

Cost: 1 epoch over ~47k queries (~600k sampled rows) is ~3 h on Apple
Silicon MPS. Smoke-test the whole path first with `--max-queries 300
--eval-every 50 --out /tmp/smoke` (~2 min).

## 3. Bundle for distribution (ADR 0012)

```
uv run python training/bundle_artifact.py $HOPTRACE_DATA_DIR/models/hoptrace-rerank-minilm-l6
```

Copies `model_int8.onnx`, `tokenizer.json` and `manifest.json` into
`models/hoptrace-rerank-minilm-l6/` in the repository (committed; ~24 MB)
and prints the `ModelSpec` registry entry with every file's sha256 for
`src/hoptrace/rerank.py`. `model.onnx` (fp32) is not committed: attach
it — and, for wheel installs, the three bundled files — to the GitHub
Release the registry URLs name. A clone then runs `--rerank` offline on
the int8 graph; `--rerank-precision fp32` downloads the release asset
under its pin. Each retrain adds ~24 MB to git history.
