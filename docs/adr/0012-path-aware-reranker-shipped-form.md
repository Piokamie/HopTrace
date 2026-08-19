---
status: accepted
date: 2026-08-18
---

# 12. Path-Aware Reranker: Shipped Form (Bundled int8 Default, Pinned Downloads, No Marginal-Completion)

## Context

ADR 0010 chose the v2 learned component: a cross-encoder over (query,
candidate, route-context) that rescores the top-N of the deterministic
pool. It also fixed how the artifact would be distributed — "hosted on a
model hub, downloaded on first use with a recorded checksum, never
committed to git" — and left open whether a marginal-completion variant
of greedy selection over learned scores would ship.

At ship time three of those clauses no longer describe the artifact:

- The fine-tuned model was not uploaded anywhere. The registry in
  `rerank.py` held only the zero-shot base
  (`cross-encoder/ms-marco-MiniLM-L6-v2`), so `--rerank` with no
  `--rerank-model` used an untrained model, and every fine-tuned number
  in `docs/results.md` required the reader to run `training/` (~4 h of
  dataset build over a 5.2M-passage index plus ~3 h of fine-tuning).
- The training pipeline generated candidate pools with 50 BM25 seeds
  (`retrieve(k=50)` → `seed_depth = max(20, 50)`), while inference seeds
  with 20 and rescores `pool_order[:50]`. The model was trained on a
  differently shaped pool than the one it scores. Detected in the
  2026-08-18 review; the model is retrained after the fix.
- The marginal-completion selection variant was never measured: the
  reranker reaches 95%+ of the pool-oracle ceiling for the candidates it
  scores, so the remaining headroom is in the rescoring budget and
  expansion coverage, not in the selection rule over learned scores.

Sizes that matter for distribution: `model_int8.onnx` 23.0 MB,
`tokenizer.json` 0.7 MB, `manifest.json` 3 KB, `model.onnx` (fp32)
91.0 MB. GitHub warns above 50 MB per file in git and blocks above
100 MB; release assets have no such limit. Every retrain adds another
copy of whatever is committed to git history.

The two evaluated graphs differ by at most 0.005 on any measured
metric and int8 is ~2× faster on CPU (`docs/results.md`, int8 section).

## Decision

This supersedes ADR 0010 in full. Points 1–3 of ADR 0010 are restated
unchanged; points 4 and 5 change; two clauses are added.

1. **Admission stays deterministic** (unchanged). The ranker rescores
   the deterministically generated pool — only `pool_order[:max(top_n,
   k)]` in interleave order — and can never reach outside it. The
   candidate window the ranker sees is defined once, in
   `Retriever.candidate_window()`, and the training-data builder uses
   that same method; a train/serve pool skew is a bug, not a setting.
2. **Interleave remains the shipped default selection** (unchanged).
   Reranking is opt-in (`selection="rerank"`, `--rerank`, MCP
   `rerank=true`) until the harness earns the flip.
3. **Inference is ONNX Runtime, not torch** (unchanged); MiniLM-class
   backbone initialised from `cross-encoder/ms-marco-MiniLM-L6-v2`;
   training code is dev-only.
4. **Distribution — changed.** The fine-tuned artifact ships in two
   parts:
   - `models/hoptrace-rerank-minilm-l6/{model_int8.onnx, tokenizer.json,
     manifest.json}` is **committed to the repository**, so a clone runs
     the reranker offline and the manifest (ADR 0011) travels with the
     graph.
   - `model.onnx` (fp32) is attached to a GitHub Release
     (`https://github.com/Piokamie/HopTrace/releases/download/<tag>/`)
     and downloaded on first use.
   - The registry pins the **sha256 of every file** for both models
     (fine-tuned and zero-shot base) and verifies it on download and on
     every load of a registry model. Record-on-first-download remains
     only for datasets.
   - When the package is installed without the repository (wheel), the
     bundled files are fetched from the same release with the same pins.
5. **No marginal-completion variant — changed.** Selection over learned
   scores is a plain sort (ties on chunk id). The variant moves to the
   roadmap, ordered behind the rescoring-budget sweep that the pool
   oracle ranks first.
6. **The bundled fine-tuned model is the `--rerank` default — new.**
   `rerank_model=None` resolves `$HOPTRACE_RERANK_MODEL`, then
   `hoptrace-rerank-minilm-l6`. The zero-shot base stays in the registry
   as `ms-marco-minilm-l6-v2` and is the control row in every table.
7. **Precision follows the artifact — new.** `rerank_precision=None`
   selects the graph the artifact ships (fine-tuned registry model:
   int8; base: fp32; local directory: fp32 if `model.onnx` exists, else
   int8). `--rerank-precision {fp32,int8}` forces one; a missing graph is
   a hard error before any download. `--rerank-int8` is removed. Headline
   tables report fp32 and int8 side by side.

Alternatives rejected:

- **Model hub (Hugging Face) as in ADR 0010**: needs a second account
  and upload step outside the repository; a clone would not run offline.
  Kept as a possible mirror, not the source of truth.
- **Git LFS**: bandwidth-quota'd on the free tier; every clone pays.
- **Both graphs in git**: 114 MB per retrain in history; the fp32 file
  sits above GitHub's warning threshold.
- **Keep the zero-shot base as the default**: ships a model that the
  measured tables show losing to the fine-tuned one on every all-recall
  cell; the bundled artifact would exist and not be used.
- **Fixed default precision (`fp32`)**: the bundled artifact would then
  fail by default (no fp32 in git) or force a 91 MB download to run the
  walkthrough.

## Consequences

- A fresh clone with `hoptrace[rerank]` runs the fine-tuned reranker with
  no network access; the manifest tripwire (ADR 0011) applies to the
  default model exactly as to any other, because it ships beside it.
- The repository carries ~24 MB of binary and grows by that on each
  retrain; retrains are therefore deliberate and documented in
  `training/README.md`, not routine.
- Publishing a new artifact is a three-step ritual — retrain, run
  `training/bundle_artifact.py` (copies the bundled files, prints the
  registry snippet with hashes), attach `model.onnx` to the release —
  and a mismatch between any of them fails at load time rather than
  silently serving the wrong graph.
- Zero-shot numbers require an explicit `--rerank-model
  ms-marco-minilm-l6-v2`; the eval commands in `docs/results.md` state
  the model and precision for every row.
- Latency defaults change: `--rerank` now means the int8 graph (~2×
  faster than the fp32 numbers ADR 0010 budgeted for).
- ADR 0010's audit story — the ranker cannot reach outside the pool — is
  now enforced by construction on the training side too, since both
  sides call `candidate_window()`.
