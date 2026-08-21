# Results

> v1 (deterministic retrieval) and v2 (path-aware learned reranker)
> measured on three corpus protocols. Every number is from a full
> (non-`--limit`) run unless marked sampled; retrieval
> configuration, corpus protocol, and the rerank model's training
> provenance are embedded in each report (`hoptrace eval … --json
> <file>` writes it; the text report prints the same notes).

## Protocol

- **Corpus construction** — distinct settings, never mixed in one
  comparison:
  - *hipporag-protocol* (`--dataset hipporag-musique|hipporag-2wiki`):
    the reportable setting. HippoRAG's published 1,000-question
    validation samples over the published candidate corpora (MuSiQue
    11,656 passages, 2Wiki 6,119), using the released files rather than a
    resample — a different 1,000 questions would not be comparable.
    Pinned to HippoRAG commit `b144c46`. Metrics are R@2/R@5 and
    "all-recall", the published name for all-gold@k.
  - *corpus-scale* (BEIR HotpotQA, 5,233,329 passages): the full
    Wikipedia corpus. Where the BM25 reproduction gate lives.
  - *pooled-all-splits* (`--corpus-pool all`): haystack pools train+dev
    paragraphs, queries and gold stay dev-only — the IRCoT-comparable
    protocol for MuSiQue/2Wiki. Train paragraphs are corpus mass, not
    training labels.
  - *pooled-dev* (`--corpus-pool dev`, default): haystack pools dev
    paragraphs only — MuSiQue 21,100, 2Wiki 56,687. A development
    setting: comparisons within it are valid (identical corpus,
    matched budget), but its haystack is 4–5× smaller than the
    published protocol (MuSiQue: 84,559 train-only, 101,962 train+dev),
    so its absolute numbers must never be set beside published
    open-domain figures. Every report prints this caveat itself.
  - *distractor* (appendix only): per-question corpora of 10–20
    paragraphs. Reranking, not retrieval; never a headline.
- **Budget-matched**: every compared system returns the same k; candidates
  examined and wall-clock latency (median/p95, warm; deterministic rows
  single-threaded, rerank rows at the shipped 4 ONNX intra-op threads with
  a single-threaded tokenizer — each report states its own condition) are
  reported next to recall.
- **Hit rule**: eval indexing is identity-preserving — one source paragraph
  is one chunk, indexed as `"title text"` (the Anserini *flat* condition) —
  and a retrieved chunk is a hit iff it is a gold paragraph.
- **Analyzer**: `english` (Lucene-style stopwords + Porter stemming).
  BM25 k1=0.9, b=0.4 (the published baseline configuration).
- **Retrieval** (HopTrace rows): propagated path scoring + per-ring interleave
  (ADR 0005–0008), hops as marked, beams 8/16/64, hub_df_ratio 0.001.
  Rerank rows rescore the top-50 of that pool (ADR 0012); the model and
  graph precision are named per table.
- **Stratification**: metrics are additionally split by the
  answerability-based stratum (ADR 0004 hybrid rule): *effectively
  single-hop* (floor top-10 contains answer span AND ≥1 gold),
  *effectively multi-hop*, and *excluded* (yes/no + comparison). One
  aggregate number cannot summarize a corpus that is one-third
  single-hop.

## Validation gate (BEIR HotpotQA) — PASSED

| System | nDCG@10 | Recall@100 | Source |
|---|---|---|---|
| BM25 flat (published) | 0.633 | 0.796 | [Pyserini BEIR matrix](https://castorini.github.io/pyserini/2cr/beir.html) |
| BM25 multifield (published) | 0.603 | 0.740 | BEIR paper ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)) |
| HopTrace BM25 floor (flat) | 0.6352 | 0.7988 | `hoptrace eval --dataset beir-hotpotqa --gate` |

Deltas +0.0022 / +0.003, tolerance ±0.02. Full test split, 7,405 queries,
5,233,329 passages.

## Headline: comparison with published systems (HippoRAG protocol)

All rows use the 1,000-question samples, corpora and metrics published
with [Gutiérrez et al., NeurIPS
2024](https://arxiv.org/abs/2405.14831). Rows marked *(paper)* are quoted
from its Table 2; rows marked *(measured here)* are from
`hoptrace eval --dataset hipporag-*`.

**MuSiQue** (11,656 passages, 1,000 questions):

| System | R@2 | R@5 |
|---|---|---|
| BM25 (paper) | 32.3 | 41.2 |
| BM25 floor (measured here) | 32.4 | 43.0 |
| ColBERTv2 (paper) | 37.9 | 49.2 |
| Proposition (paper) | 37.6 | 49.3 |
| HippoRAG, best variant (paper) | 41.0 | 52.1 |
| HopTrace 2hop + zero-shot rerank (measured here) | 42.4 | 57.8 |
| HopTrace 2hop + fine-tuned rerank (measured here) | 54.2 | 66.1 |

**2WikiMultihopQA** (6,119 passages, 1,000 questions) — the transfer
holdout, absent from training entirely (ADR 0011):

| System | R@2 | R@5 |
|---|---|---|
| BM25 (paper) | 51.8 | 61.9 |
| BM25 floor (measured here) | 57.1 | 67.5 |
| ColBERTv2 (paper) | 59.2 | 68.2 |
| HippoRAG, best variant (paper) | 71.5 | 89.5 |
| HopTrace 2hop + zero-shot rerank (measured here) | 58.1 | 76.5 |
| HopTrace 2hop + fine-tuned rerank (measured here) | 73.8 | 85.9 |

Two differences affect how the rows compare. The two BM25 rows are not
the same system: they agree to 0.1 points on MuSiQue (32.4 vs 32.3) but
differ by 5.3 on 2Wiki (57.1 vs 51.8), because this implementation indexes
`"title text"` (the Anserini flat condition), which favours
entity-centric titles. HippoRAG also builds its knowledge graph with an
LLM (OpenIE over the corpus); HopTrace uses entity expansion plus a CPU
cross-encoder (91 MB fp32, 23 MB int8).

Fine-tuned rows use the bundled reranker (`models/hoptrace-rerank-minilm-l6`,
fp32 graph). Its training set contains no evaluation-gold passage at all
(the leak audit below measures 0 of 2,648); the ablation model trained
with them kept appears only in the exclusion-ablation table.

### all-recall, @5

Fraction of queries for which *every* supporting passage is retrieved:

| | floor | 2hop | + zero-shot | + fine-tuned |
|---|---|---|---|---|
| MuSiQue | 9.7 | 13.1 | 27.2 | 33.4 |
| 2Wiki | 35.9 | 41.7 | 49.6 | 66.7 |

## Training-to-evaluation leak audit

The reranker trains on MuSiQue and HotpotQA *train* splits and is
evaluated on dev questions. Wikipedia paragraphs recur across questions,
so separate questions do not imply separate passages. The builder
therefore excludes every evaluation-gold passage from training — as the
scored candidate and as the route-context parent, matched on
whitespace-normalized text across both evaluation sets — and
`training/audit_exposure.py` verifies the result over every training
source:

| check (MuSiQue evaluation set; 2Wiki never enters training at all) | result |
|---|---|
| eval questions also in the training set | 0 / 1,000 |
| eval-corpus passages seen during training (candidate or parent) | 2,159 / 11,655¹ (18.5%) |
| eval gold passages seen during training | 0 / 2,648 (0.0%) |
| eval gold passages scored as a positive | 0 / 2,648 (0.0%) |

¹ unique passages after whitespace normalization; the corpus file holds
11,656 entries, one a duplicate.

The 18.5% of corpus passages that do occur in training are distractors
shared with train questions, none of them gold for any evaluation
question. Every evaluation question sits in the *none* exposure bucket,
so no exposure-split analysis applies to the bundled model; what the
exclusion costs is measured directly below.

### Exclusion ablation

The ablation baseline is the same trainer run on the same retrieval
pass with the exclusion switched off (`--keep-eval-gold`): its training
set contains the 3,795 evaluation-gold passages as unlabeled pool
members and negatives. The eval harness marks it NOT PUBLISHABLE.

| MuSiQue | with eval gold | excluded (bundled) | Δ |
|---|---|---|---|
| R@2 | 54.32 | 54.15 | −0.17 |
| R@5 | 65.70 | 66.09 | +0.39 |
| all-recall@5 | 33.20 | 33.40 | +0.20 |

| 2Wiki (holdout) | with eval gold | excluded (bundled) | Δ |
|---|---|---|---|
| R@2 | 73.60 | 73.80 | +0.20 |
| R@5 | 85.85 | 85.90 | +0.05 |
| all-recall@5 | 66.80 | 66.70 | −0.10 |

Movement is ≤0.4 points with mixed signs on both datasets: training
exposure to evaluation gold contributes nothing measurable, and the
shipped model has none by construction.

Reproduce the counts with
`uv run --extra eval python training/audit_exposure.py` (writes
`$HOPTRACE_DATA_DIR/eval/hipporag-musique-exposure.json` and a report
beside it). The ablation pair is one retrieval pass:
`training/build_dataset.py --keep-eval-gold` then `--from` that
directory (`training/README.md`).

## What the learned ranker changed

Full grid at the HippoRAG protocol, all-recall@k (`ag`) beside recall:

**MuSiQue**

| System | R@2 | ag@2 | R@5 | ag@5 | R@20 | ag@20 | median |
|---|---|---|---|---|---|---|---|
| floor | 32.4 | 3.9 | 43.0 | 9.7 | 58.4 | 23.7 | 0.3 ms |
| 2hop interleave (v1) | 28.6 | 5.2 | 42.2 | 13.1 | 60.8 | 32.0 | 9 ms |
| + zero-shot, path OFF | 43.4 | 9.5 | 56.2 | 21.8 | 69.6 | 39.5 | 828 ms |
| + zero-shot, path ON | 42.4 | 12.3 | 57.8 | 27.2 | 72.3 | 44.2 | 872 ms |
| + fine-tuned, path OFF | 49.7 | 11.5 | 61.5 | 26.4 | 72.8 | 45.0 | 889 ms |
| + fine-tuned (int8) | 53.4 | 18.1 | 66.3 | 33.9 | 74.0 | 46.6 | 524 ms |
| + fine-tuned (fp32) | 54.2 | 18.8 | 66.1 | 33.4 | 74.0 | 46.8 | 974 ms |

**2WikiMultihopQA** (holdout)

| System | R@2 | ag@2 | R@5 | ag@5 | R@20 | ag@20 | median |
|---|---|---|---|---|---|---|---|
| floor | 57.1 | 23.5 | 67.5 | 35.9 | 74.2 | 46.5 | 0.2 ms |
| 2hop interleave (v1) | 46.9 | 8.9 | 71.2 | 41.7 | 86.2 | 66.6 | 6 ms |
| + zero-shot, path OFF | 63.6 | 29.5 | 72.2 | 42.2 | 86.2 | 67.2 | 747 ms |
| + zero-shot, path ON | 58.1 | 26.1 | 76.5 | 49.6 | 90.2 | 75.8 | 795 ms |
| + fine-tuned, path OFF | 67.7 | 35.5 | 75.1 | 47.3 | 87.6 | 70.2 | 1027 ms |
| + fine-tuned (int8) | 73.7 | 47.5 | 85.7 | 66.3 | 90.7 | 76.8 | 585 ms |
| + fine-tuned (fp32) | 73.8 | 47.7 | 85.9 | 66.7 | 90.8 | 77.0 | 953 ms |

### Route context

The zero-shot model trained on neither input format, so its path-ON vs
path-OFF rows isolate the information in route context from any
train/test mismatch.

- Route context costs raw R@2 (MuSiQue 43.4 → 42.4; 2Wiki 63.6 → 58.1)
  and buys complete evidence (all-recall@5 21.8 → 27.2; 42.2 → 49.6).
  Bare text retrieves individually-relevant passages; route context
  retrieves the bridge partner that completes the set.
- Fine-tuned without route context loses to zero-shot with it on
  all-recall@5 (MuSiQue 26.4 vs 27.2).
- On the holdout the gap is larger: 2Wiki all-recall@5 is 47.3 with
  route context stripped, 66.7 with it.

### int8 quantization: 1.6–1.9× faster, ≤0.008 cost

Dynamic int8 nearly halves latency (524 vs 974 ms median on MuSiQue).
The largest deviation across both datasets is 0.008 (MuSiQue
all-recall@10, 42.0 → 41.2; in the grid above the largest is R@2,
54.2 → 53.4), and on some metrics int8 is marginally ahead (MuSiQue
all-recall@5 33.9 vs 33.4). Select with `--rerank-precision int8`
(`RetrievalConfig(rerank_precision="int8")`); it is what `--rerank`
loads by default from the bundled model, and the headline rows are fp32.

### Pool oracle

The ceiling for any selection over the generated candidates. It is
selection-policy independent, so it bounds every row above. MuSiQue,
HippoRAG protocol:

| level | ag@5 | ag@20 | gap above |
|---|---|---|---|
| fine-tuned achieved | 33.4 | 46.8 | — |
| ceiling over the top-50 scored | 48.7 | 48.7 | model headroom |
| ceiling over the full pool | 59.8 | 59.8 | top-N truncation |
| perfect | 100 | 100 | expansion never reached it |

At k=20 the reranker reaches 96.1% of the ceiling for the candidates it
scores (98.7% on recall@20). The remaining headroom is in the top-N
budget (`rerank_top_n`, linear in latency) and in expansion coverage,
which the miss taxonomy attributes to `hop_bound` and `seed_alias`.

## Calibration: the instrument recovers the known dataset properties

`multihop_fraction` is answerability-based (hybrid rule), not
annotation-based — all three datasets annotate ~100% of questions as
multi-hop; the literature's finding is about *sufficiency*.

| Dataset | Role | Annotated multi-hop | Effective multi-hop | Gap |
|---|---|---|---|---|
| MuSiQue | known positive | 1.000 | 0.654 | +0.346 |
| 2WikiMultihopQA | discrimination | 1.000 | 0.478 | +0.522 |
| HotpotQA (BEIR) | known negative | 1.000 | 0.377 | +0.623 |

The order matches the datasets' documented properties: MuSiQue, built to
defeat shortcuts, measures most-genuinely multi-hop; 62% of HotpotQA's
"multi-hop" questions are answerable from the floor's top-10.

## v1 deterministic retrieval, by stratum (development setting, k=20)

> Pooled-dev numbers: the development setting defined above, on a
> haystack 4–5× smaller than the published protocol. Valid within
> themselves (identical corpus, matched budget), not comparable to the
> HippoRAG tables above or to published figures.

**MuSiQue** (pooled dev, 21,100 chunks, 2,417 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.555 | 0.210 | 0.055 | 0.502 | 0.4 ms |
| HopTrace@1hop | 0.588 | 0.301 | — | — | 3.5 ms |
| HopTrace@2hop | 0.572 | 0.279 | 0.167 | 0.492 | 6.5 ms |

**2WikiMultihopQA** (pooled dev, 56,687 chunks, 12,576 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.708 | 0.415 | 0.071 | 0.422 | 0.9 ms |
| HopTrace@1hop | 0.801 | 0.575 | — | — | 4.2 ms |
| HopTrace@2hop | 0.774 | 0.529 | 0.342 | 0.478 | 8.6 ms |

**HotpotQA** (BEIR full corpus, 5,233,329 chunks, 7,405 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.710 | 0.473 | 0.072 | 0.693 | 66.8 ms |
| HopTrace@1hop | 0.701 | 0.479 | — | — | ~150 ms |
| HopTrace@2hop | 0.690 | 0.457 | 0.118 | 0.651 | 224.9 ms |

Readings, in calibration order:

- On MuSiQue and 2Wiki the floor completes no evidence at k=10 in the
  effectively-multi-hop stratum (ag@10 = 0.000 on all three datasets, by
  the stratum definition); hops raise complete-evidence rates at k=20.
  2Wiki gains most — its bridges are article titles, which the
  title-as-entity index encodes directly.
- On HotpotQA, hops cost in aggregate (r@20 0.710→0.690) and add +4.6
  points in the genuine-multi-hop stratum.
- Early-k recall drops on effectively-single-hop questions everywhere, as
  hop slots displace redundant-but-gold seeds. At k=20 the single-hop
  stratum is down 1 point on MuSiQue and 4 on HotpotQA; on 2Wiki it is
  up 5.6, since the title-as-entity bridges pay even there.
- hop-1 beats hop-2 in aggregate on every dataset: annotated chains here
  are mostly one bridge, and ring-2's reserved slots return less than
  they cost. `hops=1` is the reasonable default when the bracket shows
  chains are short.

## Diagnostics

**Miss breakdown at hops=2, k=20** (share of missed gold):

| Dataset | ranking | hop_bound | seed_alias | extraction | Pool recall |
|---|---|---|---|---|---|
| MuSiQue | 47.1% | 31.5% | 21.4% | 0% | 0.873 |
| 2Wiki | 62.3% | 35.7% | 2.0% | 0% | 0.921 |
| HotpotQA | 53.4% | 41.8% | 4.8% | 0% | 0.870 |

The pool holds 87–92% of gold everywhere; ranking (selection) is the
dominant miss class everywhere. Extraction misses are zero on these
corpora. Seed/alias
misses are corpus-dependent (21% on MuSiQue's varied surface forms, 2% on
encyclopedic 2Wiki).

**Displacement audit at k=20** (gold in hop-held slots vs gold evicted):

| Dataset | hop-slot gold | evicted-seed gold | Net gold |
|---|---|---|---|
| MuSiQue | 1.7% | 1.8% | −17 (break-even) |
| 2Wiki | 2.2% | 0.7% | +1,932 |
| HotpotQA | 0.6% | 1.0% | −295 |

The interleave's value is dataset-dependent: positive where the
calibration measures multi-hop as entity-bridged, negative elsewhere.

**Beam sweep (MuSiQue, hops=2)** — attacking hop-bound misses:

| Beams (ent/chunks/frontier) | ag@20 | hop_bound misses | ranking misses | Net gold |
|---|---|---|---|---|
| 8/16/64 (default) | 0.2793 | 933 | 1,395 | −17 |
| 8/32/128 | 0.2784 | 744 | 1,699 | −20 |
| 16/32/128 | 0.2855 | 722 | 1,686 | +25 |

Wider beams cut hop-bound misses 23% — and the misses migrate into the
ranking bucket almost 1:1. Candidate generation is not the binding
constraint; selection is. Wider beams are headroom for a better ranker,
not for v1.

Hub-entity cutoff: across hub_df_ratio ∈ {0.0005, 0.001, 0.01} on MuSiQue
(full) recall@20 moves within 0.002 (0.5709 / 0.5723 / 0.5709) and
all-gold@20 within 0.007 (0.2801 / 0.2793 / 0.2731); on BEIR (1,000-query
sample) the numbers are identical. Pool precision is within 0.0002 with
the specificity filter on or off, since the beam's IDF ordering already
excludes hubs. The cutoff stays as a safety bound but does little work at
these settings.

## Selection-policy evolution

Every selection policy tried during v1, measured on full MuSiQue at
hops=2 (development setting):

| Policy | ag@20 | Note |
|---|---|---|
| BM25 floor | 0.210 | |
| additive scorer (ADR 0005 context) | 0.276 | hop-2 priced by query similarity it cannot have |
| propagated + 2-population interleave | 0.300 | aggregate best, but ring-2 structurally starved |
| propagated + per-ring interleave (default) | 0.279 | ring-2 can surface; stratified gains intact |
| greedy submodular (ADR 0007, experimental) | 0.099 | failed gold-aware validation — see below |

Two findings:

1. Aggregate vs strata: the propagated scorer's aggregate numbers look
   mixed; stratified, the single-hop stratum pays at early k and the
   multi-hop stratum gains at all-gold.
2. The submodular trilemma (ADR 0008): an unbounded coverage universe
   floods (candidates mint their own novelty; junk routes outnumber gold
   routes); a query-anchored universe starves multi-hop targets, since
   containing no query content is what makes a target multi-hop; every
   interpolation carries a constant only settable by peeking at the eval.
   Route quality is therefore statistical, which is what the v2 ranker
   learns.

## Scaling: the same systems across three corpus sizes

MuSiQue dev queries, all-recall@20, as the haystack grows. All three
columns use the bundled model, so the comparison is one model across
corpus sizes. Strata are recomputed per corpus — the
answerability rule is corpus-dependent, and the effective-multi stratum
grows from 1,580 to 1,681 queries at 101,962 — so strata comparisons
hold within a column, not across.

| System | 11,656 (HippoRAG) | 21,100 (pooled-dev) | 101,962 (all-splits) |
|---|---|---|---|
| floor | 23.7 | 21.0 | 16.0 |
| 2hop interleave | 32.0 | 27.9 | 20.9 |
| + zero-shot rerank | 44.2 | 38.6 | 30.5 |
| + fine-tuned rerank | 46.8 | 41.0 | 32.1 |

Every system degrades as the corpus grows. Across the range the ordering
does not invert (floor < hops < zero-shot < fine-tuned at every scale).
On the effective-multi stratum, floor all-recall@20 is 0.055 at 21,100
and 0.030 at 101,962; with hops it is 0.167 and 0.111.

The 101,962 setting also carries the route-context control (zero-shot
path ON vs OFF: all-recall@20 30.5 vs 25.4).

## Appendix: distractor sanity check

Per-question corpora of 10–20 paragraphs — reranking, not retrieval.
MuSiQue 500-question sample: floor r@2 0.386, r@5 0.562 (not saturated —
MuSiQue's distractors are adversarial — but the setting cannot exercise
corpus-scale retrieval and stays out of the headline).

## Reproduction

Regenerating the v1 rows reproduces the gate (nDCG@10 0.6352) and every
hop metric across MuSiQue, 2Wiki and BEIR to within 0.002 of the tables
above.

The headline tables (`--extra eval --extra rerank`):

```
hoptrace eval --dataset hipporag-musique --hops 0 --diagnostics                              # floor
hoptrace eval --dataset hipporag-musique --hops 2 --diagnostics                              # 2hop interleave (v1)
hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-precision fp32 --diagnostics   # fine-tuned (bundled model)
hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-precision int8                 # its int8 row
hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-precision fp32 --no-path-context   # fine-tuned, route OFF
hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-model ms-marco-minilm-l6-v2   # zero-shot, route ON
hoptrace eval --dataset hipporag-musique --hops 2 --selection rerank --rerank-model ms-marco-minilm-l6-v2 --no-path-context   # zero-shot, route OFF
```

The same seven for `--dataset hipporag-2wiki`. The exclusion-ablation
baseline is a model built with `--keep-eval-gold` passed as
`--rerank-model <dir>`; the harness prints NOT PUBLISHABLE beside it.

The gate, the v1 development numbers, and the scaling arm:

```
hoptrace eval --dataset beir-hotpotqa --gate
hoptrace eval --dataset musique --hops 2 --diagnostics
hoptrace eval --dataset 2wiki --hops 2 --diagnostics --pool-ablation 500
hoptrace eval --dataset beir-hotpotqa --hops 2 --diagnostics --pool-ablation 300
hoptrace eval --dataset musique --corpus-pool all --hops 2 --diagnostics
hoptrace eval --dataset musique --hops 2 --diagnostics --beam-entities 16 --beam-chunks 32 --frontier-chunks 128
```

The reranker artifact is reproduced by `training/build_dataset.py` then
`training/train_reranker.py` (see `training/README.md`); the eval harness
refuses to score any split the model's `manifest.json` lists as training
data, any manifest that does not describe the graphs beside it, and any
dirty-tree artifact.

BEIR index: 5,233,329 passages → 13.3M entities, 50.4M mentions, 8.4 GB
SQLite, 1.14 h single-threaded build on a laptop-class CPU. Datasets download
to `$HOPTRACE_DATA_DIR/datasets/` with recorded checksums; never committed.
Rerank latency is measured at the shipped default of 4 ONNX threads,
`rerank_top_n=50`, on CPU; the deterministic rows are single-threaded.
Fine-tuned rows were measured 2026-08-21 with the bundled artifact;
floor, deterministic and zero-shot rows carry over from the prior
campaign — those models and the deterministic pipeline are unchanged, so
their metrics are reproducible bit-for-bit, and their latency medians
were measured under the same thread settings on the same machine.

## Dataset attributions

- HotpotQA (CC BY-SA 4.0), via the BEIR packaging (Thakur et al. 2021)
- MuSiQue (CC BY 4.0), MuSiQue-Ans dev split
- 2WikiMultihopQA (Apache 2.0), dev split
- HippoRAG sampled questions + candidate corpora (MIT), Gutiérrez et al.,
  NeurIPS 2024 ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831));
  pinned to repo commit `b144c46`
- Reranker backbone: `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0)
- Published baselines: Pyserini BEIR reproduction matrix; BEIR paper;
  HippoRAG Table 2 for the comparison rows
