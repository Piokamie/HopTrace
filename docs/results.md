# Results

> **Status: Phase 3 complete.** BM25 floors, HopTrace hop runs, stratified
> diagnostics, calibration, displacement audits, beam sweep, and hub
> ablation measured on all three datasets. Every number is from a full
> (non-`--limit`) run unless marked sampled. Retrieval configuration per
> run is embedded in each report file (`.hoptrace/eval/reports/`).

## Protocol

- **Budget-matched**: every compared system returns the same k; candidates
  examined and wall-clock latency (median/p95, single-threaded, warm) are
  reported next to recall.
- **Hit rule**: eval indexing is identity-preserving — one source paragraph
  is one chunk, indexed as `"title text"` (the Anserini *flat* condition) —
  and a retrieved chunk is a hit iff it is a gold paragraph.
- **Analyzer**: `english` (Lucene-style stopwords + Porter stemming).
  BM25 k1=0.9, b=0.4 (the published baseline configuration).
- **Retrieval** (HopTrace rows): propagated path scoring + per-ring interleave
  (ADR 0005–0008), hops as marked, beams 8/16/64, hub_df_ratio 0.001.
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
| **HopTrace BM25 floor (flat)** | **0.6352** | **0.7988** | `hoptrace eval --dataset beir-hotpotqa --gate` |

Deltas +0.0022 / +0.003, tolerance ±0.02. Full test split, 7,405 queries,
5,233,329 passages. A floor implemented here is a floor checked against
someone else's.

## Calibration: the instrument recovers the known dataset properties

`multihop_fraction` is answerability-based (hybrid rule), not
annotation-based — all three datasets annotate ~100% of questions as
multi-hop; the literature's finding is about *sufficiency*.

| Dataset | Role | Annotated multi-hop | **Effective multi-hop** | Gap |
|---|---|---|---|---|
| MuSiQue | known positive | 1.000 | **0.654** | +0.346 |
| 2WikiMultihopQA | discrimination | 1.000 | **0.478** | +0.522 |
| HotpotQA (BEIR) | known negative | 1.000 | **0.377** | +0.623 |

The ladder lands in the documented order: MuSiQue (constructed to defeat
shortcuts) measures most-genuinely multi-hop; HotpotQA's documented
shortcut-proneness is independently recovered (62% of its "multi-hop"
questions are effectively answerable from the floor's top-10). The
annotated-vs-effective gap — invisible to annotation-based definitions —
is the instrument's headline observation. This validates the measurement,
which is the product.

## Headline: floor vs. HopTrace, aggregate and by stratum (k=20)

**MuSiQue** (pooled dev, 21,100 chunks, 2,417 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.555 | 0.210 | 0.055 | 0.502 | 0.4 ms |
| HopTrace@1hop | 0.588 | **0.301** | — | — | 3.5 ms |
| HopTrace@2hop | 0.572 | 0.279 | **0.167 (3.0×)** | 0.492 | 6.5 ms |

**2WikiMultihopQA** (pooled dev, 56,687 chunks, 12,576 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.708 | 0.415 | 0.071 | 0.422 | 1.0 ms |
| HopTrace@1hop | **0.801** | **0.575** | — | — | ~6 ms |
| HopTrace@2hop | 0.774 | 0.529 | **0.342 (4.8×)** | 0.478 | 8.6 ms |

**HotpotQA** (BEIR full corpus, 5,233,329 chunks, 7,405 q):

| System | r@20 | ag@20 | ag@20 eff-multi | ag@20 eff-single | Latency med |
|---|---|---|---|---|---|
| floor | 0.710 | 0.473 | 0.072 | 0.693 | 66.8 ms |
| HopTrace@1hop | 0.701 | 0.479 | — | — | ~150 ms |
| HopTrace@2hop | 0.690 | 0.457 | 0.118 (1.6×) | 0.651 | 224.9 ms |

Readings, in calibration order:

- **Where the diagnostic says multi-hop is real (MuSiQue, 2Wiki), hops
  pay**: in the effectively-multi-hop stratum the floor cannot complete
  evidence at k=10 at all (ag@10 = 0.000 on all three datasets — the
  stratum definition at work) and HopTrace lifts complete-evidence rates 3–5×
  at k=20. 2Wiki gains most: its bridges are article titles, exactly what
  the title-as-entity index encodes.
- **Where the diagnostic says the corpus is shortcut-prone (HotpotQA),
  hops cost in aggregate** (r@20 0.710→0.690) and buy only +4.6 points in
  the genuine-multi-hop stratum. **This is the diagnostic working
  correctly, not a failure**: a tool that says "don't use me here" on the
  known-negative and "use me" on the known-positive is the credibility
  claim of the whole project. The per-corpus bracket exists to make this
  call before anyone pays for hop retrieval.
- **The interleave tax is real and stratified**: early-k recall drops on
  effectively-single-hop questions everywhere (hop slots displace
  redundant-but-gold seeds). At k=20 the single-hop stratum is nearly
  unharmed (−1 to −4 points).
- **hop-1 beats hop-2 in aggregate on every dataset** — annotated chains
  in these benchmarks are mostly one bridge, and ring-2's reserved slots
  return less than they cost. `hops=1` is the sensible per-corpus default
  when the bracket shows chains are short; hop-2 remains available.

## Diagnostics

**Miss breakdown at hops=2, k=20** (share of missed gold):

| Dataset | ranking | hop_bound | seed_alias | extraction | Pool recall |
|---|---|---|---|---|---|
| MuSiQue | 47.1% | 31.5% | 22.1% | 0% | 0.873 |
| 2Wiki | 62.3% | 35.7% | 2.0% | 0% | 0.921 |
| HotpotQA | 53.4% | 41.8% | 4.8% | 0% | 0.870 |

The pool holds 87–92% of gold everywhere; ranking (selection) is the
dominant miss class everywhere. Extraction misses are zero — the
rule-based extractor is not the bottleneck on these corpora. Seed/alias
misses are corpus-dependent (22% on MuSiQue's varied surface forms, 2% on
encyclopedic 2Wiki).

**Displacement audit at k=20** (gold in hop-held slots vs gold evicted):

| Dataset | hop-slot gold | evicted-seed gold | Net gold |
|---|---|---|---|
| MuSiQue | 1.7% | 1.8% | −17 (break-even) |
| 2Wiki | 2.2% | 0.7% | **+1,932** |
| HotpotQA | 0.6% | 1.0% | −295 |

The interleave's value is dataset-dependent and now priced: strongly
positive exactly where the calibration says multi-hop is entity-bridged.

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

**Hub-entity cutoff is decorative** (as predicted in review): recall
identical to 4 decimals across hub_df_ratio ∈ {0.0005, 0.001, 0.01} on
MuSiQue (full) and BEIR (1,000-query sample); pool precision identical
with the specificity filter on or off (the beam's IDF ordering already
excludes hubs). The cutoff stays as a safety bound; the data says the IDF
ranking does the work.

## Selection-policy evolution (the bracket's live catches)

All Phase 3 selection policies, measured on full MuSiQue at hops=2
(archive: `.hoptrace/eval/reports/phase3-additive-scorer/` and `*-sub.*`):

| Policy | ag@20 | Note |
|---|---|---|
| BM25 floor | 0.210 | |
| additive scorer (ADR 0005 context) | 0.276 | hop-2 priced by query similarity it cannot have |
| propagated + 2-population interleave | 0.300 | aggregate best, but ring-2 structurally starved |
| propagated + per-ring interleave (**default**) | 0.279 | ring-2 can surface; stratified gains intact |
| greedy submodular (ADR 0007, **experimental**) | 0.099 | failed gold-aware validation — see below |

Two live catches by the measurement harness:

1. **Aggregate vs strata**: the propagated scorer first read as "mixed";
   stratification showed correct behavior wrongly aggregated (single-hop
   stratum pays at early k, multi-hop stratum gains at all-gold).
2. **The submodular trilemma** (ADR 0008): an unbounded coverage universe
   floods (candidates mint their own novelty; junk routes outnumber gold
   routes); a query-anchored universe starves multi-hop targets *by
   construction* (containing no query content is what makes a target
   multi-hop); every interpolation carries a constant only settable by
   peeking at the eval. This is a measured existence theorem for the v2
   learned ranker: route quality is genuinely statistical. The ranker's
   job description follows exactly — learn route quality, inside the
   marginal-completion objective the audit validated, behind the drawn
   leakage line (the runtime proxy is never tuned on the audit's gold).

## Appendix: distractor sanity check

Per-question corpora of 10–20 paragraphs — reranking, not retrieval.
MuSiQue 500-question sample: floor r@2 0.386, r@5 0.562 (not saturated —
MuSiQue's distractors are adversarial — but the setting cannot exercise
corpus-scale retrieval and stays out of the headline).

## Reproduction

```
hoptrace eval --dataset beir-hotpotqa --gate
hoptrace eval --dataset musique --hops 2 --diagnostics
hoptrace eval --dataset 2wiki --hops 2 --diagnostics --pool-ablation 500
hoptrace eval --dataset beir-hotpotqa --hops 2 --diagnostics --pool-ablation 300
hoptrace eval --dataset musique --hops 2 --diagnostics --beam-entities 16 --beam-chunks 32 --frontier-chunks 128
```

BEIR index: 5,233,329 passages → 13.3M entities, 50.4M mentions, 7.1 GB
SQLite, 1.14 h build (Apple Silicon, single-threaded). Datasets download
to `$HOPTRACE_DATA_DIR/datasets/` with recorded checksums; never committed.

## Dataset attributions

- HotpotQA (CC BY-SA 4.0), via the BEIR packaging (Thakur et al. 2021)
- MuSiQue (CC BY 4.0), MuSiQue-Ans dev split
- 2WikiMultihopQA (Apache 2.0), dev split
- Published baselines: Pyserini BEIR reproduction matrix; BEIR paper
