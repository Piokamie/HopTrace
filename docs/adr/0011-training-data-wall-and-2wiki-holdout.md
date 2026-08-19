---
status: accepted
date: 2026-08-14
---

# 11. Training-Data Wall: Train Splits Only, Manifest Tripwire, 2Wiki Holdout

## Context

ADR 0004 made external benchmarks the only carrier of evidence and
forbade tuning on gold; ADR 0008 extended that law to the learned
ranker's training data ("external benchmarks, never self-benchmark,
never the audit"). V2 fine-tunes on benchmark data directly, so the
leakage surface changes: the published v1 numbers are frozen on MuSiQue
dev, 2Wiki dev, and BEIR-HotpotQA test, and a model that saw any of
those splits would make every v2-vs-v1 comparison circular.

Train-split data is available without touching the published splits: the
BEIR HotpotQA package already contains `qrels/train.tsv` (~85k
train-query judgments; the published gate and eval read only
`qrels/test.tsv`, `cli.py`), and MuSiQue-Ans train
(`musique_ans_v1.0_train.jsonl`, 241 MB) sits in the same pinned
repository as the dev file the registry already downloads.

A split policy that lives only in documentation is a promise, not a
control: nothing would stop a later eval run against a model trained on
the wrong split from silently producing a leaked number.

## Decision

1. **Training reads train splits only**: MuSiQue-Ans train and HotpotQA
   train (via the existing BEIR corpus index). Dev/test splits remain
   frozen as the published v1 baseline and the only reportable eval
   inputs.
2. **The wall is mechanical.** Every trained artifact ships with a
   training manifest (`manifest.json`: datasets, splits, source-file
   checksums, base model, trainer commit). The eval harness reads the
   manifest and refuses to run when the (dataset, split) under
   evaluation appears in it — a tripwire, not a convention. A missing
   manifest is likewise a hard error for eval runs.
3. **2WikiMultihopQA is held out of training entirely** — not
   train-split-only, absent altogether; the dataset builder hard-refuses
   2wiki input. 2Wiki dev becomes the never-seen transfer test:
   "improves on a dataset it never trained on" is the strongest sentence
   v2 can earn, and the holdout is unreservable after the fact.
4. Training-internal validation (early stopping, checkpoint selection)
   uses a held-back slice of the train splits, never dev or test.

Alternatives rejected:

- **Documented convention without enforcement**: one forgetful eval run
  publishes a leaked number; the harness, which already refuses
  mismatched analyzers, is the natural enforcement point.
- **Training on all three datasets' train splits**: maximizes data but
  spends the transfer test; 2Wiki train adds modest volume next to
  MuSiQue+HotpotQA train while its holdout buys the strongest available
  generalization claim.
- **k-fold over dev splits**: preserves no frozen baseline and makes
  every published number fold-dependent.

## Consequences

- Every v2-vs-v1 comparison is structurally non-circular, and the claim
  is verifiable by third parties from the shipped manifest alone.
- The 2Wiki transfer number measures generalization, not memorization —
  and it is also allowed to be unflattering; either way it is a finding.
- The harness gains a hard failure mode (manifest checks) that makes
  some legitimate-looking runs impossible by design; experiments on
  trained splits require deliberately bypassing eval (and cannot produce
  reportable output).
- Foregoing 2Wiki train data may cost some accuracy on 2Wiki — the
  price of the transfer claim.
- The manifest becomes a load-bearing artifact: a wrong manifest is now
  worse than no model, so the trainer writes it mechanically from its
  actual inputs, never by hand.
