# Specialist pilot: is a fleet of fly specialists any good? (pre-registered 2026-09-25)

Decisions 15–18 (`docs/DECISIONS-2026-09-25.md`): Bosco becomes a fleet of specialists. Each is the
same fly (one family: connectome, encoder, neuron order) with its own learned weights, answering one
fixed question. This pilot measures, on clean data, whether such specialists are good enough to
launch, and which ones.

## Anatomy check

- **The v1 brain** (`src/bosco/ratebrain3.py`, `src/bosco/v1.py`; `docs/v1-preflight-results.md`):
  - graded unit;
  - homeostatic label-free start at gain 4 (every cell type at a resting rate of 0.05, KCs 5%
    active, read neurons at 0.2);
  - KC fraction penalty (above 10% active);
  - per-type parameters and KC→MBON synapses trained (CLAUDE.md), with checkpointed runs.
- **The answer is read from his descending neurons** (decision 4): the anatomical approach and
  avoid DN groups from `v1.dn_groups`, fixed before training, 80 steps (400 ms), the last 8 read.
- **The nose:** the pinned nomic-embed-text-v1.5 (`bosco.encoders.TEXT`, prefix
  "classification: "), through bi46: 46 whitened components fit label-free on all five tasks'
  training texts plus the option words, one per non-innate glomerulus around a resting rate of 0.5.
- **The question is fixed per specialist, so it needs no channel of its own.** Each option is one
  sniff: the item's smell plus the option word's smell, from rest (as in A4 broad). He goes down
  one arm (softmax over the options).

## Data and leakage check

Clean sources only (`docs/clean-data.md`, verified licences, SHA-256 manifest). Personal columns are
dropped on read (DynaHate `annotator`, MASSIVE `worker_id`).

| task | source | options | train / val / test (sampled, seed 20260925) |
|---|---|---|---|
| hate | DynaHate v0.2.3 (CC BY 4.0), its splits | hate / not hate | 3,000 / 500 / 500 |
| hatemoji | HatemojiBuild (CC BY 4.0), its splits | hate / not hate | 3,000 / 500 / 500 |
| topic | DBpedia-14 (CC BY-SA), train → train+val, test | 14 | 3,000 / 500 / 500 |
| intent_massive | MASSIVE 1.1 en-US (CC BY 4.0), its partitions | 60 | 3,000 / 500 / 500 |
| intent_clinc | CLINC150 + out-of-scope (CC BY 3.0), its splits | 151 | 3,000 / 500 / 500 |

- Validation and test items with a near-duplicate (cosine > 0.95) among the same task's training
  items are dropped, and the count is reported.
- **The test sets are sealed** until the scoring step, and scored once. Smoke runs and preflights
  touch only training and validation items.
- Not in this pilot: Civil Comments (REVIEW), BoolQ and MultiNLI (two-text relations; A3 and the
  passage check showed a one-smell nose cannot carry them), SNIPS (328 items).

## Arms

| arm | what |
|---|---|
| **specialist, real** | one real brain per task, trained on that task only |
| specialist, layered | the same with the fair layered shuffle (own normalisation, the same 61,210 KC→MBON synapses) |
| shared, real | one real brain trained on all five tasks together (the A4-broad shape, on the v1 brain) |
| plain baseline | multinomial logistic regression on all 768 embedding numbers (C chosen on validation) |
| nose alone | cosine between the text and each option word, all 768 numbers, untrained |

**Training** (on the 3080):
- Adam 3e-3; 4 epochs; the epoch chosen on validation (balanced accuracy, full option sets).
- Items with more than 10 options are trained on the right option plus 9 others drawn at random each
  epoch. Validation and test always use the full option set.
- Seed 1.
- A temperature is fit on validation, for calibration only.

**Scoring** (decision 7): the test set on the **CPU** with deterministic algorithms, as served. The
validation logits for the temperature come from the GPU (development-level; stated).

## Metrics

Balanced accuracy per task (chance = 1/options) and ECE (15 bins, after the temperature) on the test
set. Also CPU time per sniff.

## Rules (fixed now)

- **S1, a specialist qualifies for the launch catalogue** if, on its test set: balanced accuracy
  ≥ chance + 0.15; **and** ≥ nose alone; **and** ≥ plain baseline − 0.05; **and** ECE ≤ 0.10.
- Reported, not gating: specialist vs shared (does one brain per task help?); real vs layered
  (whether the wiring matters, with the replay wobble of about 0.01 noted); per-task tables.
- If no specialist qualifies, the report says so plainly, and the fleet design goes back to Nick
  before anything else.
- Next pilot, not this one: the few-shot and fly-local-rule arms for self-hosted teach-your-own.

## Preflight (before any full run)

For every training run, 150 batches on training data only:
- KCs 2–15% active at the start;
- a raw read spread ≥ 1e-4;
- the loss falls by ≥ 0.02;
- gradients reach ≥ 90% of cell types;
- held-back training items above chance.

A failing run stops and is reported; it is not tuned and relaunched without an amendment.

Runner: `scripts/v1_pilot.py` (data: `scripts/v1_data.py`). Results: `docs/specialist-pilot-results.md`.

## Amendment 1 (2026-09-25, Nick; before any test score was read, only validation numbers seen)

**The production bar replaces S1.** "If the verdict is not 'works well' aka 0.8 or better it's
chucked. That's the production bar as well."

- **S1 (a specialist ships):** balanced accuracy on its sealed test set, scored on the CPU,
  ≥ **0.80**, **and** ECE ≤ 0.10, so the confidence shown is honest.
- Nose alone, the plain baseline and the layered shuffle are **reported, not gating** ("I don't
  care if another model can do the task better").
- The same bar applies to every future specialist before it reaches Ask Bosco or a release.
