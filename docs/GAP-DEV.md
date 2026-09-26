# Closing the training gap (development round, 2026-09-25; validation only)

**Why:** in the specialist pilot's ceilings (validation), the information reaches his 46
glomeruli: 0.97 for topic and 0.91 for CLINC through the nose. But the fly reached only 0.84 and
0.31 after 4 epochs on 3,000 examples, trained on 10 sampled options. Three of four specialists
were still climbing.

**What changes (revised before running, from the cost: each option is one brain run, so all 151
CLINC options would be about 6 h per epoch):**
- **Topic:** 10,000 training items (30,000 prepared), **all 14 options**, up to 8 epochs.
- **CLINC:** all 15,100 training items (with out-of-scope). **20 options per item:** the right one,
  the **10 labels most similar to it** (cosine of the option words' embeddings, i.e. hard negatives)
  and 9 random, redrawn each epoch. Up to 4 epochs.
- Early stop when validation has not improved for 3 epochs.
- Validation is the pilot's own items, minus near-duplicates of the new training items (topic 500,
  CLINC 396).
- Everything else is as in the pilot: the v1 brain, the DN read, the homeostatic start, the real
  wiring, seed 1.

**Scope:** development only. **Validation only**; the sealed test sets are not scored. A recipe that
works goes into a pre-registered specialist gate (production bar 0.80) before any test score is
read.

Data: `data/cache/v1-gap/` (same sources, sampler and nomic encoder as `scripts/v1_data.py`).
Runner: `scripts/v1_gap.py`. Results: `docs/gap-dev-results.md`.

## Addendum: prototype option smells (2026-09-26, before running; validation only)

CLINC plateaued at about 0.36 on validation (epochs 1–2) while its training loss kept falling. A ceiling
check on validation (no fly): matching an item to each option through the 46-glomerulus nose reaches

| option smell | CLINC (151) | MASSIVE (60) | topic (14) |
|---|---|---|---|
| its label words (now) | 0.595 | 0.469 | 0.594 |
| **the mean of its training examples' embeddings ("prototype")** | **0.918** | **0.764** | **0.949** |

**Next round:** CLINC and MASSIVE with prototype option smells (`scripts/v1_gap.py --proto`). The same
brain, read, start and hard negatives (now by prototype similarity), up to 4 epochs, validation only.
MASSIVE's data was added (11,512 training items; validation 353 after near-duplicate removal); topic and
CLINC are byte-identical to the round above.

**Honesty:** the prototypes are built from labelled training examples, so "prototype matching alone"
(0.918 / 0.764 above) is reported beside the fly every time. The fly still decides, between richer
smells: "he remembers what each option smells like".

## Addendum 2: way B, one learned preference per option (2026-09-26, before running; validation only)

CLINC with prototype option smells (way A) reached 0.548, 0.526 and 0.635 on validation over epochs 1–3,
well below the 0.918 that prototype matching alone carries. Judging a match from a *mixture* of two smells
is hard for this brain. **Way B:** the specialist holds one KC→MBON memory per option (the synapses a fly
learns with; per-type parameters shared). Each option is judged from **the item's smell alone** with that
option's memory: approach or avoid, and the answer is the option he is drawn to most. Training: the right
option plus 19 random others per item, softmax over them; validation over all 151. The same brain, start,
read and data. Nick chose it (2026-09-26): B for many-option specialists, A kept for few-option ones
(topic 0.933). Runs on the 3080 after the CLINC prototype round, before MASSIVE.
