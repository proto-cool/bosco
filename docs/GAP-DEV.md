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
