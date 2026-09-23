# Gate B3 — sweet or bitter, done the way a fly is trained (DRAFT 2026-09-23)

## Why B and B2 could not work

B2 (`GATE-B2-note.md`) gave the fly's rule 4,000 different smells, each paired
once, at 20–40% sugar, on one in five, an hour late, and asked it to generalise.
That is a regression's job. A fly is trained on a few odors, each repeated,
with full sugar, at once. Under B2's conditions the rule reached 0.56 on text
and 0.63 on pictures against ceilings of 0.59 and 0.74 — and could not reverse,
because its long-term memory was consolidating the population average instead
of any memory (two *different* items an hour apart counted as spaced
repetition of the same synapse). Every one of those is a harness choice, and
every one is reversed here.

The one thing B2 found that is not a harness choice stays stated: with the
mushroom body's architecture intact, a random Kenyon-cell wiring ties the real
one, and it will again, because the real one is random too (Caron et al.
2013). B3 is not built to beat that. It is built to find out how well the
fly's learning rule on the fly's architecture scores sweet and bitter when it
is given what a fly is given. The controls stay in, as controls.

## What "works" means (the bar)

Two numbers, fixed here:

1. **Generalisation.** Real-wiring accuracy on held-out items after training
   ≥ **0.9 ×** the ceiling, where the ceiling is a logistic regression fitted
   on the *same training items at the antenna* (the 53-channel drive, not the
   raw embedding — the antenna is the fly's, so it is the fair ceiling; the
   raw-embedding ceiling is reported beside it).
2. **Memory.** Recall gap on trained items ≥ **+0.30** (sweet-trained minus
   bitter-trained, on the 0–1 score) at the end of training, and the reversal
   (T2) reaches ≥ 0.9 × the pre-reversal accuracy within the same number of
   repetitions it took to acquire.

Both on T1 and T2, five seeds, mean. Controls (shuffle, hash) reported; the
expectation written down is real ≈ hash > shuffle, and a surprise either way
gets said.

## The five changes

### 1. Full-strength taste
Every pairing is sugar or shock at magnitude 1. The fine-grained label picks
the side only. Gradedness is still measured (Spearman of score against the
fine label on held-out items); it is now a property of the readout, not of
how much sugar was given. Biology: PAM/PPL1 firing does scale with
concentration, but a fly in training is given the concentration that works.

### 2. Training shaped like fly training
- **Train set:** 400 items (200 sweet, 200 bitter; labels ≥ 0.6 or ≤ 0.4 so
  the sides are clear). **Held-out set:** 400 more, any label, for
  generalisation and gradedness. Both drawn once with the pool seed.
- **Repetitions:** each training item is presented and paired **R = 6**
  times, in random interleaved order, with at least **1 h** (biological)
  between two presentations of the same item — the spacing that consolidates
  long-term memory in flies (Tully et al. 1994) and in the rule.
- **Every training presentation is rewarded**, at once (delay 0–5 s). The
  sparse, delayed reward of B was the Bluesky framing; a client of a decision
  API supplies the outcome, and a trainer supplies the sugar.
- **Epochs:** after each pass over the 400 (an "epoch", ~400 × 30 s ≈ 3.3 h
  biological), 200 of the held-out items are scored (8 seeds), and 30 trained
  items of each taste are recalled. Six epochs → the learning curve.

### 3. Long-term memory consolidates memories, not the average
The rule's consolidation condition becomes **per item**: a pairing
consolidates only if the *same item* was paired ≥ `spacing_h` earlier and its
synapses still carry short-term memory. Two different items sharing a Kenyon
cell no longer count as repetition. This is a change to the learning rule
(`config/plasticity_b3.yaml`, `ltm_by_item: true`), it is the biologically
literal reading of spaced training, and it is the defect that made B2's
reversal impossible. It is unit-tested against the phase-3 protocol: one odor
paired three times spaced still forms LTM; three different odors do not.

### 4. An antenna that keeps the signal
The random 512→53 projection is replaced by the **top 26 principal components
of the centred calibration embeddings, each split into a + and a − glomerulus**
(52 of the 53), so rectification loses nothing and the 53 channels carry the
53 most informative directions instead of 53 random ones. Still fixed once on
the unlabelled calibration set, still one scalar set for KC sparseness, still
no label. The ceiling at the antenna is measured and reported: the logistic
regression on the 52-d drive, same training items. (B2: a random projection
cost 0.70 → 0.63 on text; the estimate here is 0.66–0.68.)

### 5. Rewards as a trainer gives them, not as a feed does
Already implied by 1–2: no reward probability, no delay distribution, no
magnitude scaling. The protocol is a training schedule.

### 6. Run offline, on cached Kenyon-cell codes (found 2026-09-23)
The plastic synapses are KC→MBON, so the Kenyon-cell code *should* be
independent of what the fly has learned — and in the full model it is not.
Same sentence, same seed, naive brain versus a B2-trained state: 117 active
cells become 81, Jaccard overlap 0.77, the same as the spread between two
random seeds. Not chaos (a 0.1% weight nudge leaves the code identical) but
a loop: the plastic synapses change MBON firing, and MBONs feed back onto
Kenyon cells directly (11,000 synapses) and through APL and the dopamine
neurons. Removing the 225,000 DAN→KC fast synapses (dopamine is modulatory;
the Shiu sign convention runs it as excitatory drive) did **not** remove the
dependence, so it is MBON feedback, which is real biology.

Consequence for B2: a trained item stopped smelling like it did when it was
trained — a plausible part of why recall was so weak. Consequence for B3:
the codes are computed **once per item and seed from the naive brain and
cached**, and training, scoring and every control run over the cache with
the same rule. That is the feedforward mushroom body every standard model
uses (Hige, Aso, Dasgupta), it makes every run take seconds instead of a
night, and it is a stated simplification, not a fix: the loop is real and B3
does not model it.

## What does not change
Encoder (CLIP ViT-B/32, centred), presentation (500 ms from rest), scoring
(8 seeds, mean valence mapped to 0–1, sureness beside it), contrast read,
mixture credit, exposure, the three fly arms, the logistic arm (refit on the
training items seen so far, at the antenna and at the raw embedding), five
seeds, everything logged, nothing tuned after the first run.

## Tasks
- **T1 acquisition** — SST, as above. Six epochs.
- **T2 reversal** — after T1's six epochs, the training items' tastes flip and
  six more epochs run. Recovery measured per epoch against T1's final accuracy.
  This is the Drosophila reversal paradigm as actually run: same odors, same
  fly, reward swapped.
- **T3 pictures** — OASIS: 300 train (top/bottom by valence), 300 held-out,
  six epochs. Same bar.
- **T4 transfer** — T1's trained flies scored on all 900 pictures, no rewards.
  Rho against human valence; expectation from B2: weak and equal across arms
  (it is CLIP's), reported.
- **T5 / T6 probes** — the sheet as before. Picture probes are two OASIS
  images named in the yaml (a dessert and rotting food) so they exist; Nick
  may replace them.

## Cost
Per fly run, T1: 400 × 6 × 2 episodes (present + pair) + 6 × (200 × 8 + 60 × 8)
scoring ≈ 17,300 episodes ≈ 2.4 h alone, ~4 h at eight in parallel. 15 fly
runs → two waves, ~8 h. T2 the same on top. T3 about half. Text T1+T2 in one
night; T3–T6 the next.

## Implementation (a day)
- `plasticity.py`: `ltm_by_item` — `pair_counts(..., consolidate: bool)`;
  the arm tracks per-item last-pairing time. Unit test on the phase-3 odors.
- `gateb.py`: `Antenna` (PCA± projection) beside `Drive`; `Schedule` (train
  set, held-out, epochs, spacing) beside `Protocol`; the epoch loop; recall
  and held-out scoring per epoch; the antenna-level ceiling.
- `gate_b.py`: `learn --gate b3 ...`, `report` with a learning-curve table.
- `GATE = "b3"`, `runs/gate-b3/`, tag `gate-b3-prereg` before the first run.

## Provenance
B as run: `gate-b-run`. B2 as run: `gate-b2-run`. This document is locked with
`gate-b3-prereg` once Nick says FINAL; nothing in it changes after that except
by a dated amendment made before a run, as in B.
