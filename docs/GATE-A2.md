# Gate A2 — one Bosco, several questions (pre-registered 2026-09-24)

Step 3 of `BRIEF.md`, on A1's trained brain and phase 1's senses. **Can one
MaleCNS brain, with the question in what he smells, answer sweet/bitter
(text and pictures), safe/dangerous and junk near each question's ceiling;
does his wiring beat scrambled wiring; and does he stay fly-like inside?**

## The brain

A1's rate model (`src/bosco/ratebrain.py`) with three changes, all fixed now:

- **Nose:** e5-large-v2 → 26 principal components ± → 52 glomeruli (ORNs).
  Fit on 1,500 unlabelled training texts, 500 per question.
- **The question is in the smell.** Each question's text ("Is this sweet or
  bitter?", "Is this safe or dangerous?", "Is this junk?") goes through the
  same nose and is added to the item's glomerular drive (clipped to 0..1).
- **Eyes:** jina-clip-v2 → 26 components ± → 52 channels, fit on the 300
  training pictures unlabelled. They drive the 335 visual Kenyon cells
  (KCγ-d, KCαβ-p) directly, each cell the mean of 6 channels drawn once at
  random (seed fixed, the same for every arm), as v1's retina did. A picture
  is asked the sweet question through the smell.
- **Fly-like pressure:** the loss adds 10 × (mean Kenyon-cell rate − 0.01)²
  where the rate is above 0.01. Set on training data only before this was
  written: over 250 training batches, Kenyon-cell activity went 17% → 9% and
  training loss 0.69 → 0.57. No validation or test number was seen.
- **Out:** unchanged. Approach MBONs minus avoid MBONs, one scale, one offset.
  One read for every question: P(approach) = P(sweet), P(safe), P(not junk).
- **Trained:** per-neuron gain and threshold, as A1. Wiring and signs fixed.
  Init by A1's label-free rule.

## Data (from phase 1, clear labels)

| part | train | val / test (the phase-1 test set halved, seeded) |
|---|---|---|
| sweet (text, SST) | 6,254 | 157 / 157 |
| pictures (OASIS, sweet question) | 300, shown 5× per epoch | 150 / 150 |
| dangerous (Civil Comments) | 6,000 | 500 / 500 |
| junk (SMS Spam) | 3,902 | 836 / 836 |

## Training

Adam 3e-3, batch 64, 12 epochs, every question mixed in each batch. The epoch
is chosen by the **mean validation balanced accuracy** over the parts
trained; every number reported is **test**.

## Runs

- **Joint** (one Bosco, every question): real, shuffle, hash, free × seeds 1,
  2, 3. Twelve runs.
- **Solo** (one fly per question, real brain): sweet (text + pictures),
  dangerous, junk × seeds 1, 2. Six runs.
- About 7 hours on the Mac's GPU, one run after another.

## Measured, per part

Balanced accuracy at P = 0.5, Spearman, ECE. The **T-maze**, one of N:
2 options (the chance he goes to the right one, = AUROC) and 4 options (one
right among three wrong, 500 seeded sets, chance 0.25). Kenyon cells active
(rate > 0.01). Nick's probes under every question, and the two probe pictures
under the sweet question.

## Decision rule (real, joint, mean over seeds)

- **D1, one brain carries every question:** test balanced ≥ phase-1 ceiling
  − 0.05 on every part (sweet 0.867, pictures 0.880, dangerous 0.757, junk
  0.913).
- **D2, the wiring matters:** real's mean over the four parts ≥ each of
  shuffle, hash and free + 0.03.
- **D3, fly-like:** Kenyon cells active ≤ 0.10 on test items.
- **D4, no interference:** joint ≥ solo − 0.03 on every part.
- **Adoption:** the real joint brain becomes Bosco's decider if **D1 and D3**
  pass. D2 is the science claim and D4 the one-Bosco claim; both are reported
  whatever they say, and neither is re-scored after the fact.

Nothing above changes after the numbers are seen.

Runner: `scripts/gate_a2.py`. Results: `runs/gate-a2/`, `docs/gate-a2-results.md`.
