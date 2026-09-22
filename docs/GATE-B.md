# Gate B — pre-registration (DRAFT 2026-09-22; binding once marked FINAL)

## Question

With one fixed encoder and one identical sequence of sparse, delayed rewards,
does the MaleCNS mushroom body, learning by the three-factor rule it ran under
in v1, learn a decision better than

1. a degree-preserving shuffle of its own wiring, and
2. a random sparse projection of the same size and KC sparsity (the fly-hash
   baseline, Dasgupta, Stevens & Navlakha 2017),

and how does it compare with

3. an online logistic head on the raw embedding, learning from the same
   rewards and nothing else?

If the real wiring does not beat (1) and (2), the connectome is not earning
its place in a decision model under the biological rule, and phase A starts
from that fact.

## Arms

| arm | wiring | learning |
|---|---|---|
| **real** | MaleCNS central brain as built for v1 (`model.py`, `config/model_v1.yaml`) | three-factor rule, `config/plasticity_v1.yaml` as at `freeze-v2` (credit `share`, contrast on; extinction and homeostasis off) |
| **shuffle** | `scripts/make_dunce.py`: degree-preserving shuffle of the same matrix, same signs | same |
| **hash** | PN → KC random sparse projection, KC count and per-KC in-degree matched to the real MB, APL-style inhibition set to the same active fraction; KC → MBON dense, same MBON/DAN compartments | same |
| **logistic** | none | SGD logistic regression on the embedding, one update per reward received, nothing on unrewarded steps |

Nothing about the fly circuit is tuned on the task. The rule's parameters are
the published v1 ones.

## Encoder and drive

- Encoder: CLIP ViT-B/32 (`sentence-transformers/clip-ViT-B-32`, 512-d),
  frozen, CPU, deterministic; it embeds **text and images into one space**,
  so a sentence and a picture reach the glomeruli the same way. Fixed by
  name and revision here before running. It is the fly's eye and ear and
  nothing else: it never sees a label, a reward, or a decision.
- Embedding → glomerular drive: one fixed random projection 512 → the
  glomeruli v1 drove (`model.py` `v1_weights_mv`, ORN classes), rectified,
  scaled by a single scalar. **The scalar is set once, on 500 unlabelled
  items, so that mean KC active fraction lands in v1's recorded band
  (median 0.027, band 0.011–0.067, `main:config/thresholds.json`).** It is
  then frozen for every arm and every task. This is the only free parameter
  and it never sees a label or a reward.
- Presentation: 500 ms from rest per item, as `Agent.account_signature` did
  in v1 (the `credit_gate.py` harness), so short-term depression is recovered
  between items.

## Readout

The rule learns valence: reward-side minus punishment-side depression on the
active KCs' synapses, read against the compartment level
(`MushroomBody.learned_valence`). So the decision is **binary**, *good for me
or not*, and the tasks are chosen to be that shape.

- Answer: `yes` if learned valence > 0 else `no`.
- Probability: the item is presented under N = 8 kernel seeds; the fraction
  answering `yes` is the raw probability. Post-hoc calibration (isotonic, fit
  on a held-out 20% of rewarded items) is reported *beside* the raw number,
  never instead of it.
- The logistic arm's probability is its sigmoid output, calibrated the same
  way.

## Tasks (T1 and the text-or-image framing are Nick's, 2026-09-22; T2–T4 proposed, FINAL when confirmed)

Binary, text, public, small enough to run in hours, and **fly-shaped**: the
mushroom body's native decision is *sweet or bitter*, so that is the task.

- **T1 sweet / bitter (acquisition)** — SST-2 (GLUE), 4,000 sentences in a
  fixed seeded order, balanced. `positive` is sweet (reward), `negative` is
  bitter (punishment). Labels are **human**, never VADER: the question is
  whether the circuit can learn a taste from an encoder, not whether it can
  learn a lexicon. This is the comparison v1 could not make: v1 gave the
  same rule ~6,000 VADER pairings through a hash encoder and its verdict
  predicted nothing (r = −0.007 with the window's VADER, `main:
  docs/CLOSING-v1.md`). If the rule learns here, v1's failure was the
  encoder; if it does not, it is the rule.
- **T2 sweet / bitter (reversal)** — the same 4,000 sentences; at item 2,000
  the reward mapping flips (sweet is punished, bitter rewarded). Reversal
  learning is the standard Drosophila paradigm for the two memory
  timescales; it measures forgetting and relearning, which is what the rule's
  short- and long-term traces are for.

- **T3 sweet / bitter (pictures)** — OASIS (Kurdi, Lozano & Banaji 2017:
  900 images with human valence ratings on 1–7). The top and bottom thirds
  by mean valence, ~600 items, sweet and bitter; the middle third is not
  used. Same protocol as T1. Whether a picture can be a taste at all.
- **T4 bouba / kiki (cross-modal transfer)** — train on T1's sentences
  exactly as in T1; then present the T3 images **with no rewards** and
  score the verdict against human valence. The fly's bouba/kiki: does a
  taste learned from words carry to things seen? The logistic arm does the
  same transfer, so the shared CLIP space is controlled for, and what is
  measured is what each decision layer keeps of it.

Dropped: spam, topic pairs, urgency. Not fly-shaped; they can be phase A's.

## Reward protocol (identical across arms)

- Items in a fixed seeded order.
- After each decision, with probability **p = 0.2**, an outcome arrives:
  reward if the answer was correct, punishment if not, delayed by
  `U(0, 60 s)` biological time, applied by replay pairing exactly as v1
  applied outcomes.
- The same (item order, reward mask, delays) sequence for every arm and every
  seed of that arm. Five seeds per arm.
- Spacing: items arrive every 30 s biological time (idle simulated), so the
  spaced-repetition condition for long-term memory can occur naturally.

## Metrics

1. **Sample efficiency**: accuracy on the next 200 unrewarded items after
   *k* rewards received, at k = 25, 50, 100, 200.
2. **Calibration**: expected calibration error of the raw probability and of
   the calibrated one, 10 bins, over all decisions after k = 100.
3. **Forgetting / relearning** (T2 only): rewards needed after the flip to
   return to the pre-flip accuracy; accuracy 24 h biological after the last
   reward with no further input.
4. **Latency**: wall time per decision, one seed, reported not judged.

## Decision rule (fixed here)

The real wiring **earns its place** if, on **T1 and T2**, its mean accuracy
at k = 50 and at k = 200 exceeds *both* `shuffle` and `hash` by more than
the pooled standard deviation across the five seeds. Anything less is
reported as "does not," with the numbers. T3 and T4 are reported, not judged:
they say whether a picture can be a taste and whether taste crosses
modalities, for every arm, and are context for phase A. The logistic arm is
context, not a bar: if it beats every fly arm by a wide margin, that is
written down as what the biological rule costs.

## What is not done

- No tuning of rule parameters, drive scale, presentation length, or reward
  probability after seeing any result.
- No task added or dropped after running.
- No arm dropped for time. A result without the controls is not a result.

## Provenance

`scripts/gate_b.py` (to be written), one process per arm and seed; every run
logs its config, seed, item order, reward mask, decisions and probabilities to
a `.jsonl` beside `docs/gate-b-results.md`.
