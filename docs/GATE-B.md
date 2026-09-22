# Gate B — pre-registration (FINAL 2026-09-22; binding)

Locked by Nick on 2026-09-22 before any run. Nothing below changes except to
fix a typo or a reference; a change of substance is a new gate with a new
file.

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
(`MushroomBody.learned_valence`), a continuous number. So the decision is a
**taste score**, bitter to sweet, and the tasks are chosen to be that shape:
"I fucking hate you" should read more bitter than "I hate you", and a
photograph of a sweet should read sweet.

- **Score, 0 to 1: 1 is sweet, 0 is bitter, 0.5 is neutral.** Learned
  valence `v` (in [−1, 1]), mean over N = 8 kernel seeds, mapped as
  `(v + 1) / 2`. The magnitude is the answer as much as the side: "I fucking
  hate you" near 0, "I hate you" above it, "I love you" above 0.5, "I
  fucking love you" nearer 1. This is the output of the whole thing.
- Answer (binary, for accuracy only): `sweet` if score > 0.5 else `bitter`.
- Probability (how sure, distinct from how sweet): the fraction of the 8
  seeds on the same side of 0.5 as the mean. Post-hoc calibration
  (isotonic, fit on a held-out 20% of rewarded items) is reported *beside*
  the raw number, never instead of it.
- The logistic arm's score is its sigmoid output (already 0–1, 0.5
  neutral); its probability is that output's distance from 0.5, calibrated
  the same way.

## Tasks (T1 and the text-or-image framing are Nick's; T2–T6 confirmed by him 2026-09-22)

Binary, text, public, small enough to run in hours, and **fly-shaped**: the
mushroom body's native decision is *sweet or bitter*, so that is the task.

- **T1 sweet / bitter (acquisition)** — the Stanford Sentiment Treebank
  with its **fine-grained** human labels (a continuous 0–1 sentiment per
  sentence, from which SST-2 is cut), 4,000 sentences in a fixed seeded
  order, balanced about 0.5. Above 0.5 is sweet (reward), below is bitter
  (punishment), and the **reward's magnitude is the label's distance from
  0.5**, doubled — a stronger sentence is more sugar or more shock, as PAM
  and PPL1 firing scale with concentration. Labels are **human**, never VADER: the question is
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
  900 images with human valence ratings on 1–7). All 900, valence rescaled
  to 0–1 about the scale midpoint; sweet above, bitter below, reward
  magnitude by distance from the midpoint as in T1. Same protocol as T1,
  except that 900 items at p = 0.2 yield about 180 rewards, so sample
  efficiency and gradedness are read at k = 25, 50, 100 only, on the next
  100 unrewarded items. Whether a picture can be a taste at all, and how
  much.
- **T4 sweet / bitter (cross-modal transfer)** — train on T1's sentences
  exactly as in T1; then present the T3 images **with no rewards** and
  score the verdict against human valence. The output is still sweet or
  bitter; what is being recreated is the bouba/kiki *kind* of effect — a
  correspondence learned in one modality showing up in another. Does a
  taste learned from words carry to things seen? The logistic arm does the
  same transfer, so the shared CLIP space is controlled for, and what is
  measured is what each decision layer keeps of it.
- **T5 sweet / bitter (mixtures; optional, reported not judged)** — after
  T1 training, present a sentence and a picture **together**, no rewards:
  congruent pairs (sweet with sweet, bitter with bitter) and incongruent
  ones. Which modality wins, and by how much, per arm.
- **T6 the probe sheet (reported, not judged)** — after T1 training and no
  further rewards, a fixed list of sentences and pictures written by Nick
  before the run (`docs/gate-b-probes.yaml`): the kind of thing the gate is
  for. "I fucking hate you", "I hate you", "I love you", "I fucking love
  you", a photograph of a sweet, of rot, and so on. Printed as a table of
  scores per arm. Not a metric; it is what the numbers look like when you
  read them.

### Inputs

Text, image, or both. Each becomes a glomerular drive through the same
frozen CLIP encoder and the same fixed projection; a text+image input is the
two drives **summed at the nose** — a mixture, as a fly meets a smell with a
taste — not a fused embedding and not a second model.

Dropped: spam, topic pairs, urgency. Not fly-shaped; they can be phase A's.

## Reward protocol (identical across arms)

- Items in a fixed seeded order.
- After each decision, with probability **p = 0.2**, an outcome arrives: the
  item's **own taste**, sugar if the label is sweet and shock if bitter, at a
  magnitude of twice the label's distance from the midpoint, delayed by
  `U(0, 60 s)` biological time and applied by replay pairing exactly as v1
  applied outcomes. (The fly is not told whether it answered right; it is
  given the taste of the thing, as a fly is. Whether it answered right is
  what the metrics score.)
- The same (item order, reward mask, delays) sequence for every arm and every
  seed of that arm. Five seeds per arm.
- Spacing: items arrive every 30 s biological time (idle simulated), so the
  spaced-repetition condition for long-term memory can occur naturally.

## Metrics

1. **Sample efficiency**: accuracy on the next 200 unrewarded items after
   *k* rewards received, at k = 25, 50, 100, 200.
2. **Gradedness**: Spearman correlation between the score and the human
   fine-grained label over the same 200 items, at the same k. This is the
   "more bitter than" question: sign is accuracy, order is this.
3. **Calibration**: expected calibration error of the raw probability and of
   the calibrated one, 10 bins, over all decisions after k = 100.
4. **Forgetting / relearning** (T2 only): rewards needed after the flip to
   return to the pre-flip accuracy; accuracy 24 h biological after the last
   reward with no further input.
5. **Latency**: wall time per decision, one seed, reported not judged.

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
