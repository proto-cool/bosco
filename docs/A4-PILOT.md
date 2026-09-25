# A4 pilot: can he answer questions he was never taught? (pre-registered 2026-09-24)

Nick: a Jev-style engine must handle questions it has not seen. A5 showed a
real fly brain can learn four questions. This pilot asks whether training
on **many** question kinds teaches him how questions work in general, so
that he can answer **unseen kinds, cold**. Small on purpose: if the answer is
no, the full A4 (hundreds of kinds, a rented GPU) would only pay to confirm it.

## Anatomy check

As A5 (`GATE-A5.md`): the corrected v2 cut brain, the nose on 46
non-innate glomeruli, answers from the MBONs grouped by measured DAN input.
Text only; no eyes. The question and the option are **extra smells**: each
sniff is the item's smell plus the option's smell, clipped to 0..1 (a
mixture, as odours mix at the antenna).

## Data and leakage check

- **Training kinds:** 40 drawn at random (seed 20260924) from the 268 usable
  tasksource kinds (label words, not letter pointers; ≤ 20 options; ≥ 600
  rows; `docs/a4-data.md`). Up to 1,000 training and 100 validation items
  each.
- **Item text:** tasksource's own prompt (an instruction naming the options,
  then the text). Options are the label words.
- **Cold test:** BTZSC, all 22 tasks (sentiment, topic, intent, emotion,
  toxicity…), up to 200 items each, never trained on. Its items get the same
  prompt template with BTZSC's label words ("With no explanation, label the
  following with either …").
- **Leakage:** BTZSC sources were excluded from A4's data by name and exact
  text (`a4-data.md`). Here, additionally: the share of cold-test items with a
  near-duplicate (cosine > 0.95) among the pilot's training items is
  reported, and any such items are dropped from the cold test. Which training
  kinds share a *concept* with a cold task (e.g. another sentiment task) is
  listed, since that is generalisation to a new dataset, not a new idea.

## Arms

| arm | what | trained on the 40 kinds |
|---|---|---|
| **real** | the fly, per type + KC→MBON | yes |
| layered | the layered shuffle, same | yes |
| **nose alone** | no brain: e5 similarity between the item and each option, cold. Scored two ways (label words, and BTZSC's option sentences), and the **better** one counts | no training at all |
| plain net | a two-layer network on the same smells ([item, option, item×option], 256 hidden), trained on the same 40 kinds | yes |

Brains: Adam 3e-3, fly-like start (`a5.fly_init`), the KC pressure, a
softmax over each item's options (he goes to one arm of the maze). Up to 8
epochs; the epoch is chosen on the 40 kinds' validation items. Seed 1.
Every arm must pass the preflight first (maze version: KC band, live output,
loss falls, gradients reach the brain).

## Rules (fixed now)

- **G1, he generalises:** real's macro-average accuracy over the 22 cold
  tasks is ≥ the better nose-alone score **+ 0.03**, and ≥ the tasks'
  macro-average chance **+ 0.10**.
- Reported beside it, not gating: plain net and layered on the same cold
  test; real on the 40 kinds' validation; per-task cold accuracy, split by
  "concept seen in training" vs "new concept".
- If G1 fails, the full A4 is not run as planned. Instead, the report says
  plainly that Bosco answers only the kinds of question he is trained on.

Runner: `scripts/a4_pilot.py`. Results: `docs/a4-pilot-results.md`.
