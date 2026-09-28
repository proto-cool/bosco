# Specialist gate 4: generalists in their fields (pre-registered 2026-09-27, before any gate-4 test score exists)

This gate follows decisions 29–37 (`docs/DECISIONS-2026-09-25.md`). Specialists are **sharp questions pooled from
several clean sources** and **scored on a source they never trained on**.

## What is new against gates 2 and 3
- **The breadth test is the gate.** Each specialist's sealed test is the held-out source's test part (80% of it).
  That source contributed nothing to training, validation or the near-duplicate pool.
- **The temperature is fit on CPU logits** of up to 1,000 validation items (seeded), as the service computes them.
  Gate 2's intent failed on calibration when the temperature came from GPU validation logits.
- **The harm experts reuse the harm development run's checkpoints** (`runs/harm-dev`). Their training and
  validation data are checked to be identical (`scripts/v1_gate4_data.py` asserts it), and their recipe is this
  gate's recipe.

## Anatomy check
- The same brain, family v1, as gates 2 and 3:
  - the v1 rate model, homeostatic start and KC fraction penalty;
  - per-type parameters and KC→MBON synapses trained;
  - 80 steps;
  - the answer read from the anatomical approach and avoid DN groups, over the last 8 steps.
- The fixed family antenna (bi46) and the pinned nomic-embed-text-v1.5 encoder.
- A yes/no expert is a T-maze with two arms (design A). Language is one memory per option (design B).
- The fly's real circuits for these judgements are not claimed. As in every text gate, the encoder is the
  translator and the mushroom body the learner.

## Specialists, data and held-out sources
Builds: `docs/gate4-data-harm-tone.md`, `docs/gate4-data-intent-finance.md` and
`docs/gate4-data-topic-language-danger.md`. The cut into yes/no experts is `scripts/v1_gate4_data.py` →
`data/cache/v1-gate4` (items.json sha256 `85b64229…67bf7`).
- Yes/no training balances positives with negatives. For harm, half of the negatives come from `fine` and half
  from other harm types; for the rest, they come from every other class.
- Val, dev and test keep every item.

| specialist | yes means | pool (train/val) | held-out source (sealed test: all, yes) | bar |
|---|---|---|---|---|
| **harmful** | hate, harassment, threat or sexual | Civil Comments, Wikipedia Detox, DynaHate, HatemojiBuild | Aegis 1.0/2.0 human prompts (2,582 / 1,131) | disputed |
| **threat** | threat or violence | same | Aegis (2,582 / 337) | disputed |
| **sexual** | sexual | same | Aegis (2,582 / 196) | disputed |
| **hate** | hate | same | Aegis (2,582 / 343) | disputed |
| **harassment** | harassment | same | Aegis (2,582 / 255) | disputed |
| **problem** | something is wrong (report a problem, fraud or security) | CLINC, MASSIVE, SNIPS, BANKING77, NLU++, SGD, ABCD | MultiDoGO (3,092 / 461) | 0.80 |
| **social** | greeting, thanks, yes, no, small talk | same | MultiDoGO (3,092 / 974) | 0.80 |
| **credit_debt** | credit reports, loans, mortgages, student loans, debt collection | Money SE, BANKING77, CLINC, MultiDoGO finance, SGD | CFPB narratives ≤ 2022-10-31 (2,173 / 1,207) | 0.80 |
| **sport** | sport | Wikipedia Vital Articles, Stack Exchange, arXiv, MASSIVE, SGD | Wikinews (2,056 / 192) | 0.80 |
| **business** | business, economy, personal money or work | same | Wikinews (2,056 / 196) | 0.80 |
| **health** | health and medicine | same | Wikinews (2,056 / 196) | 0.80 |
| **science** | science or technology | same | Wikinews (2,056 / 402) | 0.80 |
| **politics** | politics, war or law | same | Wikinews (2,056 / 465) | 0.80 |
| **danger** | a risk of physical harm to people | Wikipedia danger set, openFDA I vs III, CPSC and NHTSA recalls, NWS, SE safety | Wikinews disasters vs ordinary news (642 / 323) | 0.80 |
| **language** | which of 36 languages | MASSIVE, Tatoeba, Wikipedia leads (38 languages) | SIB-200, 100 sentences per language (3,600) | 0.80 |

- **Language merges two near-twin pairs** into Malay or Indonesian and Danish or Norwegian. That leaves 36
  options, decided on the dev part's confusions. The hard scripts are out of scope (decision 32).
- **Hindi is out:** 5.2% of its tokens are unknown to the encoder.

## Leakage check
Done in the three builds; see their docs.
- Pool items that exactly match or near-duplicate (cosine > 0.95) a held-out item were dropped.
- Val items near-duplicating train were dropped.
- Wikipedia comment sets were deduplicated against each other.
- Training passages containing a SIB-200 or FLORES sentence were dropped.
- Splits follow the natural unit: comment, dialogue, question or Wikidata item.
- Text from live sources predates 2022-11-01 (decision 33).

## Training (the 3080; the Mac as a second machine if needed)
- As gates 2 and 3: Adam 3e-3, the KC penalty and seed 1. The epoch is chosen on pool validation.
- Design A runs up to 8 epochs with an early stop after 3 without improvement. Design B runs up to 5 with an early
  stop after 2.
- The preflight is as in gate 2; a failing run stops and is reported.
- The harm experts are already trained (harm development run). This recipe is theirs, unchanged. The harmful
  expert was still improving at epoch 8; the cap stays at 8 for every design-A expert in this gate.

## Scoring
- Each sealed test is scored **once, on the CPU, with deterministic algorithms**, with the temperature from CPU
  validation logits.
- **The runner refuses to score any sealed test until the harm floor below is fixed.**

## Rules (fixed now)
- **A specialist ships** if its sealed held-out test balanced accuracy ≥ its bar **and** its ECE ≤ 0.10.
- **Bars:**
  - 0.80, except for the harm experts, whose labels are disputed.
  - Harm experts: **max(0.9 × ceiling, floor)**, where the ceiling is the plain class-balanced logistic on all
    768 encoder numbers, trained on the pool and scored on the Aegis dev part. The ceilings were measured
    2026-09-27: harmful 0.824, threat 0.842, sexual 0.844, hate 0.831, harassment 0.635.
  - The **floor** is Nick's, fixed before any sealed test is scored and written here: **floor = 0.70** (Nick, 2026-09-27: "strict at 0.7"). Harm bars: harmful 0.742, threat 0.758, sexual 0.760, hate 0.748, harassment 0.700.
- **A shipped specialist's card** states what it was trained on and what it was tested on:
  - its held-out source and pool;
  - its disputed-labels note;
  - its limits in plain words, e.g. "harm: content moderation, not screening prompts sent to an AI"; "topic
    experts: tested on news"; "credit_debt: tested on consumer complaint letters";
  - the Civil Comments and Stack Exchange notes (decisions 30, 31).
- **Report-only, after the verdicts:** pool validation, the held-out dev part, the plain logistic baseline, nose
  alone and prototype alone on each test, and CPU time per question.

## Amendment 1 (2026-09-27, before any sealed test was scored)
- **Harassment is deferred to gate 4b.** Its dev part (0.672) sits under its bar (0.70), and Nick wants it worked
  until it meets the bar ("work it until we meet it"). That work runs on validation and the Aegis dev part only.
  Its sealed test (2,582 items, 255 yes) is scored **once**, in gate 4b, with the bar unchanged (0.70). Gate 4b's
  pre-registration will name what changed. The other 14 specialists are unchanged.

## Disclosure
Seen before this was written, all on validation or the held-out **dev** parts, never a sealed test:
- the encoder ceilings for every field (`runs/gate4-ceilings/`);
- the reframing diagnostic (`runs/gate4-ceilings/reframe.json`), which chose these questions;
- the harm encoder comparison (`runs/gate4-ceilings/harm-encoders.json`);
- the fly's harm experts on Aegis dev: harmful 0.821, threat 0.794, sexual 0.816, hate 0.843, harassment 0.672.

The questions and the language merge were chosen on dev numbers, which is what the dev parts are for. The sealed
tests were counted, never scored.

Runner: `scripts/v1_gate2.py --gate 4 …`. Results: `docs/specialist-gate-4-results.md`.
