# A4b development results (2026-09-25)

Plan: `docs/A4B-DEV.md`. Development only: no claim, BTZSC untouched. Practice test: 11
held-out tasksource kinds (dbpedia_14, dyda_e, snips intents, logical-fallacy, twitter
financial sentiment, political-media-message, ledgar, tweet irony, hate_speech_offensive,
citation_intent, poem_sentiment), 200 items each (4 near-duplicates of training dropped).
Training pool: 33 single-text kinds with meaningful label words. Macro chance on practice
0.196.

## L1: his antenna loses most of the matching

Nose alone (cosine between a text and each label word), practice macro:

| | practice |
|---|---|
| all 1,024 numbers | **0.418** |
| the A5/A4 antenna (23 components ±, fit on items) | 0.300 |
| **bi46**: 46 components, fit on items and label words, one per glomerulus around a resting rate | 0.362 (0.391 before clipping to 0..1) |
| components needed: 23 / 46 / 100 / 200 (no clipping) | 0.315 / 0.347 / 0.393 / 0.411 |

Matching needs ~100–200 channels; a fly nose has ~50 glomeruli. bi46 is the most his
nose can carry and is more fly-like than the ± split (an ORN fires at rest and a smell
pushes it up or down, Hallem & Carlson 2006).

## L2: trained matchers lose to untrained matching on unseen kinds

Plain net (two layers, 256), best epoch on training kinds' validation:

| antenna | options | form | val (training kinds) | practice |
|---|---|---|---|---|
| split23 | own | sum / cat | 0.665 / 0.685 | 0.313 / 0.316 |
| split23 | mixed-up | sum / cat | 0.621 / 0.659 | 0.236 / 0.252 |
| bi46 | own | sum / cat | 0.565 / 0.610 | 0.311 / 0.340 |
| bi46 | mixed-up | sum / cat | 0.361 / 0.562 | 0.274 / 0.340 |
| full 1,024 | own / mixed-up | cat | 0.610 / 0.531 | 0.284 / 0.355 |

Even on all 1,024 numbers, a matcher trained on 33 kinds (~290 label words) scores below
the untrained cosine (0.418). Mixed-up options did not help: rejecting another kind's
labels is easy and teaches nothing about telling a kind's own labels apart.

**Training on top of the cosine does help, briefly.** A net whose output starts as the
full cosine (a learned correction starting at zero, own options) went 0.418 → **0.485**
after one epoch, then 0.465, 0.437, 0.420 as it overfit to the training kinds. With
mixed-up options it only fell (0.392 → 0.349). The per-epoch practice scores were looked
at, so the right stopping point must be chosen on separate held-out kinds before it is
believed.

## The fly as a matcher

- Total KC drive and the untrained read (ap − av) pick options at chance (0.16–0.19).
- **The start (fly_init) loses what he smells.** fly_init picks the (gain, threshold) with
  the widest output spread: gain 8. There, the similarity between two label words
  survives with correlation **0.53 at the ORNs** (the input layer), 0.39 at the PNs, 0.37
  at the KCs. Lower gains keep it (gain 0.5–2: ORN 0.91, PN 0.75–0.78, KC 0.64–0.68). A
  real antennal lobe and KC layer are known to preserve odour similarity; the start rule
  never checked for it. This also bears on A5, which used the same rule.
- Even at the best start, choosing by KC-code similarity reaches only 0.24–0.30 on
  practice (antenna cosine 0.362, chance 0.196): the KC layer keeps a coarse version of
  the similarity, not the fine one that tells 14 dbpedia topics apart.

## Reading

Answering unseen kinds is mostly the nose's similarity. It needs more channels than a
fly nose has, and 33 kinds cannot teach a better one. The one thing that moved the needle
(+0.067) is starting from the nose's matching and adjusting it a little; the fly's own
circuit cannot start there (KC matching 0.24–0.30).

## L3: encoding the question with the text (2026-09-25, after A4 broad)

Nick: "the entire novelty is being able to encode human problems in a way a fly
understands." The combined encoding was L3 in the plan and was not tested before A4
broad; it should have been. Here each (text, candidate label) pair becomes **one smell**,
and the only judgement is "does this fit?" (approach or avoid), one skill across every
kind. Same practice test (11 held-out kinds, plain accuracy, chance 0.196). Training
pool: 16,503 pairs (6,000 items; the right label plus 3 wrong ones from the same kind).
Ceiling: a logistic "fits" probe on the smell. Runners: `scripts/a4b_joint_check.py`,
`scripts/a4b_pair_check.py`.

| encoding | practice |
|---|---|
| cosine, text vs label (the A4 nose), all 1,024 numbers | 0.418 |
| the same through bi46 (46 channels) | 0.362 |
| e5-instruct, text and label as one input, probe, 46 ch | 0.386 |
| **DeBERTa-v3-base zero-shot pair encoder, its own answer (encoder alone)** | **0.504** |
| **the same encoder's pair smell, probe, 46 channels** | **0.468** |
| the same, probe, all 768 channels | 0.431 (overfits) |

- The pair encoder (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`, MIT) reads the text
  and "This text is about {label}." together. Its internal state is a perception of the
  pair. Through 46 channels it carries **0.468**: +0.106 over his current nose (bi46)
  and +0.050 over the full-embedding cosine, on kinds never trained on.
- **Honesty line:** the encoder's own zero-shot answer is 0.504. A fly reading this smell
  must be reported beside it every time. Below it, the fly is a lossy readout of the
  encoder; the product can still be that, but it must say so.
- **Provenance:** the "-c" models train on MNLI and FEVER-NLI (documented) plus
  synthetic texts generated by Mixtral-8x7B-Instruct (an LLM with opaque training data).
  DeBERTa-v3 itself was pre-trained on web-scale text. An LLM sits upstream of the
  encoder's training, never in Bosco's decision. This goes in the provenance ledger.
- **Speed:** about 85 pairs per second on the Mac's GPU for the base size. One pass per
  option: fine for 2–10 options, slow for 77. Server CPU not measured.
