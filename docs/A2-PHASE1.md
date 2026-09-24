# A2 phase 1 — the questions and the nose (written 2026-09-24, before measuring)

Bosco answers Jev-style questions with one brain; every answer is approach vs
avoid at his output neurons (`BRIEF.md`). Phase 1 fixes **which questions**
and **which nose** (the frozen encoder), by rules written here first. No brain
is trained in phase 1.

## The questions

| question | Jev type | how he answers | data (clear labels only) |
|---|---|---|---|
| **sweet or bitter?** | score | approach − avoid, on his own zero | text: SST (6,254 clear train; the 400 held-out, clear ones scored). Pictures: OASIS valence, 300 clear to train, 300 clear held out (B3 split) |
| **safe or dangerous?** | yes/no | avoid = dangerous | Civil Comments: toxicity ≥ 0.5 toxic, ≤ 0.1 safe; 3,000 per side from train, 500 per side from test, seeded |
| **junk?** | yes/no | avoid = junk | SMS Spam (5,574, 747 spam): 70/30 stratified split, seeded |
| **which one?** | one of N | a T-maze: smells each option, goes to the one he approaches most | pairs and fours drawn from the three sets above; no new data |
| **seen it before?** | yes/no | the α'3 novelty compartment (familiar smells drive it less) | a stream with exact repeats and paraphrases; **its own gate later**: it is lifetime memory, not the nose |

## The nose: candidates

Frozen, general-purpose, none trained for sentiment, toxicity or spam:
CLIP ViT-B/32 (today's), all-mpnet-base-v2, bge-large-en-v1.5,
mxbai-embed-large-v1, e5-large-v2, and two that see pictures in the same
space as text: jina-clip-v2 and nomic-embed v1.5 (text + vision).

## What is measured, per nose and question

A logistic model on held-out data (balanced accuracy), (a) on the raw
embedding and (b) **through his antenna**: 26 principal components ± onto 52
glomeruli, fit on 1,500 unlabelled training texts pooled across the questions
(labels unused). A fly has 53 glomeruli, so (b) is the most any brain can
get from that nose. Pictures are measured only for the noses that see them.

## The rule for choosing (fixed now)

1. Score each nose by its **mean antenna ceiling over the three text
   questions**.
2. If the best multimodal nose is within **0.02** of the best text-only nose,
   take the multimodal one: one nose, text and pictures summed at the antenna,
   as now.
3. Otherwise take the best text-only nose for text, and the best picture nose
   drives the Kenyon cells that receive visual input in a fly (KCγ-d,
   KCαβ-p), which is how v1 fed him pictures. Two senses, one brain.
4. If the antenna costs more than **0.05** against the raw ceiling on any
   question, that is reported, and whether to give him more input channels
   is decided in A2's pre-registration, not here.

Runner: `scripts/a2_phase1.py`. Results: `docs/a2-phase1-results.md`.
