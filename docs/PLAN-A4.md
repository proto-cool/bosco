# A4 plan — a lot of questions (2026-09-24, a plan, not yet a pre-registration)

Nick: train him on many questions so he can answer ones he has never been
asked. "A *LOT* of them." Jev answers unseen questions because it was trained
on a great variety of them; this is the fly's version of that.

## Where the questions come from (checked to exist, 2026-09-24)

- **tasksource** (Sileo 2023; github.com/sileod/tasksource): 500+ English
  datasets harmonised into classification and multiple-choice templates with
  their labels exposed. The bulk of the training questions.
- **Super-NaturalInstructions** (Wang et al. 2022; github.com/allenai/
  natural-instructions): 1,616 tasks, each with an expert-written
  instruction. The classification tasks with a fixed label set give
  questions **already phrased in words**, which is what he smells.
- **BTZSC** (Aarab et al., ICLR 2026; huggingface.co/datasets/btzsc/btzsc):
  22 datasets across sentiment, topic, intent and emotion, built to test
  **zero-shot** classification. Published baselines include embedding models
  of his nose's kind. **Held out entirely:** the cold test.

## Shape

- Every task becomes Jev primitives: **noul** (yes/no), **choice** (one sniff
  per option, the T-maze) or **score** (one sniff per level). The question's
  words and each option's words are part of what he smells.
- **Question first, then the thing** (two sniffs): he smells the question,
  and while it is still echoing he smells what is being judged. That keeps
  the question from being squeezed into the same 52 channels as the thing.
  It depends on A3 phase 1 showing that two pieces survive the nose.
- Trained on hundreds of questions at once, question-in-smell. Training on
  that many will be long: days on the Mac's GPU, which is now acceptable.
- **Capacity:** if the 82k per-neuron settings cannot hold hundreds of
  questions, the next thing to train is the KC→MBON synapses themselves
  (61k), which is where a fly stores what it learns. Decided in the
  pre-registration, not after.

## The test that matters

**Whole questions held out**, never trained on, asked cold:
1. BTZSC's 22 datasets.
2. Jev's benchmark (routing, yes/no, rating, ambiguity), for the direct
   comparison.
3. A2's sweet, dangerous and junk questions, removed from training.

Nothing from those sources, or anything derived from them (SST, BANKING77,
BoolQ, Yelp, MultiNLI/ChaosNLI, Civil Comments, SMS Spam and BTZSC's 22),
enters training. Checked by dataset name and by text overlap.

## Controls

- **The nose alone:** the same encoder answering cold by similarity between
  the text and each option's words, with no brain. If Bosco cannot beat his
  own nose on unseen questions, the brain adds nothing there.
- **Scrambled brains** (shuffle, hash, free), trained the same way.
- Published BTZSC and Jev-bench numbers, for where he stands.

## Order

A2 (running) → A3 phase 1 (running) → the A4 data build (collect, convert to
primitives, de-duplicate against held-out sets) → A4 pre-registration → run.

## Before A4: can his nose read passages? (added 2026-09-24)

Jev scores 94% on passage yes/no (BoolQ). Bosco's one-fingerprint nose tops
out at 0.66 (A3 phase 1), and two separate fingerprints do not help. Before
A4 fixes his nose, a ceiling check with rules written first:
1. **sentence-by-sentence sniffing:** the question with each sentence, the
   fly finding and judging (the most fly-like; no new encoder);
2. **a question-aware embedder:** an instruction-conditioned fingerprint (it
   still only perceives);
3. **an LLM-backbone embedder,** measured only as a reference for how much
   reading the translator would be doing. Not a candidate.
Nick, 2026-09-24: "fair and honest".
