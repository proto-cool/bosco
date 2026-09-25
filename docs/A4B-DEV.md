# A4b development: finding a way to generalise (written 2026-09-25, before any of it runs)

Nick, after the A4 pilot failed: "I'd like for him to be a generalist, so we
need to move that needle." This is a **development** round, not a gate.
Nothing here is a claim about Bosco. It finds a recipe; the recipe then goes
into a pre-registered gate (A4b) against the same sealed cold test.

## BTZSC is sealed

The A4 pilot's cold test (BTZSC, 22 tasks) is **not used** in this round, for
anything. Every choice below is made on a separate **practice test**: whole
tasksource kinds held out from training. BTZSC is next scored in the A4b gate,
once, under rules written before it runs.

## What the pilot showed (its own numbers, `docs/a4-pilot-results.md`)

1. Training on 40 random kinds made every model worse on unseen kinds than
   the untrained nose. The collapse was on many-option questions.
2. Even on the training kinds, the brains reached only 0.54 (the plain net 0.60).
3. The 40 kinds were mostly reasoning over two texts (NLI), which the nose
   cannot carry, and each kind always had the same few option words, so
   remembering which option word usually wins was the easiest thing to learn.

## What is tried (the levers)

- **L1, the antenna.** Nose alone scores by cosine on all 1,024 numbers of
  the embedding. Bosco gets 46 channels (23 components ±). Measured: nose
  alone through his antenna, with the components fit on items only (the
  pilot's way) and on items and options together. If his antenna cannot
  carry "this option matches this text", no brain behind it can.
- **L2, what is trained on.** (a) The pilot's recipe: each kind with its own
  options. (b) **Kinds chosen to match the target**: single-text
  classification with meaningful label words (topic, sentiment, emotion,
  intent, dialogue act, stance, hate), never two-text reasoning. (c) (b) with
  **mixed-up options**: each training item is shown its right option plus
  wrong ones drawn from every kind's label words, 2 to 20 of them, so the
  only thing that pays is matching meaning, not remembering label words.
- **L3, how an option is smelled.** (a) The pilot's: item smell + option
  smell, clipped (a mixture). (b) **Item then option** where the brain's
  answer depends on both. Only if L1 and L2 leave a gap.
- Label words are made readable before embedding (`BookRestaurant` →
  "book restaurant", `hate_speech` → "hate speech"). That is translation, as
  BTZSC's own option sentences are.

## Models in this round

- nose alone (cosine), full and through the antenna
- the plain net (fast), for screening L2
- the real and layered brains, on the best one or two recipes only

## Practice test

Held-out tasksource kinds with meaningful labels, 200 items each, chosen
before any model runs (listed in `docs/a4b-dev-results.md` by the build step):
no kind with the same source dataset as a training kind.

## What counts as progress

A recipe where the **real brain beats nose alone on the practice test by
≥ 0.03** (the A4 bar). Short of that, the round reports how close each lever
got and why.

Runner: `scripts/a4b_dev.py`. Results: `docs/a4b-dev-results.md`.
