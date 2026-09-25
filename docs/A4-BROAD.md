# A4 broad: one fly taught a catalogue of questions (pre-registered 2026-09-25)

Nick, after the A4 pilot and the A4b development round: a fly cannot answer truly unseen
kinds better than his own nose (`docs/a4b-dev-results.md`), so make him **broad**: teach
him many kinds, and measure (1) whether one fly carries them all, and (2) how far what he
was taught carries to new data on the same concepts. "Let's try broad."

## Anatomy check

- The corrected v2 cut brain (`docs/GATE-A5.md`): per-type parameters and KC→MBON trained;
  answers from the MBONs grouped by measured DAN input. Text only.
- **Nose: bi46** (`a4b-dev-results.md`, L1): 46 components, fit label-free on training
  items and taught label words, whitened, one per non-innate glomerulus around a resting
  rate of 0.5. v2 reason: an ORN fires at rest and a smell pushes it up or down (Hallem &
  Carlson 2006), so a glomerulus carries a signed number. It replaces the ± split, which
  spent two glomeruli per component. Through it, nose-alone matching on the practice kinds
  was 0.362 vs 0.300 for the old antenna.
- **Start: a new label-free rule, `fly_init_keep`.** The A5 rule took the (gain,
  threshold) with the widest output spread (gain 8), which keeps only 0.53 of the input's
  similarity structure at the ORNs and 0.37 at the KCs. A real antennal lobe and KC layer
  preserve odour similarity. New rule: over gain ∈ {0.5, 1, 2, 4, 8} × threshold ∈ {0.05,
  0.2}, after the same KC (5%) and MBON (0.2) bisections, keep the points where both read
  MBON groups are neither silent nor saturated and the raw read spread is ≥ 1e-4, and take
  the one whose **KC-code similarity best tracks input similarity** (Pearson correlation
  over pairs of unlabelled calibration smells: 64 training items and 64 taught label
  words). Then the read scale is set so the logit's std is 1 (A4 pilot amendment 1).
  No answers are used.
- A sniff is the item's smell and one option's smell pushing the glomeruli from rest
  together (`0.5 + (item − 0.5) + (option − 0.5)`, clipped). He goes down one arm per
  item (softmax over its options).
- The question is carried by the options: every kind here is "which of these label words
  fits this text". Two kinds whose label words are identical but mean different things
  would collide; the only such pair, glue/cola and ethics/commonsense (both
  "acceptable/unacceptable"), is resolved by translating cola's labels to
  "grammatical/ungrammatical".

## Data and leakage check

- **Taught kinds: 44**, every single-text tasksource kind with meaningful label words
  (the 33 training-pool and 11 practice kinds of `A4B-DEV.md`). Labels made readable as in
  A4b dev. Per kind, drawn at random (seed 20260925): test min(200, 20%), validation
  min(100, 10%), training the rest up to 1,000.
- **Cold: BTZSC, 22 tasks, 200 items each**, the same items as the A4 pilot, bare text,
  label words made readable (`toxicaggregated` → "toxic"). Sealed since the pilot; not
  used in A4b dev.
- Leakage: test and cold items with a near-duplicate (cosine > 0.95) among training items
  are dropped and counted. BTZSC sources were excluded from tasksource by name and text
  (`a4-data.md`).
- **Cold tiers, fixed now from label words only:**
  - **Tier 1, taught label words:** amazonpolarity, appreviews, imdb, rottentomatoes,
    yelpreviews, financialphrasebank (sentiment), biasframes_offensive (offensive).
  - **Tier 2, taught concept, new label words:** emotiondair, empathetic (emotion);
    agnews, yahootopics (topic); banking77, massive (intent); wikitoxic_insult,
    wikitoxic_obscene, wikitoxic_threat, wikitoxic_toxicaggregated (toxicity);
    biasframes_intent.
  - **Tier 3, new concept:** biasframes_sex, capsotu, manifesto, trueteacher.

## Arms

| arm | what |
|---|---|
| **real** | the fly, per type + KC→MBON, fly_init_keep |
| layered | the layered shuffle, trained the same |
| plain net | two layers (256) on [item, option, item×option] through the same bi46 antenna, same training |
| nose alone | cosine, untrained: on all 1,024 numbers, and through bi46 |

Brains: Adam 3e-3, the KC pressure, 8 epochs, epoch chosen on the taught kinds'
validation. Plain net: Adam 1e-3, 20 epochs, chosen the same way. Seed 1. Both brains pass
the maze preflight first (as the A4 pilot: KC band, read not dead, loss falls, gradients
reach the brain), and the smoke numbers are checked against those bars before launch.

## Metric

**Balanced accuracy** (mean recall over a task's labels) per kind, then macro-averaged.
It fixes the A4 pilot's flaw, where plain accuracy let majority-label guessing look good
on skewed tasks. Chance is 1/(number of labels).

## Rules (fixed now)

- **G1, one fly carries the catalogue:** real's macro on the 44 taught kinds' test ≥
  plain net − 0.05 **and** ≥ chance + 0.15.
- **G2, taught labels carry to new data:** on tier 1, real's macro ≥ nose alone (all
  1,024 numbers).
- Reported beside them, not gating: tiers 2 and 3 for every arm; real vs layered on
  everything; per-kind tables.
- If G1 fails, the broad catalogue is not the public v1, and the report says what he
  carries. If G1 passes and G2 fails, the catalogue is served as named questions only
  (each kind with its own label words), and new data on a taught concept is not claimed.
- If the preflight marks the new start slow or broken, the run stops and that is
  reported. The A5 start rule is **not** swapped in quietly.

Runner: `scripts/a4_broad.py`. Results: `docs/a4-broad-results.md`.
