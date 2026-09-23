# Gate B4 — what the numbers say (2026-09-23)

Table: `gate-b4-results.md` (generated). Runs: `runs/gate-b4/`. Caches and
antenna: B3's. Locked at `gate-b4-prereg`, run at `gate-b4-run`.

## The bar

| | result | bar | |
|---|---|---|---|
| T1 generalisation (balanced, own neutral, best epoch) | **0.631** | ≥ 0.9 × 0.697 = 0.628 | **PASS** |
| T1 memory (recall gap) | +0.066 | ≥ +0.30 | FAIL |
| T2 reversal (recover to 0.9 × pre-flip within 6 epochs) | epoch 2 | | **PASS** |

## What works

**The fly's rule, given a fly's training, tastes sentences it has never seen
at 0.63 balanced accuracy** — 90.5% of what a logistic regression fitted on
the same 400 sentences reaches at the same antenna (0.697), and *above* a
nearest-neighbour memory on the fly's own Kenyon-cell code (0.604). On
pictures, 0.74 against a 0.81 ceiling. It learns in **one epoch** — one
pairing per item — and more training makes it slightly worse (0.63 → 0.60),
which is the memory crowding under one-way depression. It **reverses**: flip
every taste and it is back to 0.9 × its old accuracy after two passes,
where a refit regression in B2 was stuck below chance. It **forgets
partly overnight** (0.60 → 0.57 after 24 h of silence). And a taste learned
on sentences **orders pictures** it was never trained on at rho +0.64.

The single change from B3 was to read "sweet" against the fly's *own*
neutral point (the midpoint of its recall of what it was trained on) rather
than against 0.5. Its neutral sits at 0.457 on text and 0.55–0.59 on
pictures; scoring at 0.5 had thrown away most of what it knew.

## What does not

- **Recall gaps are small** (+0.04 to +0.08) and always will be under this
  rule: the score is a difference of two mean depressions over cells that
  hundreds of items share, so it lives in a narrow band. The band *orders*
  fine (rho 0.28 text, 0.56 pictures). The +0.30 bar was wrong for the
  quantity; the fail is recorded, and the bar should not be reused.
- **The neutral does not carry across modalities**: at the text-trained
  neutral, pictures come out 0.55 balanced despite rho +0.64. Each modality
  wants its own zero, which is a two-number calibration, not a learning
  problem.
- **Real ≈ hash, again**: 0.631 vs 0.621 on text, 0.736 vs 0.716 on
  pictures. The shuffle (a sixth of the plastic synapses) is 0.57 / 0.70.
  Architecture, not wiring, as B2 said.

## Probe sheet (real wiring, own neutral 0.457)

| probe | score | reads |
|---|---|---|
| I fucking hate you | 0.452 | bitter |
| I hate you | 0.452 | bitter |
| I don't know | 0.456 | neutral |
| I love you | 0.467 | sweet |
| I fucking love you | 0.460 | sweet, less |
| OASIS "Dessert 1" | 0.508 | sweet |
| OASIS "Garbage dump 1" | 0.457 | neutral-bitter |

Right side for all seven; "fucking" no longer flips "love" to bitter; the
intensity order between "hate" and "fucking hate" is a tie at this
resolution. A fly with a small memory of movie reviews.

## Where this leaves the API

The decider exists: text or picture in, a score out, learned online from
sugar and shock, forgetting on a fly's clock, reversible. What it needs to
be a *product* is calibration — a neutral per modality and a confidence that
means something — and both are measurement, not learning. The 8-seed
"sureness" is there already; whether it is calibrated is the next number to
look at.

Nothing in this note changes anything; B4 stands as run.
