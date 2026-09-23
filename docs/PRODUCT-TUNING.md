# Product tuning on the B4 fly (2026-09-23)

Nick: "a good signal; needs some tweaking." This is what was tried, what it
did, and what the decider now runs. Tuning was done on a **validation** half
(200) of the 400 held-out sentences and reported on the **test** half (200);
nothing below was chosen on the test number. Scripts: the ablation and the
antenna-level scaling study are in the session record; the product cache and
evaluation are `scripts/product.py`; the extinction sweep is
`runs/product/extinction-sweep.json`.

## The measurement floor, first

Two independent runs of the same configuration (6,101 training sentences,
swarm of 5, extinction off) gave test 0.675 and 0.706 — the training subset
alone moves the number by 0.03. **Differences under about 0.03 on a 200-item
half are noise.** Only tweaks that move it more, consistently, count.

## What was tried

| tweak | what it is | text (val → test) | pictures from text | verdict |
|---|---|---|---|---|
| B4 as built | one fly, 400 sentences, PCA± antenna | 0.63 | 0.765 | baseline |
| count-weighted read | weight a cell's synapses by how hard it fired | −0.02 | — | no |
| neutral from all trained items | instead of the last 30 | ±0.00 | — | no |
| **swarm of 5** | five mushroom bodies trained in different orders, scores averaged | **+0.03 to +0.04; rho 0.33 → 0.45** | +0.02 | **yes** |
| whitened antenna | each principal component to unit variance (label-free) | ±0.00 on text | **0.765 → 0.708** | no |
| more training sentences | 400 → 1,000 → 2,000 → 6,101 | single fly: worse (0.615 → 0.536 val); swarm: +0.00 to +0.03, inconsistent | mixed | no |
| extinction (bidirectional rule) | unreinforced activity relieves the other side's depression, rate 0.05–0.4 | −0.01 to −0.03, monotone in the rate | −0.03 to −0.05 | no |

## Why more data does not help a fly

At the antenna a nearest-neighbour memory gains from data (kNN 0.651 →
0.699 with 6,254 sentences and whitening). The fly does not, and a single fly
gets *worse*. The rule's depression is one-way: every sweet pairing
depresses the reward-side synapses of the cells that fired, every bitter one
the punishment side, and with thousands of items each cell is on both sides
many times over. The verdict on a new sentence is a difference of two nearly
saturated means. The swarm averages the noise out of that difference; it
cannot put information back. Extinction was the candidate for a memory that
unlearns, and it made things slightly worse at every rate — as implemented,
it relieves both sides' depression about equally and the signal with it.
This is the fourth appearance of the same property (B2's reversal, B3's
decline over epochs, the single-fly data curve, and this), and it is the
rule's, not the harness's.

## What the decider runs now

- **Five flies** on the real wiring, one kernel (the Kenyon-cell codes are
  shared; each mushroom body pushes its own weights before reading them),
  each trained in its own order; the score is their mean.
- **The B4 antenna**, unwhitened: it keeps pictures at 0.765 from text alone.
- **400 training sentences** by default (`bootstrap --n-per-side 200`); more
  is allowed and does no harm with the swarm, and no reliable good either.
- A fifth of a bootstrap set is held back, scored by the swarm before it is
  paired, and used to calibrate `p_right`; live rewards keep adding to it.
- Everything else as B4 and `CALIBRATION-b4.md`.

Expected, on sentences it has never tasted: **about 0.66 balanced, rho
0.44**; on pictures from text alone, **about 0.76**; the encoder's own
ceiling on this text is 0.70–0.72. That is where the tweaking stops: what is
left is the learning rule, and changing it is a gate, not a tweak.

## Live use: a one-shot memory is gone by evening (2026-09-23, evening)

Nick asked whether the saturation seen in training would be a problem as the
decider lives and sees more rewards. Simulated: 60 biological days of live
rewards, each a new sentence paired once, scored on the 400 held-out sentences
every 10 days (single fly, real wiring, product cache).

| rewards/day | | d10 | d20 | d30 | d40 | d50 | d60 |
|---|---|---|---|---|---|---|---|
| 20 | as the rule stands | 0.51 | 0.43 | 0.60 | 0.52 | 0.50 | — |
| 20 | **rehearsal +1 h, +3 h, +24 h** | 0.57 | 0.55 | 0.59 | 0.61 | 0.58 | **0.63** |
| 100 | as the rule stands | 0.60 | 0.58 | 0.52 | 0.60 | 0.57 | — |
| 100 | **rehearsal +1 h, +3 h, +24 h** | 0.62 | 0.61 | 0.62 | 0.62 | 0.62 | **0.64** |

The live problem is the opposite of saturation: short-term memory fades in
~4 h and long-term memory forms only from a spaced repetition of the *same*
item, so a fly rewarded once per item knows only its last few hours. The
bootstrap's 0.66 was 400 pairings inside 3.3 hours, all still fresh.

**Rehearsal** is what a fly gets in training and what the decider now does
on its own: every rewarded item is paired again from its cached code at +1 h,
+3 h and +24 h (`REHEARSE_H`), applied when the clock passes, at zero kernel
cost. Long-term memory then forms (half the plastic synapses by day 60), the
score holds at 0.62–0.64 for two months with no drift, and no saturation
appears at these rates. It is scheduling inside the rule, not a change to it.
