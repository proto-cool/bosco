# Calibration on the B4 fly (2026-09-23)

Two measurements on top of `GATE-B4.md`, no learning changed. Script:
`scripts/calibration_b4.py`; numbers: `runs/gate-b4/calibration.json`. Real
and hash arms, five seeds, one training epoch (B4's best). Means over seeds.

## 1. A neutral per modality

The fly's neutral is not 0.5 and not the same for sentences and pictures:
0.457 on text after text training, 0.55 on pictures after picture training.
For a modality it was **never rewarded on** there is nothing to recall, so
the label-free candidate is the **median of its scores over the items it is
shown**.

Text-trained fly scoring all 900 OASIS pictures (T4), balanced accuracy:

| neutral | real | hash |
|---|---|---|
| the text neutral (B4 as run) | 0.528 | 0.520 |
| 0.5 | 0.715 | 0.673 |
| **median of the pictures' own scores** | **0.765** | 0.753 |
| (for comparison: trained on pictures, own neutral) | 0.745 | 0.725 |

rho is +0.65 either way; the neutral only decides which side "neutral" is.
With the picture median, a taste learned on sentences scores pictures it
was never rewarded on at 0.765 — slightly *better* than training on the
pictures themselves. That is the cross-modal result of B2 (rho +0.12) with
an antenna that keeps the signal and a zero that is the fly's.

**Rule:** the neutral for a modality is the midpoint of the fly's own recall
if it has been trained on that modality, else the median of what it has
been shown. Both are label-free.

## 2. An honest confidence

Two candidates for "how sure", each mapped to a probability of being right
by an isotonic fit and judged on held-out sentences (real arm):

| confidence | fitted on | AUROC | ECE | mean p | acc, top quarter | acc, bottom quarter |
|---|---|---|---|---|---|---|
| seed agreement (8 seeds) | trained items' recall | 0.534 | 0.147 | 0.76 | 0.653 | 0.598 |
| distance from neutral | trained items' recall | 0.571 | 0.147 | 0.76 | 0.714 | 0.584 |
| **distance from neutral** | **its own pre-pairing predictions** | 0.571 | **0.057** | 0.59 | 0.714 | 0.584 |

Held-out accuracy is 0.632 throughout. Two findings:

- **Seed agreement is nearly useless** (AUROC 0.53): the eight kernel seeds
  disagree about noise, not about difficulty. **Distance from the neutral**
  carries what confidence there is (0.57; the most-sure quarter is right
  71% of the time, the least-sure 58%).
- **Fitting the map on recalled trained items over-promises** (mean p 0.76
  against 0.63 actual; ECE 0.15), because the fly recalls what it was trained
  on better than it generalises. Fitting it on the fly's **own predictions
  made during training, before each item was paired** — a genuine held-out
  prediction with a known label, costing nothing — gives ECE **0.057**, mean
  p 0.59 against 0.63 actual. When it says ≥ 0.7 (5.6% of items), it is
  right **79.5%** of the time. When it is unsure, it says about 0.55.

That is the Jev property in the fly's own terms: a weak signal, honestly
reported. The confidence is not strong, and no calibration makes it strong;
what calibration does is make the number mean what it says.

**Rule:** confidence = distance from the neutral in units of the spread of
the fly's pre-pairing predictions, mapped by an isotonic fit on those
predictions. Re-fit whenever the fly is trained.

## What the decider is now

- input: text, or a picture, or both (drives summed at the antenna);
- output: a score 0..1 (1 sweet), a side against a per-modality neutral,
  and a probability of being right that is honest to about ±0.06;
- learning: online, one pairing per item, sugar or shock, on the fly's clock,
  reversible in two passes, forgetting overnight;
- ceiling: about 0.9 × a logistic regression at its own antenna on text,
  and 0.765 balanced on pictures from text alone.

Real ≈ hash on every number here (the hash arm is within 0.01–0.02 of the
real wiring throughout): the architecture is doing it.
