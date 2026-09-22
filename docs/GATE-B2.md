# Gate B2 — pre-registration (FINAL 2026-09-22; binding)

Gate B (`GATE-B.md`, tag `gate-b-run`) was run as written and answered **NO**
for every arm at every k — real 0.49, shuffle 0.51, hash 0.51, and the
logistic baseline 0.48–0.52 (`gate-b-results.md`). That result stands. This
gate is the same question asked again with the two things B's diagnosis found
broken in the harness fixed, and nothing else changed. Everything not stated
here is as in `GATE-B.md`: arms, encoder, tasks T1–T6, reward protocol, seeds,
metrics, decision rule.

## What B's diagnosis found (on the real arm's saved state, T1 seed 1)

1. **Every sentence was the same smell.** Raw CLIP text embeddings sit in one
   cone (mean pairwise cosine 0.765). Rectified and projected, a sweet
   sentence shared 60 of its ~109 active Kenyon cells with another sweet one
   and 59 with a bitter one. 400 sweet and 400 bitter pairings therefore
   depressed the same cells on both sides, and contrast cancelled it to zero
   for everyone. The rule could not even recall the items it had been paired
   with: the last 30 sweet-trained read 0.495, the last 30 bitter-trained
   0.482.
2. **The logistic baseline did not learn.** Constant-step SGD at the step I
   fixed barely moved: 0.53 in a plain loop where a batch logistic regression
   on the same 800 labels reaches 0.70. A baseline that does not learn is not
   context.

The 53-glomerulus bottleneck itself is not the cause: a logistic regression
on the 53-d drive reaches 0.63 against 0.70 on the full embedding.

## The two changes

1. **Centre the embedding on the calibration mean before projecting.**
   `e' = (e − μ) / ‖e − μ‖`, with μ the mean CLIP embedding of the 500
   calibration sentences (label-free, fixed once, the same for text and
   images). Mean pairwise cosine falls from 0.765 to 0.009; shared Kenyon
   cells between two sentences fall from ~60 to ~11 of ~104. The one scalar
   is recalibrated exactly as before (bisection on 50 calibration sentences,
   verified on all 500, KC fraction on target). Still one fixed projection,
   still one scalar, still no label.
2. **The logistic arm refits.** At every reward received it refits a plain
   logistic regression (lbfgs, C = 1, sample weight = magnitude) on every
   reward so far, and scores with the fitted probability. No step size to
   choose. This is the ceiling of the encoder under the same rewards, which is
   what the arm was for.

## One metric added

**Recall**: at the end of a run, the last 30 rewarded sweet items and the last
30 rewarded bitter items (as trained: post-flip labels in T2) are scored
again. Mean score sweet-trained, mean bitter-trained, and the gap. This
separates "cannot remember" from "cannot generalise", which B could not.
Reported, not judged.

## The pilot, disclosed

Before writing this, the centred drive was tried once on the **real wiring
only**, 300 items every one paired, no protocol: recall sweet-trained 0.565
against bitter-trained 0.461; 100 held-out items 0.60 accuracy, rho 0.14. A
logistic regression on the same items' Kenyon-cell codes reaches 0.60. That
pilot chose nothing but whether to write this file; B2's numbers are fresh
runs under the protocol, with every arm.

## Decision rule

As in `GATE-B.md`: on T1 and T2, real beats both shuffle and hash at k = 50
and k = 200 by more than the pooled standard deviation over five seeds. The
logistic arm is context. T3–T6 reported.

## Provenance

`bosco.gateb` at the commit after tag `gate-b-run` (`GATE = "b2"`), runs under
`runs/gate-b2/`, drive under `runs/gate-b2/drive.json`.
