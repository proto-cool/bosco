# Gate A5 results

Pre-registration: `docs/GATE-A5.md`. Test; epoch chosen on validation. Mean over seeds.

| arm | seeds | sweet | pictures | dangerous | junk | mean | KC active | behaviour agrees |
|---|---|---|---|---|---|---|---|---|
| real-type | 3 | 0.909 | 0.865 | 0.773 | 0.903 | **0.863** | 0.070 | 40% |
| layered-type | 3 | 0.900 | 0.912 | 0.782 | 0.954 | **0.887** | 0.160 | 65% |
| hash-type | 3 | 0.908 | 0.907 | 0.770 | 0.919 | **0.876** | 0.058 | 47% |
| free-type | 3 | 0.920 | 0.855 | 0.797 | 0.949 | **0.880** | 0.021 | 71% |
| real-neuron | 3 | 0.917 | 0.885 | 0.793 | 0.937 | **0.883** | 0.062 | 46% |
| reference (logistic on the senses) | | 0.929 | 0.939 | 0.817 | 0.971 | | | |

## Real brain (per type): every measure

| part | balanced | AUROC (2-option T-maze) | 4-option T-maze | ECE |
|---|---|---|---|---|
| sweet | 0.909 | 0.972 | 0.923 | 0.025 |
| pictures | 0.865 | 0.951 | 0.894 | 0.105 |
| dangerous | 0.773 | 0.851 | 0.681 | 0.028 |
| junk | 0.903 | 0.981 | 0.945 | 0.022 |

## Decision

- **D1 product** (each part ≥ reference − 0.05): **FAIL**: sweet 0.909 vs 0.879, pictures 0.865 vs 0.889, dangerous 0.773 vs 0.767, junk 0.903 vs 0.921
- **D2 wiring** (real ≥ every control + 0.03): **FAIL**: real 0.863, layered 0.887, hash 0.876, free 0.880
- **D3 fly-like** (KCs active ≤ 0.10): **PASS**: 0.070
- **D4 cost of the constraint:** real-neuron 0.883 − real-type 0.863 = +0.021
- **Adopted as the first Bosco version:** no

## Seed-to-seed disagreement (share of test items on different sides)

- real-type: 6.7%
- layered-type: 5.1%
- hash-type: 4.7%
- free-type: 6.2%
- real-neuron: 6.3%

## Probes (real per type, P(approach) by seed; behaviour read in brackets)

- sweet: I fucking hate you: 0.38 (+0.01), 0.81 (-0.06), 0.48 (+0.01)
- sweet: I hate you: 0.47 (+0.01), 0.84 (-0.06), 0.75 (+0.01)
- sweet: I don't know: 0.69 (-0.00), 0.93 (-0.05), 0.62 (+0.01)
- sweet: I love you: 0.95 (+0.01), 0.98 (-0.05), 0.98 (+0.01)
- sweet: I fucking love you: 0.97 (+0.01), 0.99 (-0.05), 0.98 (+0.01)
- dangerous: I fucking hate you: 0.17 (+0.00), 0.15 (-0.07), 0.18 (+0.01)
- dangerous: I hate you: 0.16 (+0.00), 0.14 (-0.07), 0.24 (+0.01)
- dangerous: I don't know: 0.88 (-0.00), 0.95 (-0.05), 0.91 (+0.01)
- dangerous: I love you: 0.95 (+0.00), 0.98 (-0.05), 0.97 (+0.01)
- dangerous: I fucking love you: 0.88 (+0.00), 0.96 (-0.06), 0.94 (+0.01)
- junk: I fucking hate you: 0.98 (+0.01), 0.99 (-0.06), 0.99 (+0.01)
- junk: I hate you: 0.99 (+0.01), 1.00 (-0.05), 1.00 (+0.01)
- junk: I don't know: 1.00 (-0.01), 1.00 (-0.04), 0.99 (+0.01)
- junk: I love you: 0.99 (+0.01), 1.00 (-0.05), 0.99 (+0.01)
- junk: I fucking love you: 0.99 (+0.01), 1.00 (-0.05), 1.00 (+0.01)
- sweet: Dessert 1: 0.99 (-0.38), 1.00 (-0.46), 1.00 (-0.19)
- sweet: Dessert 2: 1.00 (-0.21), 1.00 (-0.37), 1.00 (-0.15)
- sweet: Dessert 3: 1.00 (-0.23), 1.00 (-0.37), 1.00 (-0.15)
- sweet: Dessert 4: 1.00 (-0.36), 1.00 (-0.50), 1.00 (-0.30)
- sweet: Dessert 5: 1.00 (-0.40), 1.00 (-0.48), 1.00 (-0.30)
- sweet: Dessert 6: 1.00 (-0.22), 1.00 (-0.37), 1.00 (-0.11)
- sweet: Dessert 7: 1.00 (-0.31), 1.00 (-0.41), 1.00 (-0.16)
- sweet: Dessert 8: 1.00 (-0.35), 1.00 (-0.44), 1.00 (-0.24)
- sweet: Garbage dump 1: 0.00 (-0.05), 0.01 (-0.13), 0.00 (+0.02)
- sweet: Garbage dump 2: 0.04 (-0.07), 0.31 (-0.12), 0.40 (+0.02)
- sweet: Garbage dump 3: 0.03 (-0.11), 0.19 (-0.18), 0.07 (-0.01)
- sweet: Garbage dump 4: 0.01 (-0.15), 0.10 (-0.24), 0.01 (-0.07)
- sweet: Garbage dump 5: 0.00 (-0.14), 0.01 (-0.19), 0.00 (-0.02)
- sweet: Garbage dump 6: 0.00 (-0.18), 0.12 (-0.25), 0.01 (-0.09)
- sweet: Garbage dump 7: 0.02 (-0.21), 0.20 (-0.27), 0.07 (-0.12)
- sweet: Garbage dump 8: 0.01 (-0.08), 0.06 (-0.13), 0.03 (+0.02)

## Reading (after the numbers; the rules are unchanged)

- **Not adopted.** D1 fails on two parts, narrowly: pictures 0.865 against a
  bar of 0.889 (−0.024), and junk 0.903 against 0.921 (−0.018). Sweet (0.909)
  and dangerous (0.773) pass.
- **Pictures fell from validation to test** (validation 0.90–0.95 per seed,
  test 0.865). The test set is 121 pictures from themes never seen in
  training, so it is both harder and noisy (about ±0.03).
- **D2 fails as in every gate:** real 0.863 is the lowest; the controls
  score 0.876–0.887. The fly's wiring gives no accuracy edge on these tasks.
- **D3 passes:** 7% of KCs active, in the fly's range.
- **D4:** the per-type constraint costs 0.021 (per-neuron 0.883).
- **He was still learning when training stopped.** The real brain's best
  epoch was 9, 10 and 10 (the limit), and per-neuron 9, 9, 9. Layered peaked
  at 5–9. The pre-registered 10 epochs likely undertrained him.
- **Held-out probe pictures are right** (themes never in training): the 8
  desserts score 0.99–1.00 sweet in every seed; the 8 garbage dumps score
  0.00–0.40 (bitter) in every seed.
- **Text probes are mixed:** "I love you" 0.95–0.99 sweet; "I hate you"
  0.47 / 0.84 / 0.75 across seeds, wrong in two. Dangerous probes are right
  (hate 0.14–0.24 safe; love 0.88–0.98). No probe sentence is junk, and all
  score not-junk (0.98–1.00).
- **Calibration:** ECE 0.022–0.028 on text, 0.105 on pictures.
- **The behaviour read agrees with the answer only 40% of the time:** his
  behaviour neurons are not trained, and they do not follow the MBON answer.
  So "what the fly would do" is not yet a meaningful readout.
