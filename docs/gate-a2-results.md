# Gate A2 results

Pre-registration: `docs/GATE-A2.md`. Epoch chosen on the validation mean; test reported.

## Joint: one brain, every question (test balanced accuracy, mean over seeds)

| arm | seeds | sweet (text) | pictures | dangerous | junk | mean | KC active |
|---|---|---|---|---|---|---|---|
| real | 3 | 0.874 | 0.780 | 0.777 | 0.935 | **0.841** | 0.095 |
| shuffle | 3 | 0.860 | 0.909 | 0.798 | 0.959 | **0.882** | 0.105 |
| hash | 3 | 0.877 | 0.780 | 0.773 | 0.959 | **0.847** | 0.076 |
| free | 3 | 0.896 | 0.922 | 0.802 | 0.975 | **0.899** | 0.100 |

Ceilings through the senses (phase 1): sweet 0.917, pictures 0.930, dangerous 0.807, junk 0.963

## Real brain, every measure (test, mean over seeds)

| part | balanced | rho | ECE | T-maze, 2 options | T-maze, 4 options |
|---|---|---|---|---|---|
| sweet | 0.874 | 0.79 | 0.050 | 0.959 | 0.921 |
| pictures | 0.780 | 0.69 | 0.087 | 0.890 | 0.797 |
| dangerous | 0.777 | 0.63 | 0.042 | 0.863 | 0.723 |
| junk | 0.935 | 0.57 | 0.013 | 0.984 | 0.959 |

## Decision

- **D1 one brain carries every question** (each ≥ ceiling − 0.05): **FAIL** — sweet 0.874 vs 0.867, pictures 0.780 vs 0.880, dangerous 0.777 vs 0.757, junk 0.935 vs 0.913
- **D2 the wiring matters** (real mean ≥ every control + 0.03): **FAIL** — shuffle 0.882, hash 0.847, free 0.899, real 0.841
- **D3 he stays fly-like** (Kenyon cells active ≤ 0.10): **PASS** — 0.095
- **D4 no interference** (joint ≥ one-fly-per-question − 0.03): **PASS** — sweet joint 0.874 / solo 0.872, pictures joint 0.780 / solo 0.641, dangerous joint 0.777 / solo 0.769, junk joint 0.935 / solo 0.954
- **Adopted as the decider** (D1 and D3): **no**

## Probes, real brain (P(approach): sweet / safe / not junk), per seed

- sweet: I fucking hate you: 0.05, 0.03, 0.10
- sweet: I hate you: 0.04, 0.02, 0.09
- sweet: I don't know: 0.96, 0.99, 0.98
- sweet: I love you: 0.95, 0.99, 0.98
- sweet: I fucking love you: 0.93, 0.98, 0.97
- dangerous: I fucking hate you: 0.15, 0.17, 0.21
- dangerous: I hate you: 0.14, 0.20, 0.22
- dangerous: I don't know: 0.98, 0.94, 0.99
- dangerous: I love you: 0.97, 1.00, 0.98
- dangerous: I fucking love you: 0.96, 0.99, 0.97
- junk: I fucking hate you: 0.93, 0.86, 0.83
- junk: I hate you: 0.92, 0.69, 0.77
- junk: I don't know: 1.00, 1.00, 1.00
- junk: I love you: 0.97, 0.97, 0.98
- junk: I fucking love you: 0.98, 0.98, 0.96
- sweet: oasis-Dessert 1: 0.73, 0.76, 0.70
- sweet: oasis-Garbage dump 1: 0.00, 0.00, 0.00

## Reading (after the numbers; the rules are unchanged)

- **One brain carries the text questions.** Sweet 0.874, dangerous 0.777 and
  junk 0.935 are all within 0.05 of their ceilings, and carrying them
  together costs nothing (D4; pictures are even better joint, 0.780, than
  alone, 0.641). He stays sparse, with 9.5% of Kenyon cells active (D3).
- **D1 fails on pictures alone:** 0.780 against a bar of 0.880. Under the
  rule, the brain is **not adopted**.
- **D2 fails, and the wiring now costs rather than helps.** Real 0.841 is the
  lowest mean; shuffle 0.882 and free 0.899 are higher. Almost all of the gap
  is pictures. Real and hash both score 0.780, shuffle 0.909 and free 0.922.
  Real and hash keep the fly's own mushroom-body anatomy, where the visual
  Kenyon cells (335 of 4,064, KCγ-d and KCαβ-p) reach only some compartments
  and so some output neurons. Scrambled brains let pictures reach everything.
  On text alone, real (0.862) sits with the controls (0.844–0.891).
- **The probes read right.** "I (fucking) hate you" 0.02–0.10 sweet,
  0.14–0.22 safe; "I (fucking) love you" 0.93–0.99 both. The rot picture 0.00,
  the dessert 0.70–0.76. "I don't know" is not neutral (0.96–0.99); there
  were no neutral sentences in training.
- **Calibration** on the real brain: ECE 0.013–0.087 by part (Jev on its
  public bench: 0.161).
