# Brain checks, 2026-09-28

> **Superseded, 2026-09-29.** See `docs/audit-2026-09-29-full.md`. Several checks below were invalid (T7 measures the
> ratio of the read scales k; T2 and T3 are confounded; T4's R² measures alignment, not share), and these claims are
> retracted:
> - "7× memory transport";
> - "14% / 19% innate";
> - "about 7× potentiation" (the real maximum is 166,000×). (BRAIN-SPEC with amendment 1)

**Verdict: the brain does not yet pass.** Thirteen of the fourteen checks pass. **T3 fails narrowly:** 78.9% of
answers flip, against a bar of 80%. The standard error is about 0.9 points (n = 1,865), so the miss is within
noise, but the bar was fixed in advance and is not re-read. The layered control learns the task about as well as
the real wiring (0.724 against 0.727).

Everything was run on the Mac CPU (label-free checks and scoring, through the served path `RateBrain3.answer`),
except the training, which ran on the 3080. Outputs:
- `runs/brain-check/`: the real brain and the layered and hash controls. Run 1, at rest 0.1 with the anatomical DN
  read, is in `run1-rest0.1/`.
- `runs/brain-train/checks.json`: the trained checks.

## Label-free checks (L1–L6)

| # | check | real | bar | result |
|---|---|---|---|---|
| L1 | starts from rest | converged; a resting sniff drifts 3e-8 | fixed point | pass |
| L2 | ORN rest rate | 0.096 | ≤ 0.15 | pass (run 1 at rest 0.1: 0.154, fail) |
| L3 | KCs active per sniff | 4.8% (range over items 0.6–13.7%) | 2–10% | pass |
| L4 | KCs ever active / Jaccard between items | 99.7% / 0.029 | ≥ 50% / ≤ 0.25 | pass (v2: about 20% / 0.26–0.55) |
| L5 | read s.d. across items, untrained | 0.00075 | amended to trained, below | reported |
| L6 | one answer, CPU, 4 threads | **0.102 s** | ≤ 1.1 s | pass (0.94 s before today) |
| — | determinism | bit-identical on repeat and at 1, 2, 3, 4 and 10 threads | — | pass |

Controls, label-free:

| arm | ORN rest | KC per sniff | ever active | Jaccard | untrained read s.d. |
|---|---|---|---|---|---|
| real | 0.096 | 4.8% | 99.7% | 0.029 | 0.00075 |
| layered | 0.015 | 4.3% | 99.8% | 0.021 | 0.00027 |
| hash | 0.096 | 4.6% | 99.8% | 0.027 | 0.00076 |
| real, mushroom body silenced | — | 0 | — | — | 0.00073 |

## Trained checks (T1–T6)

Task `harmful`: train 5,000/5,000, validation 1,865 (76% harmful). Main arm: only the KC→MBON memory, k and c
train, for 3 epochs.

| # | check | result | bar | verdict |
|---|---|---|---|---|
| T1 | val balanced accuracy | 0.727 (plain logistic on the same 46 numbers: 0.735) | ≥ 0.685 | pass |
| T2 | memory reset | 0.534, a drop of 0.193 | drop ≥ 0.170 | pass |
| T3 | flipped memory swapped in | **78.9%** of answers flip | ≥ 80% | **fail** |
| T4 | recall, harmful / not | 0.736 / 0.718 | both ≥ 0.5 | pass |
| T4 | untrained share (R²) | 0.134 | ≤ 0.2 | pass |
| T5 | KCs with trained weights | 7.7% per sniff, 99.2% ever active, Jaccard 0.115 | L3, L4 | pass |
| T5 | L5, trained read s.d. | **0.012** (k = 8.5, logit s.d. 1.0) | ≥ 0.01 | pass |
| T6 | layered control, same recipe | 0.724 (0.704 / 0.745); reset 0.50; read s.d. 0.0017 with k = 62 | reported | — |

## What this says

- **Fixed:**
  - the fly starts from rest;
  - his nose rests quietly;
  - nearly every KC is used, with distinct sets for different items;
  - the answer is read in one sniff;
  - it is 9× faster and bit-exact.
- **The memory is the learner.** Resetting it takes the answer to chance (T2), and a flipped memory flips 79% of
  answers (T3, just short of the bar).
- **The real wiring does not beat the scrambled wiring on accuracy** (0.727 against 0.724), as in v2. It differs in
  how: the real brain's answer rides on a read 7× larger (s.d. 0.012 against 0.0017), and the layered brain needs a
  read scale 7× larger to answer at all.
- **Before training, the item barely moves the DN read.** The untrained read s.d. is 0.00075, the same with the
  mushroom body silenced and in the controls, because the untrained MBONs do not tell items apart (amendment 1).

## For Nick
1. **T3:** accept a failing brain, or decide what to change. Options:
   - a longer or stronger recipe for the memory;
   - T3 on a larger validation set, so that noise cannot flip the verdict (audit decision 5);
   - rethink why 21% of answers do not follow the memory (the innate path through the lateral horn).
2. **The real-versus-layered tie** is the question BRAIN-SPEC T2/T3 were meant to answer about "what the fly adds".
   T6 has no bar. Should one be set?
3. **Speed:** 80 steps kept. The step-60 answer correlates 0.9996 with step 80 (untrained).

---

## Update: 5 seeds and the sampling-error rule (amendment 2)

`runs/brain-train/checks.json`:
- seeds 1–5 for the real brain, the real brain on flipped labels, and the layered control;
- 2,000 bootstrap draws over seeds and items.

**Verdict: 7 of 9 pass. T2 and T3 fail:** both clear their bars on the mean, but not at the 5% bound. The single
seed-1 run had passed T2 and missed T3 by a hair; with 5 seeds the picture is steadier and the same as before.

| check | mean over seeds | 5th / 95th percentile | bar | result |
|---|---|---|---|---|
| T1 brain − logistic balanced accuracy (logistic 0.735) | −0.010 | −0.027 / +0.007 | p5 ≥ −0.05 | pass |
| T2 drop on reset − 0.75 × margin | +0.014 | **−0.008** / +0.035 | p5 ≥ 0 | **fail** |
| T3 answers flipped by the swapped memory | 0.809 | **0.777** / 0.838 | p5 ≥ 0.80 | **fail** |
| T4 lower recall | 0.691 | 0.650 / 0.725 | p5 ≥ 0.5 | pass |
| T4 untrained share (R²) | 0.141 | 0.117 / 0.167 | p95 ≤ 0.2 | pass |
| T5 KCs, trained, every seed | 7.4–9.0% active, ≥ 99.2% ever, Jaccard 0.10–0.19 | — | L3, L4 | pass |
| T5 L5, trained read s.d. | 0.0120 | 0.0116 / 0.0124 | p5 ≥ 0.01 | pass |
| **T7 real ÷ layered read s.d.** | **7.10** | 6.93 / 7.27 | p5 ≥ 2 | **pass** |
| T6 real − layered balanced accuracy | +0.001 | −0.013 / +0.016 | reported | tie |

Per seed: balanced accuracy 0.715–0.730; with the memory reset, 0.503–0.583; flipped share 0.75–0.86.

**What it means:**
- **The memory carries most, not all, of the answer.** About 14% of the trained logit variance, and about 19% of
  the answers, follow the path that does not learn: the item reaches the DNs through the lateral horn and the rest
  of the brain, and k scales that path together with the memory. In a fly, innate and learned paths both drive
  behaviour, but the spec asks for the memory to dominate, and it does not reach the bars.
- **The real wiring carries the learned memory to the motor side 7× better than scrambled wiring (T7).** Accuracy
  is a tie (T6).
- **The KC drift (4.8% to 7–9% active after training) is explained.** Training potentiates KC→MBON: the avoid
  MBONs go from a mean rate of 0.041 to 0.307, and the approach MBONs from 0.038 to 0.115. Their output lowers
  APL's rate (0.029 to 0.025), which releases the KCs. The direct MBON→KC feedback works against the drift:
  cutting it raises KC activity further, from 7.7% to 8.6%. Real KC→MBON learning is mainly depression (Hige et al.
  2015); ours is unbounded both ways.

---

## Update: amendment 3, depression-only learning, fails, and is not adopted

`runs/brain-train-depress/checks.json`: 5 seeds, the same bootstrap as amendment 2.

| check | mean | 5th / 95th percentile | result |
|---|---|---|---|
| T1 balanced accuracy (logistic 0.735) | **0.515** | 0.500 / 0.539 | **fail** |
| T2 | −0.008 | −0.022 / 0.000 | **fail** |
| T3 flipped share | **0.013** | 0.000 / 0.031 | **fail** |
| T4 lower recall / untrained share | 0.068 / 0.985 | — | **fail** |
| T5 L5 trained read s.d. | 0.0007 | — | **fail** |
| T5 KCs (every seed) | 5.0% active, 99.6% ever, Jaccard 0.032 | — | pass (no drift) |
| T7 ratio | 2.77 | 2.67 / 2.88 | pass |

He did not learn: four of the five seeds answer 0.500 (always one class), and the fifth reaches 0.576.

**Why.** At the label-free operating point, the MBONs are held up by their own thresholds, not by the KCs.
- Silencing every KC→approach-MBON synapse lowers those MBONs only from a rate of 0.038 to 0.034.
- The whole range that depression can reach on the read is about ±0.0007, the size of the untrained item spread.
- Strengthening the same synapses 7.4× moves the read +0.0046.

So a weakening-only memory has almost nothing to remove. In the fly, MBON odour responses are driven by the
KCs, and learning cuts them. Our MBONs, and the KCs' weak output (active KCs fire at a median rate of 0.015), are
not at that operating point. Amendment 2's results (potentiation allowed) stand.
