# The brain must work first (written 2026-09-28, before any measurement against it)

Nick: "We have to get the Brain working first. If it's not actually working correctly the experiment is moot. Fix
it." The audit (`docs/audit-2026-09-28/model.md`) found that the fly was not doing the deciding:
- the option word and the item were blended on one nose;
- about 20% of the Kenyon cells did all the work;
- every sniff booted from zero;
- the answer rode on rate differences of about 0.001.

No specialist, gate or fleet work resumes until the brain passes every check below. Thresholds come from fly
biology or from the audit's failure modes, and are fixed here before anything is measured. A check that fails is
reported as failed.

## Scope

- **Model:**
  - the 50,140-neuron cut (decision 39: no added neurons);
  - the rate model, and one question at a time;
  - the item by smell, with no option word in the nose (the audit's F1);
  - yes/no as one sniff read as approach minus avoid, the fly's own valence.
- **Senses.** Taste and sight come after the brain passes; they are not in this spec.
- **Data:**
  - label-free checks use unlabelled item smells from `data/cache/v1-gate2` (validation and train texts only);
  - trained checks use the harm development pool (`data/cache/v1-harm-dev`, the `harmful` yes/no task,
    train/val only). No sealed test is touched.

## Label-free checks (the untrained brain at its label-free operating point)

| # | check | pass if | why |
|---|---|---|---|
| L1 | **Starts from rest** | every sniff starts from the brain's own resting state (smell at rest), not r = 0 | a fly is alive before the odour (audit F3) |
| L2 | **Quiet at rest** | mean ORN rate at rest ≤ 0.15 | real ORNs rest at a low spontaneous rate (F8: ours sat at about 0.42) |
| L3 | **Sparse KCs per sniff** | 2–10% of KCs active (rate > 0.01) per sniff, mean over 200 items | Honegger et al. 2011; Turner et al. 2008 |
| L4 | **Different items, different KCs** | over 200 different items, ≥ 50% of KCs are active at least once, **and** the mean pairwise Jaccard of active sets between different items is ≤ 0.25 | odours recruit largely distinct KC ensembles (F2: 18–21% coverage, Jaccard 0.26–0.55) |
| L5 | **The item reaches the answer** | s.d. across items of the raw read (approach − avoid DN mean, last 8 steps) ≥ 0.01 rate units | F5/§5.2: the answer rode on about 0.001 |
| L6 | **No slower** | CPU time for one yes/no answer ≤ 1.1 s (Mac, 4 threads, the brain pass only) | decision 39 and the service baseline |

## Trained checks (one yes/no task, `harmful`, train/val only)

Training follows CLAUDE.md: the graph and signs are never trained. The **main arm trains only the KC→MBON
synapses and the read's scale and offset**, which is where a fly stores what it learns. A comparison arm also
trains the per-type gain, threshold and time constant (the amendment of 2026-09-24).

| # | check | pass if | why |
|---|---|---|---|
| T1 | **It learns** | validation balanced accuracy ≥ the plain logistic on the same 46-number smell minus 0.05 | the fly must use what his nose passes (the audit measured about 0.74 on the old antenna) |
| T2 | **The mushroom body is the learner** | with the KC→MBON synapses reset to their untrained values, validation accuracy falls by ≥ 75% of its margin above chance | if the memory can be removed without loss, the fly is not learning in his memory |
| T3 | **The memory carries the answer** | a copy trained on flipped labels, with its KC→MBON memory swapped into the original, flips ≥ 80% of validation answers | the answer follows the memory, not the rest of the brain |
| T4 | **No constant lean** | recall ≥ 0.5 on both classes, and the untrained offset explains ≤ 20% of logit variance | the answer comes from the item, not a bias |
| T5 | **KCs still vary once trained** | L3 and L4 hold with the trained weights | training must not collapse the code |
| T6 | **Controls** | the layered (scrambled) wiring is trained with the same recipe on the same data and reported beside the real brain | CLAUDE.md: controls run first; no pass/fail, reported whatever it shows |

## Changes allowed to pass (each must point at real fly biology; decision 8)

- **Start from rest** (L1).
- **A lower resting drive:** ORN input = rest + a signed item signal, with rest lowered toward the ORNs'
  spontaneous rate (L2).
- **KC thresholds set per KC, label-free**, so each KC fires on a similar share of unlabelled calibration
  smells. Real KCs compensate for their own input strength (Abdelrahman, Merkler & Hige 2021). This is not
  training on labels, so it is not the per-neuron comparison arm (L4).
- **Only KC→MBON and the read scale trained** in the main arm (T2, T3).
- **One sniff for yes/no,** answered as sigmoid(k · (approach − avoid) + c).

Anything else needs Nick first.

## Amendment 1 (2026-09-28, Nick; written before the runs it governs)

**What was found** (label-free, `runs/brain-check/`, first run with the allowed changes: rest 0.1, per-KC
thresholds, start from rest). L1, L3, L4 and L6 pass: KCs 4.8% active per sniff, 99.7% ever active, Jaccard 0.029,
one answer 0.107 s. L2 fails narrowly (ORN rest 0.154). **L5 fails by 30×** (read s.d. 0.00033). The item is strong
at the ORNs (per-cell s.d. 0.19) and PNs (0.05), but the untrained MBONs barely vary with the item (0.002), and the
MBON→DN coupling is weak. The layered, hash and silenced-mushroom-body controls give the same 0.00033. No
label-free DN selection reaches past 0.00073. Stronger KC output (×3 to ×30) does not help and breaks the sparse
code. A random KC→MBON memory (log-multipliers ~ N(0, 3)) reaches 0.0044. Before learning, a mushroom body that
pools thousands of KCs gives much the same output for every odour; the valence lives in the learned KC→MBON
memory. So:

1. **L5 is measured on the trained brain** (main arm, `harmful`): the read's s.d. across the validation items is
   ≥ 0.01 rate units. The untrained value is still reported.
2. **The DN read (audit F10), label-free, fixed before training.**
   - At the label-free start built with the anatomical groups (`v1.dn_groups`), each DN's resting-state response
     to +0.1 drive on all approach MBONs, minus its response to +0.1 on all avoid MBONs, is its coupling `dR`.
     Each type takes the mean `dR` of its cells.
   - Approach types: `dR` ≥ the 95th percentile of |`dR`| over DN types. Avoid types: `dR` ≤ minus that value.
   - DNa02 and DNa03 are excluded: they are steering DNs, and averaging the left and right cells loses the sign.
   - The label-free start is then rebuilt from scratch with these groups.
3. **Rest input 0.05** (an allowed change, for L2): the recurrent ORN excitation lifted the rest rate above the
   expected rate for 0.1.
4. **80 steps are kept** (Nick). From rest, the step-60 answer correlates 0.9997 with step 80.
5. **The answer path is one sniff at a time on the CPU** (`RateBrain3.answer`). It is bit-identical at any thread
   count and between runs: softplus is written as relu(z) + log1p(exp(−|z|)), mathematically the same.
   Published numbers use this path.

**Trained checks, as they will be run** (`scripts/brain_train.py`, `harmful`, the train split to train and the
val split to score, with no model selection on val):
- **Recipe.**
  - Main arm: only `kp_logm` (KC→MBON), `log_k` and `c` train.
  - Every sniff starts from the resting state (re-settled, detached, every 20 batches).
  - Class-balanced BCE, Adam at learning rate 0.03, batch 64, 3 epochs, seed 1, on the 3080.
  - The antenna, operating point and read are the label-free ones above.
  - Scoring is on the Mac CPU through `answer`, with p(harmful) = sigmoid(logit) and yes at p ≥ 0.5.
- **T1:** val balanced accuracy ≥ that of a logistic regression (sklearn, class-balanced, C = 1) on the same
  46-number smell, fit on train, minus 0.05.
- **T2:** with `kp_logm` reset to 0 (keeping the trained k and c), balanced accuracy falls by ≥ 75% of
  (trained − 0.5).
- **T3:** a copy trained with the same recipe on flipped labels has its `kp_logm` put into the original (keeping
  the original's k and c). At least 80% of the original's val answers flip.
- **T4:**
  - recall ≥ 0.5 on both classes;
  - the untrained part's share ≤ 0.2, where the share is the R² of the trained val logits on the logits with
    `kp_logm` reset (the same k and c).
- **T5:** L3 and L4 on the 200 label-free items with the trained weights, plus L5 as amended (item 1).
- **T6:** the layered control, with its own label-free start and the real brain's read cells, trained with the
  same recipe and scored the same way. Reported beside the real brain, with no pass or fail.

## Amendment 2 (2026-09-28, Nick agreed; written before the runs it governs): seeds, sampling error, T7

Amendment 1's trained checks came from one seed on one validation set: 0.727 against 0.724, and 78.9% against 80%,
could be noise either way. From now on:

- **Seeds.** Every trained arm (real, real on flipped labels, layered) is trained with seeds 1–5. Only the
  training seed changes, and with it the batch order. The antenna, the label-free starts, the read cells and the
  layered wiring (shuffle seed 1) stay as in amendment 1. Amendment 1's runs are seed 1; they are kept as they
  are, not retrained.
- **Sampling error in every bar (audit decision 5).** Each trained statistic is recomputed over 2,000 bootstrap
  draws. Each draw resamples the 5 seeds with replacement and the 1,865 validation items with replacement (the
  same items for every seed and arm within a draw), and averages the statistic over the drawn seeds.
  - An "at least" bar passes if the draws' 5th percentile clears it.
  - An "at most" bar passes if their 95th percentile does.
  - The label-free KC measures in T5 must hold for every seed.
  - The rule is seeded (20260928) and deterministic.
- **T1–T5 as in amendment 1,** under this rule. T1, for example, passes if the 5th percentile of (brain balanced
  accuracy − logistic balanced accuracy) is ≥ −0.05.
- **T7, new: the real wiring carries the memory to the DNs.** In the fly, what an odour means is stored at KC→MBON
  and has to reach the motor side. A random expansion can learn the same items without carrying them through
  real circuits. Pass if the 5th percentile of (real trained read s.d. ÷ layered trained read s.d.) is ≥ 2.
  - The read s.d. is the s.d. across validation items of approach − avoid, taken from the served logits as
    (logit − c) / (10 k).
  - Seed 1 measured 7.0 (0.0121 against 0.0017).
  - Both arms are trained with the same recipe and scored on the same items.
- **T6 still has no pass or fail.** The real-minus-layered balanced accuracy is reported with its 90% bootstrap
  interval.
- **Cost:** 12 new trainings on the 3080 (4 seeds × 3 arms), and scoring on the Mac CPU through `answer`.

## Amendment 3 (2026-09-28, Nick: "A"; written before the runs it governs): the memory must dominate, by the fly's own rule

The question settled: the trained answer must come from what he learned. T2 and T3 keep their bars (amendment 2's
rule). Before anything is relaxed, the fly's own learning rule is tried:

- **Depression-only KC→MBON learning.** In flies, dopamine mainly weakens KC→MBON synapses (Hige et al. 2015), and
  that weakening is the memory. The trained multiplier on each KC→MBON synapse is exp(`kp_logm`) with
  `kp_logm` ≤ 0: after each step it is clipped to ≤ 0, so a synapse can only weaken, down to zero. Amendment 2's
  runs let synapses strengthen without bound. That pushed the avoid MBONs from a rate of 0.04 to 0.31 and loosened
  the KC code through APL.
- **Train as it serves.** The resting state is re-settled before every batch (it was every 20).
- **Otherwise unchanged:**
  - the arms (real, real on flipped labels, layered, all under the same rule), seeds 1–5, the recipe (Adam at lr
    0.03, batch 64, 3 epochs, class-balanced BCE), the label-free starts and the read;
  - CPU scoring through `answer`, and amendment 2's bootstrap and bars for T1–T7.
- **Outputs:** in `runs/brain-train-depress/`. Amendment 2's results stay as they are.
- **Adoption:** the rule is adopted if T1 and T7 still pass and the 5th percentiles of T2 and T3 are higher than
  in amendment 2 (−0.008 and 0.777).
- **Passing:** the brain passes only if all nine checks pass.
- **If it does not pass,** option B (the memory carries the majority, with the innate share reported) goes back to
  Nick. It is not adopted here.

## Amendment 4 (2026-09-29, Nick: "sound good"): brain v3.1, the deep-dive package (written before its runs)

This fixes the 14 flaws in `docs/audit-2026-09-28/brain-deep-dive.md` as one package, judged against
`docs/BRAIN-REQUIREMENTS.md` (R1–R7, with resemblance first). Every part points at the fly.

**P1. A silent sub-threshold.** BETA goes from 50 to 500, so the rate at threshold falls from 0.0139 to 0.0014
(R1.7 requires ≤ 0.002).

**P2. Only fast transmitters drive** (R1.6).
- **Transmitter per neuron:**
  - the consensus call, else the body's prediction, else its type's prediction;
  - DPM is GABA (as before);
  - still unknown means no fast output.
- **Fast signs:** acetylcholine +1; GABA, glutamate and histamine −1; dopamine, octopamine and serotonin 0
  (metabotropic; they are the teaching signal, not fast drive).
- **KC→KC synapses:** 0 (Manoim et al. 2022).
- **Antennal-lobe LNs:** the fix now zeroes only the confidently cholinergic ones.
- **Input totals:** a neuron's input total counts only fast-drive synapses, from any MaleCNS body, with the same
  rule applied to bodies outside the model. The controls get the same rule on their own wiring.

**P3. Operating points by stage, set label-free, by input gain rather than bias** (R1.3, R1.7).
- **One new fixed buffer per neuron:** an input scale `s_in` on its synaptic drive, set per cell type. Its
  biology is intrinsic excitability (Apostolopoulou & Lin 2020, Abdelrahman et al. 2021). Thresholds are 0 for
  every non-sensory type, so no neuron is held up by a bias. `s_in` folds into the frozen matrix, at no cost in
  speed.
- **The homeostatic start** now moves `s_in`, multiplicatively and clipped to [0.1, 100], until each type's mean
  sniff rate reaches its target:
  - 0.05 in general;
  - 0.2 for the DN read cells.
  - Types whose net drive is inhibitory stay near silent, and are reported.
  - It runs to convergence (mean residual ≤ 5% of the target, at most 80 iterations).
  - "untyped" cells are split by class and superclass.
- **ORNs:** threshold 0, so with rest input 0.05 an ORN rests near a rate of 0.05.
- **KCs:**
  - a per-cell offset keeps each KC active on about 5% of the calibration smells;
  - the KC types' `s_in` is bisected so that the median rate of an active KC is 0.3 (R1.7: ≥ 0.2).
- **MBONs:** threshold 0 and a target rate of 0.2, so their response comes from their inputs. New check L7:
  silencing KC→MBON cuts the MBONs' mean sniff rate by ≥ 80%.

**P6. The read.**
- **MBON valence by transmitter,** for the typical MBONs (MBON01–19) only (Aso et al. 2014b):
  - glutamatergic → avoid;
  - GABAergic or cholinergic → approach;
  - novelty (PPL104) not read;
  - the atypical MBON20–35 (Li et al. 2020) are not in the valence groups.
- **The DN read cells** are then re-chosen by amendment 1's coupling rule (top 5% of |coupling| by sign, steering
  DNs out), on the new wiring. The start is rebuilt after that choice.

**Label-free checks (must all pass before any training):**
- L1–L6 as before.
- L7 as above.
- R1.3 on the path PN → KC → MBON → DN read: median bias share ≤ 0.5, and item modulation (cell s.d. ÷ mean)
  ≥ 0.2.
- R1.7: the rate at threshold ≤ 0.002, and the median rate of an active KC ≥ 0.2.
- The layered, hash and silenced-mushroom-body controls are reported beside them.

**P4. Training** (only if the label-free checks pass):
- KC→MBON may weaken without limit, and strengthen at most 2×: `kp_logm` ≤ ln 2.
- The resting state is re-settled before every batch.
- Otherwise amendment 2's recipe, 5 seeds.

**P5. Trained checks.** Amendment 2's bootstrap, with T2 and T3 measured by ranking (AUC), independent of c:
- **T2:** AUC(reset) − 0.5 ≤ 0.25 × (AUC(trained) − 0.5), at the 95th percentile.
- **T3:** 0.5 − AUC(swapped) ≥ 0.8 × (AUC(trained) − 0.5), at the 5th percentile.
- T1, T4, T5 and T7 as in amendment 2.
- New T8, for requirements R4.2 and R4.3: the seed s.d. of balanced accuracy ≤ 0.01, and every seed's k ≤ 20.

**P7 (timed sniffs)** comes after the brain passes, as its own tested change (decision 39.1).
