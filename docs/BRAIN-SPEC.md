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
