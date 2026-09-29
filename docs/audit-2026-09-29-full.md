# Full brain audit, 2026-09-29

Nick asked for a deeper audit after the deep dive of 2026-09-28 was followed at once by new failures. This one covers:
- the committed brain (v3.0, the one trained and scored);
- the uncommitted v3.1 work;
- every candidate operating point.

It works in six strands, all label-free on the Mac CPU, with no training and no GPU:
1. a full inventory;
2. dynamics;
3. function (known fly experiments re-run in the model);
4. a red team;
5. read, learning and evaluation;
6. the candidates.

**Items:** exploration used 512 unlabelled items that no check, calibration or antenna fit uses.

**Evidence:** scripts and outputs are in `runs/deep-audit-2026-09-29/` (auditA–E, `candidates.py`). I re-measured every
high-severity finding marked ✔ myself.

## Verdict

**The brain does not work the way the product claims, and no candidate design works yet.**
- **The claim** is that the fly's own mushroom-body memory decides, carried through his real circuits to his
  descending neurons.
- **What was measured:**
  - The untrained answer comes almost entirely from paths *around* the mushroom body.
  - The mushroom body's output reaches the answer with almost no gain.
  - The resting brain has several stable states, so every answer depends on a hidden starting condition.
  - Several of the checks that said "pass" were measuring the wrong thing.
- **What works:**
  - the bit-exact, fast answer path;
  - the pinned encoder;
  - a stable v3.0 at its operating point;
  - KC sparseness under APL (as in the fly, though the shuffles do it equally);
  - some sensory → DN pathways that depend on the real wiring (LC4 → giant fibre).

## 1. Function: does the model do what a fly brain does? (auditor C; v3.0, untrained unless stated)

| experiment | criterion | real | layered | hash | verdict |
|---|---|---|---|---|---|
| E1 ORN → PN broadening (Bhandawat 2007; Olsen 2010) | PNs broader than ORNs, less skewed | PNs *sparser* (0.84 against 0.66) and weaker (0.049 against 0.134); each PN copies its glomerulus | same | same | **FAIL** |
| E1b glomerular identity | pulse lands on its own PNs | 98% | 2% (chance) | 98% | pass; layered fails as it should |
| E2 APL block (Lin 2014) | KCs ≥ 1.5× denser, overlap up | 3.2×; Jaccard 0.03 → 0.43 | pass | pass | pass, but not specific to the wiring |
| E3 KC sparse and decorrelated | 2–10% active; KC corr < PN corr | 5.4%; 0.051 < 0.059 | pass | pass | pass, but not specific to the wiring |
| **E4 MBONs KC-driven** (Hige 2015) | silencing KC→MBON cuts the odour response ≥ 80% | **54%** | 76% | 72% | **FAIL** (the shuffles come closer to the fly) ✔ |
| E5 odour-specific depression | MBON drop for A ≥ 50%, specific | 17/20 types | | | pass, mechanically |
| E5 does it reach behaviour? | the read moves | right direction 18/20, but the size is **1.6e-4** (1/60 of the L5 bar) | chance | 17/20 | **FAIL on size** |
| E6 MBON activation valence (Aso 2014b) | right sign ≥ 80% | 9/11 (MBON10 wrong); circular | 1/4 | 8/9 | pass, circular |
| **E7 innate path survives MB loss** (Heimbeck 2001) | read spread ≥ 50% with KCs silenced | **98%, r = 0.997** | | | "passes" because **the mushroom body contributes nothing to the untrained answer** ✔ |
| E8 LC16 → MDN (backward walking) | MDN recruited | +1% | | | **FAIL** |
| E8 LC4 → giant fibre (von Reyn 2014) | DNp01 recruited | 4.6× | 8× weaker | | pass; depends on the wiring |
| E9 memory → read at a fixed scale | real carries a memory better than the shuffles | trained 1.3e-2; random memory 3–4e-3 | about 6× worse | **as good or better** | beats layered, not hash |

**What it means:**
- The odour signal reaches the DNs through the PN → lateral horn and other paths. With PNs silenced the read keeps
  7% of its item spread; with the LH silenced, 73%; with the KCs silenced, 98%.
- The mushroom body is not in the loop until training moves the KC→MBON memory by very large factors. Only the
  trained memory, with multipliers up to e¹², reaches the L5 bar; a fly-sized, one-compartment memory moves the
  answer by 1/60 of it.
- Where the wiring does matter (E1b, E6, E8 LC4, E9 against layered), it is the sensory → DN and MBON → DN
  wiring, not the KC wiring. The hash shuffle matches the real brain on E9.

## 2. Dynamics (auditor B; the Jacobian checked by finite differences; long runs, step-to-step)

| design | stability | several resting states | notes |
|---|---|---|---|
| **v3.0 (C0), scored** | stable (ρ 0.988 at rest) | **yes: 4 attractors** ✔ | settling from r = 0 differs from the saved rest in 906 cells; the resting read moves 2.4× its item spread; trained answers move a mean of 0.15 logits (1.6% flip) ✔. Unstable at gain ×3 (×2 with BETA 500). The slowest mode relaxes over 83 steps, about one sniff |
| v3.1 per-type homeostasis (C2, amendment 4 P3) | **unstable** (ρ 1.04–1.12) | yes | about 11,800 cells oscillate, mainly in the central complex; not a step-size artefact; the answer depends on the phase |
| global gain 4 + stage settings (C1, "4.1") | **oscillates** | — | my candidate script scaled the read cells with no cap (up to 651,019) ✔; the KC targets never converged (10% active, rate 0.10); the read is lopsided (43 approach cells, 4 avoid) |
| C1 with scales capped at 100 (C1c) | stable, a single state | no | margin: +25% gain holds, +50% is unstable; onset of instability for the procedure between G = 6 and 8 |
| v3.0 wiring + the same operating point (C3) | stable, marginal | yes | **no avoid cells, so the read is NaN** ✔; the read at step 80 correlates only 0.87 with its settled value |
| a time step of tau = 5 ms (allowed by TAU_RANGE) | unstable (period-2 flips) | | a uniform tau needs to stay above about 9 ms; the per-type tau arm is at risk |

**Recurring unstable loops:**
- the posterior slope (PS068/127/177/178), a Hopf bifurcation at high gain;
- the central complex (PEN/GLNO/PFN/PFR, vDelta);
- KCγ in the homeostatic design.

## 3. Wiring and data (auditor A)
- **The answers ride on uncertain transmitter calls** (red team).
  - Flipping neurons with prediction confidence < 0.6 takes KCs from 8% to **37%** active and flips 35% of
    answers. Making glutamate excitatory flips 74%.
  - Calls made by the per-body "predicted" path agree with FlyWire only 51% of the time.
  - 186 of 241 predicted-acetylcholine cells are glutamatergic in FlyWire, so they have the wrong sign in both
    versions.
- **The 858 neurons with no transmitter:** v3.0 made them excitatory; by output, FlyWire calls them mostly
  inhibitory.
- **v3.1's fallback:** it makes 304 neurons serotonergic at a median confidence of about 0.59. FlyWire agrees for
  only 108 of them. The model has 333 "5-HT" cells, against MaleCNS ground truth of 44.
  - LHPV6q1 (28k synapses, FlyWire ACh) is wrongly silenced.
- **MBON valence is wrong in both versions** ✔ (Li et al. 2020 checked directly).
  - v3.0 puts GABAergic MBON09 and MBON10 in "avoid".
  - In v3.1, the "typical = MBON01–19" cut is wrong: MBON10 is atypical, MBON21–23 are typical.
  - v3.1 extends "every aversive MBON is glutamatergic" to "every glutamatergic MBON is aversive".
  - MBON08 is not in MaleCNS.
- **Antennal-lobe inhibition is mostly missing.**
  - 58% of LN→PN synapses carry no drive in either version, because likely GABAergic LNs (v2LN30, lLN2F_a,
    lLN2T_d) are silenced as "unknown or cholinergic".
  - 20 cholinergic LNs escape the prefix rule.
- **The <5-synapse cut:**
  - it keeps 42% of ORN→ORN, 22% of KC→DAN and 65% of LHN→LHN;
  - MBON→DN is 4,347 synapses in all, and only 88 DNs keep any MBON input.
- **v3.1's fast input totals quietly remove the outside-model deficit** (fragments fall from 4.7% to 0.15%).
  - 4,881 neurons lose more than half their total, and 149 are left with a total of 0.
- **v3.0 VPNs:** with no visual input, they are held at a rate of 0.05 by bias, feeding the central brain.
- **The controls are not fair** ✔. Both shuffles leave about a third of KCs with **no APL input** (hash 1,395,
  layered 1,283; real 1). The hash also scrambles DAN, MBON and VPN edges; it is not a PN→KC hash. The layered
  shuffle destroys glomerular identity.
- **The unit's hand-written softplus has gradient 0 at exactly x = 0.**

## 4. Read, learning and evaluation (auditor E; all ✔ from the saved logits)
- **T7 measures the ratio of the read scales k, not memory transport.**
  - The read s.d. is sd(logit)/(10k), and BCE drives sd(logit) to about 1 in both arms.
  - The depression-only brain, which did not learn, passed T7 at 2.77.
  - **My claim "the real wiring carries the memory 7× better" is retracted.**
  - T5-L5 (trained) is the same flaw (it amounts to k ≲ 10), and T8's k cap is a looser copy of it.
- **T2 by AUC (amendment 4)** is one-sided, and flips with the arbitrary yes/no convention.
- **T4's R² measures alignment, not share.**
  - The untrained part is **0.3%** of the logit variance, not 14%; the R² of 0.13 is corr(memory, innate)².
  - "19% of answers follow the innate path" was T3's offset artefact.
- **Not calibrated:** ECE is **0.16–0.25** on val (the requirement is ≤ 0.10, never measured). Training on 50/50
  data and validating at 76% harmful shifts the prior.
- **The source is a confound.**
  - A rule that uses only the source reaches 0.659 balanced accuracy.
  - The brain is near chance on DynaHate (AUC 0.60) and HatemojiBuild (0.54) ✔.
- **The brain is below a linear readout of its own 46-number input:**

  | model | balanced accuracy |
  |---|---|
  | the brain | 0.725 |
  | logistic | 0.735 |
  | random sparse expansion + logistic | 0.747 |
  | MLP | 0.770 |

- **Training did not converge.**
  - The loss was still falling steeply at the end.
  - The first ~160 steps sat at chance while Adam random-walked the memory: k starts at 1, about 100× too small.
- **The memory is extreme.**
  - Synapse multipliers: 99th percentile 704×, maximum 166,000×.
  - 23% of the memory's variance is not specific to the label.
  - Synapses from KCs that are never active still moved.
- **Seeds vary only the batch order** (the memories correlate 0.99), so the "5-seed bootstrap" is really an
  items-only one. Answers disagree between seeds on up to 14.6% of items.
- **The read:** 7 of 20 approach cells are octopaminergic, and others are song and courtship DNs. "Yes = harmful"
  is read as *approach*.

## 5. Pipeline and reproducibility (auditor D)
- **Saving a v3.1 brain would lose its operating point:** `save_start`, `load_start` and `load_brain` do not save
  `s_in`. v3.1 isn't wired into the scripts at all.
- **Nothing is keyed to what made it:**
  - the fast-tables cache is keyed only by neuron ids, so a rule change is silently ignored ✔;
  - the start, read and weight files carry no version and no hash.
- **The playback names the wrong answer cells:** `runs/brain-map/map.json` still carries v2's DN groups. Only 2 of
  the 20 approach read cells appear in it.
- **A stale fold is unguarded:** changing the memory without `freeze` serves the old memory, bit-identically.
- **Bad inputs are served silently:** a NaN smell becomes a silent "no", and nonsense inputs get confident answers.
- **A 5% change in the per-type gains flips 11.7% of answers,** so the gain/threshold/tau comparison arm acts on
  the most sensitive knob.
- **The trained weights cannot be reproduced:** CUDA training isn't deterministic, and no checksums are recorded.
- **The tests never exercise** a trained memory, the scoring path, the choice of resting state, `s_in`, or the
  playback.

## 6. Retractions (my claims that do not stand)
- "The real wiring carries the memory 7× better" (T7): an artefact of k.
- "About 14% / 19% of the answer is innate": the wrong measure; it is 0.3%.
- "Training potentiated about 7×": the multipliers reach 166,000×.
- "Global gain 4 settles, is 83% KC-driven, DN modulation 0.21": from a 64-item quick test. The full build did not
  converge, and my script's uncapped scales made it oscillate.
- The results doc's headline "13 of 14 checks pass": superseded, and several passes were invalid.

## 7. Not covered by this audit
- **Training-time dynamics:** stability with a trained memory, apart from C0.
- **Per-type tau** stability under the comparison arm.
- **The encoder's language bias,** and the antenna on text outside the training domain.
- **Sight and taste.**
- **Held-out fly experiments:** every experiment here was used to judge. If function becomes a fitting target,
  some must be kept back.
- **The dev/test splits,** deliberately untouched.

## 8. What this means for the next step (for Nick; nothing is being built)
The failures have one shape. The model has the fly's wiring but not the fly's *operating parameters*: the gains,
thresholds and time constants that make each circuit do its job. Every hand-set operating point so far has been:
- item-blind (v3.0);
- unstable (per-type homeostasis);
- or unconverged (global gain).

Meanwhile the checks rewarded the read scale rather than the circuit.

Options, none started:
- **A. Fit the fly's operating parameters to the fly's known physiology,** not to task labels, with the wiring and
  signs fixed (the approach of Lappalainen et al. 2024, which fit connectome-constrained parameters to a task).
  - The experiments above (E1–E8), plus stability and single-state rest, become the objective.
  - Some experiments are held out as the test.
  - Only then is the memory trained.
- **B. Fix the evaluation first:**
  - drop or redefine T7, T2, T4 and T5-L5;
  - add ECE, stronger baselines (MLP, random expansion), per-source AUC, and seeds that vary initialisation;
  - fair controls that keep APL;
  - canonical rests.
  
  Otherwise no brain can be judged honestly. B is needed under any option.
- **C. The transmitter question:** the answers depend on low-confidence transmitter calls. Deciding a source of
  truth (FlyWire's calls where they agree, confidence floors, which uncertain neurons get no fast output) is a
  decision for Nick, and should come before any wiring change.
- **D. Reconsider the premise:** the mushroom body may not be able to carry arbitrary text decisions to the
  descending neurons in a rate model of this cut. v1 reached a similar conclusion for a social account. A would
  test that directly.
