# Decisions locked 2026-09-24 (Nick), after the audit

Made after A2's numbers and the audit (`AUDIT-2026-09-24.md`), with the case
for and against each laid out first. They govern the next training leg.

## 1. Training on task labels: allowed, constrained

- **Main:** parameters shared **per cell type** (gain, threshold, time
  constant), plus the **KC→MBON synapses**, which are where a fly stores what
  it learns. The graph and signs are never trained.
- **Comparison arm:** per-neuron parameters (A1/A2's method), so the cost of
  the constraint, and whether the wiring matters once training cannot
  override it, are both measured.
- Why: it is the only approach that has worked (0.87 against 0.555), and it
  reads as "evolution sets the strengths" (Lappalainen et al. 2024). Per-neuron
  freedom let training override the wiring, which is why random brains
  matched him.

## 2. Vision: the visual projection neurons now, the optic lobes for Doom

- Add the 9,201 `visual_projection` neurons, the central brain's real visual
  entrance. Pictures arrive through a fixed, label-free mapping from the eyes
  model onto them. Documented as a stand-in for the optic lobes.
- The optic lobes (about 96k neurons) come in the Doom phase, where motion is
  what matters and fly vision is good at it.

## 3. The answer: corrected MBON read; behaviour recorded

- The answer is approach minus avoid over the MBONs, grouped by **measured**
  dopamine input (PAM = reward, PPL1 = punishment, from synapse counts), not
  by the v1 config.
- The behaviour neurons (proboscis: eat; walking and turning: approach;
  giant fibre and looming: flee) are recorded in every trace. How often
  behaviour agrees with the answer is reported, which decides whether a
  later leg answers from behaviour.

## Consequences written into the rules

- `CLAUDE.md` amended: what may be trained on labels; anatomy and leakage
  checks in every pre-registration; no v1 setting without a v2 reason.
- `BRIEF.md` amended: trained offline per version, never taught through the
  API.

## 4. Synapse cutoff: ≥5 synapses, every KC→MBON synapse kept (Nick, after the pilot)

The pilot rerun (`a5-cutoff-pilot2-results.md`): cut 0.737 vs full 0.745 on
validation, 2.6× faster end to end. Cut vs full disagree on 10.6–20.3% of
items, no more than the full brain disagrees with itself across seeds
(18.8%). The cut brain is steadier (9.3% across seeds), and on disputed items
neither brain is right more often. Nick: "keep the cut brain for speed."
