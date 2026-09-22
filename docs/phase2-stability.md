# Phase 2: pruning, port to MaleCNS, network stability, KC sparseness

## Summary

- Model set: 40,939 traced central-brain neurons (cb_* superclasses plus
  descending and ascending neurons), 8.49 M edges, 45.7 M synapses.
  Optic lobes and VNC dropped.  Cache digest `4ffc9e4c84806199bce839f164e15d27`.
- Shiu et al.'s LIF parameters are used verbatim except `w_syn`, which is
  re-calibrated per connectome by their own criterion (below).
- **Finding:** the Shiu LIF model, on either connectome, has no weight at
  which the sugar reflex works *and* olfactory input stays bounded.  Any ORN
  drive (even 25 random ORNs at 50 Hz) tips the network into a
  self-sustaining state of 7–11 k neurons firing at the refractory ceiling.
  Verified independently in the authors' own Brian2 code on FlyWire v783
  (8,309 neurons active, 95 k spikes in the 200 ms after input off).
- **Cause (MaleCNS):** the 157 cholinergic antennal-lobe local neurons
  (lLN1_bc, lLN2T/X/F, v2LN…; NT prediction confident, lLN1_bc has ground
  truth) form a recurrent excitatory loop (≈183 k synapses within the
  persistent set; 87 k lLN1_bc→lLN1_bc alone).  Recruitment cascade:
  ORNs → PNs → eLNs light up together at ~110 ms → 1,000+ KCs within 20 ms.
- **Fix (documented modeling decision):** the fast chemical output of these
  eLNs is set to zero.  Lateral excitation in the fly antennal lobe is
  primarily electrical (Yaksi & Wilson 2010), which the LIF has no
  representation of; as fast chemical excitation it is unphysiological.
  With this single change, at Shiu's parameters: sugar reflex intact,
  bitter does not drive MN9, 25/400 random ORNs leave no persistent
  activity, and KC codes are sparse and odor-specific (pairwise Jaccard
  ≈ 0.03).
- Things tried and rejected (all documented in `scripts/phase2_*.py`):
  global short-term depression (kills the reflex before it kills the
  runaway), refractory ceilings of 5–20 ms (leaves a smoldering
  sub-attractor), inhibitory gain ×1.5–2 (same), KC→KC removal (helps, not
  sufficient; left at connectome weight since it made no difference once the
  eLN fix was in), APL gain ×2–4 (does not address the loop).

## w_syn calibration

Shiu: "We chose W_syn such that activation of sugar GRNs at 100 Hz resulted
in roughly 80% of maximal MN9 firing."  On the v1 wiring
(`docs/phase2-wsyn-calibration.md`), with all 83 labellar sugar/water GRNs
driven:

| w_syn | MN9 @100 Hz | MN9 @200 Hz | ratio |
|---|---|---|---|
| 0.150 | 42 | 83 | 0.51 |
| 0.160 | 71 | 93 | 0.77 |
| 0.165 | 82 | 95 | 0.86 |
| 0.170 | 76 | 98 | 0.77 |

**v1: w_syn = 0.16 mV.**  Stability holds at 0.16 (no post-stimulus
activity for 25 or 400 random ORNs); 0.25 begins to smolder.

## KC sparseness (gate)

At w_syn 0.16, APL at connectome weight, odors = k random glomeruli, all
their ORNs driven at r Hz for 500 ms (4 odors each):

| k | r | KC active frac (mean, min–max) | odor-pair Jaccard | MBON Hz |
|---|---|---|---|---|
| 10 | 100 | 0.031 (0.009–0.069) | 0.03 | 3.2 |
| 10 | 150 | 0.026 (0.010–0.058) | 0.03 | 2.4 |
| 15 | 150 | 0.036 (0.031–0.043) | 0.06 | 3.4 |
| 20 | 150 | 0.051 (0.019–0.083) | 0.08 | 6.1 |
| 27 | 150 | 0.131 (0.115–0.151) | 0.32 | 17.3 |

**Gate: KC activity 2.5–5% per odor for 10–20 glomeruli at 100–150 Hz,
odor-specific.** Target was ≈5%; physiological estimates are 5–10%
(Turner et al. 2008; Honegger et al. 2011).  APL was **not** tuned: the
encoder's odor breadth (12 glomeruli, 120 Hz; see `docs/encoder.md`) is
chosen inside this range.  Larger/stronger odors (≥27 glomeruli at 150 Hz)
push KC activity past 10% and MBONs into a broad response; the encoder
stays well below that.

Residuals: some odors and the bitter stimulus leave a small persistent
population (~500 neurons, ~3–6 k spikes per 200 ms) after input off.  This
never propagates across episodes because every episode starts from rest
(v = v0, g = 0); only the plastic KC→MBON weights carry over.

## Determinism

Replays are bit-identical (same seed, same weights, same inputs) on the
same binary: verified for every configuration above.  Build flags forbid
FMA contraction and fast-math.
