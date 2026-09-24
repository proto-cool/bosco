# Gate A1 — the connectome as a trained decider (pre-registered 2026-09-24)

C1 showed that the fly's lifetime learning rule, read at his output neurons,
cannot carry sweet/bitter. The mushroom body memorises particular smells;
sweet/bitter is a concept. Phase A (`PLAN.md`) asks the other question:
**with the wiring fixed and only what evolution sets trained, does the MaleCNS
brain decide sweet/bitter, and better than the same brain with its wiring
scrambled?**

## The model (`src/bosco/ratebrain.py`)

- **The whole central brain as in every gate:** 40,939 neurons, 8.5 M
  connections. The synapse counts and signs are the connectome's and are
  never changed. W[i, j] = sign(j) × count(j→i) / total synapses onto i.
- Rate neurons, 12 steps from rest:
  `r ← r + 0.5·(−r + tanh(relu(W·(g⊙r) + b + input)))`.
- **Trained, per neuron:** the output gain g (positive) and the threshold b.
  That is 82k numbers against 8.5 M connections: how excitable each cell is
  and how hard it drives what it is wired to. Nothing about *who connects to
  whom* is learned.
- **In:** the B4 antenna (CLIP → 26 principal components ± → 52 glomeruli,
  unchanged) on the ORNs of those glomeruli.
- **Out:** the same fixed read as C1. Mean rate of approach MBONs (26 cells,
  punishment-DAN compartments) minus avoid MBONs (41, reward-DAN
  compartments), × one scale + one offset, gives P(sweet). No readout layer.
- **Init, label-free:** from a fixed grid of (gain, threshold), each arm takes
  the pair with the widest spread of the output difference over 64 unlabelled
  calibration sentences, so it is neither silent nor saturated.
  Harness check before writing this: the real brain picks gain 8, threshold
  0.3; a 10-batch smoke run lowers training loss 0.70 → 0.63 (training loss
  only; no held-out number was looked at).

## Arms

| arm | wiring |
|---|---|
| real | MaleCNS as is |
| shuffle | the whole brain degree-preserving shuffled (`dunce_v1.npz`, as in every B gate) |
| hash | only PN→KC and KC→MBON rewired at random (`hash_v1.npz`) |
| free | every connection's target drawn at random, same counts and signs, same neurons |

Same model, parameters, init rule, data, batches and epochs for all.

## Training and data

- Adam, learning rate 3e-3, batch 64, 8 epochs, binary cross-entropy on
  sweet vs bitter. Fixed now, not tuned.
- **Full:** all 6,254 clear-labelled SST training sentences (the product set).
  **Small:** 400 of them, 200 per side (sample efficiency).
- Seeds 1 and 2 (batch order) per arm and size: 16 runs, about an hour on the
  Mac's GPU, one after another.
- The epoch is chosen on the **validation** half of the 400 held-out
  sentences. Every number is reported on the **test** half.

## Measured

Test balanced accuracy at P = 0.5 (headline), Spearman, ECE (calibration),
pictures from text alone (900 OASIS, at the median and at 0.5), Nick's probes,
and the share of Kenyon cells active (does the trained brain stay sparse like
a fly's?).

## Decision rule

On the full training set, mean over the two seeds:

1. **PASS, the wiring earns its place,** if real ≥ shuffle + 0.03, real ≥
   hash + 0.03 **and** real ≥ free + 0.03 (0.03 = the measurement floor on
   200-item halves).
2. Otherwise **FAIL**, written up the same. If real beats shuffle and free but
   not hash, the reading is B's again: the mushroom body's architecture
   matters, its exact wiring does not.
3. Reported beside it, not gating: real against the antenna's logistic
   ceiling (0.697); the small-set results; the pictures; the probes; KC
   sparseness.

Nothing above changes after the numbers are seen.

Runner: `scripts/gate_a1.py`. Results: `runs/gate-a1/`,
`docs/gate-a1-results.md`.
