# Gate A5: the corrected fly, one brain, every question (pre-registered 2026-09-24)

A2 again, on the rebuilt v2 fly, with fair controls and clean data. It is
the first gate under `CLAUDE.md` as amended on 2026-09-24, so it opens with
the anatomy and leakage checks.

## Anatomy check: what carries each part of the task, and is it in the model

| step | in a fly | in the model | stand-in / missing |
|---|---|---|---|
| smell (text) | ORNs → antennal-lobe PNs | yes: 1,865 ORNs of 46 glomeruli; PNs, LNs (the eLN chemical output is off, v1's stability fix, reason kept) | the encoder (e5-large-v2) → 23 principal components ± per glomerulus. The 7 innate glomeruli (DA1, DL3, VA1d, VA1v, VL2a, DA2, V) are not driven. |
| sight (pictures) | photoreceptors → optic lobes → **visual projection neurons** → central brain | the 9,201 visual projection neurons: yes | optic lobes and photoreceptors: **missing** (Doom phase). The encoder (jina-clip-v2) → 26 PCs ± → 3 channels per VPN type. **A stand-in.** |
| the question | none (a fly is asked nothing) | the question's text is added to the smell | a stated convention, not anatomy |
| memory / valence | KCs (sparse) → MBONs; DANs teach | KCs (5% active at start, sparseness pressure), all 61,210 KC→MBON synapses, trainable; DANs present, DAN→KC fast drive off | the teaching is gradient descent, not dopamine (decision 1) |
| the answer | MBON balance → behaviour | approach MBONs (48 cells) − avoid MBONs (41 cells), grouped by **measured** PAM vs PPL1 input | a read, one step before behaviour |
| behaviour | DNs / MNs | eat, approach, flee and groom neurons, **recorded** | not the answer (decision 3) |
| dynamics | time constants per cell | per-type τ (starting at 20 ms), 40 steps × 5 ms = 200 ms of fly time | rate units, not spikes |
| wiring | 50,140 neurons in scope | connections of ≥5 synapses, plus every KC→MBON synapse (decision 4) | 24% of synapses (the weakest) dropped |
| signs | transmitters | consensus; prediction where the consensus is unclear; DPM inhibitory | glutamate is always inhibitory (a convention); neuromodulators act as fast synapses (except DAN→KC) |

## Leakage check (`docs/a5-data.md`)

- val/test items with a near-duplicate (cosine > 0.95) in training: sweet
  0.3/0.2%, dangerous 0.2/0.0%, junk 0.0/0.0% (split by message family),
  pictures 0.9/0.0% (split by theme series).
- Probe pictures: two whole themes (Dessert, Garbage dump, 16 pictures), in
  no split.
- Probe sentences: Nick's five, not in any dataset.

## Arms (all the cut brain; all passed preflight or are recorded as slow)

| arm | wiring | trained | seeds |
|---|---|---|---|
| **real** | MaleCNS | per cell type + KC→MBON | 1, 2, 3 |
| layered | shuffled within class-to-class blocks (no shortcuts) | same | 1, 2, 3 |
| hash | KC inputs and KC→MBON shuffled | same | 1, 2, 3 |
| free | every target random (it has shortcuts, by construction) | same | 1, 2, 3 |
| real-neuron | MaleCNS | per neuron + KC→MBON (the comparison) | 1, 2, 3 |

## Training

The cleaned data, all training items (pictures shown 5×), every question
mixed in each batch. Adam 3e-3, batch 64, 10 epochs, the fly-like start
(`a5.fly_init`), and the KC sparseness pressure. The epoch is chosen by
mean **validation** balanced accuracy; everything reported is **test**.
Roughly 70 minutes per run, 15 runs: about **17 hours** on the Mac's GPU,
one after another.

## Behaviour agreement, defined

His behaviour read is the mean rate of the approach and eat neurons minus
the mean rate of the flee neurons, over the last step. It agrees with his
answer when its sign matches the side of P(approach) against 0.5.

## Preflight

All five arms passed (`docs/a5-preflight-results.md`, revision 2), on the
cut brain. The remaining preflight configurations (per-neuron controls, the
full brain) are not used by this gate; they were stopped unfinished, to free
the GPU.

## Measured (test)

Per part: balanced accuracy at 0.5, AUROC (the two-option T-maze), the
four-option T-maze (ties broken at random), ECE. KC activity. How often the
behaviour neurons agree with the answer. Seed-to-seed disagreement within
each arm. The probes. The reference: a logistic model on the same senses and
splits (`runs/a5-reference.json`: sweet 0.929, pictures 0.939, dangerous
0.817, junk 0.971 on test), computed before the run.

## Decision rules (fixed now)

- **D1, the product:** real's test balanced accuracy is ≥ reference − 0.05
  on every part.
- **D2, the wiring matters:** real's mean over the parts is ≥ each of
  layered, hash and free + 0.03.
- **D3, fly-like:** KCs active ≤ 0.10 on test items.
- **D4, the cost of the constraint** (reported, no bar): real-neuron minus
  real.
- **Adoption:** if D1 and D3 pass, the real brain (the best seed on
  validation) becomes **bosco-2026-09-xx**, the first published version,
  with its test numbers, calibration and limits on its model card. D2 is
  reported whatever it says.

Nothing above changes after results are seen. Any change before the run is
an amendment, recorded here with its reason.
