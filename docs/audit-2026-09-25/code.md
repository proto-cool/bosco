# Code audit (2026-09-25)

Scope: `src/bosco/{model2,ratebrain2,senses,controls,a5,device}.py`,
`scripts/{a4_broad,a4b_dev,a4_pilot,gate_a5}.py`. Read-only: no code or existing doc was
changed. Findings already in `AUDIT-2026-09-24.md` / `ERRATA-2026-09-24.md` (the maze-loss
clamp, the gain-8 start, MPS non-repeatability of the plain net) are not repeated unless
something new was measured about them.

Every number marked **measured** comes from a probe run for this audit (throwaway scripts in
`/tmp/bosco_audit/`, CPU, `torch.set_num_threads(4)`, Apple M3 Max) or from the saved
checkpoints in `runs/`. The probes read the code as it is; they did not modify anything.

Ranking: **critical** = changes a published result or claim; **important** = a real defect or
rule breach that has not (yet) been shown to move a number; **minor** = hygiene.

---

## Critical

### C1. The control arms are not normalised like the real brain: free gets ~10× the drive and 3% of the memory synapses

- `ratebrain2.py:49` divides every edge by `in_total[post]`, and `model2.with_wiring` /
  `a5.build` pass the **real** brain's `in_total` to every control. The shuffles move synapse
  counts to new targets (`controls.py:31-43, 74-79`), so a control neuron's summed input no
  longer matches the denominator.
- **Measured** (cut brain, row sums of |w| per neuron):

  | arm | mean | p99 | max | neurons with Σ\|w\| > 1 |
  |---|---|---|---|---|
  | real | 0.458 | 0.91 | 0.98 | 0 |
  | layered | 0.748 | 2.6 | 13.5 | 14,062 |
  | hash | 0.459 | 0.92 | 2.7 (MBONs) | 67 |
  | free | **4.75** | 32.5 | **1,651** | 26,639 |

  In free, mechanosensory and other sensory cells with an `in_total` of 0–2 receive hundreds of
  synapses' worth of normalised drive.
- **The trainable memory differs too** (checkpoints, `kp_logm` size): real 61,210 KC→MBON
  synapses; layered 54,324 and hash 54,359 (duplicate edges created by the shuffle are merged
  by `csr_from_edges`); **free 1,710**, because a uniformly retargeted KC edge almost never
  lands on an MBON.
- Failure scenario: GATE-A5 D2 ("real ≥ every control + 0.03") compares real against brains
  whose per-neuron gain and plastic capacity differ from real's for reasons that are not
  wiring. D2's FAIL (real 0.863 vs layered 0.887, free 0.880) and the reading "the wiring does
  not matter" are confounded. The direction of the bias is unknown. A1/A2-era shuffle
  comparisons had the same denominator problem through `ratebrain.py` if it used the real
  totals (not checked here).
- Fix direction: normalise each control by its own in-strength (keep each neuron's deficit
  ratio `model_in / in_total` from the real brain), or shuffle stubs so that each neuron's
  synapse in-strength is kept. Report the plastic-synapse count per arm next to each result.

### C2. From the A4 start, most of the brain is silent, and so is every descending neuron. Training cannot reach them.

- **Measured**, cut real brain at A4 broad's chosen start (gain 2, threshold 0.05, then the KC
  and MBON bisections), 64 sniffs around the resting rate: **91.4% of neurons silent**
  (r < 1e-3), **780 of 8,257 cell types** with any active cell, **100% of descending neurons
  silent**, and every behaviour group (approach, eat, flee, groom) silent.
- The trained A4 broad checkpoint agrees: `runs/a4-broad/real.pt` has **7,046 of 8,257 types
  whose gain, threshold and τ never moved** (|Δ log| < 1e-4 after 8 epochs of Adam at 3e-3),
  and 15,501 of 61,210 KC→MBON multipliers untouched. On 64 test sniffs the trained brain has
  90% of neurons silent and 99% of DNs silent.
- Why: `tanh(relu(·))` (`ratebrain2.py:151`) gives exactly zero gradient to a unit below
  threshold, and a unit that stays silent contributes `g·0`, so its type's gain, threshold and
  τ get no gradient either. The "per-type" brain in A4 broad was in practice a brain of ~800
  types around the antennal lobe, the mushroom body and the MBONs.
- A5 (gain 8) is less extreme but the same in kind: trained `real-type-s1` on 64 test items has
  76% of neurons silent and **80% of DNs silent**. The "behaviour agrees" figures (real 31–55%)
  are a read of neurons that are mostly not firing (the probe behaviour values are ±0.01).
- Failure scenario: decision 2026-09-25 #4 ("the answer is read from his descending neurons")
  cannot be met by swapping the read in `ratebrain2.py:153`. With this start, the DN read would
  be identically zero and its gradient too. The start rule has to be judged on whether the
  paths to the DNs are active, not only the KCs and MBONs.

### C3. The KC sparseness pressure is inert; the A4 broad brain ended at 15% KCs active

- `a5.py:29-30`: `KC_RATE_TARGET = 0.01` is a **mean rate**, penalised as
  `10·relu(mean(r_KC) − 0.01)²` (`a5.py:244, 331`; `gate_a5.py:122`; `a4_broad.py:207`;
  `a4b_dev.py:369`; `a4_pilot.py:251`). `KC_INIT_TARGET = 0.05` (`a5.py:108`) and D3 are a
  **fraction active** (r > 0.01). The two are different quantities.
- **Measured** at the A4 start: 4.9% of KCs active, their mean rate 0.032, so the mean KC rate
  is **0.0017**, a sixth of the target. The penalty is exactly 0 and its gradient is 0.
- Trained A4 broad brain on test sniffs: **15.0% of KCs active** (above the fly's 2–10% band the
  docs cite, and above D3's 0.10 bar had D3 applied), with the mean rate 0.0085, still under
  0.01, so the penalty never fired. In A5 the trained real brain sits at 6.8% active, mean rate
  0.016: the penalty is ~4e-4 against a BCE of ~0.3, so it does almost nothing there either.
- The 0.01 target and the weight 10 were set in GATE-A2 "on training data" for the 12-step,
  per-neuron `ratebrain.py`, where KCs ran hot. They were carried into the 40-step, per-type v2
  brain with no v2 reason, which is also a breach of the carry-over rule (see I1).
- Failure scenario: any claim that sparseness is "kept fly-like by training" is false. D3 in
  A5 passes because of the start, not because of the pressure. Fix: penalise the fraction
  active (or a smooth surrogate) against the same target the start uses.

### C4. A5 was run with both flaws that were found today, and was evaluated on a GPU

- **Start rule:** every one of the 15 A5 runs chose **gain 8** (`runs/gate-a5/*.json`
  `init.gain`). A4b-dev measured that this start keeps only 0.53 of input similarity at the
  ORNs and 0.37 at the KCs. All A5 arms share it, so it does not bias real vs control directly,
  but A5's absolute numbers and D1 were produced by a brain that starts by scrambling its
  smell.
- **Mashed question:** `a5.py:81` `smell = clip(nose(X) + qz[question])`, and pictures get the
  sweet question's smell alone (`a5.py:78`: every picture has an identical smell). **Measured:**
  the question codes add 4.4–6.3 units of drive spread over the channels; after adding them,
  the pairwise item similarity correlates 0.96 with the antenna alone, and 1–2% of channels
  clip at 1. The cost to item information is modest. The larger costs are structural: the
  question is a constant offset per part, it uses the same glomeruli as the item, and the
  antenna alone keeps only 0.856 of the full embedding's similarity. This contradicts decision
  2026-09-25 #3 either way.
- **Undertrained real arm:** the real-type runs chose epochs 10, 10 and 9 of 10 (val still
  rising at 0.875), while layered chose 5, 6 and 9. D2 was decided at a budget where real had
  not finished and layered had.
- **Evaluated on the training GPU:** `device.default()` picks CUDA, then MPS
  (`device.py:13-16`), and `gate_a5.evaluate` runs on `m.device`. The published A5 test numbers
  are MPS numbers, not repeatable bit for bit (see D1 below), which is against decision #7.
- **The C1 control confound** applies to A5's D2.
- The three real-type seeds share an identical start (the init is label-free and
  deterministic; `spread 0.0485` in all three) and differ only in batch order. "Seed-to-seed
  disagreement" for real therefore measures batch order plus GPU noise, not start variability.
  That is legitimate, but the controls' seeds also change the wiring, so the disagreement rows
  are not comparable across arms.

### C5. The A4 broad smoke run wrote sealed cold (BTZSC) and test picks to disk before the real run

- `a4_broad.py:299-300, 303-305`: `--smoke` writes `test` and `cold` picks (300 each) to
  `runs/a4-broad/smoke/real.json`, which exists and is committed (`aa0b8dc`). The same pattern
  is in `gate_a5.py:140-154` (test, "never read") and `a4b_dev.py:382-386` (practice).
- This is the process flaw recorded in ERRATA for GATE-A2 ("the smoke run wrote validation and
  test scores to disk before the commit"), happening again. BTZSC was described as "sealed
  since the pilot". Nothing shows the file was read, but, as the erratum says, the record
  cannot show that it was not. Fix: a smoke run should score only training items.

---

## Important

### I1. v1/A-era settings running without a v2 reason written next to them

| setting | where | origin | status |
|---|---|---|---|
| KC penalty 10 × (mean rate − 0.01)² | `a5.py:29-30` | GATE-A2, tuned for `ratebrain.py` (12 steps, per-neuron) | inert in v2 (C3) |
| "active" = r > 0.01 | `a5.py:117, 303`, `gate_a5.py:44` | A2 | on a tanh rate scale where active KCs average 0.03, 0.01 is a third of a typical active rate; unexamined |
| logit = k · d · **10** + c | `ratebrain2.py:157` | `ratebrain.py:130` (A1) | a magic factor; with the A4 read-scale rule it is cancelled by k anyway |
| behaviour groups | `model2.py:159-165` reads `config/readout_populations.yaml` | "frozen at freeze-v1", chosen for **account actions** (follow, like, leave, post) | the DN sets for "approach/avoid" under decision #4 need choosing for v2 |
| DT 5 ms, STEPS 40, READ_STEPS 8 | `ratebrain2.py:29-31` | GATE-A5 states them; no reason for 200 ms or a 40 ms read | **measured:** not settled at step 40 (max \|Δr\| 1.3e-3 per step while active KC rates are ~0.03), so the read averages a transient |
| TAU0 20 ms, TAU_RANGE (5, 200) | `ratebrain2.py:32-33` | 20 ms is Shiu's (reason given); the range has none | see I4 |
| INIT_GRID, KEEP_GRID, MBON_INIT_TARGET 0.2, KC_BAND, spread ≥ 1e-4 | `a5.py:28, 107-109, 148, 189` | A5 preflight; written down, but chosen to make brains train, not from the fly | acceptable if stated as harness choices |
| EYE_PCS 26, VIS_FAN 3 | `senses.py:27-28` | A2 used 26 PCs and 6 channels per cell | the change of 6 → 3 has no stated reason |
| question embeddings | `a5.py:23` `cache/a2/questions.npz` | A2 | reused as-is |

### I2. The pipeline contradicts the 2026-09-25 decisions (expected before the rebuild, listed so none is missed)

- **#4, answer from DNs:** `ratebrain2.py:153` reads MBONs. The DN path is silent (C2).
- **#3, question and thing on separate senses:** A5 adds the question into the smell
  (`a5.py:81`). A4 broad and A4b sum item and option into the **same** glomeruli
  (`a4b_dev.py:315`, `0.5 + (item − 0.5) + (option − 0.5)`), so the "question" (the label word)
  and the thing share one sense.
- **#7, CPU serving and evaluation:** `device.py` prefers GPUs, and every script evaluates on
  the training device. No script has a CPU evaluation pass.
- **#2, optic lobes:** `model2.V2_SUPERCLASSES` (`model2.py:28`) still excludes `ol_*`.
  (Known; a rebuild item.)

### I3. The read scale k is parametrised linearly; in A4 broad it could not train

- `ratebrain2.py:101` `k` is a plain parameter. A4 broad sets k = 1/(10·raw spread) = **427.99**
  (`a4_broad.py:188`). With Adam at lr 3e-3, each step moves k by about 0.003, so k is
  effectively frozen (the checkpoint's k is 427.998). Use `log_k`.

### I4. The τ clamp kills its own gradient

- `ratebrain2.py:142` `torch.exp(log_tau).clamp(5, 200)`: past either bound the gradient is 0
  and Adam's momentum decays, so a type that crosses the bound stays there. **Measured:** 14
  types in A5 `real-type-s1`, 34 cells in `real-neuron-s1` and 8 types in A4 broad sit below
  5 ms. It is small, but it is the same class of bug as the maze clamp. Use a smooth bound
  (for example `5 + 195·sigmoid(·)`).

### I5. The bi46 "nose alone" baseline is a dot product, not a cosine

- `a4_broad.py:315-317` and `a4b_dev.py:216, 227` score `argmax(zl[opts] @ zi)` on whitened,
  centred codes. For the full embedding both sides are unit-norm, so that is a cosine; for bi46
  it is not, and label words with larger projected norms win more often. The docs call both
  "cosine" (`a4b-dev-results.md` L1, `A4-BROAD.md`). The bi46 nose numbers (0.362, 0.453, tier
  columns) may be under- or overstated. The G2 rule uses the 1,024 nose, so G2 is not affected.

### I6. The A4b-dev antenna was fit on the practice kinds' label words

- `a4b_dev.py:166`: bi46 is fit on training items **plus every label word** (`L` holds the
  practice kinds' labels too). It is unsupervised, but it is transductive on the held-out
  kinds' label words, and it was A4b-dev's chosen nose. A4 broad fixed this (`a4_broad.py:139`
  uses taught labels only). The bi46 practice number in `a4b-dev-results.md` (0.362) is
  slightly optimistic.

### I7. The fly_init_keep start takes about 14 minutes on the CPU

- **Measured:** one forward pass of 64 sniffs takes about 1.6 s on CPU. `fly_init_keep` runs
  10 grid points × (3 bisections × 16 + 3) forward passes plus a similarity pass, about 520
  passes, which is roughly 14 minutes before training starts. That matters for the CPU-only
  gate and for any replay on the Kimsufi. Coarser bisection (8 steps gives 0.02 resolution) or
  bisecting only at the chosen grid point would cut it several-fold.

---

## Determinism (item 3)

**Measured on CPU** (cut real brain, float32, 4 threads):

- The forward pass is **bitwise repeatable**, run to run, and **identical between 1 and 4
  threads** (max diff 0.0).
- **Batch composition:** sniff 0 run alone vs inside a batch of 64 differs by **1.9e-8**. A
  served answer therefore depends, below 1e-7, on what else was in the batch. This is harmless
  for decisions, but "exactly repeatable" (decision #7) only holds if serving fixes the batching
  (always B = number of options of one question, or always B = 1) and evaluation uses the same
  batching.
- **The backward pass is not bitwise repeatable on CPU:** the gradients of `log_g`, `b` and
  `log_tau` differ between identical runs (the backward of the per-type gather
  `param[self.unit]` is a scatter-add over threads). `kp_logm`, `k` and `c` repeat. CPU
  training is repeatable only with `torch.use_deterministic_algorithms(True)` (as the A4
  plain net already does, `a4_broad.py:321`) or with a single thread. That was not verified for
  the sparse ops here.
- **GPU:** `index_add` (`ratebrain2.py:150`), the sparse-mm backward and the gather-backward
  scatter use atomics on CUDA, and MPS was shown non-repeatable (A4-BROAD amendment 3). No
  script sets deterministic flags for the brains. Replays drift about 0.01 (a4-broad-results).
- **Seeds:** there is no torch randomness in the brain path; `torch.manual_seed` calls are
  harmless no-ops. All randomness is numpy `default_rng` with fixed seeds (batch order,
  shuffles, splits). That part is sound.
- **The CPU path that gives repeatable results exists today for inference:** `BOSCO_DEVICE=cpu`,
  float32, fixed batching. Checkpoints load on CPU (probe3 did this).

---

## CPU performance (item 6)

**Measured**, cut real brain: n = 50,140 neurons; 2,294,517 fixed edges plus 61,210 plastic
KC→MBON edges; 40 steps.

| run | time |
|---|---|
| forward, B = 64 sniffs | 1,616 ms (**25 ms per sniff**) |
| forward, B = 8 | 632 ms (79 ms per sniff) |
| forward, **B = 1** | **533 ms** |
| forward with `record=True`, B = 64 | 1,652 ms (+2%) |
| forward + backward, B = 64 | 3,324 ms |

Per step at B = 64: `torch.sparse.mm` (COO) **32.1 ms**; the same matrix as CSR 23.7 ms;
plastic `index_add` 1.8 ms; elementwise 3.6 ms. **The sparse mat-mul is about 80% of the
time**, and 40 of them are a sniff's cost. Per sniff that is about 40 × 2.36M ≈ 94M
multiply-adds, memory-bound. The Python loop itself is negligible.

What this means for Halteres: a 2-option question at B = 2 takes about 0.55 s on an M3 Max.
The Kimsufi's 4 older cores will be slower, likely around 1–2 s (not measured). A 93-option
kind takes about 2.3 s at B = 93.

Speed-ups, in order of value:

1. **Drop neurons that can never fire (exact).** At the A4 start 91% of neurons are silent and
   in trained A5 76% are. A unit whose threshold bias is ≤ 0 and that is not downstream of any
   input within 40 steps is exactly 0 forever. Removing such rows and columns (or restricting
   to the set that is active on any calibration sniff, checked exactly) could cut the matrix
   several-fold. This must be re-checked per checkpoint.
2. **CSR instead of COO**, with the plastic block folded in at inference: −26% measured on
   the mat-mul, about 1.3× overall. It is a one-line change in serving and keeps results
   deterministic (verify bitwise).
3. **Batch the options of a question, and concurrent questions, into one call.** B = 1 costs
   533 ms, while B = 64 costs 25 ms per sniff: per-call cost dominates. A micro-batching server
   is worth up to about 20× in throughput. Fix the batching to keep answers bitwise stable (see
   determinism).
4. **Record lazily.** `record=True` costs only 2% of time on CPU but stores 40 × n × B float16
   (about 4 MB per sniff). Record only what Halteres displays (groups, or every k-th step), or
   rerun a single sniff with recording when someone opens the trace.
5. **Settle or shorten time.** The state is not settled at step 40, so steps cannot simply be
   cut. If the read were taken at a fixed point, or with a larger DT after a v2 reason is
   written, cost falls linearly with steps.
6. **Do not use bfloat16/float16 on CPU:** a bfloat16 sparse mat-mul was **2.2× slower**
   (69 ms vs 32 ms) and changes the numbers. `torch.compile` will not touch the sparse kernel,
   and only the elementwise 10% could fuse. MKL or scipy sparse SpMM with multiple right-hand
   sides is an option if CSR in torch is not enough.

---

## Minor

- **M1. Shuffles create self-loops and duplicate edges:** layered 301 self-loops (real 25).
  Duplicates are merged, which is why layered and hash have about 11% fewer plastic synapses
  (C1). `controls.py:31-43`.
- **M2. `picks` breaks ties by taking the first option** (`a4_pilot.py:188-197`). A dead or
  flat read always answers option 0. That is harmless under balanced accuracy for a truly dead
  read, but it inflates plain accuracy on kinds where option 0 is common (the A4 pilot used
  plain accuracy).
- **M3. `a4b_dev.bare()`** (`a4b_dev.py:98-99`) drops everything before the first newline. On an
  input with no instruction line but a multi-line text, it drops real text. It was not counted.
- **M4. `_similarity_kept`** (`a5.py:151-169`) compares input cosines taken from rest
  (`x − 0.5`) against KC cosines taken from the batch mean (`k − k.mean(0)`): two different
  centrings. It is a defensible choice, but it is not documented in A4-BROAD.
- **M5. Hard-coded 52** (= 2 × EYE_PCS) in `a5.py:156`, `gate_a5.py:86`, `a4_pilot.py:205, 225,
  249`, `a4_broad.py:166, 177, 205`, `a4b_dev.py:326, 346, 367`. It breaks silently if the
  eyes change (they will: decision #11).
- **M6. The validation sets are not near-duplicate-checked** against training (A4 broad checks
  only test and cold, `a4_broad.py:113`). Validation only picks the epoch, so the effect is
  small.
- **M7. The state dict omits the wiring** (`W`, `kp_w` are plain attributes, not buffers). A
  control checkpoint is only meaningful with the arm and seed that rebuild its shuffle. Fine
  today; fragile for serving a versioned model. Store a hash of the wiring with the checkpoint.
- **M8. `set_kc_threshold` and `set_mbon_threshold`** write per-type thresholds. In
  `mode='neuron'` they write per cell, so the two modes start from the same values; this is
  fine and was checked.

---

## What was checked and found sound

- Epoch selection is on validation in `gate_a5.py`, `a4_broad.py`, `a4_pilot.py` and
  `a4b_dev.py` (training kinds' validation). No script selects on test, cold or practice.
- `a4_broad.balanced` (mean recall over the labels present) has expected value 1/n for a
  uniform picker, so the 1/n chance column is right.
- A4 broad near-duplicate checks cover test and cold against training across all kinds.
- `model2.mbon_groups` sides follow the stated rule. `wiring_values` applies the eLN and
  DAN→KC zeros to controls by presynaptic identity, which is correct.
- No dtype problems were found. All tensors are float32; the trace is float16 on CPU, used for
  display only.
