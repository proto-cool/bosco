# Model audit, 2026-09-28: how the fly is fed, what he uses, what to build next

An adversarial read of family v1 (`service/families/v1`), the rate brain (`src/bosco/ratebrain3.py`), its
build (`src/bosco/v1.py`), the senses (`src/bosco/senses.py`), the antenna (`scripts/v1_pilot.py:41-55`,
`scripts/export_family.py:antenna`) and the service runner (`src/bosco/service/core.py:run_cols`).
Read-only; nothing was trained and no sealed test was read.

**Provenance of the measurements.** Commit cc31052 (06:50 today) deleted every trained model and
`service/specialists/`. The only v1-family weights left are in git history. For the measurements in §3 and §5
I extracted the gate-2 checkpoints `junk.pt`, `topic.pt` (design A) and `support.pt` (design B) from tag
`specialist-gate-2-run` into `/tmp` (not into the repo), and deleted them afterwards. Items are gate-2
**validation** items (`data/cache/v1-gate2`), 16–48 per task; small samples, so these are diagnostics, not
results. Every run used the service's own arithmetic (`Family.params`, `W_csr`, `unit_fn`) with 2 CPU threads.
Scripts: `/tmp/audit/a1.py`–`a7.py` (not kept in the repo).

---

## Summary

| # | Finding | Evidence | Severity |
|---|---|---|---|
| F1 | **Design A throws most of the signal away.** The sniffs for different options differ only by the option word's smell, added to the item's. For junk, 91% of the logit variance is "how much does this item excite the DNs", which cancels in the softmax. The choice rides on 1.7% interaction plus 7.5% option bias. For topic, the interaction is 35%. | §1.3 | high |
| F2 | **The Kenyon-cell code is stereotyped.** Each sniff has 4–6% of KCs active, as intended, but only 18–21% of all KCs are ever active across 64–128 different sniffs. Two different items share 26% (junk) to 55% (topic) of their active KCs. In topic, **changing the option word moves the KC set more than changing the item does** (Jaccard 0.36 vs 0.55). The memory mostly sees the option word. | §3.2 | high |
| F3 | **Every sniff starts from r = 0, not from rest.** The docstrings say "from rest", but `ratebrain3.py:206` and `core.py:116` both zero the state. The first ~40 of the 80 steps are the brain booting. KCs are silent until about step 15, and the read's item spread reaches ~50% of its final value only by step 40 and ~95% by step 60. | §5.1 | high (cost and design) |
| F4 | **Taste reaches the DNs, not the KCs.** Random per-sniff gustatory input moves the logit by 0.37–0.99 × the item spread. Mean change at the KCs is 0.0000, at MBONs/DANs 0.001–0.003 and at DNs 0.004. Form can bias the answer through the SEZ but cannot be stored in the mushroom-body memory. | §4.2 | medium |
| F5 | **Sight is live and dominant.** The VPN path is intact and costs no extra time: the input is a bias term, so the matvec is unchanged. Random sight input moves the logit **3–9 × the item spread**, and VPNs are one synapse from the DN read. Sight skips the KCs almost entirely (KC Δ 0.0005–0.002 against DN Δ 0.04–0.05). The service has no sight argument. | §4.1 | medium |
| F6 | **The antenna is not the main bottleneck.** On gate-2 val, a linear probe on the 46 channels (Z) is within 0.6–1.6 points of the full 768-d embedding for topic, support and politeness. It loses 4 points on junk and 8 on hate. A 92-component version gains 1–4 points (linear). | §2 | low–medium |
| F7 | **The antenna is fit half on option-label words.** 226 pilot labels are repeated 17× (3,842 of 7,842 fit rows), and the texts are English only (hate, hatemoji, topic, massive, clinc). All 46 whitened channels are weighted equally, so low-variance directions are amplified as much as the top ones. | §2 | medium |
| F8 | **The resting input is at half scale.** Every glomerulus rests at 0.5, so ORNs fire tonically at about 0.4. An item moves a channel by ±0.18 (s.d.) around that. Unphysiological, and the likely cause of F2. | §1.1 | medium |
| F9 | **The encoder does not run on this Mac.** `SentenceTransformer(nomic-embed-text-v1.5)` raises `AttributeError: 'NomicBertModel' object has no attribute 'get_extended_attention_mask'` on both cpu and mps (transformers 5.17.0, sentence-transformers 6.1.0, per `uv.lock`). `Family.smell`, and so the service, cannot encode here. The nomic remote code (`nomic-bert-2048`) is not pinned by revision. | §5.4 | high (service) |
| F10 | **DN read groups are label-free but not motor-selected.** The avoid group includes steering DNs (DNa02, DNa03, averaged across sides) next to MDN. The approach group is mostly SEZ DNg/DNge types, and DNp09 (forward walking) is in the model but not in the group. | §3.4 | low–medium |

---

## 1. Input path: how a smell reaches the brain

### 1.1 Mapping

- `senses.nose` (`senses.py:59-64`): 53 glomeruli with ORNs in `pop.orns()`, minus 7 innate ones
  (`INNATE_GLOMERULI`, `senses.py:26`: DA1, DL3, VA1d, VA1v, VL2a, DA2, V) = **46 channels**. The order is
  a seeded permutation, so antenna component i lands on glomerulus `order[i]` (the top component on VM6v, and
  so on). There is no anatomical reason for any component–glomerulus pairing. That is acceptable as a stated
  stand-in, but it should be written as "arbitrary" and not described as a receptor map.
- **ORNs per channel:** 14–84 (median 34), 1,865 fed out of 2,639 in the model. The 774 unfed ORNs are the
  innate glomeruli (DA1 204, VA1d 132, VA1v 130, DL3 103, VL2a 98, V 55, DA2 48) plus 4 with no glomerulus.
- **Both sides are fed the same value** (`ratebrain3.py:194`: `inp[orn_idx] = smell.T[orn_chan]`). The fed
  ORNs' rootSide is R 949, L 717, unknown 199. The asymmetry is in the reconstruction; the two antennae are the
  same.
- **Ipsilateral share of ORN→ALPN synapses:** right PNs 73% from right ORNs, left PNs 55% from left ORNs
  (384,532 synapses). A left/right split would give partly independent PN drive, but the sides are unequal and
  the channels would mix at the PNs. It is not a clean way to double the channels.
- **Input is a bias, not a rate:** ORN drive = W·r + b + z. The homeostatic start leaves sensory types at
  threshold 0.05 (`v1.py:99`, sensory excluded from tuning). With z at rest 0.5, an ORN sits near
  tanh(0.45) ≈ 0.42. Real ORNs rest at a few percent of their maximum. This is F8.
- `senses.py`'s module docstring still describes the older "K components ± split into 2K channels" nose. The
  shipped family is "bi46" (46 signed components around 0.5, `core.py:80`). The docstring is stale.

### 1.2 Saturation and clipping (gate-2 Z, 100,508 items; ZL, 190 option words)

- Item smells alone: 0.45% of channel values clip at 0 and 0.33% at 1. **30% of items have at least one
  clipped channel.** The 1–99th percentile span per channel averages 0.84 of [0,1] (range 0.67–0.98). The
  range is well used and clipping is rare per value.
- Option words: 1.2% clipped.
- **Design A mix** `clip(z_item + z_opt − 0.5, 0, 1)` (`core.py:154`, `a4b_dev.py:315`): 3.7–8.1% of
  channel values clip (hate 8.1%, topic 5.3%, junk 4.1%, politeness 3.7%). This is a loss, but not the main
  one.

### 1.3 The blend itself (F1)

The sum is symmetric: the brain cannot tell which part of a glomerulus's drive came from the item and which
from the option. For a yes/no task, the two sniffs differ by the constant vector ZL[yes] − ZL[no], which is
**the same for every item**. The option words of a yes/no task are close together: the s.d. of the option
smell across options is 0.05 (junk, hate), against 0.16–0.17 for items. Any item-dependence of the choice
must come from the brain's nonlinearity acting on a small, fixed offset.

Measured on the trained gate-2 specialists (two-way decomposition of the logits, items × options):

| task (design) | item main effect | option main effect | interaction | logit s.d. |
|---|---|---|---|---|
| junk (A, 2 opt) | **0.908** | 0.075 | **0.017** | 10.7 |
| topic (A, 4 of 14 opt) | 0.625 | 0.021 | 0.354 | 5.4 |
| support (B, 4 of 18 opt) | 0.920 | 0.037 | 0.043 | 16.5 |

Only the option and interaction columns can change a softmax choice. For yes/no, the brain spends two sniffs
to produce a 1–2% difference and discards the rest. The approach-minus-avoid read is already a valence, and
one sniff of the item would use all of it directly (see §6).

## 2. The 46-d antenna

- **Fit data** (`v1_pilot.py:44-51`, repeated verbatim in `export_family.py:antenna`): 4,000 random training
  texts from the pilot's five tasks (hate, hatemoji, topic, intent_massive, intent_clinc, all English), plus the
  226 option labels repeated 17 times. **Labels are 49% of the fit rows.** The components are therefore shaped
  as much by one-to-three-word label strings as by texts. It is label-free in the sense that no gold answers
  were used, but it is not task-neutral: it leans toward the pilot's vocabulary.
- **Whitening** (`W = vt[:46]/s[:46]`) gives every component unit variance. Channel s.d. on gate-2 ranges from
  0.14 to 0.21, so the 46th component drives its glomerulus as hard as the first.
- **Variance kept** (orthonormal span of W) on gate-2 X: **0.445** overall (per task: topic 0.29, politeness
  0.33, junk 0.38, hate 0.38, mood 0.40, support 0.48, intent 0.51). Gate-2's own top-46 PCs keep 0.53, top-92
  0.69, top-138 0.78, top-184 0.84.
- **What the brain could at most get linearly** (logistic regression, ≤ 8,000 train rows → val, balanced
  accuracy):

| task | X 768-d | Z (family antenna, 46) | own PCs 46 | own PCs 92 |
|---|---|---|---|---|
| topic | 0.973 | 0.967 | 0.947 | 0.965 |
| junk | 0.895 | 0.853 | 0.814 | 0.842 |
| hate | 0.717 | 0.636 | 0.684 | 0.694 |
| politeness | 0.663 | 0.636 | 0.633 | 0.653 |
| support | 0.840 | 0.824 | 0.786 | 0.823 |

  The antenna costs little for topic, support and politeness and a lot for hate (8 points) and junk (4). Going
  from 46 to 92 components is worth 1–4 points linearly. It is worth doing only if it is free (timed sniffs,
  §6), not by splitting glomeruli.
- **Language bias** could not be measured: the encoder does not load (F9), and the gate-4 language data is out
  of bounds for this audit. By construction it is plausible: an all-English fit with a label-word half. Test it
  before a language specialist ships: compare the variance kept on non-English dev text with that on English.
- **More channels at zero neuron cost?** Per-ORN channels within a glomerulus do not help. All of a
  glomerulus's ORNs converge on the same uPNs, so they average to one input per glomerulus per side. Left/right
  gives about 1.3× at best (55–73% ipsilateral) with unequal sides. The seven innate glomeruli (774 ORNs)
  would add 7 channels, but they drive pheromone, CO2 and geosmin circuits; they were excluded for good reason.
  **Time, not space, is the free axis** (§6).

## 3. Which neurons are used (trained gate-2 weights, val items)

### 3.1 Per region, read window (last 8 steps)

"Varies" means the s.d. across sniffs is above 1e-3; "silent" means it never exceeds 0.01 at any step. Values
are shown as junk / topic / support.

| region | n | varies | silent |
|---|---|---|---|
| ORN | 2,639 | .71 / .71 / .71 | .29 (= the unfed innate ORNs) |
| ALPN | 686 | .78 / .78 / .76 | .20–.23 |
| ALLN | 420 | .76 / .72 / .72 | .24–.26 |
| **KC** | 4,064 | **.24 / .25 / .31** | **.81 / .76 / .69** |
| MBON | 97 | .75 / .72 / .98 | .19 / .19 / .00 |
| DAN | 340 | .15 / .14 / .46 | .24 / .43 / .07 |
| LH | 1,973 | .81 / .76 / .73 | .17–.27 |
| CX | 2,950 | .21 / .10 / .27 | .09–.35 |
| VPN | 9,201 | .01 / .01 / .02 | .05–.44 (no sight input) |
| DN | 1,314 | .07 / .04 / .13 | .20–.39 |
| other sensory | 2,785 | .00–.02 | .80–.92 (idle by design) |
| rest | 23,671 | .20–.25 | .21–.38 |

Nothing is saturated: at most 5.6% of ORNs and under 0.1% of anything else has a mean above 0.9. **Overall,
only 21–25% of the 50,140 neurons change with the input.** The VPNs (18% of the brain) and the other sensory
neurons (6%) are idle by construction. The largest gap between what is there and what is used is the KCs.

### 3.2 The KC code (F2)

| | junk | topic |
|---|---|---|
| active per sniff | 4.3% | 5.5% |
| ever active across sniffs | 21% (96 sniffs) | 18% (128 sniffs) |
| Jaccard, different items | 0.26 | 0.55 |
| Jaccard, same item, other option | 0.57 | **0.36** |

If sparse codes were independent, 96 sniffs at 5% would recruit about 99% of KCs. Here the same ~20% win
every time. The threshold is bisected to 5% on a tonic input (every glomerulus at 0.5, F8), so the KCs with
the most PN input always cross it, and the item's ±0.18 modulation only reshuffles the margin. The
**KC→MBON memory, the one place a fly learns, is written on about 800 of 4,064 KCs.** In topic, the option word
changes the KC set more than the item does.

### 3.3 Reach within 80 steps

In unweighted hops through the cut wiring, everything is close. ORN→KC is 2 hops (median 2); ORN→DN read has
a minimum of 2 and a median of 3. Gustatory→KC is 2–3 hops, gustatory→DN read 2. VPN→KC/MBON/DAN is 1–2 hops
and VPN→DN read is 1. Hop counts are not strength, though. The dynamic tests in §4 and the timing in §5.1 are
the real answer: ORN signal reaches the KCs by step ~15–20 and the read's item spread is ~95% formed by step
60 (from r = 0).

### 3.4 The read groups (F10)

`v1.dn_groups` (`v1.py:126-156`) selects DN types by their 1–2 hop signed input from the approach MBONs
minus the avoid MBONs (threshold 1e-4). The result is 16 approach cells (10 types: CB0429, DNg104, DNg13,
DNg34, DNge138/149–152, DNp68) and 30 avoid cells (13 types including MDN, DNp42, DNa02, DNa03, DNa13).

- MDN (backward walking) and DNp42 are sensible avoid cells.
- DNa02 and DNa03 are steering DNs. Averaging left and right cells throws away their sign, so they measure
  "turning", not "away".
- Most approach types are SEZ DNg/DNge types whose motor role is largely uncharacterised.
- DNp09 (forward walking toward a stimulus; 2 cells in the model) is not in the approach group.

The rule is label-free and fixed before training, which is correct. The claim that the answer is "going toward
or away" is anatomically weak and should be worded as "the DNs the MBONs drive most, by sign".

## 4. Sight and taste

### 4.1 Sight

- `vis_idx` (9,201 VPN cells, 346 types) and `vis_M` (cells × 52: each type takes the mean of 3 of 52 channels,
  `senses.py:79-89`) are intact in `RateBrain3.run` (`ratebrain3.py:195-196`).
- **Speed cost: zero.** Sight is a bias term added before the same matvec.
- `Family.run_cols` has no sight argument, so the service cannot see yet.
- What it needs:
  1. a pinned picture encoder;
  2. the 26-component eye antenna, fit label-free and saved in the family;
  3. a `sight` column in `run_cols`;
  4. specialists trained with it (decision 38: an untrained sense gets no input).
- Measured with random 52-channel sight on the trained text specialists: the logit moves 3.0–9.1 × the
  item-to-item spread. The DN mean |Δr| is 0.04–0.05, against KC 0.0005–0.002 and MBON 0.006–0.008.
  **Pictures would reach the answer almost entirely through fixed VPN→DN paths,** bypassing the mushroom-body
  memory, and would swamp smell unless scaled. Expect the per-type gains to have to learn a large damping.
  Scale the eye input (for example to rest 0 with a small range) and measure the KC share before training.

### 4.2 Taste

- 355 gustatory neurons (275 cb_sensory + 80 ascending) in 45 types (claw_tpGRN 50, LB3d 26, LgAG1 25, …);
  entry nerves MxLbN 221, aPhN 36, leg nerves 80. No receptor annotation.
- Uniform +0.5 on all gustatory neurons: large mean logit shifts (−35 junk, −50 support), but these cancel
  across options (Δ s.d. 1–2% of the item spread).
- **Random per-sniff taste patterns:** Δlogit s.d. = 0.37 (junk), 0.39 (topic), 0.99 (support) × the item
  spread. Taste is within reach of the read.
- **But mean |Δr| at the KCs is 0.0000** (MBON 0.001–0.0035, DAN 0.0003–0.0017, LH 0.001–0.004, DN
  0.0017–0.0044). In this cut and at this operating point, taste does not enter the KC code.
- **Consequence:** "taste for form" can shift the answer through the SEZ and the per-type gains, but a
  specialist cannot store form in its KC→MBON memory, and a design B bank cannot use it per option. That is
  acceptable for yes/no (one read), but it must be stated.

## 5. Dynamics, parameters, and the service runner

### 5.1 Start state (F3)

The mean |Δr| per step is 2e-3 at step 10, 8e-4 at 20, 1.4e-4 at 40 and 2.6e-5 at 60. The item spread at
the read is 1–13% of its final value at step 20, 40–75% at step 40 and 76–96% at step 60.

**Experiment, weights unchanged:** start from the specialist's resting state (80 steps at z = 0.5, which
`rest_of` already computes for the trace) and run only 40 steps:

| | 80 steps from r = 0 | 40 steps from rest | 20 steps from rest |
|---|---|---|---|
| junk, accuracy (48 val) | 0.979 | 1.000 (r = 0.97 with the 80-step logit differences) | 0.48 |
| topic, accuracy (32 val, 4 options) | 1.000 | 0.25 (r = 0.57) | 0.17 |

Junk survives, so the answer does not need the boot transient in principle. Topic's parameters were trained
on the transient and would need retraining. Starting from rest is the fly-true choice, since the fly is alive
before the odour, and it frees about half the steps.

### 5.2 Other dynamics

- **τ clamp (5–200 ms, DT 5 ms):** no type sits at a bound in junk; one type sits at 5 ms in topic. Fine. At
  τ = DT a neuron is memoryless (alpha = 1), which is stable. Gains trained to 2.7–7.4 (junk) and 1.9–17 (topic).
- **The answer rides on tiny rates.** The read's item spread is 0.0009–0.003 rate units, and the scale
  `exp(log_k)·10` (`ratebrain3.py:239`) turns that into logits with s.d. 5–16, so the gain is about 10⁴. The
  answer is deterministic on the CPU, but it is a 0.1%-of-range difference in 46 DN cells. The playback shows
  DNs that barely move, and the fly-level claim ("his DNs decide") should be modest. `c` is dead in every
  softmax over options; only a one-sniff read would use it.
- **KC penalty** acts on the last step's state only (`v1_pilot.step`). The KC fraction is stable from step 20
  (5–6%), so this is fine.
- **Per-type sharing:** 8,257 units. `untyped:unclassified` makes 898 unrelated neurons share one gain,
  threshold and τ. BM_InOm shares one unit across 745 cells, and KCg-m across 1,342 KCs (one KC threshold per
  KC type is defensible, since APL acts per lobe). The 898 unclassified cells are the only lumping that is not
  defensible. It is small, but they should be split by superclass.
- **Homeostatic start:** VPNs are tuned to TONIC with no input and ORN/gustatory types are not tuned. This is
  consistent, but a family that adds sight must re-run the start with sight present, or the VPN operating
  point is wrong.

### 5.3 Service runner against training

`run_cols` (`core.py:101-129`) has the same arithmetic as `RateBrain3.run`:

- the same input placement;
- W·(g r) through scipy CSR (the torch path in training; a different float summation order);
- the same KC→MBON term with per-column memory;
- the same read window and `k·read + c`;
- design A's mix is identical to the training mix (`core.py:154` = `a4b_dev.py:315`).

The encoding matches as well: the prefix, 200-word truncation and normalisation in `v1_data.cmd_embed` are the
same as in `Family.smell`, though training embeddings were computed on mps and the service embeds on cpu.

`tests/test_service_parity.py` covers topic, support, junk and kind on 12 test items each. It now **skips
entirely**, because `runs/` and `service/specialists/` were deleted. Before retraining, add a parity test that
needs no gate files: a stored tiny fixture of weights and inputs, `run` against `run_cols` to 1e-5. `REST_Z`
(`core.py:39`) is unused.

### 5.4 The encoder (F9)

On this Mac, the pinned encoder fails to run on both cpu and mps with transformers 5.17.0 / sentence-transformers
6.1.0 (the locked versions). The remote code for `nomic-bert-2048` is loaded without a pinned revision (only
the embedding model's revision is pinned in `family.json`), which is a determinism and provenance hole. Either
pin the remote-code revision and a transformers version that works with it, or vendor the model code. Fix this
before any new family is built, since the antenna fit and every item smell go through it.

## 6. Proposal: family v2, "rest, then sniff", one design

All four owner requirements, with no added neurons and at most the current CPU time. Each part is adopted only
if its dev test beats the current family on dev/val (decision 39).

**Family v2:**

1. **Start from rest.** Each specialist's resting state (and each option memory's, in design B) is computed
   once at load: z at rest, taste and sight at 0, 80 steps. Every sniff starts from that state. The fly is alive
   before the odour. This removes the boot transient (F3).
2. **Two timed sniff windows (decision 39.1).** The same 80 steps as now, from rest:
   - steps 0–39: the item's antenna components 1–46;
   - steps 40–79: components 47–92 of the same fit (92-component antenna, label-free).
   The read is the mean over the last 8 steps of *each* window; the read is computed every step anyway, so
   this costs nothing. The same 46 glomeruli carry 92 components.
3. **Rest drive lowered (F8).** The antenna maps to [0,1] with rest near 0.1 (for example 0.1 + 0.9·clip(p/norm)
   on a ±-split into both halves of each window), so ORNs rest near their spontaneous rate and an item's
   signal is not riding on a tonic ceiling. This targets F2.
4. **Question and item apart (decision 3), as memory, not as a blend.**
   - Yes/no (most of the fleet, decision 36): **one sniff** of the item. The answer is
     sigmoid(k·(approach − avoid) + c), the fly's native valence (F1). This halves CPU for yes/no.
   - Choose or rate: design B, one KC→MBON memory per option (already built).
   - The option word never enters the nose, which removes the blend, the clipping and the option-dominated KC
     code.
   - Decision 3 as written asks for a *sense*. The only senses left for a word are taste and sight, and
     decision 38 reserves those for form and pictures ("meaning is never duplicated"). **Nick should confirm**
     that "the question is the learned memory, the item is the smell" satisfies decision 3. It is the fly's own
     arrangement: what an odour means is stored at KC→MBON.
5. **Taste for form (decision 38).** A fixed counting rule makes about 20–45 form features (length, word-length
   mean, capitals ratio, punctuation and digit ratios, non-ASCII and script-class fractions, repeats, URLs).
   Each feature drives one gustatory type (45 types). Input is 0 at rest and on for the whole sniff. It is used
   only by specialists trained with it. State on the card that form reaches the answer through the SEZ, not the
   KC memory (§4.2).
6. **Sight via VPNs (decision 38).** Wire a `sight` column into `run_cols` (zero cost) for picture specialists
   only, with the eye antenna rest at 0 and the input range scaled so that random sight moves the logit by
   ≤ 1 × the item spread before training (F5). Re-run the homeostatic start with sight present.

**CPU:** the same 80 steps per sniff. Yes/no uses 1 sniff instead of 2, so its cost halves. Choose/rate is
unchanged (one sniff per option, as now). Rest states are computed once per specialist at load; the trace
already computes them. Net ≤ current, and ~0.5× for yes/no.

**The cheapest dev experiment for each part, in order (dev/val only):**

| part | experiment | cost | adopt if |
|---|---|---|---|
| 0 | Fix and pin the encoder (F9); fixture parity test for `run` against `run_cols`. | hours, no GPU | the parity test passes |
| 2 | Linear probe: 46 against 92 components (this audit: +1–4 points own-PC). Repeat with the new fit on the gate-4 dev sets, including language. | minutes, CPU | ≥ +1 point on most tasks |
| 3 | No training: homeostatic start at rest 0.1 against 0.5 on 256 unlabelled sniffs; measure KCs ever active and the Jaccard between items. | ~10 min, CPU | ever-active ≥ 50%, Jaccard ≤ 0.2 |
| 1 | Already partly done: 40 steps from rest keeps junk's answer (0.979 → 1.0) and loses topic's (retrain). Retrain junk and topic from rest, 80 steps, two windows. | 2 short GPU runs | val within 1 point of v1 |
| 4 | Retrain junk and hate as one-sniff yes/no against their gate-2 design A runs (same data and seeds); compare val balanced accuracy and ECE. | 2 GPU runs | ≥ design A, ECE ≤ 0.10 |
| 5 | Linear probe: Z + form features against Z on language, plain and junk dev; then one retrain on the best task. | minutes, then 1 GPU run | probe gain ≥ 1 point |
| 6 | No training: scale the eye input so that random sight gives Δlogit s.d. ≤ 1× the item spread; then one picture specialist when picture data exists. | minutes | — |

Run parts 3, 1, 2 and 4 together as a single family-v2 dev retrain of two yes/no tasks and one design B task,
against v1's gate-2 validation numbers. Then retrain the fleet.
