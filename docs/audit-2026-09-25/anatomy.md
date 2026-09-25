# Audit 2026-09-25: anatomy of the inputs and outputs

This audit covers what enters Bosco, what reaches his Kenyon cells and descending neurons, and
what adding the optic lobes would cost. Every number is computed from the MaleCNS v1.0 feathers in
`data/raw/` (annotations, `connectome-weights-...-minconf-0.5`, neurotransmitters), from
`bosco.model2.load_or_build()` (the 50,140-neuron v2 model) and from `bosco.a5.cut(b, 5)` (the
trained cut brain). "Traced" means `status == 'Traced'` (165,122 bodies), which is the set
`model2` draws from.

The queries that produced the numbers are in `docs/audit-2026-09-25/anatomy-queries/`
(`prep.py` first; `common.py` is shared). One caveat applies throughout. An `unstack` on the
photoreceptor side table gave wrong totals in an early query (`q4.py`). The corrected counts come
from a crosstab and are the ones used below.

Unless a line says otherwise, "synapses" means the minconf-0.5 counts between traced bodies. Of
the 311.8 M synapses in the weights feather, 124.0 M (25.6 M edges) are between traced bodies.

---

## 1. Sensory inputs: full MaleCNS vs the v2 model

| Sense | Class / types | Traced in MaleCNS | In v2 model | Used as an input today |
|---|---|---|---|---|
| Olfactory ORNs | `ORN_<glom>`, 53 glomeruli | 2,635 (+4 untyped) | 2,639 (all) | yes. The nose uses 46 glomeruli (53 minus the 7 "innate" ones) |
| Thermosensory | TRN_VP1m 11, VP2 7, VP3a 6, VP3b 1 | 25 | 25 | no |
| Hygrosensory | HRN_VP1d 18, VP1l 8, VP4 28, VP5 12 | 66 | 66 | no |
| Gustatory, head | labellar LB* 163, pharyngeal PhG* 48, taste peg 60, other 4 | 275 | 275 | no |
| Gustatory, leg (ascending) | LgAG* | 80 (`sensory_ascending`) | 80 | no |
| Gustatory, VNC | leg LgLG* 688, wing WG* 385 | 1,073 (`vnc_sensory`) | 0 | no |
| Mechanosensory, Johnston's organ | JO-A…F, auditory 114, wind/gravity 475, grooming 65 | 672 JO-typed | 672 | no |
| Mechanosensory, other head | BM bristles 932, pharyngeal aPhM 39, TPMN 60 | 1,061 | 1,061 | no |
| Proprioceptive/tactile, ascending | campaniform, chordotonal, haltere… | 424 | 424 | no |
| VNC mechano/proprio/tactile/chemo/unknown | | 5,292 | 0 | no |
| Other/unknown cb sensory | GNG643, SAxx, ENS… | 131 | 131 | no |
| **Photoreceptors** (`ol_sensory`) | R1–R6, R7 p/y/d, R8 p/y/d, HB eyelet | **4,114** | **0** | no |
| Optic lobe intrinsic | 271 types | 89,390 | 0 | — |
| Visual centrifugal | 108 types | 563 | 0 | — |
| Visual projection neurons | 346 types | 9,201 | 9,201 | yes. The `eyes` stand-in drives all of them |
| Ocellar neurons | OCG01a–f, OCG02b/c, OCG03 (inside `visual_projection`) | 20 | 20 | yes, and they should not be (§5) |

### Photoreceptors per eye (by `rootSide`)

| | Left eye | Right eye | Expected per eye |
|---|---|---|---|
| R1–R6 | 501 | 893 | ~5,300 (6 × ~890 columns) |
| R7 (all subtypes) | 607 | 692 | ~890 |
| R8 (all subtypes) | 625 | 704 | ~890 |
| All | 1,733 | 2,374 | |

The column count comes from lamina and medulla columnar cells, which are complete: L1 884 L / 892
R, L2–L5 and Mi1, C3 and T1 all 878–898 per side. That gives about 890 ommatidia per eye in this
male.

**Most R1–R6 are not reconstructed:** only 9% (left) and 17% (right) are there. R7 and R8 are about
70–80% present. As a result, L1 and L3 receive a median of only 10–13 synapses from traced
photoreceptors (L2: 0–1), and 18–30% of L1 and L3 cells have no traced photoreceptor input at all.
Eyes that start at the photoreceptors (option (a) or (c)) would have to synthesise the missing
R1–R6 drive, or inject at L1–L3 instead: one lamina cartridge per facet, which is complete.

There are no ocellar photoreceptors in the dataset, only the ocellar interneurons (OCG).

---

## 2. What reaches the Kenyon cells

The model has 4,064 KCs: g-m 1,342; ab-s 657; ab-m 536; ab-c 488; a'b'-ap2 291; g-d 206;
a'b'-m 205; a'b'-ap1 199; ab-p 129; 13 other or untyped.

### Direct synapses onto KCs (full dataset, % of each subtype's total input)

About 51–58% of every subtype's input is KC→KC (axo-axonal in the lobes). The rest is APL, DAN,
DPM, PNs and so on.

| KC type | ALPN | VPN | other visual-named cb neurons | ORN/GRN/JO/thermo/hygro direct |
|---|---|---|---|---|
| KCg-m | 20.9% (180,266 syn) | 0.05% (397) | — | 0 |
| KCab-s/m/c | 20.8–22.2% | ≤0.003% | — | 0 |
| KCa'b'-ap1/ap2/m | 13.2–14.0% | ≤0.004% | — | 0 |
| **KCg-d** | **0.67%** (788) | **6.84% (8,041)** | LoVP97 654, PLP095 1,352 (cb_intrinsic) | 0 |
| **KCab-p** | **1.46%** (1,052) | **2.36% (1,695)** | — | 0 |

No primary sensory neuron of any modality synapses on a KC. **Vision reaches KCs one hop from
the VPNs.** In total, VPNs make 11,094 synapses onto KCs, 9,736 of them onto KCg-d plus KCab-p.
The main types, in synapses:

| VPN type | Synapses | Note |
|---|---|---|
| aMe12 | 1,785 | also takes 3,072 synapses from R7/R8 |
| aMe26 | 1,127 | |
| MeVP41 | 1,013 | |
| LoVP42 | 923 | |
| MeVP36 | 659 | |
| aMe20 | 518 | |
| LoVP71 | 424 | |

There are 52 VPN types (123 cells) with at least 5 synapses onto KCs. This is consistent with the
visual Kenyon cells described in flies: γd and αβp (Vogt et al. 2016 eLife; Li et al. 2020 eLife).
**So yes, a real visual → KC path exists in this male brain, and it is confined to about 335 KCs
(8%).**

It is also short from the retina. R7/R8 → aMe12 → KC is only 2 hops, but in weight it is
small: 0.23% of KCg-d input is attributable to traced photoreceptors at 2 hops, and 0.47% by 4
hops. The rest of vision reaches the calyx through the medulla and lobula, 3–4 hops from the
retina.

### Multi-hop attribution

Method: for each hop, the share of a neuron's input that is attributable to a sense. At each step
the input fraction is `syn / all synapses onto the target`, and the fractions multiply along a
path. Numbers are % of total KC input, KC→KC included.

| KC type | ORN @2 hops | thermo+hygro @2 | gustatory @2 | JO @3 | VPN @1 / @2 |
|---|---|---|---|---|---|
| KCg-m | 8.55 | 0.26 | 0.005 | 0.008 | 0.06 / 0.25 |
| KCab-s/m/c | 9.2–9.5 | 0.09–0.10 | ≤0.003 | ≤0.007 | ≈0 / 0.06–0.09 |
| KCa'b'-ap1 | 3.72 | **0.77** | 0.006 | 0.007 | ≈0 / 0.05 |
| KCa'b'-ap2/m | 6.0–6.3 | 0.02–0.10 | 0.001 | ≤0.004 | ≈0 / 0.02–0.05 |
| KCg-d | 0.07 | 0.08 | 0.02 | 0.003 | **6.82 / 6.31** |
| KCab-p | 0.54 | 0.02 | 0.001 | 0.002 | **2.32 / 3.91** |

The same thing as "equivalent synapses" onto all KCs at 2 hops:

| Source | Equivalent synapses |
|---|---|
| ORN | 164,822 |
| VPN | 13,395 |
| hygro | 2,193 |
| thermo | 2,056 |
| photoreceptor | 397 |
| gustatory | 94 |
| other mechano/proprio | 16 |
| JO | 2 |

### Mixed KCs are almost absent

Only 42 of 4,064 KCs take at least 5 synapses from both ALPNs and VPNs, and 24 take at least 10
from both. 300 KCs take visual input (at least 5 VPN synapses): KCg-d 198, KCab-p 85, KCg-m 10.

### Which senses can carry a "question" into the mushroom body

- **Olfaction** reaches every KC class, but it is already the thing being judged.
- **Vision** reaches only KCg-d and KCab-p, and is 6–8% of their input. It is the only second sense
  with a real KC route of any size.
- **Thermo/hygro** reaches the calyx through the VP PNs, mostly KCa'b'-ap1 (0.77% of its input).
  It is weak but real, and the 91 TRN/HRN are already in the model, unused.
- **Taste, mechanosensation and JO** have no meaningful route to KCs (below 0.03%). Taste reaches
  the MB through the DANs: PAM and PPL1 about 0.06–0.15% at 2–3 hops. That is the fly's
  reinforcement channel, not a context channel.

**Consequence for "question and thing on different senses":** if the question comes through the
eyes and the thing through the nose, they land on almost disjoint KC populations. They first meet
at the MBONs. The γ-lobe MBONs MBON01, 05, 09, 11 and 32 each take several thousand synapses from
both KCg-m and KCg-d; MBON27, 33 and 35 are KCg-d-dominated. In a readout that is linear in the KCs
(KC→MBON weights), the MB therefore gives only an additive `f(thing) + g(question)`, with no
KC-level conjunction.

Any question × thing interaction would have to come from downstream nonlinearity: MBON→MBON,
MBON→DAN feedback, and the LH or other convergence zones. This is a structural limit, not a
tuning issue, and the architecture doc should state it. The alternative is to put the question on
a channel that modulates rather than drives: DAN or MBON-feedback gating, or a state input.
Whether that is faithful needs its own anatomy check.

---

## 3. Descending neurons: count, candidates, paths from the MB and the LH

The model holds 1,314 `descending_neuron` (481 types, 4 untyped), plus 12 `sensory_descending`
and 4 `efferent_descending`.

`readout_populations.yaml` names 37 types. All of them are in the model, and the DNs among them
are 2–16 cells each. The engage set is DNp09, DNa01–04, DNb01, 02, 05 and 06; the leave set is
DNp01–04, 06, 10 and 11, and MDN. These are literature-named; see the yaml for sources (Bidaye et
al. 2014 MDN, Bidaye et al. 2020 P9/DNp09, von Reyn et al. 2014 for the giant fiber DNp01, Namiki
et al. 2018).

Sanity checks on the data:
- DNp01 takes 11,252 synapses (26% of its input) from LC4 and LPLC2, the known looming inputs.
- DNp09 takes 3,671 (23%) from VPNs, mostly LC9 (1,977) and LC31a (763).

### Hops (unweighted shortest path)

| From → to all 1,314 DNs | Full model: 1 / 2 / 3 hops | Cut brain (≥5 syn): 1 / 2 / 3 / 4 hops / unreachable |
|---|---|---|
| any MBON | 170 / 1,103 / 41 | 88 / 759 / 457 / 9 / 1 |
| avoid MBONs | 50 / 1,039 / 225 | 16 / 562 / 693 / 42 / 1 |
| LH-named neurons (LHAV/AD/PV/PD/CENT) | 343 / 965 / 6 | 191 / 950 / 172 / 1 |

Candidate DNs in the cut brain:
- MBON → DNa02, DNa03, DNb05, DNp42, DNa13 and MDN: 1 hop.
- MBON → DNp09, DNa01, DNp01–04 and the rest of the set: 2 hops; DNb06: 3 hops.
- LH → DNp01–04, 06, 10, 11, DNp09 and DNp42: 1 hop.

### Strength

Direct MBON→DN synapses total 4,347, over 409 edges, of which 151 have at least 5 synapses. The
largest:

| Edge | Synapses | Sign |
|---|---|---|
| MBON33 → DNg104 | 362 | + |
| MBON20 → DNp42 | 334 | − |
| MBON31 → DNa03 | 251 | − |
| MBON27 → DNa03 | 203 | + |
| MBON32 → DNa13 | 140 | − |
| MBON31 → DNa02 | 133 | − |
| MBON32 → MDN | 46 | − |
| MBON27 → MDN | 33 | + |

Share of input from the MB, cut brain, summed over 1–3 hops, as % of the DN's input:
- **median DN type: 0.056%**
- DNa02: 1.4%; MDN: 2.1%; DNa03: 4.7%
- **DNp09: about 0.1%; DNp01: about 0.04%**

The DN types the MB reaches most strongly are DNp52 (6.4%), DNg104 (5.1%), DNge151 (4.9%), DNa03
(4.7%), DNge150 (3.8%), DNp42 (2.3%), DNge152, MDN, DNa13 and DNpe023. Most of these are not in
the readout yaml. The LH gives DNs 5–10× more per hop than the MB does: 0.23% of mean DN input at
1 hop against 0.05% from the MBONs.

With signs, the approach-group MBONs (which include GABAergic MBON11, 20, 31, 32 and
glutamatergic 25, 30, 34) have a net *negative* effect on MDN (−0.18% at 1 hop, −0.50% at 2
hops) and on DNa02 and DNa03. The avoid group has a net positive effect on MDN (+0.10% and
+0.15%). The direction is plausible (avoid → back up), but it depends entirely on glutamate = −1
(§5).

**MB → DN is diffuse and weak in the wiring.** A DN read has to be chosen from where MB output
actually lands, not only from the literature locomotor list. Otherwise the answer will be carried
by the LH and the direct visual paths: VPN → DNp09 is 23% of its input, far above the MB's
share.

### Timing: measured, not only estimated

Hop count gives an estimate. From the ORNs it is ORN → PN → KC → MBON → (1–2) → DN, which is
4–5 synaptic hops. In a first-order cascade with τ = 20 ms (4 steps of 5 ms) the mean delay is
about 4 steps per hop, so about 16–20 steps to the DN mean, and the 90% point later.

I also ran the trained A5 checkpoint (`runs/gate-a5/real-type-s1.pt`, cut brain) for 160 steps on
128 test items (sweet and dangerous). `STEPS` was monkeypatched in the script; the code is
unchanged. I measured the stimulus-dependent spread per group:

| Group | 50% of final | 90% of final | Spread at step 40 / step 160 |
|---|---|---|---|
| ORN | step 3 | 5 | 0.97 |
| ALPN | 6 | 13 | 0.96 |
| KC | 8 | 10 | 0.90 |
| MBON (all) | 9 | 24 | 0.95 |
| MBON approach − avoid (the read) | 11 | 19 | 1.09 |
| DN (all) | 19 | 30 | 0.99 |

Trained medians of τ: KC 23 ms, MBON 12.8 ms, DN 20 ms.

- **40 steps is just enough for the DN population to plateau when the input is the nose or the
  stand-in eyes.** The 90% point is at step 30, and the read window is steps 33–40.
- The MBON read is still transient at step 40. It peaks at about step 32 (0.134), is 0.105 at step
  39, and swings between 0.08 and 0.10 up to step 160.
- **Real optic lobes add about 3–4 hops before the VPNs** (R → L1–L3 → Mi/Tm → LC/LT or Tm → VPN).
  That is about +12–16 steps, so the DN 90% point moves to roughly steps 42–50. **40 steps would
  then be too short; plan for 64–80 steps**, and measure again.

**The trained A5 network has most candidate DNs silent.** At step 40, only 304 of 1,314 DNs are
active (rate above 1e-3 for any stimulus). The behaviour groups are almost off:

| Group | Active cells |
|---|---|
| approach | 3 / 20 |
| flee | 3 / 18 |
| eat | 1 / 13 |
| groom | 29 / 76 |

DNp09, DNa02, DNa03, MDN, DNp52 and DNg104 have exactly zero rate for every stimulus. DNp42 is
active. A DN read will need the DN thresholds and gains set up the way the A5 init sets up the
MBONs. Today they sit below threshold.

---

## 4. Optic lobes: size and cost

**Per side, traced:**

| | Left | Right |
|---|---|---|
| ol_intrinsic | 44,599 | 44,789 |
| visual_centrifugal | 283 | 280 |
| VPN | 4,589 | 4,612 |
| photoreceptors | 1,733 | 2,374 |

There are about 890 columns per eye.

**Synapses onto ol_intrinsic:** 17.76 M on the left (4.55 M edges) and 21.19 M on the right (5.16
M edges). The right lobe is more completely reconstructed. Edges that touch optic-lobe or VPN
bodies total 13.4 M, with 55.4 M synapses.

VPNs get only 20.6% (median) of their input from neurons in the current model. In synapses, 7.71
M of their 10.83 M inputs come from ol_intrinsic, 2.33 M from model neurons and 36 k from traced
photoreceptors.

**Model growth** from adding ol_intrinsic, visual_centrifugal and ol_sensory:

| | Neurons | Edges | Synapses | Edges ≥5 (+ all KC→MBON) |
|---|---|---|---|---|
| Current v2 | 50,140 | 9.77 M | 52.2 M | **2.39 M** (the trained cut) |
| + optic lobes | 144,209 (×2.88) | 21.87 M (×2.24) | 101.1 M | **5.22 M (×2.19)** |

**Rough CPU cost:**
- Per step, the sparse matvec scales with edges and the state update with neurons, so expect about
  **2.2–2.9×** per step.
- If the run also grows from 40 to 64–80 steps, the total is **about 3.5–5.8× the current cut
  brain per decision**.
- The fp16 trace memory scales with neurons × steps, so it grows about 4.6–5.8×.

These figures are estimates from the edge counts. They are not timed on the Kimsufi and need a
benchmark.

---

## 5. Things that contradict anatomy or look like mistakes

1. **The stand-in eyes drive ocellar interneurons.** The 20 OCG cells are in `visual_projection`
   and get picture channels like any VPN. The ocelli are not the compound eye; these should be
   excluded from `eyes`.
2. **The eyes drive all 346 VPN types evenly (3 random channels per type).** Only 52 types reach
   KCs, while the looming and object types (LC4, LPLC2, LC9, LC31) synapse directly on the escape
   and pursuit DNs (DNp01 26% of input, DNp09 23%). A picture encoder on the VPNs therefore has a
   2-hop route to the DNs that bypasses the MB. Once the answer is read from the DNs, this becomes
   a decider-shaped shortcut: the encoder can steer the DNs through innate visual circuits. It
   needs an explicit control, such as a DN read with the MB lesioned, or VPN→DN edges masked as an
   arm.
3. **Some visual PN-type cells sit in `cb_intrinsic`** and are neither driven nor excluded as
   visual: LoVP37, 81, 91, 94, 97, 105; aMe13, 15, 22, 23, 24; LCNOp; 30 cells in all. LoVP97
   gives KCg-d 654 synapses. With real optic lobes they would be driven naturally; with the
   stand-in they get no picture input.
4. **91 thermo/hygro receptor neurons are in the model but unused, and are the only other sense
   with a calyx route.** The nose selects `ORN_*` only, which is correct for smell, but it means
   the VP glomeruli are silent.
5. **Innate glomeruli.** The 7 excluded glomeruli (DA1, DL3, VA1d, VA1v, VL2a, DA2, V) are all
   present in the data, and the nose uses all of the remaining 46. The remaining set still
   includes glomeruli with narrow innate roles; for example DC4 and DP1m carry Ir64a acid sensing
   (Ai et al. 2010). The exclusion list is therefore "pheromone, geosmin and CO2", not "all
   innate". This is minor, but the doc should say it.
6. **Glutamate = −1 everywhere.** This is Shiu et al.'s convention. Glutamate is inhibitory
   through GluClα in parts of the fly CNS (Liu & Wilson 2013), but that is not established for
   every MBON target.
   - It fixes the sign of the 7 glutamatergic avoid MBONs (01–07) and of MBON25, 30 and 34 onto
     their targets.
   - The MBON sign assignments themselves agree with the literature: 01–07 glutamate, 09–11 GABA,
     12–19 ACh, all as consensus calls in this dataset.
   - Today this does not matter, because the read is taken at the MBONs. With a DN read, the
     approach/avoid meaning downstream depends on it. Flag it as a stated assumption; a test would
     compare with glutamate = +1 on MBON outputs.
7. **The DAN→KC edges are zeroed** (PAM08 → KCg-d alone is 4,733 synapses; APL → KCg-m is 79,151
   and is kept as inhibition). This is documented in `ratebrain2` and defensible, but DPM is also
   set to −1 by override. Both choices are stated; I found no error.
8. **DN input deficit is small:** a median of 91.6% of DN input comes from model neurons, and 1–3%
   from the VNC. For DNp01 only about 81% does, so its drive is under-represented by about a
   fifth.
9. **The read is not at steady state (§3), and the literature DN set is silent in the trained
   brain (§3).** Neither is an anatomy error, but both block the planned DN read as the model is
   now.
10. **Two numbers to state on the model card:**
    - In this male, R1–R6 are 9% and 17% reconstructed (§1).
    - The left optic lobe has about 16% fewer synapses than the right, and fewer than half the
      right's R1–R6. Any real-eye pathway will be asymmetric through reconstruction, not biology.

## Uncertainty

- "LH neurons" means types prefixed LHAV, LHAD, LHPV, LHPD or LHCENT. That includes some LH
  input and local neurons, not only output neurons.
- Multi-hop attribution uses unsigned products of input fractions. It measures route strength,
  not activity.
- The timing numbers come from one trained seed (real-type-s1) and one input set.
- The CPU multipliers come from edge counts, not a benchmark.
- The expected ~5,300 R1–R6 per eye assumes 6 per column × about 890 columns.
