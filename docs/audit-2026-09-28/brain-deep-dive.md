# Brain deep dive, 2026-09-28: every flaw found, ranked

Nick: "we've had like 6 of this. do another deep dive and make sure this is the only one."

**Answer: it is not the only one.** There are 14 flaws. They come from three root causes, plus a read that needs
review and a nose that costs accuracy.

**Method.** Three independent strands, all label-free, with no training and no GPU:
1. **Measurements** (`scripts/brain_audit.py`, `runs/brain-audit/audit.json`): every stage, silencing, and drive
   decomposition.
2. **An adversarial code review.**
3. **A literature check** (citations checked against PubMed / Europe PMC).

Every high-severity item was re-measured by me. Evidence marked "review" or "lit" was not re-measured.

## Root cause 1: the signals are too weak past the first relays (the operating regime)

| # | flaw | evidence | fly biology | severity |
|---|---|---|---|---|
| 1 | **The unit leaks below threshold.** unit_fn(0) = 0.0139 with BETA = 50, so "silent" cells still send output. | 59% of all KC output comes from KCs below the 0.01 "active" line (measured). | KCs fire about 0.1 Hz spontaneously, about zero (Turner 2008). | high |
| 2 | **Active KCs barely fire.** | The median rate of an active KC is 0.015 of max (measured). | A responding KC fires a burst of 2–5 spikes from about zero, nearly binary (Turner 2008). Compensation raises response *magnitude* (Apostolopoulou & Lin 2020). | high |
| 3 | **The MBONs are not driven by the KCs.** | Silencing all KC→approach-MBON input moves them 0.038 → 0.034. MBON item modulation is 0.05 (measured). | Depressing KC→MBON alone removes 80–90% of the MBON odour response (Hige et al. 2015). | high |
| 4 | **Thresholds, not inputs, set almost every neuron's rate.** One set point (rate 0.05) for every type, reached by bias. | For 70% of neurons, \|bias\| is larger than the synaptic drive (review). Median bias share is 0.79–0.86 at the DN read and 0.9 at the DANs. Past the PNs, LH and KCs, item modulation is ≤ 0.06 everywhere (measured). | Rates differ more than 100× between types. There is no evidence of one shared set point (lit). | high |
| 5 | **The KCs' input is diluted by KC→KC in the denominator.** | 53% of a KC's input total is KC→KC and only 18% is PN (measured). The cut drops 82% of KC→KC, but those synapses stay in the total, so PN drive is squeezed about 5× (review). | — | medium |
| 6 | **The homeostatic start is unfinished.** | The residual error is still 17% of the target at the last of its 40 iterations. 898 unrelated "untyped:unclassified" cells share one unit, left untuned because 20 of them are sensory (review). | — | medium |

## Root cause 2: three kinds of synapse run as fast excitation, and they are not

| # | flaw | evidence | fly biology | severity |
|---|---|---|---|---|
| 7 | **Dopamine, octopamine and serotonin act as fast excitation everywhere** except DAN→KC. | DAN input onto the read MBONs (0.0075) is larger than KC input (0.0063) (measured). 757 neurons and 1.28M synapses (review). | These act through metabotropic receptors: slow and modulatory (Takemura 2017; Cohn 2015). Shiu et al. 2024 use the same convention, so it is the field's shortcut, but it is wrong for a brain whose answer rides on MBONs. | high |
| 8 | **KC→KC axo-axonal synapses act as excitatory drive onto KCs.** | They supply about 20% of KC activity (silencing them: 5.0% → 4.0% active) (measured). | They act locally in the axon through mAChR-B and *suppress* their neighbours (Manoim et al. 2022). | medium |
| 9 | **An unknown transmitter defaults to excitatory.** | 876 neurons (699k synapses). The antennal-lobe LN fix then deletes 27 likely-GABAergic LNs that got +1 by default (review). | — | medium |

## Root cause 3: learning and how it is measured

| # | flaw | evidence | fly biology | severity |
|---|---|---|---|---|
| 10 | **T2 and T3 are confounded by the read offset c.** | c absorbs the memory's shift in the mean read. Re-scored with the flipped copy's own k and c, **94–95% of answers flip on every seed** (as scored: 75–86%). By ranking, the flipped memory reverses the items (AUC 0.21 against 0.79 trained) (measured). The reset arm's "0.5 accuracy" is an offset lottery; its AUC is 0.609 on every seed. **T3's failure was a flaw in the measure.** The reset arm's AUC shows the innate path carries about 37% of the ranking margin (T2 by rank would still fail). | — | high (spec) |
| 11 | **Potentiation is unbounded.** | Training potentiated KC→MBON about 7×, and that loosened the KC code through APL (measured). | Depression dominates (−55 to −90%). The largest potentiation measured is +63% (Hige 2015; Yamada et al. 2024). | medium |
| 12 | **Training ran from a stale rest** (re-settled every 20 batches). | Fixed in the depress variant (every batch). | — | low |

## The read, and the nose

| # | flaw | evidence | severity |
|---|---|---|---|
| 13 | **The approach DN group is not a locomotor "approach".** 6 of its 12 types are octopaminergic SEZ DNs (running as fast excitation, see #7), and the others are song and courtship DNs (pIP10, pMP2, DNp13) (review). The avoid group is plausible (MDN, DNp42, DNa13). The MBON grouping (PAM → avoid) misfiles the GABAergic MBONs in PAM compartments (γ3, γ3β′1, β′1) and the atypical MBONs (Aso et al. 2014b; Li et al. 2020) (lit). | medium |
| 14 | **The nose costs about 5 points.** Logistic on the 46 channels 0.735, on the 768-d embedding 0.786 (measured). Timed sniffs (decision 39.1) are the free fix. | medium |

## Checked and sound
- `freeze`, `settle`, `answer`, the checkpointed read steps and the tau clamp.
- Determinism (bit-exact at any thread count).
- The cut: PN→KC is 1.3% dropped, APL↔KC 0.5%.
- APL works as in the fly: silencing it gives 16% of KCs active and a Jaccard of 0.46.
- The resting state converges.
- Neurotransmitter signs follow the field's convention (Eckstein et al. 2024, 94% accurate; Shiu et al. 2024).

## What this means
The brain's shape is right, and the size of its signals is wrong. Four things together leave the item's signal
dying within a few synapses of the nose:
- a leaky unit;
- KCs that barely fire;
- one threshold-set operating point for every type;
- input totals diluted by synapses that are not fast drive.

Modulators running as fast excitation then mask what is left. Every earlier failure today is a symptom of this:
- L5, the flat untrained read;
- the 7× potentiation that learning needed;
- the loosened KC code;
- the innate path's share;
- depression-only learning with nothing to remove.

**The fix is one package, not another single patch** (a proposal, for Nick). Each part points at the biology above.
It is checked against `docs/BRAIN-REQUIREMENTS.md` before any training.
