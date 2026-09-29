# Brain requirements (DRAFT for Nick's sign-off, 2026-09-28; Q1 and Q2 answered)

Nick: "set requirements for brains: small, performant, still resembling maleCNS for ability to watch/replay
thinking. Optimized for ease of training and malleability, but also accuracy."

Once signed, these become the checks every candidate brain is scored against, before any specialist work (CLAUDE.md,
"spec before build"). BRAIN-SPEC's L and T checks become part of them. The numbers marked *proposed* are mine and
wait on Nick. Where a requirement touches a locked decision, the decision is named, and the conflict goes to Nick
rather than being settled here.

## R1. Resembles MaleCNS: you can watch and replay him think
- **R1.1 Real cells.** Every unit is a real MaleCNS neuron or a real MaleCNS cell type, and is named by it in the
  trace. The playback shows named cells and types only.
  - *Open:* "neuron" (as now) or "cell type" (see Q1).
- **R1.2 Real wiring.** The connectivity and the signs come from MaleCNS (with the documented fixes). They are
  never trained.
- **R1.3 The decision travels through real circuits.**
  - The item's signal reaches the answer through the fly's pathways, and the stages on that path are driven by
    their inputs, not held up by tuned thresholds.
  - *Proposed:* on the path smell → PN → KC → MBON → answer DNs, each stage's median bias share is ≤ 0.5 and its
    item modulation (cell s.d. ÷ mean) is ≥ 0.2.
  - Today the MBONs sit at 0.05–0.06 and the DNs at about 0; they fail.
- **R1.4 The real wiring matters.** It carries the learned memory to the answer ≥ 2× better than the layered and
  hash shuffles (T7, measured 7.1). It must also either match or beat them on accuracy, or the cards say plainly
  that it ties.
- **R1.5 Replayable.** The trace of any served answer can be regenerated bit for bit from its inputs (decision 7).
- **R1.6 Only fast synapses are fast drive.** Transmitters that act through metabotropic receptors are not fast
  excitation: dopamine, octopamine, serotonin, and KC→KC axo-axonal acetylcholine (mAChR-B, Manoim et al. 2022).
  A neuron's input total counts only the synapses that drive it. An unknown transmitter is never defaulted to
  excitatory (deep dive, #7–9).
- **R1.7 Each stage sits at the fly's operating point.**
  - **Below threshold is silent:** a unit's rate at threshold is ≤ 0.002.
  - **KCs are sparse and burst:** 2–10% active per sniff, and the median rate of an active KC is *proposed*
    ≥ 0.2 of max (Turner 2008).
  - **MBONs are KC-driven:** silencing KC→MBON cuts the MBONs' odour-evoked response by ≥ 80% (Hige et al. 2015).
    This is a new label-free check, L7.
  - No single set point is imposed on every type.

## R2. Small
- **R2.1** The 50,140 neurons, one unit each (decision 39; Nick, Q1): no more, and no fewer.
- **R2.2** A served brain with its memories fits in *proposed* ≤ 200 MB of RAM. One specialist's own memory takes
  *proposed* ≤ 1 MB (today the KC→MBON memory is 61,210 values, 245 KB).

## R3. Performant
- **R3.1** One yes/no answer (encoder excluded) takes *proposed* ≤ 0.15 s on 4 CPU threads (today 0.10 s on the
  Mac). The gate is the Kimsufi (decision 7).
- **R3.2** Bit-exact: the same answer on repeat, at any thread count, and between scoring and serving (holds today
  via `RateBrain3.answer`). The encoder is bit-exact at its pinned thread count.
- **R3.3** Several questions per request (decision 37): *proposed* 10 yes/no answers in ≤ 1 s.

## R4. Easy to train
- **R4.1 Converges.** A specialist trains in *proposed* ≤ 20 minutes on the 3080, with no tuning per task: one
  recipe for every yes/no task.
- **R4.2 Stable.** Across 5 seeds, validation balanced accuracy has *proposed* s.d. ≤ 0.01, and the answers
  disagree on ≤ 10% of items (v2 measured 18.8%).
- **R4.3 The memory has leverage.** A change to the memory can move the answer across its whole range: the trained
  read s.d. is ≥ 0.01 (amended L5), with no read scale above *proposed* k ≤ 20 (today 8.5; layered 60–74).
- **R4.5 The learning checks do not depend on the read's offset.** T2 and T3 are measured by ranking (AUC), not by
  thresholded answers, since c absorbs the memory's shift (deep dive, #10).
- **R4.4 The fly's own learning rule works** (depression at KC→MBON, Hige et al. 2015), or the departure is stated
  on the card with its reason. Amendment 3 failed this.

## R5. Malleable
- **R5.1** One brain, many questions. A new specialist is a new memory (KC→MBON, plus k and c) on the shared brain.
  Nothing else changes (decision 27, one memory per option).
- **R5.2** Adding a sense (taste, sight) or a new specialist does not retrain or break the shipped ones (decision
  38: an untrained sense gets no input).
- **R5.3** Teach-your-own by the fly's local rule (decision 16) reaches *proposed* ≥ 90% of gradient training's
  accuracy on the same task.

## R6. Accurate
- **R6.1** The brain reaches the ceiling of what it smells: balanced accuracy ≥ a plain logistic on the same smell
  minus 0.02 (T1 is −0.05; *proposed* tightening).
- **R6.2** The nose is not the bottleneck: logistic on the smell ≥ logistic on the full embedding minus 0.02.
  Today it is 0.735 against 0.786, which fails; timed sniffs (decision 39.1) are the free fix.
- **R6.3** Calibrated: ECE ≤ 0.10 (decisions 23, 34).
- **R6.4** The production bar stays 0.80 on the sealed test (decision 23), per specialist, later.

## R7. Honest (unchanged, from CLAUDE.md)
- **Controls in the same script:** the layered and hash shuffles and a silenced mushroom body.
- **Pre-registration and sealed tests,** with sampling error in every bar (BRAIN-SPEC amendment 2).
- **Claims** as in CLAUDE.md.

## Answered by Nick (2026-09-28)
- **Q1. Size:** keep the 50,140 neurons (decision 39).
- **Q2. Priority:** resemblance first, then speed and determinism, then accuracy, then ease of training.

## Open
- **Q3.** Are the proposed numbers right?
