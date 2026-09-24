# bosco v2

A fruit fly's brain as a decision model. The MaleCNS v1.0 connectome (HHMI
Janelia / Google Research / Cambridge, Cell 2026), the same LIF kernel and
mushroom-body plasticity that ran v1, now pointed at one question: **is a real
brain's wiring diagram a useful prior for learning decisions?** Typed decisions
with a probability, no text, in the shape of a System One model (Jev).

v1 — the fly as a Bluesky account — ran 2026-09-13 to 2026-09-22 and is
archived whole on `main` (tag `end-v1`, `docs/CLOSING-v1.md` there). This
branch starts from an empty tree and takes only what stood on its own: the
kernel, the model build, the plasticity rule, the phase 0–3 gates and their
reports. Nothing here is a fix to v1; v1 is answered.

## What is being asked

- **B, first.** With a fixed encoder and identical sparse rewards, does the
  real mushroom-body wiring under the biological three-factor rule learn
  decisions better than a degree-preserving shuffle of itself, and than a
  random sparse projection of the same size (the fly-hash baseline)? Cheap,
  reuses v1 code unchanged, days. Pre-registered in `docs/GATE-B.md`.
- **A, if B earns it or fails informatively.** Fix the graph and the signs,
  train the magnitudes and neuron parameters by gradient on decision tasks,
  and compare with shuffled and free wiring of the same size on sample
  efficiency, calibration and forgetting (cf. Lappalainen et al. 2024 for the
  visual system). Pre-registration to be written after B.
- **Deliverable:** a writeup with numbers, and the decision service.
  Amended 2026-09-24 (Nick, after A2): the service no longer waits on the
  connectome beating the shuffle. The controls still run, and their numbers
  are published beside his (`docs/GATE-A2b.md`).

## Non-negotiables

- **A model may perceive; it never decides and never writes.** One frozen
  embedding model turns the state into a vector, and that is the whole of
  ML in the loop for B. No LLM anywhere. Nothing generates text, ever.
- **Every gate is pre-registered** in `docs/` before it runs: arms, tasks,
  reward protocol, metrics, decision rule, seeds. The fly circuit is never
  tuned on the task; the one free scale (embedding → PN drive) is set once,
  on unlabelled data, so KC activity lands in the band v1 recorded (2–7%).
- **Controls run first, not later.** v1's dunce never ran; here the shuffle
  and the random projection are arms of the same script as the real wiring,
  and a result without them is not a result.
- **Deterministic and replayable.** Seeded everything; the kernel is fixed
  point or strictly ordered float; a run is reproducible from its config and
  seed. Parallelism across flies, never within one.
- **Report what happened.** A gate that fails is written up the same as one
  that passes. No metric is chosen after seeing the numbers.
- **Not a character, not on a feed.** "Ask him on Bluesky" is a possible
  front end for a decider that exists; it is not this project.

## Model (unchanged from v1 where it stands)

- LIF neurons, parameters from Shiu et al. 2024; weights = synapse count ×
  sign from neurotransmitter prediction; central brain, optic lobes and VNC
  dropped (`docs/phase0-coverage.md`, `docs/phase2-*.md`). Time step 0.1 ms.
- Mushroom body: three-factor rule, KC activity coincident with DAN firing
  depresses KC→MBON synapses in that compartment; two timescales; credit
  confinement as at `freeze-v2` (`config/plasticity_v1.yaml`,
  `docs/plasticity-v3.md`). Extinction and homeostasis in
  `config/plasticity_v2.yaml`, disabled, with the caveat in that doc.
- Reproductions that stand: Shiu's sugar → proboscis (`docs/phase1-shiu-gate.md`),
  KC sparseness (`docs/phase2-kc-sparseness.md`), learn/forget
  (`docs/phase3-plasticity-gate.md`).

## Layout

```
kernel/        C99 LIF kernel (make -C kernel)
src/bosco/     paths, data, kernel, model, populations, sim, plasticity, brain
config/        model_v1, mb_compartments, plasticity_v1/v2, readout_populations
scripts/       phase1/2/3 gates, make_dunce, cpu_determinism  (+ gate_b.py to come)
docs/          v1 gate reports kept as provenance; PLAN.md; GATE-B.md
ops/fetch_data.sh   the three MaleCNS feathers + FlyWire annotations
```

Python 3.12, `uv`, `ruff`, `pytest`. `uv run pytest tests/test_kernel.py` needs
only a C compiler; everything else needs `ops/fetch_data.sh` (~1.1 GB).

## Things Claude Code must not do

- Put a model anywhere but the encoder. No LLM, no classifier deciding.
- Tune the fly circuit, the projection, or the rule on task labels or rewards.
- Run a gate that is not written down first, or change its decision rule after.
- Drop a control arm to save time.
- Touch `main`: it is the v1 archive.
