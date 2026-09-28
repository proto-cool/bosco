# bosco v3

A computer model of a real male fruit fly's central brain, answering typed questions (yes/no, choose, rate) with
calibrated probabilities, so that you can watch his real cell types decide. The front end is Ask Bosco, a separate
repo (`~/projects/arclight/halteres`) that you do not edit.

**Start here: `docs/HANDOFF-v3.md`.** v2 ended on 2026-09-28 (tag `end-v2`): the audit found the fly was not
doing the deciding, and every result and trained model was deleted. v3 is an empty tree that carries only what the
audit did not fault.

## What the model is (say exactly this, nothing grander)
- **Connectome:** MaleCNS v1.0 (Janelia FlyEM et al., CC BY 4.0). **50,140 of about 165,000 traced neurons**:
  the central brain plus the visual projection neurons. The optic lobes and the nerve cord are left out
  (decision 39: no added neurons, speed rules). Edges under 5 synapses are dropped; every KC→MBON synapse is kept.
- **Dynamics:** a rate model (`src/bosco/ratebrain3.py`): graded softplus-tanh unit, 5 ms steps, 80 steps per
  sniff. Weights are synapse count × sign from the neurotransmitter prediction, normalised by input. It is **not**
  a LIF model at 0.1 ms; the C kernel in `kernel/` is v1-era and unused by the brain.
- **Senses:**
  - smell: 46 glomeruli, fed by a frozen text encoder (nomic-embed-text-v1.5, pinned), the translator;
  - taste: 355 gustatory neurons; sight: 9,201 visual projection neurons. Both are in the model but not yet
    fed (see the handoff).
- **Answer:** read from descending-neuron approach and avoid groups (`v1.dn_groups`). The audit flags that
  steering neurons are counted as avoid, and that this needs fixing.
- **Learning:** by gradient descent on task labels, allowed only on the KC→MBON synapses (the main arm) and on
  per-cell-type gain, threshold and time constant (a comparison arm). The graph and the signs are never trained.

## How to work (the reason v2 failed, so these come first)
1. **Every locked decision is a requirement.** Check the code against `docs/DECISIONS-2026-09-24.md`,
   `docs/DECISIONS-2026-09-25.md` and the handoff's requirement list before building anything. Never defer,
   narrow, reinterpret or drop one on your own; if two conflict, stop and ask Nick.
2. **Tell Nick exactly what you will run, and why, before running it.** Wait for his go before anything that
   trains, scores, runs longer than about a minute, or touches the 3080. Say what is running and when it stops.
3. **Don't re-run what is already known.** No "baseline" runs of configurations already measured to be broken.
   Read the audits first.
4. **The brain first.** No specialist, gate, fleet or service work until the brain passes `docs/BRAIN-SPEC.md`.
5. **Spec before build.** Requirements become checkable criteria that Nick signs off; conformance tests gate
   training.
6. **Report what happened, failures first,** with numbers. Never report anything unverified as done. That
   includes "nothing is running": check every machine and show the output.

## Non-negotiables
- A model may perceive (the encoders); it never decides and never writes. No LLM anywhere; nothing generates
  text.
- **Every gate is pre-registered** in `docs/` before it runs: arms, tasks, data, metrics, the decision rule
  (including sampling error) and seeds. It opens with an anatomy check and a leakage check (exact and near).
  Sealed tests are scored once.
- **Controls run in the same script** as the real wiring: layered and hash shuffles, and a silenced mushroom body.
  Their numbers are published.
- **Deterministic and replayable.** Published numbers are the served numbers: CPU, the same code path, and the
  same encoder environment.
- **Clean data** (`docs/clean-data.md` and the gate-4 data docs):
  - the licence must allow commercial use and redistribution of the weights;
  - provenance is recorded and checksummed;
  - no NC data, no LLM-generated text, no Twitter/X;
  - text from live sources predates 2022-11-01;
  - personal data is scrubbed.
- **Honest claims.** No "full brain" and no "all 50,000 mapped neurons". The encoder's training data is
  disclosed. Controls are shown beside the results.
- **Branches:** do not touch `main` (the v1 archive) or `v2` (the v2 archive, tag `end-v2`).

## Machines
- **The Mac:** CPU scoring, and the service.
- **magnetar** (RTX 3080, Linux): `ssh magnetar`, repo at `~/bosco` (rsync, not git), `~/.local/bin/uv`.
  Start long jobs with `setsid nohup … &`.

## Layout
```
src/bosco/   paths, data, model, model2 (connectome build), populations, senses, controls, ratebrain2/3, v1 (build,
             homeostatic start, DN groups), encoders (pins), device, kernel (v1 LIF binding)
scripts/     brain_check.py (label-free brain checks, a draft never run), brain_render.py
reference/   v2-data-builders/: v2's data fetch and build scripts, kept for their provenance logic (not importable)
docs/        HANDOFF-v3.md, BRAIN-SPEC.md, DECISIONS-*, audit-2026-09-28/, API.md, SERVICE-R10.md, data docs, BRAIN-VIEWS.md
runs/        brain-map/ (the Ask Bosco brain views)
ops/         fetch_data.sh
```
Python 3.12, `uv`, `ruff`, `pytest`. Data in `data/` (gitignored; raw and cached embeddings are local on the Mac).
