# Bosco v3: closed, 2026-09-30

Nick ended v3: "it's over, I think." This is the record of what was learned: failures first, with numbers. The
detail is in `docs/audit-2026-09-29-full.md`, `docs/audit-2026-09-28/brain-deep-dive.md`, `runs/minibrain/`,
`runs/tastebrain/` and the commit messages on this branch.

## The conclusion
The idea was a fruit fly's brain, from the MaleCNS connectome, that learns a judgement about content (text) and
expresses it through his own behaviour: his descending neurons, or his proboscis. **It cannot be built honestly with
the data and models that exist today.**
- **Which links fail.** The two needed to turn the fly's learning into behaviour are too thin in the connectome to
  carry a signal:
  - MBON output reaches the descending neurons weakly: even the most MB-driven DNs take only about 5% of their input
    from the MB side;
  - taste reaches the KCs with at most 1.1% of any KC's input;
  - MBONs reach MN9 (the proboscis) with about 0.01%.
- **What that means.** The real fly does all of this (Kirkhart & Scott 2015; Aso et al. 2014b), so whatever makes
  those links strong in a living fly is not in the wiring data.
- **Other projects.** The many community MaleCNS projects since the September 2026 release have the same limit.
  Those that report learning train a readout outside the fly, or retrain every synapse's strength (flyvis-style).

## What failed, and why (in the order it was found)
- **The full brain (v3.0) never worked as claimed.**
  - Neurons were held up by tuned biases, not by their inputs.
  - Dopamine, octopamine and serotonin ran as fast excitation.
  - The unit leaked below threshold.
  - The resting brain had 4 stable states.
  - As a result the untrained answer bypassed the mushroom body (with every KC silenced, the read correlated 0.997
    with intact), and the MBONs were only 54% KC-driven.
  - Training reached 0.725 balanced accuracy on harm only by pushing KC→MBON synapses up to 166,000×.
- **Several checks passed for the wrong reason.**
  - T7 measured the ratio of the read scales k; a brain that learned nothing passed it.
  - T2 and T3 were confounded by the read offset.
  - T4's R² measured alignment, not share.
- **The brain was not calibrated** (ECE 0.16–0.25), and it was near chance on two of four harm sources.
- **Hand-set operating points** (per-type homeostasis; one global gain with stage targets) were unstable or did not
  converge.
- **The descending-neuron answer failed,** even with the relay layer (139 LAL/SMP/CRE/VES neurons on the MBON→DN
  paths): with every KC silenced, 98% of the answer's variation remained, and a fly-sized memory moved it at most
  0.2 item s.d.
- **The taste brain latches.** After tasting, 88 cells (MN9 among them) stay on. Two prototypes (adaptation and
  depression; Shiu-style raw weights) made it worse.
- **The process failures** are in the audits and in memory (`feedback-audit-depth`). Chief among them: fixes were
  written into the spec before they were prototyped, and flaws were fixed one symptom at a time.

## What worked (verified, committed)
- **A minimum smell brain** (`src/bosco/minibrain.py`, 7,910–11,780 neurons):
  - MaleCNS transmitters only, fast synapses only;
  - synapse compartments from MaleCNS synapse regions (`src/bosco/compartments.py`);
  - settings fitted to fly physiology rather than labels (gain 5, APL ×16, KC threshold 1.97);
  - stable, one resting state, and it returns to rest;
  - PNs broader than ORNs; APL block about 2×; MBONs 91–98% KC-driven;
  - a fly-sized KC→MBON memory moves the MBON answer the right way for every MBON type;
  - it holds with the lateral horn and the convergence neurons.
- **A minimum taste brain** (`src/bosco/tastebrain.py`, 1,189 neurons). It reproduces Shiu et al. 2024: sugar and
  water drive MN9, bitter suppresses the sugar response, bitter alone does nothing, and the dose-response rises.
  Each taste must start from rest.
- **A fast, bit-exact answer path;** the pinned encoder (`encoders.load_text`/`embed_text`); the clean-data pipeline
  and its provenance docs.

## What was never done
- **No specialist was trained** on the minimum brain.
- **The evaluation fixes are specified but not built:**
  - calibration;
  - per-source scores;
  - MLP and random-expansion baselines;
  - controls that keep APL.
- **Sight and timed sniffs** were not started.

## Loose ends
- **Uncommitted work, now committed as "superseded".** The v3.1 work in progress (`src/bosco/wiring3.py`, and edits
  to `ratebrain3.py` and `v1.py`) is committed for the record and is not validated. Its per-type homeostasis makes
  the brain unstable, and its MBON typical/atypical cut is wrong (Li et al. 2020: MBON10 is atypical; MBON21–23 are
  typical).
- **Local data** (gitignored):
  - the MaleCNS synapse table (`data/raw/synapses/`, 6.5 GB) and caches in `data/cache/`;
  - trained weights in `runs/brain-train*/` (kept local: their data licences).
- **magnetar** `~/bosco` holds v3 code and start files. Nothing is running.
