# Plan (written 2026-09-22)

v1 is over and answered: a fly connectome cannot carry a social account. What
ran was VADER, an n-gram, a clock and rate caps, with the brain deciding only
*when*. Its closing note is on `main`. This plan is for what is novel about the
connectome, and it is in phases with gates because that discipline was the part
of v1 that worked.

## Decisions taken (2026-09-22)

- This repo, this branch (`v2`, orphan). `main` is the archive and is not
  touched.
- A frozen embedding model is in bounds as the *encoder*. A model may
  perceive; it never decides and never writes. No LLM anywhere.
- Order: gate B (biological rule, untouched wiring) first, then A
  (connectome as a trainable prior).
- Deliverable: a writeup first. A decision service only if the connectome
  beats the shuffle.

## Phase B — the biological rule on the wiring as it is

*Does the real wiring, learning by the fly's own rule, decide better than a
shuffle of itself or a random projection?* Pre-registered in `GATE-B.md`.

- Reuses v1's kernel, model, plasticity rule and `make_dunce.py` unchanged.
- New: `scripts/gate_b.py` — one script, four arms, same encoder, same
  reward sequence, seeded.
- Output: `docs/gate-b-results.md`, numbers only, whatever they say.
- Gate: see `GATE-B.md` §Decision rule. Written before running.

Estimated: one to two weeks of evenings; the runs are hours of CPU.

## Phase A — the connectome as a trainable prior

*Is the wiring diagram a useful inductive bias when the strengths are
learned?* Fix the graph and the signs; train magnitudes and neuron
parameters by gradient descent on decision tasks; compare with (i) the
same graph shuffled degree-preserving, (ii) a free network of the same size
and sparsity. Precedent for the method: Lappalainen et al. 2024 (fly visual
system). Scope to the paths that can train — PN → KC → MBON → DAN → DN
populations, a few thousand neurons — not the whole central brain.

- Pre-registration `GATE-A.md`, written after B's numbers are in and before
  any training run.
- Rate-based surrogate first; spiking with surrogate gradients only if the
  rate model shows an effect worth the cost.
- Metrics: sample efficiency, calibration, forgetting/relearning under
  drift, and — the one only a connectome model can offer — whether the
  trained model's internal activity resembles anything known about the real
  circuit (MBON valence, KC sparseness).

## Phase C — the service (built 2026-09-23: `src/bosco/decider.py`, `scripts/decider.py`; earned by B4)

`decide(state, questions) → {answer, probability}`, `reward(decision_id,
value)`, `state()`. Stateful by default (habituation, appetite, clock,
biological time), stateless mode from a fixed snapshot. Probability from N
seeds, post-hoc calibrated against held-out rewards and said so. Only built if
A or B shows the connectome earning its place.

"Ask him on Bluesky" is one possible front end for this and nothing more.

## What carries over from v1, and what does not

Carries: kernel, model build, plasticity rule, the phase 0–3 reproductions,
determinism, pre-registered gates, the habit of writing down what failed.

Does not: the account, the encoder (hashes into twelve glomeruli), the
readout thresholds (quantiles of his own activity, which normalised away
whatever the wiring carried), the corpus, the phrasebook, the generator, the
ledger schema (Bluesky-shaped). A new ledger is written when the service is.

## Open

- Which decision tasks are *fly-shaped* (valence: is this good for me?) rather
  than generic benchmarks. `GATE-B.md` names candidates; Nick picks.
- Whether the Kimsufi box is kept for A's training runs or a laptop GPU does.
