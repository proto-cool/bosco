# bosco

A simulated male fruit fly (MaleCNS v1.0 connectome) running as an
autonomous Bluesky account, `bosco.proto.cool`.  The question is whether a
fly-brain policy over a constrained action set can hold its own against the
bottom of the feed.  See `CLAUDE.md` for the rules and `EXPERIMENT.md` for
the pre-registration.  This file documents the model as built.

## Pipeline

```
poller -> encoder -> kernel (C, ctypes) -> plasticity -> readout -> poster
                 \_______ SQLite ledger + stimulus log (features only) _______/
```

- `src/bosco/data.py`, `populations.py` — raw feathers, named populations.
- `src/bosco/model.py` — pruned network as CSR; v1 wiring rules.
- `kernel/lif.c`, `src/bosco/kernel.py` — deterministic LIF kernel.
- `src/bosco/encoder.py` — (DID, VADER, mention) → sensory drive. `docs/encoder.md`.
- `src/bosco/sim.py` — episodes from rest; `plasticity.py` — mushroom body.
- `src/bosco/readout.py` — DN/MN populations → behaviour → action.
- `src/bosco/agent.py` — one event → one episode → one ledger row.
- `src/bosco/bsky.py` — Bluesky loop; `cli.py` — `bosco poke|replay|run ...`.

## Modeling decisions

Every number below has a gate report under `docs/`.

1. **Neuron set** (`model.CB_SUPERCLASSES`): 40,939 traced central-brain
   bodies: cb_intrinsic/sensory/motor/endocrine/efferent, plus descending
   and ascending neurons.  Optic lobes dropped (no visual input in v1).  VNC
   dropped; descending neurons are the readout.  8.49 M edges, 45.7 M synapses.
2. **LIF parameters** are Shiu et al. 2024 verbatim (`config/model_v1.yaml`),
   except the per-synapse weight, which Shiu calibrate per connectome:
   "W_syn such that activation of sugar GRNs at 100 Hz resulted in roughly
   80% of maximal MN9 firing".  Same criterion on MaleCNS gives
   **w_syn = 0.16 mV** (`docs/phase2-wsyn-calibration.md`).
3. **Synaptic sign** from the MaleCNS consensus NT prediction, Shiu's
   convention: ACh and monoamines excitatory, GABA/glutamate/histamine
   inhibitory; 91 bodies with no prediction default excitatory.
4. **Antennal-lobe excitatory LNs: fast chemical output zeroed.**  With
   Shiu's model on either connectome, any olfactory drive (25 random ORNs at
   50 Hz) tips the network into a self-sustaining state of ~10 k neurons at
   the refractory ceiling; verified in the authors' own Brian2 code.  On
   MaleCNS the loop is the 157 cholinergic AL local neurons.  Lateral
   excitation in the AL is primarily electrical (Yaksi & Wilson 2010), which
   the LIF cannot represent; removing their chemical output is the single
   change that makes olfactory input bounded (`docs/phase2-stability.md`).
5. **KC→MBON gain 5.**  At connectome weight the plastic pathway contributes
   nothing to MBON odor responses (zeroing it changes nothing).  Gain 5 is
   the smallest at which MBON odor responses are ≥ 80% KC-driven
   (`docs/phase3-plasticity-gate.md`).  APL is left at connectome weight.
6. **Plasticity**: three-factor depression of KC→MBON synapses gated by DAN
   compartment (`config/mb_compartments.yaml`, Aso et al. 2014), on two
   timescales (`config/plasticity_v1.yaml`).  Short-term: one pairing,
   e-fold 4 h.  Long-term: forms only by spaced repetition (a pairing on a
   synapse still carrying short-term memory from a pairing ≥ 1 h earlier;
   Tully et al. 1994), e-fold 30 days.  Effective weight = connectome ×
   short × long.  Dopamine is modulatory only: a replay pairing re-presents
   the stored stimulus and gates the change; DANs are not driven as fast
   excitation.
7. **Encoder**: 12 of 35 valence-neutral glomeruli per account, 120 Hz;
   VADER → sugar/bitter GRNs up to 150 Hz; mention → JO-A/B 100 Hz; clock
   neurons on a 24 h cosine (`docs/circadian.md`).  KC activity 2–5% per
   account odor, odor-specific.  Background: clock neurons and bristle
   debris (landings; see 11).
8. **Readout** populations are cell-type lists with citations
   (`config/readout_populations.yaml`): engage → follow, reply → reply,
   like → like, leave → unfollow, groom → post.  Thresholds are provisional
   (synthetic battery, 85th percentile) until re-set from the dev-period
   distribution of real activity (`scripts/calibrate_thresholds.py`).
12. **Learned valence gates the readout.**  MBON output does not reach the
   descending neurons at any stable gain in the LIF (scans in
   `scripts/phase3_mbon_out_gain.py`: gains that propagate also smolder and
   move approach the wrong way).  Instead the readout does what summed MBON
   output does in the fly (Aso et al. 2014): v (decision 16) scales approach
   populations by (1 + 2v) and avoid by (1 − 2v) before thresholds.  `bosco memory --did X` prints v, the synapses carrying it,
   and the outcomes that caused it; `bosco people` ranks every account.
10. **Text**: `src/bosco/textgen.py`, a word trigram with absolute-discount
   backoff over `corpus/` plus phrasebook lines.  The fly supplies the
   corpus subset (tags), temperature (arousal) and seed.  Not an LLM.
13. **Moderation**: labels from the account's subscribed moderation services
   are innate bitter and block approach (`config/moderation_v1.yaml`,
   `src/bosco/moderation.py`). The generator's vocabulary is closed.
11. **Scope**: Bosco reads his timeline and the discover feed; each unseen
   post is a stimulus; he may like, follow, unfollow, reply, and post, within
   per-kind caps (`config/caps_v1.yaml`).  Own posts come from grooming:
   debris lands on his bristles as discrete, seeded events (about 1.5 an
   hour); a landing is an onset, his grooming neurons answer onsets and
   adapt to held input, and if the answer crosses threshold he grooms, which
   is a post, and the debris resets.  There is no timer.  Operator commands
   by mention: `src/bosco/control.py`.
9. **Determinism**: single thread, fixed accumulation order, no FMA or
   fast-math, splitmix64 RNG.  His full state (voltages,
   conductances, adaptation, delay ring, RNG, plastic weights, debris drive)
   is snapshotted hourly; any span replays bit-identically from the snapshot
   before it plus the logged inputs (`bosco replay`).
14. **Bosco runs continuously** (decided 2026-09-13; `src/bosco/brain.py`).
   No episodes from rest.  One biological second per wall second, about half
   a core.  Idle time is simulated in one-second slices with only the
   background drive (clock neurons, bristle debris); events are presented
   for one second on top; decisions are read from those windows; pairing
   depresses the synapses of the KCs that actually fired.
15. **Spike-frequency adaptation** (adaptive threshold, +0.1 mV per spike,
   1 s decay) so that activity settles after input instead of smouldering
   for ever in a continuous run.  Cost: sugar reflex 25 Hz instead of 69,
   MBON odor responses 6 Hz instead of 22; KC code unchanged
   (`scripts/phase2_sfa_scan.py`).  Denormal floats are flushed in the
   kernel; without that, long runs slow down a hundredfold.
17. **Topics as smells** (`config/topics_v1.yaml`, `src/bosco/topics.py`).
   A hand-written keyword list, published: whole-word matches on the post
   text name up to three topics, the text is discarded, the names are
   logged.  Each topic is three neutral glomeruli chosen by its name,
   driven at 100 Hz on top of the account odor, so a post is a mixture of
   who and what, and the same mushroom body that learns people learns
   subjects.  Not a classifier.
18. **Learning what to say** (`Agent.voice_update`).  When a generated reply
   or post gets an outcome, the corpus documents that matched its register
   get their preference nudged by 25% up (reward) or down (punishment),
   bounded 0.2 to 4, decaying toward 1 over two weeks.  Preferences scale
   how often each document's sentences enter the trigram pool.  Counts
   only; the vocabulary never changes.  Stored in `state/voice.json` and
   logged as `voice` control rows.
16. **Learned valence is read from the weights**, not from MBON rates: a 1%
   uniform weight change flips a single-realisation MBON rate from 9 to
   6 Hz (deterministic chaos), so rates are averaged over seeds in the gates
   and the behavioural signal is the depression on the active KCs' synapses
   (reward-side minus punishment-side, Aso 2014 sign).

Not in v1: short-term depression (implemented in the kernel, off), the
topic→odor map, visual input.

## Running

```
uv sync && make -C kernel
uv run python scripts/phase0_coverage.py     # needs data/raw/*.feather (see paths.py)
uv run pytest
uv run bosco poke --did did:plc:x --text "hello fly" --mention
uv run bosco spontaneous && uv run bosco say -n 5
uv run bosco memory --did did:plc:x && uv run bosco people
uv run bosco spontaneous --at 2026-09-14T12:00   # simulates the gap; reports grooms
uv run bosco replay                              # from the latest snapshot to now
BOSCO_HANDLE=... BOSCO_APP_PASSWORD=... uv run bosco run --dry-run
```
