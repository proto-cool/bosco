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
   corpus subset (register tags, and `topic=` tags for what the post
   smelled of), temperature (arousal) and seed.  Not an LLM.
21. **Questions and being asked about oneself.**  A post with a `?` is a
   stronger touch (JO drive x1.5) and approach on it is a reply.  "what do
   you think of me" from anyone is answered with his real memory of their
   smell (`config/identity_v1.yaml` `memory_question`): unknown, neutral,
   sweet, or bitter, with how many times they came.
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
   subjects.  Not a classifier.  Sixty topics since 2026-09-14 (v1.2); the
   list is frozen with the encoder, so he cannot acquire a new subject, only
   a new opinion of one.
18. **Learning what to say** (`Agent.voice_update`).  When a generated reply
   or post gets an outcome, the corpus documents that matched its register
   get their preference nudged by 25% up (reward) or down (punishment),
   bounded 0.2 to 4, decaying toward 1 over two weeks.  Preferences scale
   how often each document's sentences enter the trigram pool.  Counts
   only; the vocabulary never changes.  Stored in `state/voice.json` and
   logged as `voice` control rows.
19. **Habituation** (`config/model_v1.yaml` `std_u`, `std_tau_rec`): short-term
   depression on sensory afferents only, 0.4% of resources per spike,
   recovery e-fold 3 min.  The same smell every two minutes settles near
   70%; a different account is fresh; twenty quiet minutes restore it.
   Central synapses are not depressed (docs/phase2-stability.md).  Costs
   the sugar reflex a little more: MN9 15 Hz at 100 Hz drive.
20. **Outcomes** (`src/bosco/bsky.py`): likes, reposts and follows from anyone
   reward the last window with that account (2 per account per day) and
   make the account known; a reply or quote to one of his posts punishes if
   VADER < −0.05, rewards if VADER > 0.3 from anyone or if the account is
   known; a block from someone he acted toward punishes.  Labeled accounts
   never reward.
16. **Learned valence is read from the weights**, not from MBON rates: a 1%
   uniform weight change flips a single-realisation MBON rate from 9 to
   6 Hz (deterministic chaos), so rates are averaged over seeds in the gates
   and the behavioural signal is the depression on the active KCs' synapses
   (reward-side minus punishment-side, Aso 2014 sign).

21. **Familiarity is a register** (`src/bosco/textgen.py`, decided 2026-09-14):
   corpus documents may carry `familiarity=new|known|familiar` (the
   phrasebook's bins over prior interactions with the account) and join the
   pool only when the account he is answering falls in that bin.  What he
   says to a stranger and to someone who keeps coming back differ by the
   ledger, not by a rule in the generator.  Corpus v2 (`corpus/090`–`129`)
   adds the twenty topic files, questions, night, the three familiarity
   registers, a file about his own learning, and sweet/bitter variants of
   the first-week topics.
22. **Stranger rail** (`src/bosco/bsky.py` `act`): toward an account he only
   browsed past (not mentioned by it), a like, follow or reply goes out only
   if that account has interacted with him before or his learned valence for
   its smell is above the cut.  The second condition is his memory; the first
   is a spam rail for a new account, logged as a withheld `leave`.

23. **The panel** (`panel/`, `src/bosco/panel.py`): the process writes a sparse
   list of the neurons that spiked in the last simulated second and a status
   JSON from the ledger; a static page (ARC UI, Vite) draws every neuron at
   its soma and lights the ones that fire. Read-only: nothing on the page can
   reach him. Post text is fetched from the public API by URI, never stored.

24. **Activity blocks are eight neurons and the flush is 1e-4 mV** (kernel, decided
   2026-09-14). A block of neurons is skipped while all of them are at rest;
   with 64-neuron blocks and a 1e-9 mV flush, 34 spikes a second kept 82% of
   the brain "awake" and an idle second cost 0.42 wall s on a laptop. Eight-neuron
   blocks, rest flushed below 1e-4 mV and the adaptive threshold below 1e-2 mV
   (the spike threshold sits 7 mV above rest) bring that to 0.20 wall s with
   the readout bit-identical in the tests we ran; a 1e-3 mV flush changed
   which spikes fired and was rejected. The loop lives between polls instead of
   sleeping, checks notifications every 20 s and browses on the interval.

25. **Grooming is an onset** (`config/encoder_v1.yaml` `spontaneous`, decided
   2026-09-14). Day-one data: 25 grooms in 23 h, every landing crossing a
   10.5 Hz threshold, because dust never settled and the driven bristle
   subset was redrawn every second, so held debris was a new onset each
   second. Now one seeded permutation per landing picks the bristles (the
   subset shrinks as the debris settles, never adds an onset), dust settles
   passively with tau 600 s, landings come 0.5/h, and the 15 s after each
   landing are logged as `landing` windows so the groom threshold is set
   from real responses (peak per landing). No timer anywhere.
26. **Answering has a circuit: pC1** (`config/encoder_v1.yaml` `courtship`).
   The `reply` population is the courtship-song descending neurons, and in
   day-one data they sat at 0.0 Hz in every mention: nothing sensory reaches
   them (JO-B synapses onto the giant fibre). Being addressed now also
   excites the male courtship command neurons pC1/P1 (types `pC1*`, 156
   bodies; von Philipsborn 2011, Kohatsu 2011, Hoopfer 2015, Zhang 2016)
   at 12 Hz × appetite (× 1.5 for a question), the input we chose for the
   neurons that make the reply outputs fire, as bristle debris is for
   grooming. JO drops to 50 Hz (100 Hz made every mention a startle) and the
   question gain moves to the courtship drive. Walking toward whoever spoke
   to him is a reply; toward someone he only browsed past, a follow. Not a
   guarantee: threshold from the real distribution, learned valence still
   gates it, caps and the stranger rail still apply.
27. **Appetite for contact** (`config/appetite_v1.yaml`): a scalar in
   [0, 1] on the agent, rising toward 1 with e-fold 6 h without a social
   reward and taking a 30% bite on each reward; punishment leaves it
   alone. It scales the approach populations ×(1 + 0.5 (a − 0.5)), centred
   so it can neither force nor silence, and sets the courtship drive. It is
   state (saved, snapshotted, in the digest, logged on every window,
   replayed), ticks on his own clock, and is not a pain signal. It does
   nothing to the idle brain, which is the clock drive plus the minutes
   after a landing. A basal ORN drive for the panel's sake was measured
   (+43% idle CPU for spikes that reach no readout) and rejected.
28. **Thresholds follow a pre-registered policy** (`config/thresholds_policy.yaml`):
   engage, like and leave at the 85th percentile over event windows;
   reply at the 40th over windows in which he was addressed; groom at the
   70th over per-landing peaks. `scripts/calibrate_thresholds.py --policy`
   applies it; `ops/fetch_ledger.sh` copies the live ledger (features and
   URIs only) for it. Likes and follows removed in the app as him are
   reconciled on each sweep (`unliked_in_app`, `unfollowed_in_app`); the
   learning they earned stays, because the reward was for the window.

29. **Words as smells** (`config/words_v1.yaml`, decided 2026-09-14). Every
   content word of his closed vocabulary (850 words from the corpus and
   phrasebook, minus a published stop list) is an odor: two neutral glomeruli
   and a rate chosen by its hash, exactly as an account is an odor. A post's
   words that are in his vocabulary (up to eight, in order) are driven on
   top of the account and topic odors; the sentence is never seen. The
   mushroom body then learns, per word, what came with it. A word's
   Kenyon-cell signature (`data/word_kc_v1.npz`, `scripts/build_word_atlas.py`:
   the word presented alone to a quiet network, once, a property of the
   wiring) lets that memory be read from the weights with no simulation;
   the generator multiplies a word's chance by (1 + learned valence). A
   thread lingers: the words of its last three posts stay in the air for
   15 minutes and are smelled again, fainter, with the next post in it,
   and his own posts draw on whatever is in the air. His own posts are also
   tied to his state by corpus tags: the hour (`time=`), his appetite
   (`appetite=`), and the grooming he is doing when he posts. The
   reflective corpus file that read as thought was removed. What he says
   is stitched from fragments by a trigram; none of this is understanding,
   and it is not meant to look like it.

Not in v1: visual input.

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
