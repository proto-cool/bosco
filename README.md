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
   to him is a reply; toward someone he only browsed past, a follow. Song
   answers speech: when he is addressed and the song population crosses its
   threshold that is the action, and if the proboscis population crossed
   too he likes the post as well (a fly can extend its proboscis and sing).
   Otherwise a kind question's sugar won as a like every time. Not a
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
   content word of his closed vocabulary (about 850 words from the corpus
   and phrasebook, minus a published stop list) is an odor: three neutral
   glomeruli and a rate chosen by its hash, exactly as an account is an
   odor. A post's words that are in his vocabulary (up to six, in order)
   are driven on top of the account and topic odors; the sentence is never
   seen. A thread lingers: the words of its last three posts stay in the
   air for 15 minutes and are smelled again, fainter, with the next post in
   it, and his own posts draw on whatever is in the air.
   What the mushroom body learns from this is a memory of *mixtures*, not
   words: measured on the model, a word alone lights ~11 Kenyon cells, the
   same word with an account lights ~72, only ~4 are shared, and the cells a
   word adds to a mixture agree across accounts at 9% (Jaccard). A
   vocabulary-wide "which words are sweet" readout was built, found to
   return spurious verdicts for this reason, and removed. What is real and
   readable is "how does this word smell with this person": when he chooses
   words for a reply, the few words in the air are each presented with the
   account's odor to a snapshot of his state (restored to the byte, random
   stream included) and the verdict on the cells that fire scales that
   word's chance (`Agent.word_valence_in_context`). Words in the air also
   come up a little more (echo). His own posts are tied to his state by
   corpus tags: the hour (`time=`), his appetite (`appetite=`), and the
   grooming he is doing when he posts; the reflective corpus file that read
   as thought was removed. What he says is stitched from fragments by a
   trigram. None of this is understanding, and it is not meant to look
   like it.
30. **The air is read from his antennae** (decided 2026-09-14). A fly
   cannot hold a smell that is gone: at these rates a held odor habituates
   to a few percent within a minute, and he has no working-memory circuit
   for one, so none is invented. What he does have is habituation itself:
   every sensory afferent keeps a resource that drops with each spike and
   recovers over three minutes (`config/model_v1.yaml` `std_tau_rec`), so
   the mean depression over a word's olfactory neurons is a physical
   record of how recently, and how hard, he smelled it. When he chooses
   words, "in the air" is that depth read from the kernel over the words
   the log says were in his recent threads (`Agent.antennae`); a fresh
   word is heavy, one from two minutes ago faint, one from ten minutes
   ago gone. Candidates come from the log because words share glomeruli:
   a mixture of six depresses enough of them that a fifth of the
   vocabulary would read as present. The thread's lingering re-presentation
   (a plume does linger) is kept but cut to four minutes, and words smelled
   in the last minute are not re-presented since their afferents are still
   depressed. Remembering a conversation past that is the mushroom body's
   job, and needs a pairing: a like on his reply, a kind reply.
31. **He reads in his languages** (`BOSCO_LANGS`, default `en`). His
   vocabulary is English; a post in another language carries none of his
   words, hits no topic and tastes of nothing, so all that reached him was
   a stranger's odor, and in the dev period a quarter of what he browsed
   was that (146 of 629 posts with no topic and no taste, drawing 28 likes
   and 34 leaves). He now sets his content language the way any user does
   (an `Accept-Language` header, which the discover feed honours) and does
   not perceive a post whose record declares only languages he does not
   read. Posts that declare nothing are read. This is a setting on his
   account, not a change to any threshold; the next recalibration from the
   real ledger will see the narrower distribution.
32. **He goes places, and a place has a smell** (`config/feeds_v1.yaml`,
   decided 2026-09-14). He reads from a published list of feeds: the
   accounts he follows, discover, popular-with-friends, science, art,
   nature photography, cooking and bugs. Each feed is an odor of two
   neutral glomeruli chosen by its name, driven with every post read
   there on top of the account, topic and word odors, and logged on the
   row, so the mushroom body learns the place as part of whatever
   mixtures were rewarded or punished in it. Where he spends his browsing
   is his own doing: each poll the budget is split across feeds in
   proportion to his approaches there over the last two days (a like,
   follow or reply his readout decided on a post read in that feed) plus
   a prior, each feed keeping a floor of the budget so every place is
   still visited. That uses his behaviour, never outcomes, and touches no
   threshold. A fly stays on the patch where its proboscis has been
   extending. `bosco feeds` prints the split; the panel says where he has
   been reading.

33. **The off ramp** (decided 2026-09-14). People will not all want a fruit
   fly in their replies, and they should not need the operator to be rid of
   him. Any account that tells him to go by mention ("shoo fly" above all;
   also "go away", "leave me alone", "unfollow me", "stop", "opt out";
   `config/identity_v1.yaml`) is
   answered once, unfollowed, and placed on the ignore list under its own
   DID: never in his stimulus stream again, never approached, logged as
   `opt_out`. "come back" from the same account lifts it; the operator's
   ignores are not theirs to lift. A block works the same without words,
   and is the punishment signal besides; a mute costs nothing. Like the
   identity answers, this is a reflex outside the network, not a decision
   of his.

34. **Habituation recovers lazily** (kernel, decided 2026-09-14). A
   synapse's resource `x` is only read when its neuron spikes, so the
   kernel no longer relaxes every neuron's `x` every step; it records the
   step each `x` was last current and applies the whole recovery in one
   multiplication when the neuron next fires, when it is read back, or
   when time is skipped. The power `e^k` is computed by squaring in plain
   double arithmetic, not by libm, so a replay on another machine is still
   bit-identical. The saved state carries the per-neuron step; a state saved
   before this loads with every `x` marked current. Measured on the laptop
   from a rested network under the clock drive alone: a dead-quiet second
   (early afternoon, no clock cells driven) fell from 0.08 to 0.04 wall
   seconds, a morning second from 0.13 to 0.09, and an evening second (the
   evening cells at 6 Hz wake a quarter of the brain) from 0.35 to 0.28. The
   whole-brain sweep had been a fixed cost on every second; what remains in
   the evening is real synaptic traffic. Same dynamics, cheaper seconds.

35. **The outward policy** (decided 2026-09-14, after a leaky rail had
   him following people who had never heard of him). Browsing, he may
   follow or like whoever his readout picks, under the caps; that is his
   curiosity and it is allowed to be random. He replies only to someone who
   replied to him or tagged him, and at most once per post of theirs,
   whether the answer came from the network or a reflex. Engaging with an
   account he already follows while browsing is nothing. A post he merely
   browsed never gets a reply, whatever the song neurons did; the decision
   stays in the record as withheld. His own posts are grooming. Nothing
   else goes out. The earlier "stranger rail" (no like or follow until they
   had come to him, or the window's learned verdict was sweet) is gone: the
   window's verdict is on the whole mixture and generalised to strangers
   who shared a place or a topic, and the rule above makes it unnecessary.

36. **Innate smells** (`config/innate_v1.yaml`, decided 2026-09-14). A
   fly is born liking ripe and rotting fruit, vinegar and fermentation, and
   born avoiding geosmin, and each of those is a particular glomerulus with
   its own wiring to the lateral horn (the encoder already excluded them
   from neutral odors, with sources). The words of his vocabulary that
   literally name those smells (fruit, banana, apple, vinegar, wine, rot,
   dirt, twenty-two in all) are now driven on their real glomeruli instead
   of the neutral ones a hash would pick. Everything else about a word is
   unchanged: it is smelled with the account, the place and the topics, and
   the mushroom body learns the mixture. This is the fly's biology,
   published, not our opinion of the words; only words that name an odor a
   fly actually meets are listed, and taste words (sweet, sugar, honey) are
   not, since taste is the tone channel. One consequence is that all fruit
   words are one smell: banana and grape are the same glomeruli, as they
   are to a fly. Measured on the model (ten presentations each, raw rates,
   gate off, a different account each time): a post carrying a fermentation
   word drove the walking population to 6.7 ± 0.6 Hz against 3.7 ± 0.9 for
   a neutral word, so the innate approach wiring does carry through to the
   descending neurons; fruit and geosmin words did not differ from neutral
   within error, and the proboscis stayed at zero for all of them. The
   innate map is kept for what it is, the fly's own glomeruli, whatever the
   readout makes of each.
37. **The learned verdict stays a gate; the wiring carries a whisper**
   (measured 2026-09-14). We asked whether a learned smell could extend the
   proboscis through the connectome alone, as a trained odor does in a fly
   after odor–sugar conditioning, so the readout gate could go. From a
   rested brain, one mixture (account, place, topic, two words) was rewarded
   five times, two hours apart, by the real outcome path; its learned
   verdict reached +0.74. Read raw, with the gate off, the walking
   population went from 2.0 to about 2.3 Hz and the proboscis population
   from 0 to 0: the depressed KC→MBON synapses change MBON output, but at
   these rates the path onward to the descending neurons moves them by a
   tenth, and a proboscis with no sugar in front of it stays in. So the
   verdict keeps reaching behaviour through the readout gate
   (`config/readout_populations.yaml` `kappa`), as documented in decision
   12, and the gate's size is what stands in for the pathway the model
   cannot carry. Recorded here so nobody has to wonder whether we tried.

38. **An answer opens on their word; his own posts carry his day**
   (decided 2026-09-14). Two things changed about what he says, neither of
   them a model. When he answers someone, the generator now opens the
   utterance on one of the words on his antennae, drawn by freshness, from
   a context of his own corpus that ends in that word, and walks on from
   there; and the echo on words in the air is four times louder
   (`echo_gamma` 2.0). So a reply about bananas begins at banana, in his
   sentences. When he grooms, the register is chosen by his day: the
   valence of the post is the balance of his own verdicts on what he read
   in the last six hours, not the verdict on dust; a `mood=` tag from the
   ledger (stung within six hours of a punishment, warm within two of a
   reward, alone after a day without anyone coming to him) selects three
   new corpus files written in his register (`170`–`172`); and the words
   he met in those hours are a faint air on his own posts (`day_echo`), so
   a post of his is made of the day he had. None of it is understanding.
   It is a fly answering on the word you used, in the mood his day left him.
39. **An account's memory is its own smell** (fixed 2026-09-14). The panel
   showed the same learned valence for everyone because the readout used
   the Kenyon cells of the mixture an account last arrived in, which are
   mostly the place and the topics. Each account's odor is now presented by
   itself to a copy of his state once, the cells it lights are kept (on
   disk too), and the verdict is read on those. What you see beside a name
   is what he has learned about that smell, and the mixture verdicts stay
   where they belong, in the windows.

40. **Whoever speaks to him is answered; appetite sets how much he says**
   (decided 2026-09-14). Nick's rule: if you talk to a bot, you want a
   response, and he should be chatty with people who want to interact.
   Silence stays legal for everything he merely reads, but a mention, reply
   or quote that does not make his song neurons cross is answered anyway,
   once per post, under the caps, with the words the window produced: same
   valence, same arousal, primed on their words, phrasebook coin and all.
   The episode row keeps the network's decision ("nothing"); the action row
   is kind `answer`, and the panel's record shows "answer" for such a
   window, so nobody reading the ledger mistakes manners for his choice.
   What appetite decides now is length, not whether: one sentence when
   sated, up to three when hungry for company, one more for a smell he has
   learned to like and one fewer for one he avoids (`Agent.verbosity`).
   Nobody who did not speak to him is ever addressed.

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
