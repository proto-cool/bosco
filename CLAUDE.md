# bosco

Bosco is a simulated male fruit fly (MaleCNS v1.0 connectome, HHMI Janelia /
Google Research / Cambridge, Cell 2026) running as an autonomous Bluesky
account at `bosco.proto.cool`. The question: can a fly-brain policy over a
constrained action space, loosed on bsky.app, hold its own against the bottom
of the feed? Whether it sinks or swims is the community's call — there is no
rater panel and no headline metric. What is binding is that the fly is
genuinely the one acting. The account runs indefinitely. It is part of the
experiment that it is live while being built.

Companion to this file: `EXPERIMENT.md` (binding sections lock at tag
`freeze-v1`).

## Non-negotiables

These exist so the fly does the work and we don't. Do not violate them for
convenience, "just for testing," or because a result would look better.

- **No LLM anywhere in the runtime loop.** No embeddings, no classifiers, no
  sentiment models. The only text scorer is the VADER lexicon (deterministic,
  public). The only text generator is an n-gram model over the published
  `corpus/` (`src/bosco/textgen.py`), seeded by the episode and conditioned
  on the fly's state. If a task seems to need a model, stop and ask.
- **Encoder, readout thresholds, phrasebook, and action set are frozen at
  `freeze-v1`.** Before the tag they may change; after it they may not.
  Thresholds are set from dev-period real activity, never from outcomes.
- **Humans influence Bosco only through the network.** No manual edits to
  weights, the outcome ledger, or the stimulus log. Bug fixes to code are
  fine; hand-nudging state is not.
- **Silence is a legal action** and must be a common one. Never add a floor
  on posting to make the account look alive. The one exception (decided
  2026-09-14): whoever speaks to him (mention, reply, quote) is answered,
  once per post, in the register the window produced; the action is logged
  as `answer` so the record never mistakes it for the network's choice. His
  appetite and what he has learned of the account set how much he says.
  Exception to the exception (decided 2026-09-16): a reply under the pinned
  primer thread (`config/identity_v1.yaml` `primer`, posted by the
  operator's `primer` command, logged as `primer`) is read like a browsed
  post and not answered unless it tags him.
- **Log everything, replay anything.** Every window records its features,
  activity, decision, and the brain digest after it; he is
  snapshotted hourly. Any span must replay bit-identical from the snapshot
  before it plus the log.
- **Store features and URIs, never post text.** Other people's posts do not
  live in our database, and never enter the generator's corpus. Words
  outside his vocabulary are kept as hashes (`h:<16 hex>`), never as words;
  images are never stored, only the retina's channel names.
- **Rate caps are hard** but by kind (`config/caps_v1.yaml`): they are loop
  guards, not a schedule. Timing of posts and replies is the network's;
  never add a timer. Self-label as a bot.
- **Bosco lives on the network like anyone else** (decided 2026-09-13). He
  reads his feeds, and may like, follow, unfollow, and post on his own.
  **He replies only to someone who replied to him or tagged him, once per
  post of theirs; never to a post he merely browsed** (decided 2026-09-14:
  no spam). Everything he reads is a
  stimulus; everything he does is an episode in the ledger.
- **Operator override** from `@proto.cool` (EXPERIMENT.md §2a) by mention:
  sleep, wake, delete, ignore, unfollow, reload, restart, status, people,
  memory, forget. Only `forget` touches state, and it is logged and flagged.
- **Anyone can send him away** (EXPERIMENT.md §2a): "shoo fly" (or "go
  away", "stop", "opt out") by mention is a reflex: answered once, unfollowed,
  ignored for good under their own DID; "come back" from them lifts it.
- **Moderation labels are bitter** (EXPERIMENT.md §2b): labeled posts and
  accounts are never approached and never reward. The vocabulary is closed
  (`corpus/`), so words cannot drift; associations can, and labels plus the
  ignore list plus `bosco people` are the guard. Since 2026-09-15 the
  closed vocabulary is his *mouth*: perception is open (every word is a
  smell, by hash), production is the corpus and phrasebook only.

## Architecture

One Python process, one container (Podman quadlet), one dedicated x86-64-v3
server (Kimsufi KS-5-A from 2026-09; a dedicated-vCPU VPS before). No GPU.

```
poller  ->  encoder  ->  kernel (C, ctypes)  ->  plasticity  ->  readout  ->  poster
                  \______________ SQLite ledger + stimulus log ______________/
```

- **Python 3.12**, `uv`, `ruff`, `pytest`. Deps: `pyarrow`, `numpy`,
  `neuprint-python`, `atproto` (MarshalX), `vaderSentiment`.
- **Kernel**: C99, single-threaded per fly, CSR sparse matrix, fixed-point
  or carefully ordered float so runs are deterministic. Parallelism is across
  flies (bosco, dunce, offline controls), never within one.
- **State**: SQLite for ledger/log; weights as `.npy` on disk. Each episode
  records MBON vector, DN winner, action, stimulus id, seed, and weight
  digest in the ledger. Nightly: dump ledger + weight snapshot to the repo
  (`snapshots/`) so anyone can audit that action N followed from state N.
- **No custom lexicon.** Bosco writes ordinary `app.bsky.feed.post` records
  and nothing else. Auditability comes from the published dumps, not from
  ATProto.

## Data

Bulk files under `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` — full connection
  graph, segment-to-segment synapse counts
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` — cell types, sides,
  classes, `flywireType` cross-reference
- aggregate neurotransmitter predictions per neuron (check exact filename on
  https://male-cns.janelia.org/download/)

Do not download the EM volume. `neuprint-python` against
`neuprint.janelia.org` dataset `male-cns:v1.0` for ad-hoc queries.

## Model

- **Neurons**: leaky integrate-and-fire, parameters from Shiu et al. 2024
  ("A Drosophila computational brain model reveals sensorimotor processing").
  Their code is public; take parameters from it, do not invent them.
- **Weights**: synapse count × sign from neurotransmitter prediction.
- **Pruning (v1)**: central brain only. Optic lobes dropped; since
  2026-09-15 a hand-built retina (`config/retina_v1.yaml`) drives the
  visual Kenyon cells directly: a fixed, parameter-free transform from a
  thumbnail to a few channel names (hues, brightness, edges, saturation),
  never a classifier, never the image stored. VNC dropped; descending
  neurons are the behavioral readout. Documented as modeling decisions in
  the README.
- **Short-term depression** on sensory synapses only (habituation; on).
- **Thresholds** follow `config/thresholds_policy.yaml` (pre-registered
  quantiles per population over named window sets); calibrate with
  `scripts/calibrate_thresholds.py --policy`, never by hand.
- **Circadian**: drive the annotated clock neurons with a 24 h rhythm. Do
  not add a posting-time gate; let the network produce the schedule.
- **Time step** 0.1 ms. Bosco runs continuously, one biological second
  per wall second; idle time is simulated, not skipped (`--simulate-gaps`
  in the CLI; the loop always simulates). Spike-frequency adaptation keeps
  activity from smouldering. Nothing is reset.

## Learning (mushroom body)

- Three-factor rule: KC activity coincident with dopaminergic (DAN) firing
  in a compartment depresses KC→MBON synapses in that compartment.
- **Extinction** (decided 2026-09-18, `config/plasticity_v2.yaml`
  `extinction`, `Agent.PLASTICITY_VERSION` 3): a compartment whose DANs did
  *not* fire while its KCs did relaxes back toward baseline, by the same
  rate fraction, LTM slower by `ltm.eta / stm.eta`. Depression alone is
  one-way, and under a diet 2.7:1 sweet it ratcheted: every odor inherited
  the same positive offset, no account could read bitter, and the verdicts
  converged (+0.259 → +0.332 in a day). The fly's own counterweight
  (Berry et al. 2012, 2015; Felsenberg et al. 2018; Cohn et al. 2015).
  Measure with `scripts/plasticity_v2_gate.py`; `eta` comes from the
  behavioural extinction protocol, never from his feed and never from
  outcomes.
- **Replay pairing**: outcomes arrive hours late. When an outcome lands for a
  past action, re-present the stored stimulus encoding and fire the
  appropriate DANs. This is the lab protocol; do not invent delayed credit
  assignment.
- **Two memory timescales** (decided 2026-09-13). Short-term: one pairing,
  fades over hours. Long-term: only spaced repetition consolidates (a
  second pairing while the first is still fresh, ≥ 1 h later), fades over a
  month. One insult is forgotten by evening; a week of them is not.
- **Learned valence reaches behaviour through the readout**: the mushroom
  body's verdict on a stimulus, read from the depression on the active KCs'
  synapses, scales approach vs avoid populations before thresholds.
- **What he remembers about you** (`config/associations_v1.yaml`, decided
  2026-09-15): a decaying association between an account and the tokens
  that arrived with it, faintly in the air when they speak to him again.
  A named tool outside the connectome, tokens only, replayed, in the
  digest. Asked something, `seen=met|fresh` is his α'3 familiarity with
  the words he was asked with; the corpus carries his yes and his no.
- **Identity is a reflex**: who/what/why/creator are answered from
  `config/identity_v1.yaml` regardless of the network. `bosco memory --did X`
  shows what he has learned about an account and why.
- **Reward DANs**: likes, reposts, follows from anyone; kind replies
  (VADER > 0.3) to his posts from anyone; any reply or return visit from a
  known account (≥1 prior interaction). Two per account per day. Labeled
  accounts never reward.
- **Punishment DANs**: block records (public, real) and VADER-negative
  replies or quotes to his posts. Punishment is sparse; the innate bitter
  channel carries most aversion. Do not "fix" the sparsity by adding a model.
- **Taste while browsing** (decided 2026-09-15): the sugar or bitter of a
  post he reads pairs that window's mixture with reward or punishment at a
  small published strength (`config/plasticity_v2.yaml` `taste`), in the
  window, as sugar drives PAM and bitter drives PPL1 in the fly. Not a
  social reward: no appetite bite, no outcomes row. Sizes come from the
  gate (`scripts/phase8_taste_gate.py`), never from outcomes.
- **Familiarity is the α'3 compartment** (decided 2026-09-15; Hattori et
  al. 2017): every stimulus window depresses the exposure trace of the KCs
  that fired; the mean over a smell's KCs is his familiarity with it, no
  valence, e-fold a day. A novel smell alerts (scales walking). PPL104 is in
  the `exposure` set, not punishment. The learning rule is a frozen artifact
  at `freeze-v1`; a change of rule bumps `Agent.PLASTICITY_VERSION`, which
  writes a `plasticity` control row and a snapshot.
- **Habituation**: short-term depression on sensory afferents only; the
  same smell every few minutes fades, a new one is fresh.
- **Appetite** (`config/appetite_v1.yaml`): a scalar that rises with hours
  since his last reward and falls on reward; it scales approach in the
  readout and the courtship (pC1) drive of a mention. Not a pain signal,
  not a floor, never set from outcomes. Being addressed maps engage to
  reply; browsing maps it to follow.

## Encoder (frozen at tag)

- Account DID → sparse code over a **valence-neutral** subset of ORN classes.
  Account is the odor. Never map onto innately valenced ORNs.
- VADER compound score → bitter GRNs (negative) / sugar GRNs (positive).
- Being mentioned → mechanosensory channel.
- Words → odors (`config/words_v1.yaml`): his own vocabulary's content
  words in a post, up to six, each three neutral glomeruli by hash, except
  the words that name a smell a fly is born to answer, which take their
  real glomeruli (`config/innate_v1.yaml`: fruit, vinegar, rot, dirt); a
  thread's last posts linger for a few minutes. Never the sentence. What
  is "in the air" when he speaks is read from his antennae (habituation
  depth over the logged words), not from a list. Words outside his
  vocabulary are smells too, up to four per post, as `h:` hashes
  (decided 2026-09-15); a hash is in no sentence of his, so it is never said.
- Other people and places in a post (decided 2026-09-15): a mention facet
  or quoted author is that account's odor at half rate (up to three); a
  link card's site is a place by domain; alt text, cards and quoted posts
  are read as context words; `img`/`video` tokens go to the retina.
- Feeds → places (`config/feeds_v1.yaml`): a published list of feeds, each
  an odor of two neutral glomeruli driven with every post read there. His
  browse budget is split across them by his own recent approaches (like,
  follow, reply decided), with a floor per feed; never by outcomes.
- Languages (`BOSCO_LANGS`, default en): his content-language setting, sent
  as Accept-Language like any user's, and a post declaring only other
  languages is not perceived. His vocabulary is English; nothing else of
  such a post could reach him.
- Topics: a hand-authored 60-topic keyword→odor map, published
  (`config/topics_v1.yaml`); up to three topics per post, each three
  neutral glomeruli on top of the account odor (decided 2026-09-14).
- Pictures → the retina (`config/retina_v1.yaml`, decided 2026-09-15): a
  thumbnail becomes channel names (two dominant hues, brightness, edges,
  saturation, `img`, `motion`) by a fixed transform; each channel drives
  fifteen visual Kenyon cells; colour words and "picture" are the same
  channels. Nothing tells him what he is looking at.

## Readout (frozen at tag)

- Behavior classes defined over **annotated DN populations**, never single
  neurons: engage (walking DNs), reply (courtship-song DNs), like (proboscis
  extension), leave (avoidance DNs), groom (grooming DNs → spontaneous post).
- Winner-take-all over population activity; silence if nothing crosses
  threshold. Thresholds from dev-period real activity distribution
  (provisional synthetic-battery thresholds until then).
- Action set: `reply`, `like`, `follow`, `walk`, `leave`,
  `spontaneous_post`, `nothing`. engage → follow while browsing (a walk if
  already followed), reply when addressed; a walking rate over the lower
  `walk` rung but under engage's threshold → walk (read a few more of the
  account's posts; not outward; capped; decided 2026-09-15); leave →
  unfollow. A reply is only ever an answer, and it embeds nothing: a quote
  notifies someone who never spoke to him (removed 2026-09-16).
- Own posts are grooming. Debris lands on his bristle neurons as discrete
  seeded events; a landing is an onset; his grooming neurons answer onsets
  and adapt to held input; if the answer crosses threshold he grooms, which
  is a post. This is the input we chose for the neurons that make the
  grooming outputs fire, not a body simulation.

## Phrasebook and generator

`phrasebook.yaml`, hand-authored by Nick, keyed by
`(behavior, valence, arousal, familiarity)`. Flat declarative fly register.
Frozen and published at tag. Claude does not write phrasebook lines.

`corpus/*.txt` is the generator's training text. Register: a fly's world,
a person's wants, flat declaratives, no jokes (`docs/corpus-audit.md`). `textgen.py` is a word n-gram (four words of context, backing off) with
absolute-discount backoff: the fly picks the corpus subset (tags), the
temperature (arousal) and the seed; the corpus never contains other
people's posts. Utterance policy (revised 2026-09-18, `Agent.UTTERANCE_VERSION`
2): **retrieval steers, it does not speak.** He reads a thread through
before answering in it (the posts above, oldest first, as context).
Retrieval still finds the thought of his that smells most like the moment
(lexical, over his corpus, weighted by freshness and learned word valence),
but he does not say it: he opens on the word of theirs that lit it up, with
that thought as his context, and walks on in his own words. He may run
`COPY_RUN_MAX` (9) tokens alongside one line of his; past that, the words
that would carry that line on are struck out wherever he has anywhere else
to go, so a page never comes back whole. He stays coherent inside a sentence
(`UTTERANCE_VERSION` 3): he carries the clause up to their word rather than
opening mid-phrase, finishes the clause he is in before changing lines
(`STAY_BIAS`), and may stop where one of his own sentences stops. The
phrasebook is his last resort, not a coin: it speaks only when the walk comes
back with nothing. Measure both with `scripts/quotation_gate.py` — what he
quotes and what he splices — and never tune either on outcomes. When he grooms, the register is his day:
the balance of his own verdicts on what he read, a `mood=` tag from the
ledger, and the day's words as faint air. Corpus and phrasebook digests are
frozen artifacts.

## Controls

- **dunce**: degree-preserving shuffle of the connectivity matrix, same
  plasticity, **live** on its own account for weeks 3–4 after `freeze-v1`,
  offline replay otherwise.
- **frozen-MB**: real wiring, plasticity off, offline replay.
- **random policy**: offline replay.
Controls replay against Bosco's logged stimulus stream, so determinism
matters.

## Build phases and gates

0. Pull feathers. Verify MB annotations (KC types, MBON types, DAN
   compartments) are populated in v1.0. If thin, map through `flywireType`
   and say so. **Gate: annotation coverage report.**
1. Kernel. Reproduce Shiu's sugar → proboscis extension on their FlyWire
   model with our kernel. **Gate: reflex reproduces.**
2. Prune, port to MaleCNS, measure KC sparseness. **Gate: ~5% KCs active per
   stimulus. If not, tune APL inhibition and document.**
3. MB plasticity + replay pairing. **Gate: learn/forget an odor in a
   synthetic protocol.**
4. Encoder, readout populations, synthetic stimulus set as unit tests.
5. atproto loop with `--dry-run`, snapshots, nightly ledger publish. Go live
   (dev period).
6. Threshold calibration from real activity. Tag `freeze-v1`. Publish
   EXPERIMENT.md.
7. dunce account. Monthly descriptive report script (see EXPERIMENT.md §6).

## Ops

- systemd/quadlet with restart-on-failure.
- Nightly: ledger + weights rsync off-box. Weekly: dated tarball of state
  off-box. numpy pinned to x86-64-v3; a change of numerics level is a logged
  boundary (`docs/MIGRATE.md`).
- Dead-man alert if no episode has run in 4 h.
- Bio says "in development" until `freeze-v1`.

## Things Claude Code must not do

- Add an LLM, embedding, or classifier call.
- Tune any threshold using outcome data.
- Edit weights, ledger, or stimulus log by hand.
- Store post text.
- Add a minimum-posting floor.
- Change encoder/readout/phrasebook after `freeze-v1` without a new tag and
  a new pre-registration.
