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
  on posting to make the account look alive.
- **Log everything, replay anything.** Every window records its features,
  activity, decision, and the brain digest after it; he is
  snapshotted hourly. Any span must replay bit-identical from the snapshot
  before it plus the log.
- **Store features and URIs, never post text.** Other people's posts do not
  live in our database, and never enter the generator's corpus.
- **Rate caps are hard** but by kind (`config/caps_v1.yaml`): they are loop
  guards, not a schedule. Timing of posts and replies is the network's;
  never add a timer. Self-label as a bot.
- **Bosco lives on the network like anyone else** (decided 2026-09-13). He
  reads his timeline and the discover feed, and may like, follow, unfollow,
  reply in any thread he reads, and post on his own. Everything he reads is a
  stimulus; everything he does is an episode in the ledger.
- **Operator override** from `@proto.cool` (EXPERIMENT.md §2a) by mention:
  sleep, wake, delete, ignore, unfollow, reload, restart, status, people,
  memory, forget. Only `forget` touches state, and it is logged and flagged.
- **Moderation labels are bitter** (EXPERIMENT.md §2b): labeled posts and
  accounts are never approached and never reward. The vocabulary is closed
  (`corpus/`), so words cannot drift; associations can, and labels plus the
  ignore list plus `bosco people` are the guard.

## Architecture

One Python process, one container (Podman quadlet), one Vultr dedicated-vCPU
VPS. No GPU.

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
- **Pruning (v1)**: central brain only. Optic lobes dropped (no visual
  input). VNC dropped; descending neurons are the behavioral readout.
  Document this as a modeling decision in the README.
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
  words in a post, up to eight, each two neutral glomeruli by hash; a
  thread's last posts linger for 15 min. Never the sentence.
- Topics: a hand-authored 60-topic keyword→odor map, published
  (`config/topics_v1.yaml`); up to three topics per post, each three
  neutral glomeruli on top of the account odor (decided 2026-09-14).

## Readout (frozen at tag)

- Behavior classes defined over **annotated DN populations**, never single
  neurons: engage (walking DNs), reply (courtship-song DNs), like (proboscis
  extension), leave (avoidance DNs), groom (grooming DNs → spontaneous post).
- Winner-take-all over population activity; silence if nothing crosses
  threshold. Thresholds from dev-period real activity distribution
  (provisional synthetic-battery thresholds until then).
- Action set: `reply`, `like`, `follow`, `leave`, `spontaneous_post`,
  `nothing`. engage → follow (reply if already following); leave → unfollow.
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
a person's wants, flat declaratives, no jokes (`docs/corpus-audit.md`). `textgen.py` is a word trigram with
absolute-discount backoff: the fly picks the corpus subset (tags), the
temperature (arousal) and the seed; the corpus never contains other
people's posts. Utterance policy: when a phrasebook line exists for the key
a seeded coin uses it verbatim half the time, otherwise the generator
speaks. Corpus and phrasebook digests are frozen artifacts.

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
- Nightly: ledger + weights rsync off-box. Weekly: Vultr snapshot.
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
