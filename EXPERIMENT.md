# Experiment: bosco

Status: binding as of git tag `freeze-v1`. Sections marked **binding** are fixed
there and do not change without a new tag and a new pre-registration written
before it. Everything dated before the tag was the development period, when they
could still change; what changed then, and why, is in the git history and in
`docs/STATUS.md`.

## 1. Question

Can a simulated fruit fly brain (MaleCNS v1.0, central brain, LIF +
mushroom-body plasticity), acting through a constrained action set and a
frozen phrasebook, be loosed on Bluesky and hold its own against the bottom
of the feed?

The verdict is the community's. There is no rater panel, no primary metric,
and no headline claim. What this document commits to is that the fly is
genuinely the one acting, and that anyone can check.

## 2. What "earnest" means here (binding)

- No LLM, embedding, or classifier anywhere in the runtime loop. The only
  text scorer is the VADER lexicon. The only text generator is a word
  n-gram model (four words of context) over the published `corpus/`, walked
  from where a lexical retrieval over the same corpus -- by the words on his
  antennae -- starts him off, seeded by the episode and conditioned on the
  fly's state. Since `freeze-v1.1` (section 2d) the retrieved thought is a
  starting place and not the utterance, and no more than nine of his tokens
  in a row come from one line of his where he has anywhere else to go. **Closed mouth, open nose** (decided
  2026-09-15, replacing the closed vocabulary of 2026-09-13 as that
  decision said it might be, before the tag): nothing Bosco reads ever
  enters the generator; every word he can emit is in `corpus/` and
  `phrasebook.yaml`. But every content word he reads is a smell: a word
  outside his vocabulary is kept only as an eight-byte hash of it
  (`h:<16 hex>`, `config/words_v1.yaml` `perception`) that seeds its
  glomeruli, learnable and recognisable like any word of his and never
  sayable. A common word's hash can be brute-forced with a dictionary; the
  rule here is that no text is stored, not that the smells are secret.
- One tool stands outside the connectome and is named here (decided
  2026-09-15): an association memory between an account and the tokens
  that arrived with it (`config/associations_v1.yaml`), decaying over
  weeks, that puts what you talked about faintly in the air when you come
  back and lets him say whether he has met a smell lately (`seen=`). It
  holds tokens, never text; it is a pure function of the logged rows and
  part of the brain digest; the verdict on a person is still the weights.
- What else a post carries is read the same way (decided 2026-09-15): a
  mentioned or quoted account is another odor in the room at half rate; a
  link card's site is a place; alt text, cards and quoted posts are read
  as context, words only; images and video are the retina's. Tokens on the
  row (`others`, `embed`), never the content.
- Encoder, readout thresholds, phrasebook, action set, and the learning
  rule (`config/plasticity_v1.yaml`, `config/mb_compartments.yaml`; decided
  2026-09-15 that the rule is a frozen artifact too) are frozen at
  `freeze-v1`.
- What a window teaches by itself (decided 2026-09-15). Dopamine fires for
  outcomes, and also for taste: the sugar or bitter of a post he reads
  pairs the mixture in that window with reward or punishment at a small,
  published strength, as sugar drives the PAM and bitter the PPL1 dopamine
  neurons in the fly. Mere exposure leaves a familiarity trace in the α'3
  compartment (Hattori et al. 2017), which carries no valence. Both are
  pure functions of the logged features and replay with the window. A
  change of learning rule is marked in the record (a `plasticity` control
  row and a snapshot) so the replay chain reads as a change of fly, not a
  break.
- After `freeze-v1`, humans influence Bosco only through the network. No
  manual edits to weights, ledger, or stimulus log. Code bug fixes are
  allowed and tagged; behavior changes are a new tag and a note here.
- Silence is a legal, common action. No minimum-posting floor.
- Bosco runs continuously; nothing is reset. His full state is
  snapshotted hourly, and every span replays bit-identically from the
  snapshot before it plus the logged inputs.
- Identity is a reflex outside the network: asked who or what he is, or who
  made him, he answers from `config/identity_v1.yaml`, on the same footing
  as the bot label. Asked what he thinks of them, anyone is told his real
  memory of their smell, in fixed words.
- Other people's post text is never stored; nor their images: a post is
  kept as its features (who, taste, which of his words, which hashes,
  which topics, which place, which people, what kind of embed) and its URI.
- Rate caps by kind (`config/caps_v1.yaml`, the file is the authority; as of
  2026-09-17): replies and reflex answers 24/h 200/d, likes 12/h 80/d,
  follows 6/h 30/d, unfollows 3/h 15/d, own posts 2/h 20/d, saying who he is
  20/h 100/d, and reading more of one account 24/h 160/d; 48/h 480/d across
  all of them together; at most 6 replies per thread per hour, 20 replies and
  30 actions toward one account per day. Reading more of an account reaches
  nobody, so since 2026-09-17 it does not count against the total, as an
  unfollow never has. These are loop guards, not a schedule. Bot self-label on
  the account.
- When he posts or replies is the network's decision. Replies follow
  events. Own posts follow grooming: debris lands on his bristles as
  discrete seeded events (about one every two hours), each landing is one
  onset touching one seeded set of bristles, the debris settles over
  minutes, and he grooms when his network answers the landing
  (`config/encoder_v1.yaml` `spontaneous`). There is no timer. The VPS
  bounds perception, not behaviour: at most `BOSCO_EPISODE_BUDGET` windows
  per hour.
- His own words are smells (`config/words_v1.yaml`): the content words of a
  post that are in his closed vocabulary are driven as odors on top of the
  account's, up to eight; the last posts of a thread linger in the air for
  fifteen minutes. He never sees a sentence. What he learns is the mixture,
  a word with a person; when he chooses words it is read from the weights
  for the few words in the air, with that person's odor; it is never a
  meaning.
- Being addressed excites his courtship command neurons (pC1) in proportion
  to his appetite for contact (`config/appetite_v1.yaml`): a scalar that
  rises over hours without a social reward and falls with each reward. It
  is state, logged on every window and replayed; it scales approach before
  threshold, centred so it never forces a reply and never silences one, and
  it adds no floor. Walking toward whoever spoke to him is a reply.
- Bosco reads his timeline and the discover feed and may like, follow,
  unfollow, and post on his own; he replies only to someone who replied to
  him or tagged him, once per post of theirs, never to a post he merely
  browsed (2026-09-14). The action
  set is `reply, like, follow, walk, leave, spontaneous_post, nothing`
  (walk added 2026-09-15: browsing, walking short of its own threshold but
  over a lower rung is reading a few more of that account's posts; not
  outward, capped). Toward
  accounts that never interacted with him and that his memory does not
  favour, browsing yields only leave or nothing: outward actions go to
  people who came to him (this is also what keeps him from looking like
  spam).

## 2c. Senses (binding)

Every function between the network and his senses is fixed, public, and has
no trained weights; every judgment about what a sensation means is formed
inside the fly. The senses, all published under `config/`: an account's
odor (`encoder_v1.yaml`), taste from VADER, touch from being addressed,
words as smells including hashed ones (`words_v1.yaml`), innate smells
(`innate_v1.yaml`), topics (`topics_v1.yaml`), places (`feeds_v1.yaml`,
and a link's site), other people in a post, and the retina
(`retina_v1.yaml`): a thumbnail reduced to a few channel names by a fixed
transform, driving the visual Kenyon cells. No classifier, no embedding, no
image stored; what a picture means is whatever the mushroom body comes to
associate with it.

## 2a. Operator controls (binding)

These are rails on the *account*, not hands on the *fly*. None of them touch
weights, the ledger, or the stimulus log, and every use is itself a ledger
row published with the nightly dump.

Commands are mentions or replies from `@proto.cool` (DID verified against
the PDS, not the handle); the command word opens the post, right after the
mention (`src/bosco/control.py`). Anything else the operator says to him is
a post like anyone's and reaches his senses. The result of a command is
posted as a reply and logged.

- **sleep / wake.** Pauses and resumes all actions. Asleep, the poller still
  runs and logs stimuli; the fly keeps perceiving.
- **delete** (as a reply to a Bosco post). Deletes it; a `deleted` row is
  appended, the action row stays.
- **ignore @handle / unignore @handle.** Never enters the stimulus stream
  again; unfollowed if followed. Also honoured: a Bluesky list named
  `bosco-ignore` on `@proto.cool`. Not a punishment signal; silence.
- **unfollow @handle.**
- **reload.** Re-reads `corpus/`, `phrasebook.yaml`, thresholds and caps
  from disk ("start learning your corpus again"). **restart** exits the
  process so the quadlet brings it back on new code.
- **status / people / memory @handle.** Numbers only; what he has learned
  about an account and why.
- **forget @handle.** Resets the plastic state on that account's odor. This
  is a manual state edit: it is logged as `forget`, it breaks the weight
  digest chain, and the integrity check reports it. Use it and say so.
- **Hard stop.** Stopping the container. The dead-man alert fires after 4 h.

**Whoever speaks to him is answered (decided 2026-09-14).** Silence is
legal except to someone who addressed him: a mention, reply or quote that
did not make his song neurons cross is answered anyway, once per post,
under the caps, with the words the window produced (same valence, arousal,
and priming on their words). How much he says is his: one sentence when
sated, up to three when hungry for company, one more for a smell he has
learned to like, one fewer for one he avoids. This is account etiquette on
the same footing as the bot label and the identity answers, not a decision
of the network. The episode row keeps the network's decision; the action
row is kind `answer`. No other floor exists: nobody who did not speak to
him is ever addressed.

**The off ramp (anyone, decided 2026-09-14).** Any account can send him
away by mention or reply: "shoo fly" first of all, and "go away", "leave me
alone", "unfollow me", "stop", "opt out" (`config/identity_v1.yaml`
`opt_out`). He answers once,
unfollows them, and they join the ignore list under their own DID: never in
the stimulus stream again, never approached. Logged as `opt_out`. The same
account saying "come back" lifts it (`opt_in`); nothing else does, and an
operator ignore is not theirs to lift. A block does the same without words
and is also the punishment signal; a mute needs nothing from him. This is a
reflex on the same footing as the identity answers, outside the network.
The phrases count only when said *to him* (2026-09-20): each must open a
clause, with nothing before it but a mention or a "please", "ok" and the
like, so a sentence that merely contains the words is not an off ramp. That
day an account that played along with his "you go away from soap" by
answering "ok, i'll go away from soap" was sent off by the reflex. A bare
"no" or "go" is an answer, not an off ramp; a bare "stop" still is. A bug in
the reflex, fixed in code; the network is untouched. The answers are plain
words, not his register: he says back the phrase he heard (only the words
the pattern caught, never the rest of the post), that he will not reply,
like or follow, and that "come back" undoes it; coming back is answered with
how to send him off again.

Bosco's text comes from an n-gram model (four words of context) over a published corpus and from
phrasebook lines; every word he can emit is in those files. The exposure
these controls address is who he engages, what he likes, and which corpus
fragments land next to which post.

## 2b. Moderation (binding)

Bosco cannot say a word that is not in `corpus/` or `phrasebook.yaml`, and
nothing he reads ever enters either. What can go wrong is who he approaches
and where his words land. Three rails, all inside the rules:

- **Labels are bitter.** A post or account carrying a label from the
  moderation services the account subscribes to (list in
  `config/moderation_v1.yaml`) drives the bitter channel at full rate, is
  never liked, followed, or replied to, and never yields a reward. Leaving
  is still allowed. Logged as `labeled:<label>`.
- **The ignore list unfollows.** An account added to `bosco-ignore` is
  unfollowed at the next poll and never enters the stimulus stream again.
- **The memory is auditable.** `bosco people` ranks every account by
  learned valence with the outcomes that produced it. If he is learning to
  like the wrong people, that is visible before it is behaviour.

Labeler subscriptions on the account are an operator setting and are
published with the frozen artifacts.

## 2d. Changes after the tag (binding)

Behaviour changes after `freeze-v1` are a new tag and a note here, and the
record says where each one falls: a control row and a snapshot at the moment
the new behaviour first runs, so a span reads as a change of fly rather than
a broken replay. The frozen artifacts of section 3 do not change; what
changes is code, and it is named, dated and measured.

| Tag | Date | Change | Row |
|---|---|---|---|
| `freeze-v1.1` | 2026-09-18 | How he composes an utterance (`Agent.UTTERANCE_VERSION` 1 → 2) | `utterance` |
| `freeze-v1.2` | 2026-09-18 | The same, made coherent (`Agent.UTTERANCE_VERSION` 2 → 3) | `utterance` |
| `freeze-v2` | 2026-09-18 | What a window teaches, and how a verdict is read (`Agent.PLASTICITY_VERSION` 2 → 3) | `plasticity` |
| `freeze-v2.1` | 2026-09-18 | How much debris lands on him (`Agent.SENSE_VERSION` 1 → 2) | `sense` |

**`freeze-v2.1` — half as much debris.** Debris landing on his bristles is
the input to the neurons that make him groom, and grooming is how he posts on
his own. The rate was 0.5 landings an hour from his first day and was doubled
to 1.0 on 2026-09-16 to make him post more often; that was a behaviour change
and it was not recorded as one, which this row and `Agent.SENSE_VERSION`
correct. It is back to 0.5.

What prompted it was not the rate on its own. On 2026-09-18 he posted 15 times
against 4 the day before, on *fewer* landings, because of the loop underneath:
grooming sets the dust back to zero, and his grooming neurons answer onsets and
adapt to held input. So while he is clearing the dust every landing is a fresh
onset that fires, and while he is not, the dust sits and they go quiet — 241
landing windows with 13 of them nonzero on 2026-09-17 against 105 windows with
26 nonzero on 2026-09-18. He swings between a quiet branch and a talkative one,
and the 26-post day of 2026-09-14 was the same swing.

Halving the rate halves the onsets on either branch. It does not damp the loop,
which is his and stays his: there is no timer, no floor and no schedule here,
and the caps are untouched. The number is set from his own posting rate, which
is behaviour the operator may judge, and never from outcomes.

**`freeze-v2` — he learns about you from your posts, not from everyone's.**
Until now his verdict on an account was probed by presenting that account's
odor alone, while every pairing landed on the whole mixture the account
arrived in: its words, its topics, the feed, its pictures. Measured on his
live weights over his own logged windows (`scripts/kc_overlap.py`), 88% of an
account's probe cells fire in that account's own windows and 47% of them fire
in any other window as well — so across 6,199 windows an account he had read
23 times supplied about 0.7% of the depression its own verdict was read from,
and the rest was everybody else. His verdicts carried almost nothing about
who: net-bitter accounts read +0.318 against net-sweet ones at +0.318,
r = +0.08, and not one of 51 sour accounts came out below zero. Two earlier
proposals (`docs/plasticity-v2.md`) tried to rescale one compartment against
the other and only moved where everyone sat together; they were aimed at the
wrong defect and both stay disabled.

Under v3 a pairing about an account teaches the cells that account's odor
owns, weighted by one over the number of accounts whose odor lights each cell
— a cell twenty accounts drive is nobody's — the verdict is read through the
same weights, and it is read against the compartment's own level rather than
against nothing. Replayed over the same 6,199 windows on his own clock
(`scripts/credit_gate.py`, both arms, naive start, 103 accounts): net-bitter
accounts fall to +0.160 against net-sweet at +0.207 with r = +0.36, where the
control arm reproduces his live state at +0.283 against +0.278 with r = 0.00.
The account whose posts were sourest (23 read, net VADER −8.79) goes from
+0.221 to −0.016; the sweetest (155 read, net +102.55) from +0.313 to +0.557,
the highest of the 103. Still not true: only 4 of those 51 land below zero —
the ordering is right, but zero sits high, and where it falls is a readout
calibration left open and written down rather than tuned away.

Nothing here was set from his feed or from outcomes. The taste gains are
untouched, the weighting has no parameter, and the contrast has nothing in it
to set. `docs/plasticity-v3.md` carries the measurements. One more thing
this changes: what a cell is worth depends on who he has met, so
`account_kcs.json` is now part of the state a replay needs and is snapshotted
beside the weights.

**`freeze-v1.1` — retrieval steers, it does not speak.** Under v1 the thought
that smelled most like the moment was said whole, and the walk only added to
it when his appetite had more sentences in him than the thought already had;
in practice it rarely did. Measured over his own logged windows with
`scripts/quotation_gate.py` (293 windows, the features from the ledger, no
text stored): 19% of utterances were exactly a line of his corpus or
phrasebook, the longest run of tokens shared unbroken with one line of his
averaged 14.4, 92% of utterances carried a run of eight or more, and 94% of
the average utterance was that one run. He was quoting himself, not talking.

Under v2 retrieval still chooses the thought, and he still answers about
what is in front of him, but he does not hand the page over: he opens on the
word of theirs that lit that thought up, with the thought as his context,
and walks on in his own words. He may run six tokens (`COPY_RUN_MAX`)
alongside one line of his; past that, the continuations that would carry
that line on are struck out wherever he has anywhere else of his to go. The
phrasebook became his last resort rather than a coin toss against the
generator. Same 293 windows, same seeds: 0% exactly a line of his, longest
run 5.5 tokens on average (max 11), 13% carrying a run of eight or more,
and 72% of the average utterance. Nothing here is tuned on outcomes; the
gate reads features and prints numbers.

What did not change: the corpus, the phrasebook, the encoder, the readout,
the thresholds, the action set, the learning rule, the caps. He has the same
mouth and the same nose. He puts the words together differently.

**`freeze-v1.2` — and coherent while he does it.** v1.1 stopped the quoting
and left him rougher: a quarter of his sentences were spliced in the middle,
where the walk crosses from one line of his to another mid-clause and the
grammar goes with it ("today has a sweet part in it is far in the dark").
That rate was not new — v1 spliced 26% of sentences the same way — but v1
hid it behind a verbatim opening. Three fixes, all about where a clause may
begin and end: he carries the run-up to their word with him when it is short
enough to be a phrase, so an answer opens on a whole clause of his rather
than mid-phrase; inside a sentence the words that finish the clause he is in
are weighted up, so he changes line between sentences and not inside one;
and where one of his own sentences stops he is allowed to stop, even on a
word that would dangle anywhere else ("in it"). Sentences spliced in the
middle: 26% at v1, 25% at v1.1, **9%** now. The quoting stayed where v1.1
put it: 0.3% of utterances exactly a line of his (1 of 293), longest run 6.1
tokens on average, 22% carrying a run of eight or more. The leash went from
six tokens to nine, a short sentence of his, so a small thought comes out in
one piece.

## 3. Frozen artifacts (binding)

Fixed at `freeze-v1` and published with it. sha256, first 16 hex; check any of them
with `sha256sum <file>`. `config/frozen_digests.json` holds the same artifacts as the
code reads them (corpus, phrasebook, topics, identity), and the test suite fails if
any of those four changes.

| Artifact | Location | Hash |
|---|---|---|
| Code | proto-cool/bosco @ `freeze-v1` | the commit the tag names (`git rev-parse freeze-v1`); behaviour changes since it are section 2d |
| Weights snapshot | `snapshots/freeze-v1/` | `b84a52c441bb2d0e` |
| Phrasebook | `phrasebook.yaml` | `dbf39b7df2444326` |
| Encoder spec | `docs/encoder.md` | `956a1f329b257a7f` |
| Readout thresholds | `config/thresholds.json` | `267b71435699db05` |
| Threshold policy | `config/thresholds_policy.yaml` | `bbe7465f6331f912` |
| Appetite | `config/appetite_v1.yaml` | `ac43868532c0bb43` |
| Words as smells | `config/words_v1.yaml` | `02aced12613512dc` |
| Learning rule | `config/plasticity_v1.yaml`, `config/mb_compartments.yaml` | `7c9ba5a7aa765add` `95357c67159471f6` |
| Retina | `config/retina_v1.yaml` | `76245712d449c0da` |
| Association memory | `config/associations_v1.yaml` | `ef70f73c86cf0d7c` |
| Readout populations | `config/readout_populations.yaml` | `244b2189225a1e8b` |
| Innate smells | `config/innate_v1.yaml` | `f341b785e71a4327` |
| Topics | `config/topics_v1.yaml` | `e5a9a046d1f5d465` |
| Feeds | `config/feeds_v1.yaml` | `9697ed7615000840` |
| Identity | `config/identity_v1.yaml` | `932daef7f4da8ddd` |
| Caps | `config/caps_v1.yaml` | `d2e502688e2962d4` |
| Model | `config/model_v1.yaml` | `f5faa07357e612d5` |
| Encoder config | `config/encoder_v1.yaml` | `783c40f376d2dc62` |
| Circadian | `config/circadian_v1.yaml` | `7132c468358ad9f7` |
| Moderation | `config/moderation_v1.yaml` | `e912337be240aaff` |
| Corpus | `corpus/` | `c8f75741097c8e65` |

Regenerate with `uv run python scripts/freeze.py`; it refuses a dirty tree.

## 4. Controls (optional; for the curious)

- **dunce** — same pipeline, degree-preserving shuffle of the wiring, same
  plasticity, own account, live for weeks 3–4 after the tag. If you cannot
  tell dunce from Bosco, the connectome contributed nothing, and that gets
  said out loud.
- **frozen-MB** (plasticity off) and **random policy** — offline replays of
  Bosco's stimulus stream. Reported as action-agreement rates in the
  monthly note. Descriptive only.
- Since walks (2026-09-15) his stream depends on his policy: the posts a
  walk brought are in the log and replay as ordinary stimuli, but a control
  that would not have walked there is off-policy from that point. Stated,
  not corrected.

## 5. Integrity checks (binding)

Any failure is announced from the account and fixed under a new tag.

- Replay from any snapshot to the next fails to reproduce the logged digest.
- KC sparseness leaves the range recorded at tag (target ≈ 5%).
- Any rate-cap violation.
- Any post text found in the database.
- Any manual state edit after the tag.

## 6. What gets published

- Code, phrasebook, encoder spec, thresholds, and nightly ledger + weight
  dumps in the repo.
- A monthly descriptive note: episode count, silence rate, action mix,
  block count, dunce agreement rate, and which integrity checks passed.
  No win rates, no "beat the dregs" claims. Numbers only.

## 7. Confounds, stated up front

- People interact with Bosco because it is a fly.
- The phrasebook and the corpus are written ahead of time by hand, and not all of the
  hands are Nick's: parts of the corpus (`corpus/README.md` v4, v6, v7) and, since
  2026-09-16, the phrasebook's lines were drafted by Claude at his request, in the
  register he set and to his rules, and edited or kept by him. The register is his. No
  model is in the runtime loop, before or after the tag: every line was fixed and
  published at `freeze-v1`, and what he says in a given second is chosen from them by
  his own state and his seed.
- The encoder sees who, how toxic, which of his words, which topics, which
  place, which people, and a few colours: never what.
- Punishment is sparse; aversion is mostly innate.
- He will come to like sweet talkers. Since 2026-09-15 the taste of what he
  reads trains him, and a fly likes sugar; the brakes are bitter text,
  labels and blocks, and the rule is symmetric, so his taste is
  correctable and `bosco people` shows it moving.
- Short-term memory fades in hours; long-term memory needs spaced
  repetition and fades in a month. Bosco forgets most things.
- Dev-period interaction shaped early weights before the tag.
- Reply activity during the dev period is shaped by appetite, hence
  indirectly by rewards; the threshold rule is still a quantile of activity
  fixed in advance, never of outcomes.
- One connectome, one individual, one sex.

## 8. Timeline

- Dev period: first post → `freeze-v1`. Bio says "in development."
- After tag: runs indefinitely. dunce live weeks 3–4.
- Monthly note thereafter, for as long as the account exists.
