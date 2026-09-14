# Experiment: bosco

Status: DRAFT. Sections marked **binding** lock at git tag `freeze-v1` and do
not change without a new tag. Everything before the tag is development.

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
  trigram model over the published `corpus/`, seeded by the episode and
  conditioned on the fly's state. **Closed vocabulary** (decided
  2026-09-13): nothing Bosco reads ever enters the generator; he learns
  people, not words. Letting him acquire words from accounts he has
  learned to like was considered and deferred; it would be a new
  pre-registration, decided before `freeze-v1` or not at all.
- Encoder, readout thresholds, phrasebook, and action set are frozen at
  `freeze-v1`.
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
- Other people's post text is never stored.
- Rate caps by kind (`config/caps_v1.yaml`): replies 12/h 100/d, likes
  12/h 100/d, follows and unfollows 3/h 15/d, own posts 1/h 6/d, all
  actions 24/h 240/d; at most 6 replies per thread per hour, 20 replies and
  30 actions toward one account per day. These are loop guards, not a
  schedule. Bot self-label on the account.
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
  unfollow, reply in any thread he reads, and post on his own. The action
  set is `reply, like, follow, leave, spontaneous_post, nothing`. Toward
  accounts that never interacted with him and that his memory does not
  favour, browsing yields only leave or nothing: outward actions go to
  people who came to him (this is also what keeps him from looking like
  spam).

## 2a. Operator controls (binding)

These are rails on the *account*, not hands on the *fly*. None of them touch
weights, the ledger, or the stimulus log, and every use is itself a ledger
row published with the nightly dump.

Commands are mentions or replies from `@proto.cool` (DID verified against
the PDS, not the handle); free text around the keyword is fine
(`src/bosco/control.py`). The result is posted as a reply and logged.

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

**The off ramp (anyone, decided 2026-09-14).** Any account can send him
away by mention or reply: "go away", "leave me alone", "unfollow me",
"stop", "opt out" (`config/identity_v1.yaml` `opt_out`). He answers once,
unfollows them, and they join the ignore list under their own DID: never in
the stimulus stream again, never approached. Logged as `opt_out`. The same
account saying "come back" lifts it (`opt_in`); nothing else does, and an
operator ignore is not theirs to lift. A block does the same without words
and is also the punishment signal; a mute needs nothing from him. This is a
reflex on the same footing as the identity answers, outside the network.

Bosco's text comes from a trigram model over a published corpus and from
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

## 3. Frozen artifacts (fill at tag)

| Artifact | Location | Hash |
|---|---|---|
| Code | proto-cool/bosco @ `freeze-v1` | |
| Weights snapshot | `snapshots/freeze-v1/` | |
| Phrasebook | `phrasebook.yaml` | |
| Encoder spec | `docs/encoder.md` | |
| Readout thresholds | `config/thresholds.json` | |
| Threshold policy | `config/thresholds_policy.yaml` | |
| Appetite | `config/appetite_v1.yaml` | |
| Words as smells | `config/words_v1.yaml` | |
| Corpus | `corpus/` | |

## 4. Controls (optional; for the curious)

- **dunce** — same pipeline, degree-preserving shuffle of the wiring, same
  plasticity, own account, live for weeks 3–4 after the tag. If you cannot
  tell dunce from Bosco, the connectome contributed nothing, and that gets
  said out loud.
- **frozen-MB** (plasticity off) and **random policy** — offline replays of
  Bosco's stimulus stream. Reported as action-agreement rates in the
  monthly note. Descriptive only.

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
- The phrasebook is human-written; its voice is Nick's.
- The encoder sees who and how toxic, not what.
- Punishment is sparse; aversion is mostly innate.
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
