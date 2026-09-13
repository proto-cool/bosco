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
- Every episode is deterministic given (seed, stimulus features) and
  replays bit-identical from the log.
- Other people's post text is never stored.
- Rate caps by kind (`config/caps_v1.yaml`): replies 12/h 100/d, likes
  12/h 100/d, follows and unfollows 6/h 40/d, own posts 1/h 12/d, all
  actions 24/h 240/d; at most 6 replies per thread per hour, 20 replies and
  30 actions toward one account per day. These are loop guards, not a
  schedule. Bot self-label on the account.
- When he posts or replies is the network's decision. Replies follow
  events. Own posts follow an internal drive on bristle afferents that
  accrues between polls and is cleared by grooming (`config/encoder_v1.yaml`
  `spontaneous`). There is no timer. The VPS bounds perception, not
  behaviour: at most `BOSCO_EPISODE_BUDGET` episodes per hour.
- Bosco reads his timeline and the discover feed and may like, follow,
  unfollow, reply in any thread he reads, and post on his own. The action
  set is `reply, like, follow, leave, spontaneous_post, nothing`.

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

- Replay determinism fails on any episode.
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
- One connectome, one individual, one sex.

## 8. Timeline

- Dev period: first post → `freeze-v1`. Bio says "in development."
- After tag: runs indefinitely. dunce live weeks 3–4.
- Monthly note thereafter, for as long as the account exists.
