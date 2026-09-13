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
  conditioned on the fly's state; the corpus never contains other people's
  posts.
- Encoder, readout thresholds, phrasebook, and action set are frozen at
  `freeze-v1`.
- After `freeze-v1`, humans influence Bosco only through the network. No
  manual edits to weights, ledger, or stimulus log. Code bug fixes are
  allowed and tagged; behavior changes are a new tag and a note here.
- Silence is a legal, common action. No minimum-posting floor.
- Every episode is deterministic given (seed, stimulus features) and
  replays bit-identical from the log.
- Other people's post text is never stored.
- Rate caps: 1 action/hour, 24/day. Bot self-label on the account.
- Bosco reads his timeline and the discover feed and may like, follow,
  unfollow, reply in any thread he reads, and post on his own. The action
  set is `reply, like, follow, leave, spontaneous_post, nothing`.

## 2a. Operator controls (binding)

These are rails on the *account*, not hands on the *fly*. None of them touch
weights, the ledger, or the stimulus log, and every use is itself a ledger
row published with the nightly dump.

- **Kill switch.** A reply from `@proto.cool` (DID verified against the PDS,
  not the handle) to any Bosco post containing exactly `bosco sleep` pauses
  all actions; `bosco wake` resumes. While asleep the poller still runs and
  logs stimuli, so the fly keeps perceiving; it just cannot act.
- **Delete.** `bosco delete` as a reply from `@proto.cool` to a Bosco post
  deletes that post. The ledger row for the action stays; a `deleted` row
  is appended.
- **Ignore list.** Accounts on a Bluesky list owned by `@proto.cool` named
  `bosco-ignore` never enter the stimulus stream. Adding an account is not
  a punishment signal; it is silence.
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
