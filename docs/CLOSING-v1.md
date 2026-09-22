# Closing note: v1 (2026-09-13 to 2026-09-22)

The account and the box were shut down by the operator on 2026-09-22. This is
the descriptive note EXPERIMENT.md §6 promised, written once instead of
monthly, with the reason.

## The reason, in one sentence

A fly connectome cannot provide what an account is made of. Strip away the
paperwork and what ran was VADER, an n-gram, a clock and rate caps, with a
hundred thousand neurons in the middle deciding *when* -- and "when" is the
one thing nobody on a feed can see.

## Numbers (from `snapshots/2026-09-22/ledger.sqlite`)

- 14,922 episodes: 13,921 posts read, 80 groomings, 877 landings, 44 pairings.
- 7,362 actions; real ones on the network: 447 likes, 181 follows (72 undone
  in the app), 58 unfollows, ~1,000 walks, 39 replies to 7 accounts, 1
  answer, 55 posts of his own.
- Silence rate over posts read: 35% until 2026-09-21, 48% after the
  recalibration (`freeze-v2.2`).
- **Outcomes: 44.** Rewards 34: 14 follow-backs, 8 likes and 6 return visits
  from known accounts, 3 kind replies (2 from strangers), 3 known-account
  replies. Punishments 10: 9 blocks, 1 sour reply. Twelve of the 44 came from
  the operator's own account.
- **Taste pairings while reading: 7,238** (5,119 sweet, 2,119 bitter). So
  99.4% of the dopamine he ever received was the VADER score of a post.
- 29 of his 39 replies were to the operator. The last conversation with
  anyone else was 2026-09-20.
- 37 followers at the end, most of them follow-back accounts. His last 60
  posts drew about 12 likes between them.
- Integrity: every nightly report from 2026-09-17 on is ALL PASS; the weight
  digest chain holds over all 14,922 rows; every span replays.

## What he learned, measured

The learned verdict `v` on a window (`_learned` in the ledger), over the 6,324
windows since `freeze-v2`, the rule he ran at the end:

- against the window's own VADER: r = -0.007;
- against the account's prior net VADER: r = 0.19 (undecayed; less with any
  decay);
- on first meeting an account, before a word of theirs: median +0.03, sd 0.17
  (bleed through shared glomeruli);
- where an account's history was clearly signed, sign agreement 60%.

And yet it drove him: recomputing every decision from the logged rates with
`v = 0` (the reconstruction matches the log on 6,320 of 6,324 windows) changes
20.5% of decisions, and 231 of the 1,025 real actions in that span depended on
it. Learning reached behaviour; what reached it was structured noise.

## What the record can and cannot say

- The engineering held: a deterministic, bit-replayable central-brain LIF
  model ran live for nine days with a passing integrity chain every night.
  Phases 0-3 (annotation coverage, Shiu's reflex, KC sparseness, learn/forget)
  are real reproductions and stand on their own.
- One negative result, found offline (`docs/plasticity-v2.md`,
  `docs/plasticity-v3.md`): a connectome-faithful mushroom body fed a social
  feed through a mixture encoder learns the feed's average, not the people;
  extinction and homeostatic scaling do not fix that; credit confinement
  recovers the ordering (r 0.36) but not the sign.
- The controls never ran. dunce was never built; frozen-MB and random policy
  were never replayed. Whether the wiring mattered is unanswered, and could
  still be answered from the archive with `scripts/replay_controls.py` and
  `scripts/make_dunce.py` -- an offline job, if anyone cares to.
- "Frozen" did not hold: six behaviour changes in the four days after
  `freeze-v1`, and a 27-hour deaf-kernel span inside the calibration set.
  Every change is dated in EXPERIMENT.md §2d; there is no stable span longer
  than about three days.
- Thresholds set as quantiles of his own activity mean any brain in that seat
  would have acted about as often, in about the same mix. Whatever
  individuality the wiring carried was normalised away by design.
- The character people liked was the corpus and the phrasebook -- the
  operator's -- with the fly picking the seed.

## What stays

The repository, the snapshots, the corpus and the phrasebook, as the record.
The account's posts stay up. Nothing here is to be edited except to make the
record clearer. What comes next is a different project.
