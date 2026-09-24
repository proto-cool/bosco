# A3 phase 1 — can his nose carry Jev's tasks? (written 2026-09-24, before measuring)

Nick: a Jev-like decision engine should be benched on Jev's tasks. Jev's own
four workflow evals are not public. The independent, reproducible benchmark
is `manjunathshiva/jev-frontier-bench` (MIT): 200 items, 50 per task, rebuilt
locally with its `bench.py sample` (one line patched: ChaosNLI's
`label_counter` is now a JSON string); **all 200 match its published
manifest hashes.** Jev 1.13's logged answers on the same items come with it.
Local copy: `data/raw/jev-bench/` (not committed; dataset licences).

| task | Jev type | Jev 1.13 | Claude Fable 5.1 (best) |
|---|---|---|---|
| route: BANKING77 intent, 77 + "other" | choice | 76% | 88% |
| yesno: BoolQ, passage + question | yes/no | 94% | ~ties |
| rate: Yelp stars | score, 5 levels | 62% | up to 74% |
| ambig: ChaosNLI premise/hypothesis | choice, 3 | 58% | 80% |

Jev answers cold; **Bosco would be trained per task** on each source's
training split (items in the 200 excluded). Any comparison says so.

## The question

Bosco's nose turns an input into **one** vector (e5-large-v2), then 52
channels. Tasks that turn on how *two* texts relate (yesno: passage vs
question; ambig: premise vs hypothesis) may lose that relation in the
squeeze. Phase 1 measures, per task, the most any brain could get:

- **single:** one vector of the whole input (state + instructions), a
  logistic model on it — raw, and through the 52-channel antenna.
- **pair** (yesno, ambig only): the two pieces embedded separately, and a
  logistic model on [a, b, a·b, |a − b|] — raw, and on the antenna channels
  of each. This is what a brain could compute if it smelled both (two sniffs
  or split channels). It is the ceiling for any fix that keeps the decision
  in the brain.

Trained on: BANKING77 train (10,003); BoolQ train (9,427); Yelp Review Full
train, 2,000 per star, ≤ 1,500 characters; MultiNLI train, 20,000 sampled.
Scored on: the 50 bench items per task (the comparison with Jev), and on a
larger held-out set for stability (the rest of BANKING77 test; BoolQ
validation minus the bench items; 2,000 Yelp test; 2,000 MultiNLI
validation-matched minus ChaosNLI's items). Metrics as the bench: accuracy;
for rate also mean absolute error in stars; for ambig also Jensen–Shannon
divergence from the 100 human labels.

## Rules (fixed now)

1. A task goes into A3 if its best ceiling through the antenna, on the
   large held-out set, is **within 0.10 of Jev's score** on the bench.
   Otherwise the fly cannot reach Jev on it whatever the brain does, and it
   is reported and left out.
2. For yesno and ambig: if **pair** beats **single** through the antenna by
   ≥ 0.05, the two-sniff fix is worth a gate. If not, the relation is not
   recoverable from these vectors and no fix of ours brings it back.
3. Only e5-large-v2 (phase 1's nose). No new nose here.

Runner: `scripts/a3_phase1.py`. Results: `docs/a3-phase1-results.md`.
