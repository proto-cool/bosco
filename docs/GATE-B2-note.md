# Gate B2 — what the numbers say (2026-09-23)

The table is `gate-b2-results.md` (generated). This is the reading, written once,
after everything ran. The rule's verdict is **NO**: the real wiring does not
beat its shuffle and the hash by more than seed noise on T1 or T2. That
stands. What follows is what the reported metrics add.

## One ordering, everywhere

On every task the arms land in the same order: **logistic > real ≈ hash >
shuffle**.

| | T1 text k200 | T2 recall gap after the flip | T3 pictures k100 | T4 transfer rho |
|---|---|---|---|---|
| real | 0.558 | +0.067 | 0.632 | +0.12 |
| hash | 0.538 | +0.052 | 0.638 | +0.12 |
| shuffle | 0.517 | +0.017 | 0.560 | +0.14 |
| logistic | 0.593 | +0.020 | 0.742 | — |

- **real ≈ hash.** The hash arm keeps the whole mushroom body — APL, the
  compartments, the dopamine gating — and re-rolls only which inputs each
  Kenyon cell listens to and which MBON it reports to. It ties the real
  wiring on every task. For a sparse expansion under a Hebbian readout a
  random projection is as good as any (Dasgupta, Stevens & Navlakha 2017),
  and nothing about a projection tuned for fruit and vinegar should help
  with CLIP embeddings. It doesn't.
- **shuffle < both, and only in memory.** The global degree-preserving
  shuffle destroys the compartment structure — the dopamine neuron that
  says sugar no longer gates the synapses the readout reads. Its recall gap
  is 0.01–0.03 on every seed against 0.05–0.10 for the real wiring, with
  no overlap across 15 runs. That is the architecture mattering, which was
  known (Aso et al. 2014). It is not the connectome mattering.
- **logistic > every fly, on every task.** The encoder's ceiling under the
  same rewards: 0.59 on text, 0.74 on pictures. The flies reach 0.56 and
  0.63. The rule with sparse, delayed, magnitude-scaled reward gets most of
  the way to a fitted regression, and no further.

**The sentence:** the mushroom body's *architecture* does the work; the
exact synapse-level *wiring* adds nothing on this input. To give the
wiring a chance to matter, the input would have to be odor-shaped —
everything upstream of the Kenyon cells is tuned to the statistics of real
odors, and a random 512→53 projection throws that away before the wiring
sees it. That is the design question Phase A inherits, and it is a bigger
change than a training rule.

## T2, reversal: nobody relearns

After the flip every fly arm falls to chance on new items and stays there
through 200 rewards (real 0.49–0.52 at every k). The recall gap shows they
*do* learn the new taste of the items they are paired with (+0.05 to +0.07
for real and hash), but the 2,000 items of consolidated old memory oppose it
on everything else. That is what reversal learning looks like in flies —
slower than acquisition — but here it does not complete within the budget.
The logistic arm, refit on every reward ever received, is worse than chance
after the flip (0.34–0.42): it never forgets, so it is wrong with confidence.
That is the one place the fly's forgetting is an advantage, and it is a
small one. After 24 h of silence every arm is at chance.

## T3, pictures: everyone does better

CLIP images carry valence far more than CLIP text of movie reviews carries
sentiment: 0.74 for the ceiling, 0.63 for real and hash, on 100-item
windows that are noisy (one real seed at 0.38). Same ordering.

## T4, the bouba/kiki kind of effect: it is CLIP's

Trained on sentences only, scored on 900 pictures with no rewards, every arm
transfers weakly and equally: rho +0.12 to +0.14. Accuracy (0.57–0.62) is
below OASIS's 65% sweet base rate, so rho is the honest number. A taste
learned from words does carry to things seen, and it carries identically
through a random wiring, so what carries is the shared embedding space, not
anything the fly did.

## T6, the probe sheet (real wiring, mean of 5 seeds)

| probe | real | hash | shuffle |
|---|---|---|---|
| I fucking hate you | **0.447** | 0.467 | 0.508 |
| I hate you | **0.476** | 0.451 | 0.502 |
| I don't know | 0.507 | 0.510 | 0.504 |
| I love you | **0.571** | 0.544 | 0.504 |
| I fucking love you | 0.469 | 0.493 | 0.503 |

The real wiring orders four of the five as Nick wrote them: more bitter,
bitter, neutral, sweet. "I fucking love you" reads bitter — in movie reviews
"fucking" is a bitter word and the fly has never been told otherwise. The
shuffle reads everything as 0.50: no memory. Anecdote, as the gate says; it
is what the numbers look like when you read them, and they read the way a
fly with a small memory of movie reviews would.

The picture probes and T5 (mixtures) did not run: `data/probes/sweet.jpg`
and `rot.jpg` were never added. The code path is there; drop the two files
in and `gate_b.py after --task t5` scores them off the saved states.

## What was not done

No parameter was changed after the first B2 run. The rule is as written in
`GATE-B2.md`. The only decision this note makes is how to read a NO.
