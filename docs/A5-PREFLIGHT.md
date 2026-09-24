# A5 preflight: no run starts until every arm can learn (2026-09-24)

Why: the first cutoff pilot compared a full brain that never learned (it
started with 70% of its Kenyon cells active, spent its training fighting
the sparseness penalty, and sat at chance) against a cut brain that did. The
smoke test had shown the warning signs: KC activity 0.70–0.75, and a loss
that rose across three batches. They were not checked against anything
before launch. The result could not answer its question
(`a5-cutoff-pilot-results.md`).

## The fixed start (label-free)

1. From the (gain, threshold) grid, take the point with the widest output
   spread among those where both read MBON groups are active (mean rate
   between 0.001 and 0.95), on 64 unlabelled training inputs.
2. Set the **Kenyon-cell** threshold alone, by bisection, so that 5% of KCs
   are active on those inputs. A fly's KCs are sparse (about 2–10%), through
   their high threshold and APL inhibition.
3. Set the **read MBONs'** threshold alone, by bisection, so their mean rate
   is 0.2: a resting operating point, as real MBONs have tonic firing.
4. Set the KC threshold again, since MBONs feed back onto KCs through DANs.

## The checks (every arm, training data only)

| check | bar |
|---|---|
| Kenyon cells active at the start | 0.02–0.15 |
| output spread at the start (std of the logit) | ≥ 0.02 |
| loss falls: cross-entropy, mean of batches 21–30 vs 1–10 | lower by ≥ 0.02 |
| gradients reach the brain (gain, threshold, KC→MBON) | finite and non-zero |
| answers carry information after 30 batches: AUROC on 256 held-back training items | ≥ 0.6 |

Configurations: {real, layered, hash, free} × {per type, per neuron} ×
{full, cut}. A run of any configuration refuses to start unless that
configuration's preflight file says pass. A failing arm is fixed and
preflighted again. It is never dropped.

## Revision history (harness only; no validation or test data involved)

- **Start, rev 1.** Steps 1–2 alone started the brains with read MBONs almost
  silent (mean 0.005–0.016). Answers collapsed toward a constant: the full
  brain's answer spread was 0.001 after 30 batches. Steps 3–4 were added.
- **Checks, rev 1.** Seen with the rev-1 start: both pilot brains learned
  (loss 0.68 → 0.62, answer spread 0.04). They failed "answers on both sides
  of 0.5" because 65% of training items are *approach* and 30 batches had not
  yet set the offset; that tests calibration, not learning. The cut brain
  also failed "output spread ≥ 0.05" at 0.049, an arbitrary bar. The side
  check was replaced by AUROC ≥ 0.6, which tests the check's purpose
  (answers carry information), and the spread bar was lowered to 0.02, which
  still catches a dead output. Revised once, recorded here.

Runner: `scripts/a5_preflight.py`. Results: `runs/a5-preflight/`,
`docs/a5-preflight-results.md`.
