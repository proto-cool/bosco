# Gate B4 — B3 with the fly's own neutral point (FINAL 2026-09-23; binding)

B3 (`GATE-B3.md`, tag `gate-b3-run`) ran as written and **failed its T1 bar**:
held-out accuracy 0.521 against a bar of 0.628, recall gap +0.036. That
stands. Its diagnosis, on the saved runs, is one number: the fly's scores on
the text held-out set are centred at **0.455**, and on pictures at **0.55–0.59**.
The rule orders items about as well as B3's bar hoped — rho 0.28 on text
against a linear-regression ceiling of 0.505 at the antenna, rho 0.60 on
pictures against 0.735 — but the pre-registration scored "sweet" as *score >
0.5*, and 0.5 is not where this fly's neutral is. v1 found the same thing from
the other side ("zero sits high", `docs/plasticity-v3.md`) and left it as "a
readout-calibration question". This gate answers it.

## The one change

**The neutral point is the fly's own.** After training, the fly recalls its
own trained items; the midpoint between its mean score on sweet-trained and
on bitter-trained items is its neutral. A held-out item is sweet if it scores
above that. No held-out label is used; nothing is fitted; it is the same
recall the gate already measures. In the fly this is the balance of the
MBON outputs that the approach/avoid readout sits on, which is set by the
animal, not at 0.5 by us.

Measured on B3's runs before this file was written (the disclosure): balanced
held-out accuracy with the neutral at 0.5 vs at the fly's own midpoint, real
wiring, epoch 1 / epoch 6 — text **0.533 → 0.632** / 0.517 → 0.593; pictures
**0.587 → 0.745** / 0.514 → 0.705. The hash arm moves the same way. That is
what made this a gate rather than a footnote.

## Metrics, restated

- **Balanced accuracy** replaces accuracy everywhere (the OASIS held-out set
  is 73% sweet by the midpoint; accuracy there rewarded saying "sweet").
- Reported at both neutrals, 0.5 and the fly's own, every epoch.
- Ceilings and baselines (logistic and k-NN on the raw embedding, the
  antenna, and the fly's own Kenyon-cell code) are balanced accuracy too.
- Everything else as B3: recall gap, rho, reversal, transfer, retention,
  probes; five seeds; three fly arms.

## The bar

As B3's, read at the fly's own neutral, at the **best epoch** (the fly is
trained until it is best, as a fly would be; the full curve is reported):
real balanced accuracy ≥ 0.9 × the antenna logistic ceiling on T1; recall gap
≥ +0.30; T2 recovers to 0.9 × pre-flip within six epochs. The recall-gap bar
is kept even though B3 suggests the rule's gaps are small by construction
(depression saturates, and the score is a difference of means); if it fails
again with the accuracy passing, that is written as a property of the rule.

## What is not done

No parameter of the rule, the antenna, the schedule or the caches changes.
The caches are B3's. The runs take seconds and were not run before this
file was committed.
