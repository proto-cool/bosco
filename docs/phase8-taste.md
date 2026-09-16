# Phase 8 gate: taste and exposure while browsing

## One account, forty sweet posts in a day

- account-alone verdict after the day: +0.370 (want 0.1 .. 0.9)
- familiarity after the day: 0.660 (want > 0.5)
- mean KC->MBON multiplier after the day: 0.9874
- three quiet days later: verdict +0.007, familiarity 0.032 (want < 0.2)

**Check 1: PASS**

## Two hundred mixed posts from forty accounts in a day

- mean multiplier over all plastic edges: 0.9736 (want >= 0.9); min 0.300
- edges below 0.5: 171 of 61210
- KC fraction active per window: mean 0.0519, max 0.1134
- exposure depression over KCs: mean 0.230, max 0.700; KCs never met: 2195 of 4064
- account verdicts: mean -0.120, min -0.210, max -0.031

**Check 2: PASS**

**GATE: PASS**

Notes (2026-09-15, rerun after familiarity was restricted to the cells a smell adds and probes
moved to rest).  The sweet talker ends the day at +0.37, familiar (0.66), and is nearly new
again three quiet days later.  In the mixed day, 3% of posts were labeled and pair punishment
at labeled_gain 0.3, full strength, which is what pulls the mean stranger verdict below zero
(-0.12): labels are the strongest brake in the rule, as intended.  KC sparseness (mean 5.2%)
is unchanged by the pairings.  Parameters were not moved.
