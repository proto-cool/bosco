# The v2 learning rule: two attempts, both refused

**Neither fix ships, and the measurements say why.**  Two rules were proposed against the
one-way ratchet in his verdicts and both were replayed against his own logged windows from his
own live weights.

*Extinction* (a compartment whose DANs did not fire while its KCs did relaxes towards baseline)
raised his verdicts instead of levelling them: +0.310 -> +0.392 where the rule he runs reached
+0.345.  On a diet 2.7:1 sweet the side that keeps going unreinforced is the punishment side,
and relieving it lifts a difference of two depressions.

*Homeostasis* (each compartment scales itself back towards the drive it had) overshoots into the
mirror image of the bug: at equilibrium every one of the 33 accounts reads **bitter**, mean
-0.345.  It moves the common mode, and the verdict is a difference of two means, so rescaling
either side just relocates where everybody sits together.

**What both runs agree on, and what actually matters.**  Under every rule the accounts whose
posts were bitter and the accounts whose posts were sweet come out the *same*: +0.107 against
+0.111 under the rule he runs, -0.344 against -0.346 under homeostasis, with r between -0.07 and
+0.17 depending on the span.  The defect is not the sign of his verdicts, it is that his verdicts
carry almost no information about *who* -- an account odor is a few glomeruli inside a pairing
that lands on the whole mixture (its words, its topics, the feed, the hour), so what the mushroom
body learns is moments, not people.  No rescaling of compartments can put information there that
the code never carried.  That is an encoder question, and a pre-registration.

**And the ratchet is smaller than it looked.**  Replayed on his real clock rather than compressed
time, his verdicts fall on their own: +0.303 to +0.109 over eight hours under the rule he runs,
as short-term memory decays.  The climb measured across two days is real, but a snapshot taken
during heavy reading overstates it.

The rule that runs is unchanged: `config/plasticity_v1.yaml`, `Agent.PLASTICITY_VERSION` 2.
Both proposals are kept in `config/plasticity_v2.yaml`, disabled, with their machinery and unit
tests, so the next attempt starts from something runnable rather than from scratch.


`scripts/plasticity_v2_gate.py`: 800 of his logged windows replayed under each
rule from his live weights, 33 accounts read at least 5 times in that span.
`extinction.eta` is set from the behavioural protocol (config/plasticity_v2.yaml); nothing
here feeds back into it.

## What every odor inherits

The offset a smell picks up whatever its own posts said: mean depression over the whole
population on each side, and the verdict that falls out of the difference.

| rule | reward side | punishment side | floor |
|---|---|---|---|
| v1 (one-way) | 0.0588 | 0.0306 | +0.056 |
| v2 (homeostasis, tau 12.0 h) | -0.8736 | -0.6388 | -0.469 |

## Where each arm started

Both arms begin from the same weights, so the difference below is the rule and nothing
else.  From his live state this is the drift already in him.

- standing floor +0.095
- the twelve probe accounts, mean verdict +0.303

## Which way it was moving

| rule | windows | floor | sourest six | sweetest six |
|---|---|---|---|---|
| v1 (one-way) | 200 | +0.084 | +0.197 | +0.210 |
| v1 (one-way) | 400 | +0.074 | +0.164 | +0.172 |
| v1 (one-way) | 600 | +0.069 | +0.128 | +0.133 |
| v1 (one-way) | 800 | +0.056 | +0.103 | +0.105 |
| v2 (homeostasis, tau 12.0 h) | 200 | +0.051 | +0.188 | +0.202 |
| v2 (homeostasis, tau 12.0 h) | 400 | +0.049 | +0.165 | +0.175 |
| v2 (homeostasis, tau 12.0 h) | 600 | +0.057 | +0.141 | +0.147 |
| v2 (homeostasis, tau 12.0 h) | 800 | -0.024 | +0.052 | +0.054 |

## What he thinks of the people he read

| rule | verdicts | mean | sd | net-bitter accounts | net-sweet accounts | r | below zero |
|---|---|---|---|---|---|---|---|
| v1 (one-way) | +0.079 .. +0.141 | +0.109 | 0.013 | +0.107 (n=11) | +0.111 (n=22) | -0.07 | 0 of 33 |
| v2 (homeostasis, tau 12.0 h) | -0.390 .. -0.259 | -0.345 | 0.026 | -0.344 (n=11) | -0.346 (n=22) | -0.02 | 33 of 33 |

## Where homeostasis settles

The gains move over tau, and a replay is shorter than tau, so this is the same state with
each compartment's gain put at the value it is heading for.  It is a projection, and it is
labelled as one.

| rule | gains | floor | verdicts | mean | r | below zero |
|---|---|---|---|---|---|---|
| v2 (homeostasis, tau 12.0 h) | punishment x1.679, reward x2.000 | -0.469 | -0.390 .. -0.259 | -0.345 | -0.02 | 33 of 33 |

