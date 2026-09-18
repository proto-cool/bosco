# The v2 learning rule: extinction

**Result: refused.**  Extinction does not level his verdicts, it raises them.  Replayed from his
live weights over 400 of his own windows it took the probe accounts from +0.310 to +0.392, while
the rule he runs took them to +0.345; nobody came out below zero under either.  The reason is in
the readout: a verdict is reward-side depression minus punishment-side depression, and on a diet
that is 2.7:1 sweet the compartment that keeps going unreinforced -- and so keeps being relaxed
-- is the punishment one.  Relieving it lifts the difference.  Any relief scheme gated on "which
side missed its dopamine this window" will do the same thing under a skewed diet.

The ratchet is still there and still unexplained-away: see the 'what every odor inherits' floor
below, and note that it climbs under both rules.  The rule that runs is unchanged
(config/plasticity_v1.yaml, `Agent.PLASTICITY_VERSION` 2).

`scripts/plasticity_v2_gate.py`: 400 of his logged windows replayed under each
rule from his live weights, 14 accounts read at least 5 times in that span.
`extinction.eta` is set from the behavioural protocol (config/plasticity_v2.yaml); nothing
here feeds back into it.

## What every odor inherits

The offset a smell picks up whatever its own posts said: mean depression over the whole
population on each side, and the verdict that falls out of the difference.

| rule | reward side | punishment side | floor |
|---|---|---|---|
| v1 (one-way) | 0.1423 | 0.0841 | +0.117 |
| v2 (extinction, eta 0.1) | 0.1144 | 0.0590 | +0.111 |

## Where each arm started

Both arms begin from the same weights, so the difference below is the rule and nothing
else.  From his live state this is the drift already in him.

- standing floor +0.095
- the twelve probe accounts, mean verdict +0.310

## Which way it was moving

| rule | windows | floor | sourest six | sweetest six |
|---|---|---|---|---|
| v1 | 100 | +0.121 | +0.346 | +0.354 |
| v1 | 200 | +0.124 | +0.345 | +0.352 |
| v1 | 300 | +0.121 | +0.343 | +0.350 |
| v1 | 400 | +0.117 | +0.341 | +0.348 |
| v2 | 100 | +0.126 | +0.393 | +0.405 |
| v2 | 200 | +0.115 | +0.391 | +0.404 |
| v2 | 300 | +0.114 | +0.389 | +0.401 |
| v2 | 400 | +0.111 | +0.386 | +0.397 |

## What he thinks of the people he read

| rule | verdicts | mean | sd | net-bitter accounts | net-sweet accounts | r | below zero |
|---|---|---|---|---|---|---|---|
| v1 (one-way) | +0.252 .. +0.442 | +0.350 | 0.051 | +0.341 (n=5) | +0.355 (n=9) | +0.47 | 0 of 14 |
| v2 (extinction, eta 0.1) | +0.301 .. +0.478 | +0.395 | 0.047 | +0.386 (n=5) | +0.400 (n=9) | +0.54 | 0 of 14 |

