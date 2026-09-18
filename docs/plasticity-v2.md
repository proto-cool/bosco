# The v2 learning rule: extinction

`scripts/plasticity_v2_gate.py`, 120 logged windows replayed from a naive mushroom
body under each rule, 5 accounts read at least 5 times.  Nothing here is
tuned: `extinction.eta` is set from the behavioural protocol (config/plasticity_v2.yaml).

### v1 (one-way depression)

- verdicts: -0.041 .. -0.017, mean -0.030, sd 0.009
- accounts he read that were net bitter (n=2): mean verdict -0.028
- accounts he read that were net sweet (n=3): mean verdict -0.032
- separation between the two: -0.003
- correlation with what he read from them: r = -0.41
- verdicts below zero: 5 of 5 (100%)

### v2 (extinction)

- verdicts: -0.040 .. -0.017, mean -0.029, sd 0.008
- accounts he read that were net bitter (n=2): mean verdict -0.028
- accounts he read that were net sweet (n=3): mean verdict -0.031
- separation between the two: -0.003
- correlation with what he read from them: r = -0.37
- verdicts below zero: 5 of 5 (100%)

