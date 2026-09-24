# Gate C1 results

Pre-registration: `docs/GATE-C1.md`. One fly per row.

| arm | seed | read | held-out balanced (own neutral) | at 0 | rho | pictures | picture rho | recall gap | silent | KC test / naive |
|---|---|---|---|---|---|---|---|---|---|---|
| hash | 1 | output neurons | 0.567 | 0.555 | 0.08 | 0.500 | -0.04 | +0.067 | 0.26 | 0.0187 / 0.0228 |
| hash | 1 | weights (old) | 0.627 | 0.525 | 0.36 | 0.780 | 0.60 | +0.089 | 0.26 | 0.0187 / 0.0228 |
| hash | 2 | output neurons | 0.564 | 0.555 | 0.08 | 0.467 | -0.02 | +0.017 | 0.26 | 0.0187 / 0.0228 |
| hash | 2 | weights (old) | 0.629 | 0.520 | 0.35 | 0.793 | 0.61 | +0.116 | 0.26 | 0.0187 / 0.0228 |
| real | 1 | output neurons | 0.556 | 0.539 | 0.06 | 0.527 | 0.06 | +0.119 | 0.28 | 0.0146 / 0.0228 |
| real | 1 | weights (old) | 0.615 | 0.518 | 0.28 | 0.753 | 0.59 | +0.133 | 0.28 | 0.0146 / 0.0228 |
| real | 2 | output neurons | 0.554 | 0.544 | 0.06 | 0.493 | 0.04 | +0.075 | 0.28 | 0.0146 / 0.0228 |
| real | 2 | weights (old) | 0.624 | 0.518 | 0.30 | 0.767 | 0.56 | +0.132 | 0.28 | 0.0146 / 0.0228 |
| shuffle | 1 | output neurons | 0.489 | 0.498 | 0.02 | 0.420 | -0.26 | -0.059 | 0.00 | 0.0496 / 0.0496 |
| shuffle | 1 | weights (old) | 0.594 | 0.500 | 0.20 | 0.633 | 0.29 | +0.007 | 0.00 | 0.0496 / 0.0496 |
| shuffle | 2 | output neurons | 0.482 | 0.500 | 0.01 | 0.407 | -0.26 | +0.036 | 0.00 | 0.0496 / 0.0496 |
| shuffle | 2 | weights (old) | 0.586 | 0.500 | 0.20 | 0.600 | 0.25 | +0.009 | 0.00 | 0.0496 / 0.0496 |

## Decision

- Real arm, output-neuron read, held-out balanced at own neutral, mean of 2 seeds: **0.555** vs bar 0.56: **FAIL: written up as it stands**
- Same flies, weight read: 0.619 (gap -0.064; under 0.03 counts as the same)
- hash, output-neuron read: 0.566
- shuffle, output-neuron read: 0.486

## Probes (output-neuron read, score minus own neutral, real arm)

- I fucking hate you: -0.652, -0.631
- I hate you: +0.466, +0.489
- I don't know: -0.165, +0.406
- I love you: +0.591, +0.614
- I fucking love you: -0.590, -0.556
- oasis-Dessert 1: -0.846, -0.832
- oasis-Garbage dump 1: -0.672, -0.661

## Reading (written after the numbers, not changing the rule)

- **FAIL, by 0.005.** The fly's output neurons say sweet or bitter at 0.555
  on sentences he has never tasted. That's barely above chance, and the
  ordering is nearly flat (rho 0.06).
- **The learning is there; it does not reach his outputs.** On the very same
  flies, the synapses read 0.62 (rho 0.29) and pictures 0.76. The output
  neurons' firing is dominated by how strongly each smell drives them before
  any learning. The learned change is small next to that, so the probes swing
  hard and inconsistently ("I hate you" +0.47, "I fucking love you" −0.57).
- **28% of held-out sentences leave his outputs silent** (the hash is 26%, the
  shuffle 0%). With feedback on, a trained fly's Kenyon-cell activity falls
  from 2.3% to 1.5%, and many sentences then barely reach the output neurons.
- Real ≈ hash > shuffle, as in every gate before.
- By the pre-registration, the decider does **not** go back to the weight
  read quietly. The next step is a separate, pre-registered diagnosis.
