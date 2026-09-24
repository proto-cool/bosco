# A5 pilot: does cutting weak synapses change his answers? (pre-registered 2026-09-24)

Nick: "I think it makes sense to empirically test it." Keeping every synapse
(9.77M connections) is truest but about 4× slower to train and to serve.
Cutting connections of fewer than 5 synapses keeps 76% of all synapses, and
every KC→MBON synapse is kept regardless (his memory).

## Arms

The corrected v2 brain (`ratebrain2`, per-cell-type parameters plus KC→MBON,
the main configuration), real wiring:
- **full:** every connection.
- **cut:** fewer than 5 synapses dropped, except KC→MBON.

Denominators stay every synapse onto each neuron, so dropped inputs count as
missing, not rescaled away.

## Protocol

- The cleaned data (`docs/a5-data.md`). 1,500 training items per text part
  plus all 387 training pictures (shown 5×); every question at once.
- 2 epochs, batch 64, Adam 3e-3, the KC sparseness pressure. Init by the
  label-free grid.
- Seeds 1 and 2 for each arm: four short runs.
- Scored on **validation only**. The test sets are not touched.

## Rule (fixed now)

- **Use the cut** if its mean validation balanced accuracy (over sweet,
  pictures, dangerous, junk; averaged over both seeds) is within **0.02** of
  the full brain's.
- Otherwise **keep everything**, and look for speed elsewhere.

Also reported: time per batch for each arm, and how often the two arms give
the same side on the same validation item.

Runner: `scripts/a5_cutoff_pilot.py`. Results: `docs/a5-cutoff-pilot-results.md`.
