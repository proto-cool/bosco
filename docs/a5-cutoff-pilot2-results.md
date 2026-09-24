# A5 cutoff pilot results

Pre-registration: `docs/A5-CUTOFF-PILOT.md`. Validation only.

| arm | seed | sweet | pictures | dangerous | junk | mean | KC active | s per epoch |
|---|---|---|---|---|---|---|---|---|
| cut | 1 | 0.839 | 0.861 | 0.668 | 0.568 | **0.734** | 0.096 | 162 |
| cut | 2 | 0.854 | 0.868 | 0.647 | 0.591 | **0.740** | 0.097 | 160 |
| full | 1 | 0.837 | 0.800 | 0.652 | 0.496 | **0.696** | 0.066 | 419 |
| full | 2 | 0.844 | 0.853 | 0.710 | 0.768 | **0.794** | 0.075 | 416 |

## Decision

- full 0.745, cut 0.737: difference -0.008 (bar: within 0.02)
- same side on the same item: 84.4%
- **use the cut (≥5 synapses, KC→MBON all kept)**

## Reading

- By amendment 1's rule: **use the cut.** It is 0.008 below the full brain,
  inside the 0.02 band.
- The full brain varies far more between seeds (0.696 vs 0.794; junk 0.496
  vs 0.768) than the cut differs from it. After two short epochs, the cut's
  cost is within noise. Neither can be called better.
- Speed: 160 s per epoch against 417 s, **2.6× faster** end to end (about 6×
  per training batch).
- Junk is weak in every arm at this stage. Only 10% of junk items are spam,
  and after two short epochs the offset is not yet set. The main leg trains
  much longer.
