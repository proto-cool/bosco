# A5 cutoff pilot results

Pre-registration: `docs/A5-CUTOFF-PILOT.md`. Validation only.

| arm | seed | sweet | pictures | dangerous | junk | mean | KC active | s per epoch |
|---|---|---|---|---|---|---|---|---|
| cut | 1 | 0.829 | 0.890 | 0.634 | 0.614 | **0.742** | 0.156 | 170 |
| cut | 2 | 0.825 | 0.857 | 0.588 | 0.648 | **0.730** | 0.158 | 212 |
| full | 1 | 0.504 | 0.656 | 0.500 | 0.499 | **0.540** | 0.065 | 423 |
| full | 2 | 0.500 | 0.500 | 0.500 | 0.500 | **0.500** | 0.075 | 469 |

## Decision

- full 0.520, cut 0.736: difference +0.216 (bar: within 0.02)
- same side on the same item: 62.4%
- **use the cut (≥5 synapses, KC→MBON all kept)**

**VOID** (amendment 2 in `A5-CUTOFF-PILOT.md`): the full brain never learned in this run; the rerun is `a5-cutoff-pilot2-results.md`.
