# Gate B results

20 runs under `runs/gate-b/`. Pre-registration: `GATE-B.md`.


## t1

| arm | seeds | k25 acc | k25 rho | k50 acc | k50 rho | k100 acc | k100 rho | k200 acc | k200 rho | ECE raw | ECE iso |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.498 ± 0.044 | 0.04 | 0.480 ± 0.046 | 0.01 | 0.490 ± 0.041 | 0.03 | 0.490 ± 0.022 | 0.03 | 0.477 | 0.043 |
| shuffle | 5 | 0.522 ± 0.049 | 0.04 | 0.511 ± 0.047 | -0.01 | 0.475 ± 0.030 | 0.00 | 0.507 ± 0.014 | 0.05 | 0.465 | 0.046 |
| hash | 5 | 0.521 ± 0.046 | 0.02 | 0.500 ± 0.054 | -0.02 | 0.485 ± 0.049 | 0.01 | 0.511 ± 0.036 | 0.02 | 0.472 | 0.009 |
| logistic | 5 | 0.495 ± 0.046 | 0.06 | 0.489 ± 0.051 | 0.04 | 0.520 ± 0.025 | 0.05 | 0.484 ± 0.050 | 0.08 | 0.041 | 0.009 |

## Decision rule (GATE-B.md)

- t1 k50: real 0.480 (sd 0.046); shuffle 0.511 (not beaten by more than pooled sd 0.047); hash 0.500 (not beaten by more than pooled sd 0.050)
- t1 k200: real 0.490 (sd 0.022); shuffle 0.507 (not beaten by more than pooled sd 0.018); hash 0.511 (not beaten by more than pooled sd 0.030)

**Real wiring earns its place: NO** (0 of 2 comparisons).

Drive: scale 242.4 Hz, KC fraction over 500 calibration sentences mean 0.0273 (target 0.027).
