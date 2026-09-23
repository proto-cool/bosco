# Gate B2 results

20 runs under `runs/gate-b2/`. Pre-registration: `GATE-B2.md`.


## t1

| arm | seeds | k25 acc | k25 rho | k50 acc | k50 rho | k100 acc | k100 rho | k200 acc | k200 rho | ECE raw | ECE iso |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.522 ± 0.023 | 0.07 | 0.515 ± 0.028 | 0.08 | 0.542 ± 0.028 | 0.15 | 0.558 ± 0.038 | 0.13 | 0.409 | 0.028 |
| shuffle | 5 | 0.518 ± 0.011 | 0.10 | 0.514 ± 0.020 | 0.11 | 0.490 ± 0.031 | 0.05 | 0.517 ± 0.049 | 0.11 | 0.435 | 0.021 |
| hash | 5 | 0.516 ± 0.016 | 0.04 | 0.523 ± 0.034 | 0.07 | 0.522 ± 0.026 | 0.11 | 0.538 ± 0.013 | 0.11 | 0.430 | 0.016 |
| logistic | 5 | 0.531 ± 0.069 | 0.17 | 0.536 ± 0.064 | 0.25 | 0.590 ± 0.023 | 0.33 | 0.593 ± 0.056 | 0.36 | 0.075 | 0.069 |

Recall of the last 30 rewarded items of each taste, at the end: real sweet-trained 0.543 / bitter-trained 0.459 (gap +0.084); shuffle sweet-trained 0.515 / bitter-trained 0.497 (gap +0.018); hash sweet-trained 0.531 / bitter-trained 0.468 (gap +0.063); logistic sweet-trained 0.547 / bitter-trained 0.469 (gap +0.079)

## Decision rule (GATE-B2.md)

- t1 k50: real 0.515 (sd 0.028); shuffle 0.514 (not beaten by more than pooled sd 0.024); hash 0.523 (not beaten by more than pooled sd 0.031)
- t1 k200: real 0.558 (sd 0.038); shuffle 0.517 (not beaten by more than pooled sd 0.044); hash 0.538 (not beaten by more than pooled sd 0.028)

**Real wiring earns its place: NO** (0 of 2 comparisons).

Drive: scale 183.8 Hz, centred, KC fraction over 500 calibration sentences mean 0.0271 (target 0.027).
