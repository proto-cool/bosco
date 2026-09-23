# Gate B2 results

60 runs under `runs/gate-b2/`. Pre-registration: `GATE-B2.md`.


## t1

| arm | seeds | k25 acc | k25 rho | k50 acc | k50 rho | k100 acc | k100 rho | k200 acc | k200 rho | ECE raw | ECE iso |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.522 ± 0.023 | 0.07 | 0.515 ± 0.028 | 0.08 | 0.542 ± 0.028 | 0.15 | 0.558 ± 0.038 | 0.13 | 0.409 | 0.028 |
| shuffle | 5 | 0.518 ± 0.011 | 0.10 | 0.514 ± 0.020 | 0.11 | 0.490 ± 0.031 | 0.05 | 0.517 ± 0.049 | 0.11 | 0.435 | 0.021 |
| hash | 5 | 0.516 ± 0.016 | 0.04 | 0.523 ± 0.034 | 0.07 | 0.522 ± 0.026 | 0.11 | 0.538 ± 0.013 | 0.11 | 0.430 | 0.016 |
| logistic | 5 | 0.531 ± 0.069 | 0.17 | 0.536 ± 0.064 | 0.25 | 0.590 ± 0.023 | 0.33 | 0.593 ± 0.056 | 0.36 | 0.075 | 0.069 |

Recall of the last 30 rewarded items of each taste, at the end: real sweet-trained 0.543 / bitter-trained 0.459 (gap +0.084); shuffle sweet-trained 0.515 / bitter-trained 0.497 (gap +0.018); hash sweet-trained 0.531 / bitter-trained 0.468 (gap +0.063); logistic sweet-trained 0.547 / bitter-trained 0.469 (gap +0.079)

## t2

| arm | seeds | k25 acc | k25 rho | k50 acc | k50 rho | k100 acc | k100 rho | k200 acc | k200 rho | flip-k25 acc | flip-k25 rho | flip-k50 acc | flip-k50 rho | flip-k100 acc | flip-k100 rho | flip-k200 acc | flip-k200 rho | ECE raw | ECE iso |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.522 ± 0.023 | 0.07 | 0.515 ± 0.028 | 0.08 | 0.542 ± 0.028 | 0.15 | 0.558 ± 0.038 | 0.13 | 0.501 ± 0.029 | -0.00 | 0.491 ± 0.039 | -0.02 | 0.515 ± 0.023 | 0.08 | 0.506 ± 0.046 | 0.04 | 0.421 | 0.009 |
| shuffle | 5 | 0.518 ± 0.011 | 0.10 | 0.514 ± 0.020 | 0.11 | 0.490 ± 0.031 | 0.05 | 0.517 ± 0.049 | 0.11 | 0.515 ± 0.046 | 0.13 | 0.531 ± 0.030 | 0.14 | 0.530 ± 0.012 | 0.08 | 0.515 ± 0.038 | 0.10 | 0.433 | 0.018 |
| hash | 5 | 0.516 ± 0.016 | 0.04 | 0.523 ± 0.034 | 0.07 | 0.522 ± 0.026 | 0.11 | 0.538 ± 0.013 | 0.11 | 0.503 ± 0.038 | -0.00 | 0.507 ± 0.019 | 0.01 | 0.521 ± 0.029 | 0.06 | 0.510 ± 0.017 | 0.07 | 0.429 | 0.017 |
| logistic | 5 | 0.531 ± 0.069 | 0.17 | 0.536 ± 0.064 | 0.25 | 0.590 ± 0.023 | 0.33 | 0.593 ± 0.056 | 0.36 | 0.338 ± 0.043 | -0.43 | 0.367 ± 0.061 | -0.41 | 0.344 ± 0.038 | -0.41 | 0.418 ± 0.039 | -0.33 | 0.056 | 0.034 |

Recall of the last 30 rewarded items of each taste, at the end: real sweet-trained 0.538 / bitter-trained 0.471 (gap +0.067); shuffle sweet-trained 0.548 / bitter-trained 0.531 (gap +0.017); hash sweet-trained 0.537 / bitter-trained 0.485 (gap +0.052); logistic sweet-trained 0.497 / bitter-trained 0.478 (gap +0.020)

Retention 24 h after the last reward (T2): hash 0.500; hash 0.500; hash 0.470; hash 0.475; hash 0.510; real 0.510; real 0.470; real 0.485; real 0.435; real 0.500; shuffle 0.505; shuffle 0.520; shuffle 0.475; shuffle 0.525; shuffle 0.505

## t3

| arm | seeds | k25 acc | k25 rho | k50 acc | k50 rho | k100 acc | k100 rho | ECE raw | ECE iso |
|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.600 ± 0.043 | 0.23 | 0.606 ± 0.077 | 0.29 | 0.632 ± 0.134 | 0.24 | 0.346 | 0.115 |
| shuffle | 5 | 0.624 ± 0.041 | 0.35 | 0.604 ± 0.057 | 0.48 | 0.560 ± 0.088 | 0.40 | 0.395 | 0.115 |
| hash | 5 | 0.638 ± 0.056 | 0.30 | 0.662 ± 0.068 | 0.39 | 0.638 ± 0.131 | 0.30 | 0.344 | 0.144 |
| logistic | 5 | 0.680 ± 0.049 | 0.53 | 0.694 ± 0.042 | 0.75 | 0.742 ± 0.070 | 0.82 | 0.218 | 0.066 |

Recall of the last 30 rewarded items of each taste, at the end: real sweet-trained 0.527 / bitter-trained 0.479 (gap +0.048); shuffle sweet-trained 0.461 / bitter-trained 0.402 (gap +0.059); hash sweet-trained 0.521 / bitter-trained 0.476 (gap +0.045); logistic sweet-trained 0.711 / bitter-trained 0.519 (gap +0.192)

## Decision rule (GATE-B2.md)

- t1 k50: real 0.515 (sd 0.028); shuffle 0.514 (not beaten by more than pooled sd 0.024); hash 0.523 (not beaten by more than pooled sd 0.031)
- t1 k200: real 0.558 (sd 0.038); shuffle 0.517 (not beaten by more than pooled sd 0.044); hash 0.538 (not beaten by more than pooled sd 0.028)
- t2 flip-k50: real 0.491 (sd 0.039); shuffle 0.531 (not beaten by more than pooled sd 0.035); hash 0.507 (not beaten by more than pooled sd 0.031)
- t2 flip-k200: real 0.506 (sd 0.046); shuffle 0.515 (not beaten by more than pooled sd 0.042); hash 0.510 (not beaten by more than pooled sd 0.034)

**Real wiring earns its place: NO** (0 of 4 comparisons).

Drive: scale 183.8 Hz, centred, KC fraction over 500 calibration sentences mean 0.0271 (target 0.027).
