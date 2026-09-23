# Gate B3 results

45 runs under `runs/gate-b3/`. Pre-registration: `GATE-B3.md`.


## t1

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.537±0.014 | 0.528±0.010 | 0.522±0.006 | 0.519±0.003 | 0.515±0.004 | 0.521±0.006 | 0.28 | +0.036 | 0.698 | 0.703 | 0.642 | 0.605 |
| shuffle | 5 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.21 | +0.004 | 0.698 | 0.703 | 0.642 | 0.610 |
| hash | 5 | 0.552±0.035 | 0.545±0.006 | 0.531±0.006 | 0.530±0.003 | 0.529±0.005 | 0.531±0.005 | 0.31 | +0.036 | 0.698 | 0.703 | 0.642 | 0.608 |

real: T4 transfer to 900 pictures rho +0.636, acc 0.543; retention 24 h: held-out acc 0.516

shuffle: T4 transfer to 900 pictures rho +0.192, acc 0.346; retention 24 h: held-out acc 0.505

hash: T4 transfer to 900 pictures rho +0.656, acc 0.775; retention 24 h: held-out acc 0.528

Probe sheet, real wiring, mean over seeds (0 bitter .. 1 sweet):

- I fucking hate you: 0.452
- I hate you: 0.452
- I don't know: 0.456
- I love you: 0.467
- I fucking love you: 0.460
- oasis-Dessert 1: 0.508
- oasis-Garbage dump 1: 0.457

## t2

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | ep7 acc | ep8 acc | ep9 acc | ep10 acc | ep11 acc | ep12 acc | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.537±0.014 | 0.528±0.010 | 0.522±0.006 | 0.519±0.003 | 0.515±0.004 | 0.521±0.006 | 0.496±0.018 | 0.562±0.041 | 0.574±0.041 | 0.582±0.033 | 0.574±0.030 | 0.549±0.012 | 0.29 | +0.014 | 0.698 | 0.703 | 0.642 | 0.605 |
| shuffle | 5 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.505±0.000 | 0.497±0.004 | 0.499±0.005 | 0.505±0.010 | 0.506±0.003 | 0.505±0.000 | 0.505±0.000 | 0.15 | +0.003 | 0.698 | 0.703 | 0.642 | 0.610 |
| hash | 5 | 0.552±0.035 | 0.545±0.006 | 0.531±0.006 | 0.530±0.003 | 0.529±0.005 | 0.531±0.005 | 0.503±0.014 | 0.514±0.034 | 0.520±0.037 | 0.524±0.043 | 0.525±0.045 | 0.516±0.025 | 0.25 | +0.012 | 0.698 | 0.703 | 0.642 | 0.608 |

## t3

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.768±0.017 | 0.746±0.003 | 0.742±0.002 | 0.739±0.001 | 0.739±0.002 | 0.737±0.001 | 0.56 | +0.040 | 0.778 | 0.770 | 0.742 | 0.767 |
| shuffle | 5 | 0.605±0.200 | 0.748±0.012 | 0.690±0.108 | 0.741±0.004 | 0.746±0.014 | 0.742±0.010 | 0.50 | +0.006 | 0.778 | 0.770 | 0.742 | 0.723 |
| hash | 5 | 0.749±0.006 | 0.747±0.004 | 0.746±0.004 | 0.741±0.004 | 0.741±0.002 | 0.740±0.002 | 0.56 | +0.036 | 0.778 | 0.770 | 0.742 | 0.727 |

## The bar (GATE-B3.md)

- T1 generalisation: real 0.521 vs 0.9 x antenna ceiling 0.698 = 0.628: FAIL
- T1 memory: recall gap +0.036 vs +0.30: FAIL
- T2 reversal: pre-flip 0.521; post-flip by epoch [np.float64(0.496), np.float64(0.562), np.float64(0.574), np.float64(0.582), np.float64(0.574), np.float64(0.549)]; recovers to 0.9 x pre within 6 epochs: PASS (epoch 1)

Antenna: 26 PCs x (+,-), scale 328.3 Hz, KC fraction over 500 calibration sentences mean 0.0299.
