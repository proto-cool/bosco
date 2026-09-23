# Gate B4 results

45 runs under `runs/gate-b4/`. Pre-registration: `GATE-B4.md`.


## t1

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | final at 0.5 | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.631±0.012 | 0.612±0.025 | 0.595±0.025 | 0.603±0.013 | 0.596±0.016 | 0.601±0.024 | 0.517 | 0.28 | +0.036 | 0.697 | 0.702 | 0.642 | 0.604 |
| shuffle | 5 | 0.569±0.009 | 0.575±0.013 | 0.579±0.014 | 0.557±0.020 | 0.559±0.021 | 0.577±0.018 | 0.500 | 0.21 | +0.004 | 0.697 | 0.702 | 0.642 | 0.608 |
| hash | 5 | 0.621±0.024 | 0.616±0.015 | 0.604±0.016 | 0.588±0.015 | 0.602±0.015 | 0.616±0.018 | 0.526 | 0.31 | +0.036 | 0.697 | 0.702 | 0.642 | 0.606 |

real: T4 transfer to 900 pictures rho +0.636, balanced 0.553 at own neutral; retention 24 h: held-out balanced 0.572

shuffle: T4 transfer to 900 pictures rho +0.192, balanced 0.596 at own neutral; retention 24 h: held-out balanced 0.500

hash: T4 transfer to 900 pictures rho +0.656, balanced 0.525 at own neutral; retention 24 h: held-out balanced 0.595

Probe sheet, real wiring, mean over seeds (0 bitter .. 1 sweet):

- I fucking hate you: 0.452
- I hate you: 0.452
- I don't know: 0.456
- I love you: 0.467
- I fucking love you: 0.460
- oasis-Dessert 1: 0.508
- oasis-Garbage dump 1: 0.457

## t2

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | ep7 acc | ep8 acc | ep9 acc | ep10 acc | ep11 acc | ep12 acc | final at 0.5 | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.631±0.012 | 0.612±0.025 | 0.595±0.025 | 0.603±0.013 | 0.596±0.016 | 0.601±0.024 | 0.548±0.015 | 0.594±0.029 | 0.615±0.014 | 0.607±0.010 | 0.623±0.009 | 0.612±0.011 | 0.545 | 0.29 | +0.014 | 0.697 | 0.702 | 0.642 | 0.604 |
| shuffle | 5 | 0.569±0.009 | 0.575±0.013 | 0.579±0.014 | 0.557±0.020 | 0.559±0.021 | 0.577±0.018 | 0.523±0.025 | 0.553±0.021 | 0.566±0.016 | 0.559±0.018 | 0.568±0.020 | 0.561±0.016 | 0.500 | 0.15 | +0.003 | 0.697 | 0.702 | 0.642 | 0.608 |
| hash | 5 | 0.621±0.024 | 0.616±0.015 | 0.604±0.016 | 0.588±0.015 | 0.602±0.015 | 0.616±0.018 | 0.519±0.014 | 0.589±0.016 | 0.615±0.018 | 0.604±0.016 | 0.615±0.012 | 0.594±0.028 | 0.512 | 0.25 | +0.012 | 0.697 | 0.702 | 0.642 | 0.606 |

## t3

| arm | seeds | ep1 acc | ep2 acc | ep3 acc | ep4 acc | ep5 acc | ep6 acc | final at 0.5 | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | 0.736±0.023 | 0.735±0.012 | 0.720±0.011 | 0.702±0.019 | 0.713±0.016 | 0.692±0.017 | 0.514 | 0.56 | +0.040 | 0.813 | 0.827 | 0.786 | 0.750 |
| shuffle | 5 | 0.699±0.017 | 0.690±0.023 | 0.692±0.006 | 0.693±0.027 | 0.698±0.017 | 0.701±0.017 | 0.520 | 0.50 | +0.006 | 0.813 | 0.827 | 0.786 | 0.752 |
| hash | 5 | 0.716±0.031 | 0.690±0.011 | 0.707±0.016 | 0.691±0.033 | 0.705±0.024 | 0.685±0.036 | 0.522 | 0.56 | +0.036 | 0.813 | 0.827 | 0.786 | 0.750 |

## The bar (GATE-B4.md)

- T1 generalisation (balanced, own neutral, best epoch 1 of [np.float64(0.631), np.float64(0.612), np.float64(0.595), np.float64(0.603), np.float64(0.596), np.float64(0.601)]): real 0.631 vs 0.9 x antenna ceiling 0.697 = 0.628: PASS
- T1 memory: recall gap +0.066 vs +0.30: FAIL
- T2 reversal: pre-flip 0.631; post-flip by epoch [np.float64(0.548), np.float64(0.594), np.float64(0.615), np.float64(0.607), np.float64(0.623), np.float64(0.612)]; recovers to 0.9 x pre within 6 epochs: PASS (epoch 2)

Antenna: 26 PCs x (+,-), scale 328.3 Hz, KC fraction over 500 calibration sentences mean 0.0299.
