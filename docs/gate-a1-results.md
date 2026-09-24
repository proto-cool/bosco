# Gate A1 results

Pre-registration: `docs/GATE-A1.md`. Epoch chosen on validation, reported on test.

| arm | n_train | seed | epoch | val | **test** | test rho | ECE | pictures (median) | pictures at 0.5 | KC active |
|---|---|---|---|---|---|---|---|---|---|---|
| free | 400 | 1 | 7 | 0.631 | **0.641** | 0.42 | 0.092 | 0.654 | 0.509 | 0.632 |
| free | 400 | 2 | 5 | 0.655 | **0.685** | 0.46 | 0.079 | 0.676 | 0.541 | 0.665 |
| hash | 400 | 1 | 8 | 0.519 | **0.624** | 0.30 | 0.108 | 0.721 | 0.699 | 0.270 |
| hash | 400 | 2 | 7 | 0.509 | **0.574** | 0.26 | 0.066 | 0.699 | 0.682 | 0.284 |
| real | 400 | 1 | 4 | 0.600 | **0.640** | 0.38 | 0.124 | 0.725 | 0.700 | 0.343 |
| real | 400 | 2 | 2 | 0.589 | **0.594** | 0.26 | 0.054 | 0.634 | 0.606 | 0.366 |
| shuffle | 400 | 1 | 8 | 0.624 | **0.679** | 0.40 | 0.072 | 0.636 | 0.586 | 0.521 |
| shuffle | 400 | 2 | 8 | 0.620 | **0.685** | 0.42 | 0.091 | 0.604 | 0.552 | 0.647 |
| free | 6254 | 1 | 4 | 0.694 | **0.709** | 0.54 | 0.080 | 0.759 | 0.545 | 0.420 |
| free | 6254 | 2 | 5 | 0.694 | **0.704** | 0.54 | 0.065 | 0.749 | 0.538 | 0.435 |
| hash | 6254 | 1 | 6 | 0.690 | **0.725** | 0.49 | 0.097 | 0.676 | 0.505 | 0.409 |
| hash | 6254 | 2 | 7 | 0.696 | **0.696** | 0.50 | 0.082 | 0.735 | 0.500 | 0.455 |
| real | 6254 | 1 | 3 | 0.675 | **0.665** | 0.46 | 0.106 | 0.681 | 0.500 | 0.660 |
| real | 6254 | 2 | 5 | 0.680 | **0.710** | 0.48 | 0.068 | 0.700 | 0.506 | 0.659 |
| shuffle | 6254 | 1 | 6 | 0.704 | **0.704** | 0.55 | 0.074 | 0.747 | 0.531 | 0.462 |
| shuffle | 6254 | 2 | 3 | 0.700 | **0.716** | 0.54 | 0.067 | 0.744 | 0.500 | 0.528 |

## Decision (full training set, mean over seeds)

- real 0.687 vs shuffle 0.710: not ahead by 0.03
- real 0.687 vs hash 0.710: not ahead by 0.03
- real 0.687 vs free 0.707: not ahead by 0.03
- real vs the antenna's logistic ceiling 0.697: 0.687
- **FAIL: the wiring does not beat every control by 0.03**

## Probes, real arm (P(sweet), best epoch)

- seed 1: I fucking hate you 0.36, I hate you 0.30, I don't know 0.26, I love you 0.76, I fucking love you 0.67, oasis-Dessert 1 0.87, oasis-Garbage dump 1 0.88
- seed 2: I fucking hate you 0.25, I hate you 0.28, I don't know 0.15, I love you 0.62, I fucking love you 0.59, oasis-Dessert 1 0.83, oasis-Garbage dump 1 0.83

## Reading (after the numbers; the rule is unchanged)

- **FAIL.** Every trained brain reaches the encoder's ceiling: real 0.687,
  shuffle 0.710, hash 0.710, free 0.707, against the antenna's logistic
  ceiling of 0.697. With each neuron's gain and threshold free, any big
  recurrent network learns this, and the fly's specific wiring adds nothing
  on CLIP input. That is gate B's answer again, reached independently.
- **It does learn.** The real brain puts all four love/hate probes on the
  right side in both seeds, far above the lifetime rule's 0.555 in C1.
  Pictures from text alone come out at 0.68–0.70 at their median, but the
  rot photo scores sweet (0.83–0.88), and nothing is calibrated for
  pictures at 0.5.
- **Not fly-like inside.** The trained real brain has 66% of its Kenyon
  cells active; a fly has 2–7%. Training made it a dense network.
- **The limit is the nose.** A plain logistic model on stronger frozen text
  encoders reaches 0.885 (all-mpnet-base-v2) and 0.915 (bge-large-en-v1.5)
  on the clear held-out sentences, against CLIP's 0.763 (ceiling check,
  2026-09-24, not part of this gate).
