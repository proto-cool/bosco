# A5 preflight

`scripts/a5_preflight.py`; the checks and bars are in `docs/A5-PREFLIGHT.md`. Training data only.

| config | pass | KC at start | MBON at start | output spread | loss first 10 → last 10 | AUROC after 30 batches | KC after |
|---|---|---|---|---|---|---|---|
| real-type-full | pass | 0.050 | 0.201 | 0.113 | 0.679 → 0.625 | 0.714 | 0.021 |
| layered-type-full | **slow**: answers_carry_information | 0.050 | 0.201 | 0.038 | 0.676 → 0.634 | 0.571 | 0.022 |
| hash-type-full | pass | 0.050 | 0.202 | 0.127 | 0.664 → 0.625 | 0.603 | 0.071 |
| free-type-full | pass | 0.050 | 0.200 | 0.335 | 0.665 → 0.610 | 0.727 | 0.045 |
| real-neuron-full | pass | 0.050 | 0.201 | 0.113 | 0.679 → 0.623 | 0.746 | 0.029 |
| layered-neuron-full | pass | 0.050 | 0.201 | 0.038 | 0.680 → 0.634 | 0.619 | 0.043 |
| hash-neuron-full | pass | 0.050 | 0.202 | 0.127 | 0.663 → 0.615 | 0.666 | 0.065 |
| free-neuron-full | pass | 0.050 | 0.200 | 0.335 | 0.661 → 0.582 | 0.793 | 0.042 |
| real-type-cut | pass | 0.050 | 0.200 | 0.049 | 0.677 → 0.622 | 0.754 | 0.045 |
| layered-type-cut | pass | 0.050 | 0.200 | 0.063 | 0.684 → 0.619 | 0.772 | 0.098 |
| hash-type-cut | pass | 0.050 | 0.200 | 0.122 | 0.668 → 0.622 | 0.641 | 0.064 |
| free-type-cut | pass | 0.050 | 0.200 | 0.327 | 0.672 → 0.619 | 0.724 | 0.033 |
| real-neuron-cut | pass | 0.050 | 0.200 | 0.049 | 0.676 → 0.615 | 0.783 | 0.053 |
| layered-neuron-cut | pass | 0.050 | 0.200 | 0.063 | 0.681 → 0.608 | 0.782 | 0.080 |
