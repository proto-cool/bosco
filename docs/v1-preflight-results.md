# v1 training preflight (2026-09-25; development, not a gate)

Runner: `scripts/v1_preflight.py`. The rebuilt brain (`ratebrain3`, `v1.py`: graded unit, own-input
controls, homeostatic start at gain 4, KC fraction penalty, 80 steps = 400 ms), trained on the 3080
for 150 batches on the A4 broad development smells (training split only). Held-back accuracy on
300 other training items; chance 0.343; "slow" bar chance + 0.05 = 0.393.

| read | held-back before | after 150 batches | loss first 25 → last 25 | cell types with gradient |
|---|---|---|---|---|
| **descending neurons** (46 cells, 23 types; anatomical groups) | 0.357 | **0.437** | 1.551 → 1.482 | 99.3% |
| MBONs (as before) | 0.323 | 0.463 | 1.588 → 1.493 | 99.3% |

- **Training reaches the whole brain.** 99.3% of cell types get a gradient on the first batch. In
  A4 broad, 7,046 of 8,257 types never changed.
- **The answer read from descending neurons trains** and clears the slow bar. It is about level
  with the MBON read: the difference is under one standard error at 300 items (±0.028).
- For reference, A4 broad's old brain reached 0.443 at the same point (MBON read).
- The DN groups (fixed before training): approach = CB0429, DNg104, DNg13, DNg34, DNge138,
  DNge149, DNge150, DNge151, DNge152, DNp68; avoid = DNa02, DNa03, DNa13, DNde002, DNde005,
  DNge127, DNp42, DNp52, DNpe021, DNpe023, DNpe026, DNpe028, MDN. Not yet checked against the
  behavioural literature.
- Memory: 80 steps did not fit on the 3080 without checkpointing (the gradient is identical to
  7e-7).
