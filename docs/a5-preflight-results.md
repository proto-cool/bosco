# A5 preflight

`scripts/a5_preflight.py`; the checks and bars are in `docs/A5-PREFLIGHT.md`. Training data only.

| config | pass | KC at start | MBON at start | output spread | loss first 10 → last 10 | AUROC after 30 batches | KC after |
|---|---|---|---|---|---|---|---|
| real-type-full | pass | 0.050 | 0.201 | 0.113 | 0.679 → 0.625 | 0.714 | 0.021 |
| real-type-cut | pass | 0.050 | 0.200 | 0.049 | 0.677 → 0.622 | 0.752 | 0.045 |
