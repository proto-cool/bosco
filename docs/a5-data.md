# A5 data: the cleaned sets

`scripts/a5_data.py`. Leakage checked by embedding: the share of val/test items whose nearest training item has cosine > 0.95.

| part | train | val | test | positive (test) | val/test near-duplicates of train |
|---|---|---|---|---|---|
| sweet | 4614 | 977 | 968 | 0.52 | 0.3% / 0.2% |
| dangerous | 7997 | 1015 | 982 | 0.49 | 0.2% / 0.0% |
| junk | 3663 | 758 | 738 | 0.10 | 0.0% / 0.0% |
| pictures | 387 | 108 | 121 | 0.69 | 0.9% / 0.0% |

Probe pictures (whole themes held out: Dessert, Garbage dump): 16, never in any split.
