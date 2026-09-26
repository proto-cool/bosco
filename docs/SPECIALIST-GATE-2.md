# Specialist gate 2 (DRAFT, 2026-09-26; locked before any run, after the prototype round reports)

The recipe from the gap-closing round (`docs/GAP-DEV.md`) on the specialists that can reach the
production bar (decision 23: balanced accuracy ≥ 0.80 on the sealed test scored on the CPU, ECE ≤ 0.10;
disputed-label tasks at 0.9 × their information ceiling).

## What is fixed for every specialist (one family)
- **Family v1** (`service/families/v1`, `scripts/export_family.py`): the v1 brain, the homeostatic start at
  gain 4, the DN read (80 steps, the last 8 read), and **the family's fixed antenna** (bi46 fit on the pilot's
  data). Development rounds refit their own antenna; shipped specialists must not, so they all share one brain
  and one brain map.
- The pinned nomic text encoder; clean data only (`docs/clean-data.md`).

## Candidates and recipes (to be finalised from the prototype round)
| specialist | data | option smells | training |
|---|---|---|---|
| **topic** | DBpedia-14, 10,000 train | label words (validation 0.933 in the gap round) | all 14 options, up to 8 epochs, early stop |
| **intent (CLINC)** | CLINC150, 15,100 train | **prototypes** if the prototype round clears ~0.80 on validation; else not in this gate | right + 10 hardest + 9 random options, up to 4 epochs |
| **support area (MASSIVE scenarios)** | MASSIVE en-US, 18 coarse scenarios | prototypes | as above |
| **hate** | DynaHate v0.2.3, all 32k train | label words | 2 options, up to 8 epochs |

## Arms
Each specialist real (the one that ships). Reported beside it, not gating: nose alone, prototype
matching alone (for prototype specialists, so it is clear how much the stimulus carries), the plain
baseline. No shuffled twins (decision 22).

## Rules
- The test sets stay sealed until scoring (CPU, deterministic, once). Smoke runs and preflights use
  training and validation only.
- The preflight is as in the specialist pilot.
- A specialist ships to Ask Bosco only if it passes the bar. Its model card lists its data and licences,
  its numbers beside the report-only comparisons, and plain-language limits.

Runner: to be written (`scripts/v1_gate2.py`), reusing `v1_pilot.py` and `v1_gap.py`.
