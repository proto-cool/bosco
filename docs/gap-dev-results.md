# Closing the training gap: results (development, validation only; 2026-09-26)

Plan: `docs/GAP-DEV.md` (with addenda). All numbers are balanced accuracy on validation; the sealed test sets
were not touched.

| run | design | option smells | data | best validation (epoch) |
|---|---|---|---|---|
| topic, specialist pilot | A | label words, 10 sampled | 3,000 | 0.838 (4) |
| **topic, gap** | A | label words, **all 14** | 10,000 | **0.933** (7) |
| CLINC, specialist pilot | A | label words, 10 sampled | 3,000 | 0.311 (4) |
| CLINC, gap | A | label words, 20 (10 hardest + 9 random + right) | 15,100 | 0.429 (4) |
| CLINC, gap | A | **prototypes**, 20 | 15,100 | 0.635 (3) |
| **CLINC, gap** | **B** (one memory per option, item alone) | none (the item's smell alone) | 15,100 | **0.823** (3) |
| MASSIVE, gap | A, prototypes | — | — | cancelled: superseded by B (decision 27) |

Ceilings through the 46-glomerulus nose (validation): topic 0.967; CLINC 0.911; prototype matching alone 0.918
(CLINC) and 0.949 (topic).

**Reading:** the pilot's shortfalls were the fly's training, not his senses. More data and every option fixed
topic. For many options, judging a match from a mixed smell was the bottleneck; one memory per option (B)
removed it. Both recipes are in specialist gate 2 (`docs/SPECIALIST-GATE-2.md`).
