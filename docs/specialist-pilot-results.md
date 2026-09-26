# Specialist pilot results

Pre-registration: `docs/SPECIALIST-PILOT.md`. Test sets, balanced accuracy (ECE after temperature); brains scored on the CPU.

Near-duplicates dropped from val/test: 753.

| task | options | chance | nose alone | plain baseline | specialist real | specialist layered | shared real | S1 |
|---|---|---|---|---|---|---|---|---|
| hate | 2 | 0.500 | 0.525 | 0.682 | 0.625 (ECE 0.037) | 0.594 (ECE 0.044) | 0.589 (ECE 0.074) | no |
| hatemoji | 2 | 0.500 | 0.531 | 0.581 | — | — | 0.539 (ECE 0.109) | — |
| topic | 14 | 0.071 | 0.749 | 0.970 | 0.801 (ECE 0.039) | 0.824 (ECE 0.036) | 0.715 (ECE 0.098) | QUALIFIES |
| intent_massive | 60 | 0.017 | 0.600 | 0.776 | 0.257 (ECE 0.059) | 0.284 (ECE 0.067) | 0.358 (ECE 0.068) | no |
| intent_clinc | 151 | 0.007 | 0.641 | 0.901 | 0.294 (ECE 0.069) | 0.290 (ECE 0.058) | 0.253 (ECE 0.114) | no |

CPU time per sniff (80 steps, batched): median 52.0 ms.

## Decision (S1)

- hate: does not qualify
- hatemoji: —
- topic: qualifies
- intent_massive: does not qualify
- intent_clinc: does not qualify

## Reading (after the numbers; rules as amended before any test score was read)

- **One specialist qualifies: topic,** at 0.801 (bar 0.80) with ECE 0.039, a real fly brain
  telling 14 topics apart on sealed test data, scored on the CPU as served.
- **hate misses its disputed-labels bar by 0.001** (0.625 vs 0.9 × 0.696 = 0.626). The rule stands:
  it does not qualify in this pilot. Its calibration is good (ECE 0.037).
- **Both intent specialists fail by far** (MASSIVE 0.257, CLINC 0.294), as their validation
  predicted: 4 epochs on 3,000 examples, trained on 10 sampled options and tested on 60 and 151.
- **hatemoji** failed preflight on both brains; only the shared brain scored it (0.539).
- **One brain per task beats one brain for all** on topic (0.80 vs 0.72) and hate (0.63 vs 0.59),
  but not on MASSIVE (0.26 vs 0.36): sharing helped the two intent sets a little.
- **Real vs layered: tied** (topic 0.80 vs 0.82, hate 0.63 vs 0.59, CLINC 0.29 vs 0.29, MASSIVE
  0.26 vs 0.28), all within replay noise. The first fair comparison, now with equal learning synapses
  and own-input scaling, says the fly's *kind* of brain carries these tasks, not its exact map
  (decision 22: model-card footnote).
- **Speed:** 52 ms per sniff on the Mac's CPU (80 steps, batched). A 2-option question is about
  0.1 s of brain; 14 options about 0.7 s; 151 options about 8 s.
- **Where the gap is (development ceilings, `docs/a4b-dev-results.md` style, on validation):** the
  nose carries topic to 0.97 and CLINC to 0.91, so the intent failures are the fly's training, not
  his senses. The gap-closing round (`docs/GAP-DEV.md`, running) already reached **0.925 on topic's
  validation** with more data and every option in training. That recipe goes into the next
  pre-registered specialist gate.
