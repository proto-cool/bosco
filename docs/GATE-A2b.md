# Gate A2b — broader eyes (pre-registered 2026-09-24)

A2 carried every text question near its ceiling, but pictures reached 0.780
against a bar of 0.880. The cause was the fly's own anatomy: pictures entered
only the 335 visual Kenyon cells, which reach a few compartments. Scrambled
brains, where pictures reach everything, scored 0.91–0.92.

## Decision recorded (Nick, 2026-09-24, after A2's numbers)

"We need better picture accuracy. I don't care if it's less fly." Two
consequences, both stated here because they come after the numbers:

1. **The eyes may be wired more broadly than a fly's.** The docs and the API
   say which part is not fly anatomy.
2. **Adopting Bosco as the decider no longer requires the real wiring to beat
   the scrambled wiring** (`CLAUDE.md`, "a decision service only if the
   connectome beats the shuffle"; A2's D2). The controls still run, and their
   numbers are published beside his. It is a finding, not a gate.

## Arms

A2 exactly (brain, nose, question in the smell, KC sparseness pressure,
training, data, splits), with only the eyes changed:

| eyes | what a picture drives |
|---|---|
| **all_kc** | every Kenyon cell (4,064), each the mean of 6 of the 52 picture channels, drawn once at random (the same seed as A2). Not fly anatomy. |
| **antenna** | the 52 smell channels, added to the question's smell as text is (pictures use their own jina-clip-v2 projection). Not fly anatomy. |

Runs: real × seeds 1 and 2, and shuffle × seed 1, for each: six runs, about
three hours.

## Decision rule (real, mean over seeds)

1. The eyes are chosen by **validation** pictures balanced accuracy.
2. **Adopt** the chosen brain as Bosco's decider if its **test** pictures are
   ≥ **0.88** (A2's bar) **and** no text question falls more than 0.03 below
   A2's real brain (sweet 0.844, dangerous 0.747, junk 0.905).
3. Otherwise report it, and keep A2's brain with pictures as its stated
   weakness.

Runner: `scripts/gate_a2.py run --eyes {all_kc,antenna}`, `report-b`.
Results: `runs/gate-a2b/`, `docs/gate-a2b-results.md`.
