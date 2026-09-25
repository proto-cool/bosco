# Effort pilot: does thinking longer help him? (DRAFT, 2026-09-25)

**Status: draft.** It becomes a pre-registration once A4 broad has reported, because
which brain and which question kinds it runs on depend on that result. The numbers
below are proposals until then. Nick: "what if we added a high/medium/low effort, à la
you."

## The idea

An **effort** setting on every question: low, medium or high. At higher effort he keeps
sniffing and adds up the evidence until he is sure enough, or his budget runs out.
- **The fly behind it:** flies trade speed for accuracy. On harder smell
  discriminations they take longer to decide, accumulating evidence before they commit
  (DasGupta, Ferreira & Miesenböck, *Science* 2014, FoxP and evidence accumulation).
- **The product:** the brain panel can show him wavering and then committing. The
  answer reports how many sniffs he took.
- **The rule it must pass:** an effort level that is slower without being better does
  not ship. Showing it would misstate what the setting does.

## Why this needs a pilot

The brain is deterministic: the same sniff twice gives exactly the same answer. More
sniffs only add information if each sniff differs. Earlier evidence is not
encouraging:
- A3 phase 1: two separate sniffs (question, then text) gave no gain.
- Passage check: sniffing sentence by sentence gave no gain.

## Anatomy check

- **Where the variability comes from:** ORNs are noisy from trial to trial, and a
  fly's sniff of the same odour never drives its receptors exactly the same way twice.
  Here: independent noise on each glomerulus's input, per sniff, around the item's
  smell. It stays inside the nose; the brain and the read are untouched.
- **Where the adding up happens:** across sniffs, outside the brain. His lean per
  option is summed over sniffs. A fly's accumulator is thought to sit downstream of the
  mushroom body, in circuits this model does not simulate; the sum stands in for it,
  and the docs will say so.
- **Out of this pilot:** a longer settling time (400 or 800 ms per sniff). It needs
  retraining at the new length, so it gets its own gate if this pilot shows effort
  helps at all.

## Arms (on the brain that A4 broad ships, frozen; nothing is retrained)

| level | sniffs | stops when |
|---|---|---|
| **low** | 1, clean (no noise) | at once (today's answer) |
| **medium** | up to 4, noisy | the top option's probability from the summed lean ≥ 0.80 |
| **high** | up to 16, noisy | ≥ 0.90 |
| fixed-N (reference) | exactly 4 and exactly 16, noisy, no early stop | never early; separates "more sniffs" from "stopping when sure" |

Beside them, the layered brain under the same levels (does effort help any brain, or
this wiring?). As a reference only, the plain net under the same input noise, averaged
over the same numbers of draws.

## Noise and determinism

- **Noise size σ:** a grid of {0.02, 0.05, 0.10} on the 0–1 glomerulus scale, chosen
  once on the taught kinds' **validation** set by high-effort balanced accuracy. The
  test set is never used to choose it.
- **Same input, same answer:** the noise is drawn from a generator seeded by a hash of
  (version, the smelled input, the question, effort level). So a repeat request gives
  the identical answer and trace. This keeps the API's determinism promise (`API.md`),
  which Jev does not make.

## Data

The A4 broad taught kinds: the test split for scores, the validation split for σ. The
BTZSC cold tiers are reported for every level, not gating. No new data.

## Metrics

- **Balanced accuracy,** macro over kinds.
- **ECE** (calibration), since averaging noisy sniffs may mainly improve how well he
  knows when he is unsure.
- **Sniffs used,** mean and 90th percentile, and the share stopped early.
- **Server time per question** at each level, on the Kimsufi's CPU (or its measured
  stand-in).
- For the UI: the per-sniff lean trajectory, kept on a sample for the brain panel.

## Rules (proposed; fixed when this becomes a pre-registration)

- **E1, effort helps:** on the test set, high ≥ low + 0.02 macro balanced accuracy,
  **or** high improves ECE by ≥ 0.02 while its accuracy is no worse than low − 0.005.
- **E2, it is affordable:** high's mean server time ≤ 8× low's.
- **What ships:**
  - If E1 and E2 pass, low, medium and high ship. Medium ships only if it sits between
    them on the E1 measure.
  - If E1 fails, there is no effort setting; the report says plainly that more sniffs
    did not help this brain.
  - If E1 passes and E2 fails, only low and medium ship, with medium's own E1 checked.
- **Reported beside, not gating:** real vs layered; fixed-N vs early-stop; the cold
  tiers.

## Open until pre-registration

- The brain and kinds, from A4 broad's result.
- The server-time measurement (task #6, CPU speed) has to exist first for E2.
- Whether `approach` (two sides) and `choose` (many options) share one stopping rule.
  Proposed: yes, via the top option's probability.

Runner (to write): `scripts/effort_pilot.py`. Results: `docs/effort-pilot-results.md`.
