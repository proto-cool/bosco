# Specialist gate 4b: p(harassment) (pre-registered 2026-09-27, before its sealed test has been scored)

Gate 4 deferred harassment in its amendment 1. This is that follow-up. Nick chose route 1: harassment means **"is
this text harassing, insulting or attacking someone?"** It is content moderation (decision 35), not Aegis's
"requests about harassing someone". The development work is in `docs/harassment-dev.md`, on validation and dev
parts only.

## Anatomy check
As gate 4:
- the v1 brain, family v1, the pinned encoder and the fixed antenna;
- a yes/no T-maze (design A), read from the approach and avoid DN groups.

## What changed from gate 4's harassment, and why (all from dev work, none from a sealed test)
- **Held-out source: Wikipedia Detox Personal Attacks** instead of Aegis. It matches the definition. Aegis
  stays sealed and unused for harassment.
- **Pool: Civil Comments only.** DynaHate and HatemojiBuild are dropped for this expert. They label insulting
  text "no", which cut transfer from 0.828 to 0.713 on Wikipedia val. They stay in the other harm experts.
- **Labels (recipe B):**
  - **Yes:** toxicity ≥ 0.5 and insult ≥ 0.5, including comments that are also hate, threat or sexual.
  - **No:** clean comments (toxicity and every subtype 0), and hate, threat or sexual comments with no insult.
  - **Held-out labels:** yes when most raters marked a personal attack, no when none did. Mixed items are left out.
- **One source in the pool.** Decision 29 asks for several; this expert trains on one platform and is tested on
  another. The card says so.

## Data and leakage check
`scripts/v1_gate4b_data.py` → `data/cache/v1-gate4b` (items.json sha256 `3b06845d…ca126`).

| part | source | items | yes |
|---|---|---|---|
| train (balanced) | Civil Comments train | 9,958 | 4,979 |
| val | Civil Comments validation | 2,961 | 1,967 |
| held-out dev (20%) | Wikipedia Detox | 5,291 | 2,936 |
| **sealed test (80%)** | Wikipedia Detox | **21,995** | **12,438** |

- The Wikipedia split is by rev id: gate 4's val hash plus a new hash makes the 20% dev part.
- Civil Comments rows near-duplicating any Wikipedia row (cosine > 0.95) were dropped.
- Civil Comments val rows near train were dropped.
- Wikipedia dev rows near the sealed test were excluded: 272 in all.
- The same PII scrub as gate 4. Detox `worker_id` is never read.

## Training and scoring
- As gate 4:
  - Adam 3e-3, the KC penalty and seed 1;
  - up to 8 epochs, early stop after 3, the epoch chosen on Civil Comments validation;
  - the preflight as gate 2;
  - the temperature from CPU logits of up to 1,000 validation items.
- The sealed test is scored once, on the CPU, deterministically.

## Rule (fixed now)
- **Ships** if its sealed test balanced accuracy ≥ **0.754** and its ECE ≤ 0.10.
- The bar is gate 4's disputed-labels rule, **max(0.9 × ceiling, floor 0.70)**. The ceiling is the plain
  class-balanced logistic on 768 encoder numbers, trained on this pool and scored on the held-out dev part:
  0.838 (`runs/harassment-dev/ceilings.json`). So the bar is 0.9 × 0.838 = 0.754.
  - Gate 4's amendment 1 said "bar unchanged (0.70)" while Aegis was the test. With the new held-out source the
    same rule gives 0.754, and we hold it to the rule, which is the stricter bar.
- **Report-only:** Civil Comments val, the Wikipedia dev part, and the specificity on non-insulting hate, threat
  and sexual comments (the share it calls "no").

## Disclosure
The dev-part ceilings for every recipe tried were seen before this was written (68 threshold recipes, pools and
ConvAbuse dev; `runs/harassment-dev/`). The sealed Wikipedia test was never scored.

Runner: `scripts/v1_gate2.py --gate 4b …`. Results: `docs/specialist-gate-4b-results.md`.
