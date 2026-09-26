# Specialist gate 2 (pre-registered 2026-09-26, before any gate-2 test score exists)

The first specialists meant for Ask Bosco. It uses the recipes from the development rounds
(`docs/GAP-DEV.md`) and decisions 22–27 (`docs/DECISIONS-2026-09-25.md`).

## Anatomy check
- **The v1 brain** (`ratebrain3`, `v1.py`): graded unit, homeostatic start at gain 4, the KC fraction
  penalty, per-type parameters and KC→MBON synapses trained, 80 steps (400 ms). **The answer is read
  from the descending neurons:** the anatomical approach and avoid groups (`v1.dn_groups`), the last 8
  steps.
- **One family:**
  - **the fixed family antenna** (`service/families/v1/antenna.npz`: bi46, fit label-free on the
    specialist pilot's data), applied to every specialist's text, so they all share one brain and
    one brain map;
  - the pinned nomic-embed-text-v1.5 encoder (prefix "classification: ").
- **Two designs** (decision 27):
  - **A, T-maze** (few options): one sniff per option, the item's smell mixed with the option word's
    smell from rest; he goes down one arm. Trained on **every option**.
  - **B, one memory per option** (many options): each option has its own KC→MBON memory, and each
    sniff is **the item's smell alone** with that option's memory. Trained on the right option plus
    19 random others per item.

  In both, the answer is a softmax over the item's options (the full option set at validation and
  test).

## Data and leakage check
`scripts/v1_gate2_data.py` builds `data/cache/v1-gate2`. Clean sources only (`docs/clean-data.md`);
personal data is removed on read (SMS numbers → "<number>"). Validation and test items that
near-duplicate a training item of the same task (cosine > 0.95) are dropped: 1,525 in all.

| specialist | design | source | options | train / val / test (before the near-duplicate drop) | bar |
|---|---|---|---|---|---|
| **topic** | A | DBpedia-14 | 14 | 10,000 / 500 / 1,000 | 0.80 |
| **intent** | B | CLINC150 + out of scope | 151 | 15,100 / 500 / 1,000 | 0.80 |
| **support** (area) | B | MASSIVE 1.1 en-US scenarios | 18 | 11,514 / 500 / 1,000 | 0.80 |
| **junk** | A | SMS Spam Collection (scrubbed) | 2 | 4,074 / 500 / 1,000 | 0.80 |
| **hate** | A | DynaHate v0.2.3, all training | 2 | 32,924 / 500 / 1,000 | **0.658** (disputed labels: 0.9 × 0.731) |
| **politeness** | A | SE Politeness, top vs bottom quartile | 2 | 2,402 / 300 / 600 | **0.600** (disputed labels: 0.9 × 0.666) |

**Mood is deferred** (Nick): DynaSent round 2 is adversarial by construction (2-way ceiling 0.72). It
waits for a non-adversarial clean dataset. **Disputed labels** (Nick) apply to hate and politeness. Their
ceilings are the best logistic readout on all 768 encoder numbers, on validation, C ∈ {0.1, 1, 10}
(measured 2026-09-26, before any gate-2 test score).

## Training (the 3080)
- Adam 3e-3, the KC penalty and seed 1; the epoch is chosen on validation (balanced accuracy, full
  option set).
- **A:** up to 8 epochs, early stop after 3 without improvement.
- **B:** up to 5 epochs, early stop after 2.
- A temperature is fit on validation, for calibration only.
- **Preflight as in the specialist pilot** (KCs 2–15% at the start, a live read, the loss falls,
  gradients reach ≥ 90% of cell types, held-back training items above chance). A failing run stops
  and is reported.
- Smoke runs touch training and validation only.

## Scoring
Each specialist's sealed test is scored **once, on the CPU, with deterministic algorithms**, as the
service will serve it.

## Rules (fixed now)
- **A specialist ships to Ask Bosco** if its test balanced accuracy ≥ its bar **and** ECE ≤ 0.10.
- **Report-only:** nose alone (option words, A); prototype matching alone (B), so it is clear how much the
  stimulus carries; the plain baseline (logistic on 768); CPU time per question.
- **Every shipped specialist gets a model card:**
  - its data and licences, and its disputed-labels note if any;
  - its numbers beside the report-only comparisons;
  - plain-language limits.

Runner: `scripts/v1_gate2.py`. Results: `docs/specialist-gate-2-results.md`.
