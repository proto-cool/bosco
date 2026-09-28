# Audit 2026-09-28: evaluation methodology

Adversarial audit of how the specialists were evaluated: gates 2, 3, 4, 4b and harm-dev, the runner
(`scripts/v1_gate2.py`), the metric code (`v1_pilot.py`, `a4_broad.py`, `a4b_dev.py`), the data builders and the
docs. It was read-only. No training ran and no gate-4 or 4b sealed test was scored or read. Gate-2 and gate-3 test
scores were inspected. Their files were deleted from `runs/` during the audit (commit `cc31052`), so they were
recovered read-only from `cc31052^` with `git archive` into `/tmp/aud`. The gate-3 weights were never committed and
are gone, so the gate-3 brain checks below use proxies. Those proxies are shown to agree with the brain on the test.

## Verdicts on the shipped specialists (registry as of `cc31052^`)

| specialist | gate | test n (per class) | BA | 95% CI (stratified bootstrap, 2000) | bar | ECE | verdict |
|---|---|---|---|---|---|---|---|
| topic | 2 | 998 (≥57/class, 14) | 0.8837 | 0.864–0.903 | 0.80 | 0.027 | **SOUND** |
| support | 2 | 738 (≥12/class, 18) | 0.8288 | **0.795**–0.859 | 0.80 | 0.035 | **SOUND-WITH-CAVEAT** |
| junk | 2 | 816 (**61 junk** / 755) | 0.9064 | 0.859–0.948 | 0.80 | 0.023 | **SOUND-WITH-CAVEAT** |
| hate | 2 | 817 (438 / 379) | 0.6938 | 0.662–0.725 | 0.658 | 0.021 | **SOUND-WITH-CAVEAT** |
| politeness | 2 | 599 (308 / 291) | **0.59958** | 0.560–0.636 | 0.5994 (code) / **0.600 (pre-reg)** / **0.59962 (unrounded)** | 0.028 | **UNSOUND** |
| kind | 3 | 1084 (≥140/class, 7) | 0.9402 | 0.925–0.954 | 0.80 | 0.020 | **SOUND-WITH-CAVEAT** |
| food | 3 | 934 (468 / 466) | 0.9914 | 0.985–0.997 | 0.80 | 0.006 | **SOUND-WITH-CAVEAT** |
| danger | 3 | 1081 (525 / 556) | 0.9739 | 0.964–0.983 | 0.80 | 0.009 | **UNSOUND** (as "Is this dangerous?") |

BA and ECE were recomputed independently from `score-test-*.json` (`/tmp/aud1.py`). They match the cards and results
docs to 4 decimals.

---

## Findings, ranked by severity

### S1 (critical). politeness did not meet its pre-registered bar, but it shipped
- **The bar as written.** `docs/SPECIALIST-GATE-2.md` (the table, pre-reg tag `specialist-gate-2-prereg`) gives
  politeness's bar as **0.600** ("0.600 (disputed labels: 0.9 × 0.666)"). The brain scored **0.59958**.
- **The bar in code.** The runner hard-codes a rounded ceiling: `scripts/v1_gate2.py:31`
  `"politeness": 0.9 * 0.666` = 0.5994. The export recomputes the verdict with the same constant
  (`export_family.py gate_verdict`).
- **The unrounded ceiling.** I re-measured it exactly as the doc defines it: the best logistic readout on 768 numbers,
  on validation, C ∈ {0.1, 1, 10} (`/tmp/aud2.py`). It is **0.66624**, so the bar is **0.59962**. That is
  **above** 0.59958.
- **So** politeness passes only under the code's rounded constant. It fails under the pre-reg text and under the
  unrounded rule.
- **The margin is misreported.** The results doc says "passes by 0.0006", and the card's limit says "Passed its bar by
  0.0006". The true margin is 0.00018 against 0.5994, −0.00004 against the unrounded bar, and −0.0004 against the
  written 0.600.
- **It is within noise either way.** The 95% CI is 0.560–0.636, about ±0.04 around a 0.0002 margin. The validation
  curve (0.571, 0.578, **0.617**, 0.553, 0.581, 0.608 on 299 items) shows epoch selection is mostly noise.
- **Recommendation:** withdraw politeness, or re-gate it on fresh data with a pre-registered margin or CI rule.

### S2 (high). danger (gate 3) measures "which kind of thing", not "is it dangerous"
- **How the set was built.** `scripts/v1_gate3_fetch.py` (build, `negatives()`, lines ~794–856) draws positives from
  disasters (events), infectious diseases, toxic chemicals, venomous animals (species) and weapons. Negatives come
  only from the kind classes person, place, organisation, creative work and product (`neg_by_kind` in
  `data/raw/clean/wiki/SUMMARY.json`). **No negative is ever an event, a species, a chemical or a disease.**
- **The test cannot see the problem.** The test has the same construction, so 0.974 only says the model separates
  those categories.
- **What the learned rule does off that distribution.** The data's own linear readouts are good stand-ins for the
  brain: the prototype agrees with the brain's test picks 97.3% of the time and the logistic 97.8% (`/tmp/aud12.py`).
  Trained on danger/train and applied to kind val and test subjects that are not in the danger set (`/tmp/aud4.py`),
  they call **89% (prototype) / 59% (logistic) of ordinary species "dangerous"**, and 25–35% of events. Some events
  (wars, battles) arguably are dangerous.
- **The card does not say this.** Its limits cover Wikipedia leads and debatable members, not the confound.
- **Gate 4 inherits it.** Gate 4's danger pool includes "Wikipedia danger set".
- **Verdict: UNSOUND** for the question it is served under.
- **food has the milder form.** Negatives also exclude species, and the proxies call 23% (prototype) / 34%
  (logistic) of non-food species "food or drink". Caveat, not unsound, because "is this species food" is partly
  ambiguous.

### S3 (high). "Passes" decided by point estimates on small tests, with no uncertainty rule
- **No gate has an uncertainty rule.** None of the gate 2, 3, 4 or 4b pre-regs says what to do about sampling error.
- **Three shipped passes sit inside noise:**
  - politeness: CI 0.560–0.636 around a bar of 0.599;
  - support: CI 0.795–0.859, which crosses 0.80;
  - hate: CI lower bound 0.662 against a bar of 0.658.
- **junk's positive class is tiny.** It has 61 test items, which gives a CI half-width of about 0.045.
- **The docs never report test n.** They quote pre-drop sizes (1,000). The real sizes are junk 816, hate 817,
  support 738 and intent 686.
- **Recommendation:** pre-register a margin, or require the lower CI bound ≥ bar, and report n per class.

### S4 (medium). The label-blind near-duplicate drop biased two gate-2 scores
- **What the drop does.** `v1_data.py cmd_embed` removes val/test items whose cosine to any training item is > 0.95,
  whatever the label. I rebuilt the pre-drop samples deterministically, embedded the dropped items with the pinned
  encoder and scored them with the committed gate-2 weights (`/tmp/aud10.py`, `/tmp/aud11.py`). The kept-set scores
  reproduce exactly, which checks the reconstruction.
- **hate was inflated:**
  - 183 of 1,000 test items were dropped;
  - 148 of them are DynaHate perturbations whose `acl.id.matched` partner is in **train with the opposite label**;
  - the brain scores 0.642 on the dropped items, and **0.684 on all 1,000** against the reported 0.694;
  - it still passes (bar 0.658), but by 0.026, not 0.036.
  - DynaHate's official split puts contrast pairs across train and test: 448 of the kept 817 test items have an
    opposite-label partner in train. This deflates the score and is by design, but the gate docs never say it.
- **junk was deflated:** 184 dropped (87 of them spam; the brain scores 0.995 on them). On all 1,000 it scores 0.951.
  The test's spam falls from 148 to 61, the least templated spam.
- **Recommendation:** drop test items near train only when the labels agree, or report both numbers.

### S5 (medium). Test items were reused across the pilot and gate 2, including a retest after a failure
- **The overlap.** The specialist-pilot test sets were scored and read (`docs/specialist-pilot-results.md` at
  `cc31052^`). Their exact texts overlap gate-2 tests (`/tmp/aud9.py`): **support 121/738, hate 108/817,
  intent 64/686**, topic 8.
- **The redesign followed the pilot results:**
  - the gate-2 recipes (all options, more data, design B) came after the pilot test results;
  - hate failed the pilot by 0.001 and was re-gated on a test set that shares 13% of its items.
- **Size of the problem.** The exposure is aggregate only, with no per-item tuning, so it is small. It breaks "a fresh
  sealed test per gate" and is not disclosed in `SPECIALIST-GATE-2.md`.

### S6 (medium). Gate 4 and 4b procedure (nothing has shipped)
- **Harm experts trained before the pre-reg.** They were trained in harm-dev before gate 4 was pre-registered (harm-dev
  weights 12:33–14:37, 27 Sep; pre-reg final 21:12). Their Aegis-dev scores (0.821, 0.794, 0.816, 0.843, 0.672) were
  seen before the floor (0.70) was fixed. That is disclosed, and the floor only raised bars. Still, the gate "ran"
  before it was written down (CLAUDE.md: "Run a gate that is not written down first").
- **Dev and test are not independent for MultiDoGO and SIB-200.** Near-duplicates across held-out dev and sealed test
  (text only, no labels read; `/tmp/aud8.py`):
  - **MultiDoGO (problem/social): 801 of 3,092 test items have a dev twin at cos > 0.95**, and there are 5,255
    test–test pairs;
  - SIB-200 (language): 231;
  - CFPB: 18.
  
  The questions and the language merge were chosen on dev, so dev is not independent of test for these. The
  harm-tone build moves dev twins into test; the intent-finance and topic-language-danger builds do not.
- **The gate-4 sealed tests are no longer pristine.** 13 of 14 were scored before the halt, never read, and the files
  are now deleted. Any re-run is a second scoring and has to be declared in the next pre-reg.
- **4b chose its held-out source and recipe after looking.** It chose the held-out source (Wiki Detox over Aegis, CC
  and ConvAbuse) and the recipe (B, from 68) after seeing the dev ceilings for each (disclosed).
- **The 4b test keeps only clear-cut items.** Items where a minority of raters marked an attack are excluded
  (`lab_wiki`), which makes it easier than natural text. `harassment-dev.md` measures this at +0.02–0.03.
- **4b's sealed rows were used as training rows elsewhere.** They trained the exploratory Wiki→CC runs and the gate-4
  harm pool. That is disclosed, and it is not leakage into the CC-only expert.

### S7 (medium-low). The hate ceiling was not computed as the doc says
The pre-reg says hate's ceiling is "the best logistic readout on all 768 encoder numbers". 0.731 reproduces only
when the logistic is fit on the **first 20,000 of 32,924** training items. On all of train the ceiling is 0.740, so
the bar would be **0.666**, not 0.658. hate still passes (0.694). A stricter reading shrinks the margin to 0.028.

### S8 (low). Calibration and training details
- **ECE code is correct** (`v1_pilot.py:337`): top-label, 15 equal-width bins, weighted by bin share. `linspace` float
  error leaves open gaps at exactly 0.4 and 0.8, so a pmax exactly equal to either is dropped. That is negligible.
- **The temperature grid is adequate** (`v1_pilot.py:245`): 60 log-spaced points over [0.05, 20], with a 10.7% step.
  Every fitted T is interior (0.78–1.05).
- **Gates 2 and 3 fit T on GPU validation logits** from the early-stopping epoch, the same small val used to pick the
  epoch (politeness 299, hate 376). This is acknowledged in the gate-2 reading, and gate 4 fixed it.
- **Balanced accuracy is correct.** `B.balanced` takes the mean recall over the gold classes present, grouped by
  `kind`. In the runner `kind == task`, so `D.macro` averages a single key. An independent implementation matches on
  all 9 files.
  - For intent, only 141 of 151 classes appear in the test, some with a single item. This is not shipped.
- **Ties in argmax go to option 0.** They are immaterial with continuous logits.

### S9 (low). Leakage checks that came back clean or near-clean
- **The family antenna** (label-free PCA/whitening on 4,000 pilot-train texts, `export_family.antenna`) shares exact
  texts with at most one sealed-test item per task: one support item, and one each in gate-4 problem, social and
  intent. It is negligible.
- **Exact duplicates across splits are none,** for gates 2 and 3, test ∩ train. Val and test share 0–9 near-twins
  per task.
- **Gate 3 splits** are by a QID hash with cross-task consistency, a page/text dedupe and 27 near-dups dropped.
- **Template siblings remain at cos 0.90–0.95.** These are same-template paraphrases: support 303/738 (97% same
  label), danger 129/1081 and topic 25. This is mild optimism, not leakage.

### S10 (low). Overclaims and omissions in docs and cards
- **politeness margin:** "passes by 0.0006" (S1).
- **Results docs give no test n and no uncertainty** (S3).
- **kind** persons are only people born before 1900 and dead. The card does not say so.
- **danger and food** cards omit the negative-class confound (S2).
- **hate** card omits that DynaHate contrast pairs straddle train/test, and the drop effect (S4).

## Sealing and pre-registration adherence (gates 2 and 3): clean
- **Order of events is right.** Every `score-test-*` file was created after its pre-reg tag:
  - gate 2: pre-reg 26 Sep 14:47, scores 20:42–02:15;
  - gate 3: pre-reg 26 Sep 18:06:57, scores 03:26–03:51.
- **Each test was scored once.** The scorer skips tasks that already have a score (`runs/g2-scorer.sh`,
  `g3-scorer.sh`).
- **Nothing changed after the pre-reg.** The `BAR` dict has not changed since the runner commit (`47cb1a4`, 26 Sep
  15:00). The pre-reg docs have not changed since their tags.
- **Gate 3's baselines were computed on its test 22 s before its pre-reg commit.** This is disclosed.
- **Training matched the pre-reg:**
  - epochs and patience match (A ≤ 8, stop after 3; B ≤ 5, stop after 2), per the `hist` arrays;
  - preflights passed, and plain failed on `loss_falls` and was reported;
  - scoring ran on the CPU with deterministic algorithms.

## Commands
Analyses are in `/tmp/aud1.py`–`/tmp/aud12.py`. Gate-2/3 runs were restored with
`git archive cc31052^ runs/specialist-gate-2 runs/specialist-gate-3 … | tar -x -C /tmp/aud`. For S4, the dropped items
were embedded with
`uv run --with transformers==4.46.3 --with sentence-transformers==3.3.1 --with einops python /tmp/aud11.py`.
No gate-4 or 4b labels were read. `/tmp/aud8.py` loads only task, split, source and text.
