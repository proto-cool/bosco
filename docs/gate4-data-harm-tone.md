# Gate 4 data: harm and tone

Plan: `docs/research-gate4/harm-tone.md`. Decisions: 23 (bar), 28 (share-alike), 29 (fields, breadth test),
30 (Civil Comments accepted, ethics gaps on the card). Script: `scripts/v1_gate4_harm_tone.py`
(`fetch`, `build`, `embed`, `ceilings`). Output: `data/cache/v1-gate4-harm-tone/` (`items.json`, `emb.npz`).
Provenance rows: `docs/gate4-data-manifest-harm-tone.json`.

This is data and an encoder-ceiling preflight only. Nothing here trains the fly.

**Sections 1–5 were written on 2026-09-27 before `build` was run and before any model score was
computed.** At that point, the only things read from the data were column names, label value counts and
the HH-RLHF match rate for Aegis. Section 6 onward was filled in afterwards.

---

## 1. Taxonomies

### Harm: 5 options, `hate` / `harassment` / `threat` / `sexual` / `fine`

- `hate`: attacks a group, or a person as a member of a protected group.
- `harassment`: insults or attacks a person or a non-protected target.
- `threat`: threatens or incites violence. Aegis "Violence" is mapped here as a stated choice, although it is
  mostly requests ("how do I hurt…").
- `sexual`: explicit sexual content or sexual harassment.
- `fine`: none of these.

**Precedence for multi-label rows (fixed): threat > sexual > hate > harassment.**

**Excluded:**
- **Self-harm is excluded.** Any row that any source marks as suicide or self-harm is dropped, not
  relabelled. The same applies to Aegis "Sexual (minor)".
- **Profanity is tone, not harm.** A row whose only signal is obscene or profane is dropped from harm.

### Tone: 4 options, `warm` / `neutral` / `curt` / `hostile`

This is one axis, from friendly to aggressive.

- `warm`: polite, friendly, appreciative.
- `neutral`: matter-of-fact.
- `curt`: impolite, brusque, mildly rude or profane, but not attacking anyone.
- `hostile`: aggressive, insulting, attacking.

## 2. Sources and verdicts (licence re-verified at source on 2026-09-27)

| source | role | licence (where verified today) |
|---|---|---|
| Civil Comments (HF `google/civil_comments` rev `f2970eb`, already local) | harm pool, tone pool | CC0 (TFDS catalog, HF card). **Decision 30**: accepted; commenter consent and rater pay stay open, on the card |
| Wikipedia Detox: Personal Attacks (figshare 4054689 v6) | harm pool | CC0 labels (figshare API `license: CC0`, fetched today). Text **CC BY-SA** (Wikimedia Detox page) |
| Wikipedia Detox: Aggression (figshare 4267550 v5) | tone pool | same: CC0 labels (figshare API, today), text CC BY-SA |
| ConvoKit Wikipedia politeness (`wikipedia-politeness-corpus.zip`) | tone pool | CC BY 4.0 (convokit.cornell.edu/documentation/wiki_politeness.html, "Data License", fetched today). Text CC BY-SA |
| DynaHate v0.2.3 (commit `36f9dc8`, already local) | harm pool | CC BY 4.0 (README) |
| HatemojiBuild (commit `a626f63`, already local) | harm pool | CC BY 4.0 (repo LICENSE) |
| Aegis 1.0 (HF rev `bd96d86`) | **harm held-out** | CC BY 4.0 (card `license: cc-by-4.0`, "License: CC-BY-4.0", fetched today); prompts from HH-RLHF (MIT) |
| Aegis 2.0 (HF rev `d86bb8b`) | **harm held-out** | CC BY 4.0 (card, today). Only prompt-only rows whose prompt exactly matches an HH-RLHF human turn |
| Anthropic HH-RLHF (HF rev `09be8c5`; GitHub LICENSE at `c72f5ce`) | the match list for Aegis (text only; no rows used directly) | MIT (LICENSE file and HF card, today) |
| Stack Exchange politeness (ConvoKit, already local) | **tone held-out** | CC BY 4.0 (ConvoKit); SE text CC BY-SA. Decision 31 (SE accepted); the corpus is from 2013, before 2022-11 |
| HateCheck (commit `3490854`) | report-only, not scored now | CC BY 4.0 (repo LICENSE) |
| XSTest (commit `d7bb5bd`) | report-only, not scored now | CC BY 4.0 (LICENSE, readme "prompts are subject to CC-BY-4.0") |
| SimpleSafetyTests (commit `d7aee9a`) | report-only, not scored now | CC BY 4.0 (LICENSE; README "Licence is CC-BY") |

**Planned but not used:**
- **Wikipedia Toxicity Subtypes** (Jigsaw 2018). No copy from the publisher could be fetched:
  - the TFDS bucket `gs://jigsaw-unintended-bias-in-toxicity-classification/wikipedia_toxicity_subtypes.zip`
    returns HTTP 403;
  - Kaggle needs a login and acceptance of the competition rules;
  - HF `google/jigsaw_toxicity_pred` is a loader script that needs a manual download.

  Its comments are a subset of the Detox comments already used, with Jigsaw's subtype labels added. What
  is lost is the Wikipedia `threat` and `identity_hate` examples. Civil Comments covers both. Re-fetch
  with a Kaggle login if Nick wants it.
- **ConvAbuse**: the research verdict was CAUTION (implied consent on ELIZA). It is not fetched and not
  used, even as a report-only set, so the `hostile` end has **no held-out test** in this build.
- **Wikipedia Detox Toxicity**: secondary in the plan. It is not needed for the mapping.

## 3. Mappings (pre-registered)

Every threshold is on the source's own label: a fraction of raters, or a mean rater score.

### Harm

| source | hate | harassment | threat | sexual | fine | dropped |
|---|---|---|---|---|---|---|
| Civil Comments | toxicity ≥ 0.5 and identity_attack ≥ 0.5 | toxicity ≥ 0.5 and insult ≥ 0.5 | threat ≥ 0.5 | sexual_explicit ≥ 0.5 | **toxicity = 0 and every subtype = 0** | everything else, including 0 < toxicity < 0.5 and obscene-only |
| Detox Personal Attacks | — | mean(attack) ≥ 0.5 | — | — | mean(attack) = 0 (no rater marked an attack) | 0 < mean < 0.5 |
| DynaHate v0.2.3 | label = hate and type ∈ {Animosity, Derogation, Dehumanization, Support, notgiven, none} | — | label = hate and type = Threatening | — | label = nothate | — |
| HatemojiBuild | label_gold = 1 | — | — | — | label_gold = 0 | — |
| Aegis 1.0 `user_message` rows (held-out) | Hate/Identity Hate | Harassment | Violence, Threat | Sexual | Safe | see below |
| Aegis 2.0 prompt-only rows (held-out) | Hate/Identity Hate | Harassment | Violence, Threat | Sexual | prompt_label = safe and no violated category | see below |

DynaHate type names are matched case-insensitively: the v0.2.3 file writes them in lowercase (`threatening`,
`derogation`, …). This was found at the first build (0 threat rows) and fixed before any score.

**Aegis label rule.**
- **Aegis 1.0.** A category counts when more than half of the row's annotators (`num_annotations`) name it
  in their `labels_k`. `fine` needs a Safe majority and no mapped harm category.
- **Aegis 2.0.** An `unsafe` prompt uses its `violated_categories`.
- **Dropped in both:**
  - any row where any annotator (1.0) or the category list (2.0) names **Suicide and Self Harm** or
    **Sexual (minor)**;
  - unsafe rows with no mapped category (Criminal Planning, PII, Needs Caution, Profanity, substances,
    weapons, …);
  - 2.0 safe rows that still list a category;
  - rows whose text is not an HH-RLHF human turn after whitespace normalisation. This covers all of 1.0
    and 2.0, and drops DAN, AART and the REDACTED Reddit rows.
- The same prompt can appear in 1.0 and 2.0, or twice. If every copy maps to the same option, one copy is
  kept. If the copies disagree, all of them are dropped.

**DynaHate `nothate` as `fine`: a stated imperfection.** These rows are adversarial contrasts, and some of
them insult non-protected targets, which would be `harassment` here. They are kept as `fine`, capped like
every other class, because they are the only clean hard negatives for hate. This caveat goes on the card.

### Tone

| source | warm | neutral | curt | hostile | dropped |
|---|---|---|---|---|---|
| Detox Aggression (mean `aggression_score`, −3…+3) | mean ≥ +1.0 | −0.3 ≤ mean ≤ +0.3 | −1.5 < mean ≤ −0.7 | mean ≤ −1.5 | the gaps between bands |
| Wikipedia politeness (ConvoKit `Binary`) | 1 (top quartile) | 0 | −1 (bottom quartile) | — | — |
| Civil Comments | — | toxicity = 0 and every subtype = 0 | obscene ≥ 0.5 and insult < 0.2 and threat < 0.5 and identity_attack < 0.5 | toxicity ≥ 0.66 and insult ≥ 0.5 | everything else |
| **Stack Exchange politeness (held-out)** | `Binary` = 1 (top quartile of the normalised score) | `Binary` = 0 | `Binary` = −1 (bottom quartile) | — (not testable here) | — |

**Two scoring rules for Stack Exchange, fixed now:**
- Balanced accuracy on the held-out source is the mean recall over the **3 options present** (warm,
  neutral, curt).
- A `hostile` prediction counts as wrong.

`curt` on Stack Exchange is "bottom quartile of politeness", a request that raters found impolite. That is
a milder, and different, reading than rude or profane. This is a known mismatch with the pool's `curt`
(aggression −1.5…−0.7, obscene), and it is exactly what the breadth test measures.

## 4. Splits, dedupe, leakage, caps

**Pool (train / val).**
- The split is by natural unit:
  - Detox: `rev_id`;
  - Wikipedia and SE politeness: utterance `id`;
  - Civil Comments: its own train/validation files (the HF release has no comment id);
  - DynaHate and HatemojiBuild: their own splits (train → train; dev/validation → val; their test sets
    are not used).
- For Detox and Wikipedia politeness, `val` = units whose `sha256(source:unit)` falls in the lowest 10%.
  The same comment therefore lands in the same split in harm and in tone.
- **Dedupe.** Within each field, exact duplicates by normalised text (lowercased, whitespace collapsed) are
  dropped across all sources, keeping the first occurrence in source order. Detox Attacks and Aggression
  share `rev_id`s, and they are used in different fields, so no rev_id sits in two splits.
- **Near-duplicates** are computed in `embed` with nomic cosine > 0.95:
  1. pool `val` items that near-duplicate any pool `train` item of the same field are dropped;
  2. pool items (train or val) that near-duplicate **any** held-out item (dev or test, either field) are
     dropped;
  3. held-out `dev` items that near-duplicate a held-out `test` item of the same field are **moved to
     `test`**, so every copy is on one side (the sealed side).

  All counts are reported (section 6).

**Held-out.**
- Aegis (harm) and SE politeness (tone) are each split **20% `dev` / 80% `test`**, stratified by option,
  with seed 20260927.
- **`test` is sealed.** It is counted and embedded (for the leakage step) but never scored.
- Held-out sets are not capped.

**Caps (fixed now).**
- Train: at most **3,000 per option per source**, then at most **5,000 per option per field** (a random
  sample of the union). That gives at most 25,000 train for harm and 20,000 for tone.
- Val: at most **300 per option per source**, then **500 per option per field**.

**Personal data.**
- Never read: `worker_id` and the worker demographics files (not fetched), HH-RLHF `red_team_member_id`,
  ConvoKit `user`, and DynaHate `annotator`.
- Scrubbed on read:
  - e-mail addresses → `<email>`;
  - phone-like runs and any run of 7+ digits → `<number>`;
  - URLs → `<url>`;
  - `User:Name` / `User talk:Name` → `User:<user>`;
  - Detox `NEWLINE_TOKEN` / `TAB_TOKEN` → a space.
- Only HH-RLHF human turns are read, and only to build the match list. Assistant turns are never used.

## 5. Ceilings (report-only; pool val and held-out `dev` only)

Encoder: pinned nomic-embed-text-v1.5 (`classification: ` prefix, first 200 words). Antenna: the family's
fixed `service/families/v1/antenna.npz` → Z (46-dim).

- **LR-768.** Multinomial logistic regression on X (768-dim, class_weight = balanced, max_iter 2000). C is
  chosen from {0.1, 1, 10} by pool-val balanced accuracy. It is then scored on pool val and held-out dev.
- **LR-Z.** The same, on Z (46-dim).
- **Nearest prototype.** Class means of pool train, cosine on X, scored on held-out dev (and pool val).
- **Per-class recall** on held-out dev, for LR-768.
- The ceiling used for a disputed-labels bar is the **LR-768 held-out-dev balanced accuracy** (the
  breadth test is on the held-out source). The bar would be **0.9 × that ceiling**. Both fields are
  pre-flagged as disputed labels (research doc §2, §4). The pool-val figure is reported beside it.
- Chance: harm 0.20 (5 options); tone 0.25 on the pool, and on SE 1/3 over the three present options.

---

## 6. Results (filled in after build, embed and ceilings, 2026-09-27)

### Counts per split (final, after every drop)

| field | split | total | per option | by source |
|---|---|---|---|---|
| harm | train | 21,498 | hate 4,998, harassment 5,000, threat 3,500, sexual 3,000, fine 5,000 | Civil Comments 11,617; Detox Attacks 3,830; DynaHate 3,592; HatemojiBuild 2,459 |
| harm | val | 1,865 | hate 450, harassment 494, threat 275, sexual 195, fine 451 | CC 959; Detox 383; DynaHate 326; Hatemoji 197 |
| harm | **dev** (held-out) | 616 | hate 86, harassment 62, threat 80, sexual 49, fine 339 | Aegis 1.0 331; Aegis 2.0 285 |
| harm | **test (SEALED)** | 2,582 | hate 343, harassment 255, threat 337, sexual 196, fine 1,451 | Aegis 1.0 1,425; Aegis 2.0 1,157 |
| tone | train | 16,768 | warm 2,099, neutral 5,000, curt 4,669, hostile 5,000 | Detox Aggression 8,483; Wiki politeness 3,193; CC 5,092 |
| tone | val | 1,649 | warm 222, neutral 499, curt 432, hostile 496 | Detox 862; Wiki politeness 328; CC 459 |
| tone | **dev** (held-out) | 1,313 | warm 327, neutral 657, curt 329 | SE politeness |
| tone | **test (SEALED)** | 5,290 | warm 1,324, neutral 2,644, curt 1,322 | SE politeness |

**Pool availability after mapping** (train side, before caps):
- Civil Comments harm: harassment 96,955; hate 11,850; sexual 4,575; threat 4,238; fine 1.22M.
- Detox Attacks: harassment 13,991; fine 51,254.
- DynaHate: hate and threat 17,705 in total (606 of them `threatening` in the raw file); fine 15,172.
- HatemojiBuild: hate 2,384; fine 2,337.
- Tone warm is the scarce class: Detox Aggression gives 1,124 at mean ≥ +1.0, and Wikipedia politeness
  978. Civil Comments curt gives only 684, because obscene ≥ 0.5 with insult < 0.2 is rare. Warm is
  therefore at 2,099 and curt at 4,669, below the 5,000 cap.
- The full per-source table is in `items.json` (`build_report`) and in the manifest.

**Aegis filtering:**
- **Aegis 1.0:** 3,458 `user_message` rows, of which 3,439 match an HH-RLHF human turn. The other 19 are
  fragments ("isn't a gender.").
- **Aegis 2.0:** 16,608 prompt-only rows, of which 5,618 match. This drops DAN, AART and the 990 REDACTED
  Reddit rows.
- **Merged:** 1,754 duplicate copies were collapsed and 38 groups with conflicting labels were dropped.
  126,192 distinct HH-RLHF human turns were in the match list (harmless-base plus red-team-attempts).

**Dedupe and leakage:**

| step | harm | tone |
|---|---|---|
| exact duplicates (normalised text) across pool sources | 24,244 | 24,964 (mostly repeated Civil Comments) |
| pool items exactly equal to a held-out text, dropped | 1,625 (1,624 Civil Comments: "no.", "why?", "yes.", "thank you" … which are also Aegis prompts) | 0 |
| pool near a held-out item (cosine > 0.95), dropped | train 2, val 1 | train 5, val 1 |
| val near train (cosine > 0.95), dropped | 109 (90 of them HatemojiBuild, whose items are templated) | 24 |
| held-out dev near held-out test, moved to test | 23 | 7 |
| empty after scrub | 5 | 5 |

### Encoder ceilings (report-only; pool val and held-out DEV; the sealed test was never scored)

Written to `runs/gate4-ceilings/harm-tone.json`.

| field | method | pool val bal. acc | held-out dev bal. acc | chance (pool / held-out) |
|---|---|---|---|---|
| harm | LR-768 (C = 10) | **0.740** | **0.628** | 0.20 / 0.20 |
| harm | LR-Z46 (C = 1) | 0.641 | 0.606 | |
| harm | prototype-768 | 0.659 | 0.674 | |
| harm | prototype-Z46 | 0.601 | 0.581 | |
| tone | LR-768 (C = 10) | **0.752** | **0.413** | 0.25 / 0.333 (3 options present) |
| tone | LR-Z46 (C = 1) | 0.702 | 0.407 | |
| tone | prototype-768 | 0.683 | 0.388 | |
| tone | prototype-Z46 | 0.642 | 0.375 | |

**Held-out dev recall per option:**

| field | method | recall by option |
|---|---|---|
| harm | LR-768 | hate 0.56, **harassment 0.26**, threat 0.80, sexual 0.67, fine 0.85 |
| harm | LR-Z46 | hate 0.56, harassment 0.32, threat 0.83, sexual 0.65, fine 0.67 |
| tone | LR-768 | warm 0.28, neutral 0.73, curt 0.23 (predicted hostile 2 times out of 1,313) |
| tone | LR-Z46 | warm 0.40, neutral 0.60, curt 0.22 |

**Pool-val recall (LR-768):**

| field | recall by option |
|---|---|
| harm | hate 0.68, harassment 0.78, threat 0.82, sexual 0.79, fine 0.63 |
| tone | warm 0.89, neutral 0.68, curt 0.60, hostile 0.84 |

**Disputed-labels bars** (0.9 × the LR-768 held-out-dev ceiling, as pre-registered in section 5):

| field | bar | chance |
|---|---|---|
| harm | 0.9 × 0.628 = **0.566** | 0.20 |
| tone | 0.9 × 0.413 = **0.372** | 0.333 |

For reference, 0.9 × the pool-val ceiling would be 0.666 (harm) and 0.677 (tone).

### Reading

- **Harm** carries across register: 0.63 on Aegis requests, against 0.74 on the comment pool.
  - The weak option is `harassment` (0.26 on Aegis). Aegis "Harassment" prompts are requests *about*
    harassing someone. The pool's harassment is direct insult.
  - Z46 keeps most of the held-out signal (0.606 against 0.628).
  - A 5-option harm specialist is feasible under the disputed bar.
- **Tone fails its breadth preflight.** The research doc said: "Stop if the ceiling is near chance." The
  held-out dev ceiling is 0.413 against a chance of 0.333, even though the pool val ceiling is 0.75.
  Wikipedia-trained tone does not transfer to Stack Exchange requests:
  - the classifier calls most SE requests `neutral`;
  - SE "impolite" (the bottom quartile of politeness among requests) is not the pool's curt, which is
    aggression or obscenity.

  This is the mismatch that section 3 flagged before scoring. A 0.372 bar would be meaningless (0.04
  above chance).
- **C = 10 is at the edge of the grid** for LR-768 in both fields. A larger C might add a little on pool
  val. The grid was pre-registered, so it stays as run.

### Needs Nick's decision

1. **Tone:** do not take it to a gate as built. Options:
   - (a) drop the `warm`/`curt` breadth claim and hold out a different source (there is no other clean
     non-Wikipedia tone source with warm);
   - (b) move SE politeness into the pool and hold out Wikipedia politeness; the result would be weaker,
     because Wikipedia also dominates the pool;
   - (c) narrow tone to `hostile` vs not, which ConvAbuse could test if its CAUTION is accepted;
   - (d) commissioned labels (the open budget item).
2. **Wikipedia Toxicity Subtypes** needs a Kaggle login to fetch from the publisher. It is not required;
   Civil Comments covers threat and hate.
3. **ConvAbuse** (CAUTION) is the only clean test for the hostile end. It was not used.
4. **Aegis "Violence" → `threat`** and the request register pull `threat` toward "how do I hurt…". The
   recall is good (0.80), but the card should say that `threat` includes requests for violence.

### Report-only sets (fetched and counted, not scored)

| set | rows | notes |
|---|---|---|
| HateCheck `test_suite_cases.csv` | 3,728 cases, 29 functionalities | hateful 2,563, non-hateful 1,165 |
| XSTest | 450 prompts | safe 250, unsafe 200 |
| SimpleSafetyTests | 100 prompts | 20 on self-harm, outside the taxonomy |

None of these is in `items.json`. When they are scored, run a near-duplicate check against the pool
first.

### Card notes (for any specialist built on this data)

- **Civil Comments** (decision 30): commenter consent rests on the platform terms, and the raters'
  platform and pay are undocumented.
- **Share-alike text** (Wikipedia Detox, Wikipedia politeness, and SE on the tone side) means the weights
  are CC BY-SA 4.0 under decision 28. The attribution manifest is built from each item's `unit` (the
  rev_id or utterance id).
- **Stack Exchange** download terms, as in decision 31.
- **DynaHate `nothate` as `fine`** is imperfect (section 3).
- **Self-harm is out of scope.**
