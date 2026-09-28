# Harassment development: a recipe that carries across platforms (2026-09-27)

Development only, for gate 4b (amendment 1 of `docs/SPECIALIST-GATE-4.md`). Nothing here trains the fly. No
sealed test was scored: not the Aegis test, not any gate-4 held-out test, and not the sealed parts proposed
below. Script: `scripts/harassment_dev.py` (`fetch`, `build`, `embed`, `explore --stage grid|pool|propose`,
`summary`). Numbers: `runs/harassment-dev/ceilings.json` (summary) and `explore-{grid,pool,propose}.json` (every
run). Cache: `data/cache/harassment-dev/`.

**The question (route 1, Nick).** Harassment means "is this text harassing, insulting or attacking someone?"
(content moderation). Aegis "Harassment" is requests *about* harassing someone, a different construct, so Aegis is
no longer the right held-out test. Which data recipe lets a plain classifier on the pinned encoder carry to a
platform it never saw?

**Method.**
- The encoder is embedded exactly as `v1_data.py cmd_embed` does it: pinned nomic-embed-text-v1.5, the `classification: `
  prefix and the first 200 words. Z is the family antenna (`service/families/v1/antenna.npz`, 46-d), which is what the
  fly smells.
- The classifier is logistic regression with `class_weight=balanced`, C ∈ {0.1, 1, 10}, chosen on the training
  platforms' own val.
- Every number is balanced accuracy on a val or dev part.
- Civil Comments (CC): only its HF train and validation files are read.
- Wikipedia Detox Personal Attacks (Wiki): the gate-4 hash split. `worker_id` is never read.
- Candidates were sampled by band so that every threshold has enough rows: 24.9k CC train and 4.7k val, 30.2k Wiki train
  and 3.3k val.
- **Leakage handling:**
  - a training row that near-duplicates (cosine > 0.95) an eval row on another platform is dropped;
  - a val row that near-duplicates a training row on the same platform is dropped;
  - a held-out dev row that near-duplicates its own sealed test is excluded (223 Wiki, 96 ConvAbuse).

## 1. Why the current ceilings were low

The harm-dev harassment expert's data gives the old numbers (reproduced: without CC → CC val X 0.638 / Z 0.626;
without Wiki → Wiki val X 0.662 / Z 0.670). Most of the loss is **not** the platform gap:

| train on CC (gate-4 definition), score Wiki val | X | Z |
|---|---|---|
| CC alone | **0.828** | **0.834** |
| + DynaHate/HatemojiBuild hate as "not harassment" | 0.742 | 0.751 |
| + their `fine` as well (the gate-4 pool) | 0.713 | 0.721 |
| in-platform reference: Wiki → Wiki val | 0.952 | 0.924 |

- **DynaHate and HatemojiBuild are the main cause.** Their hate rows are insulting text labelled "no", and some
  DynaHate `nothate` rows insult targets that are not protected groups (`docs/gate4-data-harm-tone.md` §3, the
  stated imperfection). Together they teach "insulting but not in CC style → no". Recall on Wiki falls from 0.71–0.82
  to about 0.5.
- **Other harm types as negatives** cost a further 0.06 on transfer (CC→Wiki 0.828 → 0.896 without them). They buy
  specificity (section 3).
- **Label strictness on the training side barely matters.** Across 48 CC recipes (insult ≥ 0.5/0.6/0.7/0.8;
  negatives toxicity = 0, < 0.1 or < 0.3; other harms in or out; hate/threat/sexual insults in or out), CC→Wiki
  ranges from 0.79 to 0.90. The only big factor is other harms in or out. Raising the insult threshold to 0.8 lowers
  transfer (0.79–0.87), because fewer positives are left. Across 20 Wiki recipes (attack fraction ≥ 0.5 to 0.9,
  recipient-attack only, negatives 0 attackers or < 0.2), Wiki→CC is 0.65–0.76. The majority threshold is as good as
  any other.
- **Strictness on the test side does matter,** because it removes the borderline items. Scoring Wiki at ≥ 0.7 attackers
  adds about +0.02 to +0.03, and CC at insult ≥ 0.7 adds about +0.03 to +0.05. This is a change of definition, not a
  better model; it is listed for Nick below.
- **Wiki → CC is the hard direction** (X 0.740, Z 0.689 on the gate-4 CC definition). CC's negatives there include
  hate, threat and sexual comments with no insult, and Wiki has none of those to learn from. Without those negatives
  it rises to X 0.857 / Z 0.785.

## 2. New sources (licence checked at source, 2026-09-27)

| source | platform | licence (where verified) | verdict |
|---|---|---|---|
| **ConvAbuse** (Cercas Curry et al., EMNLP 2021), commit `c0a9469` | users talking to chatbots: CarbonBot on Facebook Messenger (2019–20), ELIZA (2002–07) | **CC BY 4.0**: the LICENSE file in the repo at that commit, fetched today. `data/raw/clean/convabuse/FETCH.json` records the url and sha256 | **CAUTION, flagged for Nick** (ELIZA consent is implied: "unclear how user consent was obtained"). Fetched and used as a **candidate held-out source only**. `annotator_id` is never read |
| ParlAI Dialogue Safety (BBF, Dinan et al. 2019) and Bot-Adversarial Dialogue (Xu et al. 2021) | crowd-written offensive chat messages | **Unverified.** The ParlAI repo is MIT *for code* ("This source code is licensed under the MIT license"). The task and project pages state no data licence. "Apache 2.0" appears only in third-party summaries | not fetched. It would be a good crowd-written third platform if the authors confirm a licence |
| Wikipedia Toxicity Subtypes `insult` (Jigsaw 2018) | Wikipedia talk (same platform as Detox) | labels CC0 / text CC BY-SA (as in gate 4); the publisher copy needs a Kaggle login (TFDS bucket 403) | not fetched. Same platform and same comments as Detox (dedupe by rev_id), so it adds no breadth |
| Detox Aggression / Toxicity | Wikipedia talk | CC0 labels / CC BY-SA text (gate 4) | same platform; not used here |
| GitHub locked-issue incivility sets (e.g. arXiv 2402.04183), Gerrit code-review toxicity (ToxiCR) | open-source dev chat | not verified; the text is GitHub/Gerrit users' own comments, scraped | **NO** (the same reasoning as the ledger's scraped-platform drops) |
| Reddit, Twitter/X, Gab, YouTube sets (CAD, MHS, OLID, …) | — | — | NO (as in `docs/research-gate4/harm-tone.md`) |

**The ConvAbuse split, made without reading labels.**
- Units are user turns, deduplicated by text (2,757 kept).
- The split is **by conversation**: sha256(`convabuse:conv_id`), stratified by bot, 20% dev (615) and 80% **sealed
  test (2,142)**.
- Labels are computed only for dev rows. Test rows carry text only; they are embedded so that pool rows near them can
  be dropped.
- Dev mapping:
  - yes = more than half the annotators mark abusive (−1…−3), and the abuse is not a hate type (racist, sexist,
    homophobic, transphobic, ableist) or sexual harassment;
  - no = more than half mark "not abusive";
  - hate types and sexual harassment count as "no" (other harms).
- Dev has only **31 yes** after the leakage exclusion (27 ELIZA, 4 CarbonBot), so its numbers carry about ±0.05. The
  sealed part should hold about 130 yes.

## 3. Proposed gate 4b: hold out Wikipedia, pool Civil Comments

- **Held-out source: Wikipedia Detox Personal Attacks,** a new split:
  - **dev** = the gate-4 val-hash rows plus sha256(`gate4b:wiki_detox:rev_id`) < 0.111 of the rest, which is 20% of
    comments;
  - **sealed test** = the other 80%, never scored by anything. Some of those rows served as *training* rows in the
    Wiki→CC runs above, and they were in the gate-4 harm experts' training pool. The 4b expert trains without Wiki, so
    neither is test leakage. The 4b pre-registration should say so.
- The dev-part counts in the candidate sample are 2,936 yes and 2,355 no. The full dev part has all of Wiki's yes rows
  and more of its no rows.
- **Pool: Civil Comments only, with no DynaHate or HatemojiBuild.** This is one training source, against the
  several-source spirit of decision 29. ConvAbuse could become a second pool source, or a second held-out source, if
  Nick clears it.
- ConvAbuse dev is reported beside Wiki as a second platform (chatbot users).

**Ceilings on the held-out DEV parts** (report-only; plain logistic):

| recipe (pool = CC; yes = insult ≥ t and toxicity ≥ 0.5, any harm type) | Wiki 4b dev X / Z | Wiki (≥ 0.7 attackers) X / Z | ConvAbuse dev X / Z | specificity: CC other harms called "no" X / Z |
|---|---|---|---|---|
| **A**: t = 0.5; no = toxicity 0 only | **0.895 / 0.885** | 0.927 / 0.916 | 0.865 / 0.850 | 0.57 / 0.53 |
| **B (recommended)**: t = 0.5; no = toxicity 0 + CC hate/threat/sexual without insult | **0.838 / 0.842** | 0.859 / 0.864 | 0.835 / 0.839 | 0.90 / 0.83 |
| **B′**: t = 0.7; no = toxicity 0 + other harms with insult < 0.2 | 0.847 / 0.844 | 0.884 / 0.877 | 0.845 / 0.835 | 0.89 / 0.82 |
| the gate-4 recipe (CC + DynaHate/Hatemoji) | 0.742 / 0.752 | 0.736 / 0.758 | 0.734 / 0.743 | 0.91 / 0.87 |

- **Why B, not A.** A gets its transfer by calling about half of hate, threat and sexual comments that carry no insult
  "harassment". The Wiki test cannot see this, because it has no other harm types. B keeps 0.90 (X) of them as "no" and
  still transfers at 0.84.
- **Would a fly plausibly clear 0.70?** On Aegis dev the fly matched or beat the plain ceiling (0.672 against 0.635,
  and 0.606 on Z). Here the Z ceiling is 0.84, so 0.70 leaves a margin of 0.14.
- **Other held-out choices, from the same runs:**
  - hold out CC, pool Wiki: gate-4 CC definition X 0.740 / Z 0.689. That is too close to the bar on Z;
  - hold out ConvAbuse, pool CC + Wiki: X 0.892 / Z 0.801. This is CAUTION data with a small dev part.

## Needs Nick

1. **Change harassment's held-out test from Aegis to Wikipedia Detox** (80% sealed, above). Amendment 1 named Aegis
   as harassment's sealed test. Moving it is a pre-registration change for gate 4b to state before any score. Aegis
   stays sealed; it could be scored once, report-only, as "requests about harassment".
2. **Drop DynaHate and HatemojiBuild from the harassment expert's pool** (the other harm experts keep them).
3. **What does "no" include?** B (hate, threat and sexual without insult are "no") or A (only benign text is "no";
   higher transfer, weak specificity). Either way, hate/threat/sexual comments that *also* insult count as "yes".
   Gate 4's precedence (hate > harassment) made them "no" for the harassment expert.
4. **Test-side strictness:** keep the majority of raters (≥ 0.5, the gate-4 rule), or ≥ 0.7 attackers (+0.02 to 0.03,
   drops the borderline cases).
5. **ConvAbuse** (CC BY 4.0, CAUTION on ELIZA consent): accept as a second held-out source or second pool source,
   CarbonBot-only (cleaner consent but only about 20 yes in all), or leave it out.
6. **ParlAI BBF/BAD**: ask the authors for the data licence. If it is open, it would be a crowd-written third platform.

## Card notes

These are for a shipped harassment expert trained on B:
- pool Civil Comments only (decision 30 ethics gaps);
- tested on Wikipedia talk (CC BY-SA text, so the weights are BY-SA under decision 28);
- "harassment includes hateful or threatening insults";
- self-harm out of scope.
