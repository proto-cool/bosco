# Training data for the harm and tone specialists (gate 4 research), checked 2026-09-27

Research only. Nothing was trained, and nothing was fetched into `data/`. To count labels, a few small
public files (under 5 MB each) were read in `/tmp`: Aegis 1.0 parquets, Aegis 2.0 test/validation, HateCheck
cases, XSTest prompts, ConvAbuse splits. The local Civil Comments copy (`data/raw/clean/civil_comments/`)
was also counted. None of this is legal advice.

This builds on `docs/free-datasets.md`, `docs/clean-data.md`, `docs/audit-2026-09-25/provenance.md` and
`docs/specialist-gate-2-results.md`. Verdicts: **USE** means the licence is verified at source and the
provenance is clean. **CAUTION** means usable only with a stated risk accepted. **NO** means do not use.
"Unverified" means I could not confirm the licence at the original source.

**Standing caveat.** Civil Comments is still **REVIEW** in the ledger: commenter consent under the Civil
Comments terms, and rater platform and pay, are unanswered. Both proposals lean on it. If it is not cleared,
the fallback pools are given below.

---

## 1. Candidate datasets: harm

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **DynaHate** v0.2.3 | Vidgen, Thrush, Waseem, Kiela | github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset | CC BY 4.0, stated in the README; there is no LICENSE file. Commit `36f9dc8` (already on the ledger) | Crowd-written on Dynabench, adversarially. Rounds 3–4 were adapted from real posts with "substantial adjustment". £16/h | none expected | 41k | hate/nothate; `type` = Animosity, Derogation, Dehumanization, **Threatening**, Support | **USE** (already KEEP) |
| **HatemojiBuild** | Kirk et al. (Oxford) | github.com/HannahKirk/Hatemoji | CC BY 4.0 (repo LICENSE, commit `a626f63`) | Crowd-written, adversarial, emoji-based | none | 5.9k | hateful 0/1 | **USE** (already KEEP) |
| **Civil Comments** | Jigsaw / Borkan et al. 2019 | tensorflow.org/datasets/catalog/civil_comments; HF google/civil_comments rev `f2970eb` | CC0 (TFDS catalog: "released under CC0", fetched today) | Real comments from about 50 news sites via the Civil Comments plugin, 2015–17; about 10 crowd raters per comment | IDs stripped; the text may name people | 2.0M | fractions for toxicity, severe_toxicity, obscene, **threat, insult, identity_attack, sexual_explicit** | **CAUTION** (ledger REVIEW). Local counts at ≥0.5: insult 118,079; identity_attack 14,761; sexual_explicit 5,127; threat 4,725; obscene 10,671; toxicity==0: 1,401,760 |
| **Wikipedia Toxicity Subtypes** (Jigsaw Toxic Comment 2018) | Jigsaw / Wulczyn et al. | tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes; Kaggle jigsaw-toxic-comment-classification-challenge | TFDS says "CC0, as is the underlying comment text". **Wikimedia's Detox page says the comment text is CC BY-SA** (meta.wikimedia.org/wiki/Research:Detox/Data_Release). Jigsaw cannot relicense Wikipedians' text, so treat it as **labels CC0, text CC BY-SA** | Wikipedia talk-page comments, CrowdFlower raters | usernames sometimes in the text | 159,571 train + 63,978 test | toxic, severe_toxic, obscene, **threat, insult, identity_hate** | **USE (BY-SA)**. Ledger KEEP. Attribute |
| **Wikipedia Detox: Personal Attacks** | Wulczyn, Thain, Dixon (Wikimedia/Jigsaw) | figshare (Wikipedia Talk Labels); meta page above | CC0 labels, CC BY-SA text (Wikimedia Detox page, fetched today) | Talk-page comments, about 10 CrowdFlower raters each; worker demographics in a separate file | usernames in the text; worker demographics are anonymised (do not read that file) | ~100k | attack (recipient / third-party / quoting / other) | **USE (BY-SA)**. Maps to harassment. Overlaps with the Toxicity Subtypes comments: dedupe by `rev_id` |
| **HateCheck** | Röttger et al. 2021 | github.com/paul-rottger/hatecheck-data | **CC BY 4.0** (GitHub licence API; commit `3490854`) | **Hand-written templates by the authors**, filled with identity terms; 5 annotators each; cases kept only with ≥4/5 agreement | none | 3,728 cases, 29 functionalities | hateful/non-hateful per functionality. Incl. `threat_dir_h`, `threat_norm_h` (273), `target_indiv_nh` / `target_group_nh` / `target_obj_nh` (abuse at non-protected targets, 192), `profanity_nh`, `counter_*_nh`, `slur_reclaimed_nh` | **USE**, as a **report-only functional suite**, never as a training source. The rows are templated near-duplicates (leakage) |
| **XSTest** | Röttger, Kirk, Vidgen et al. 2024 | github.com/paul-rottger/xstest | **CC BY 4.0** (LICENSE file; commit `d7bb5bd`) | Hand-written prompts: 250 safe prompts that use unsafe-sounding words ("kill a Python process"), plus 200 unsafe contrasts | none | 450 | type (homonyms, figurative, safe_targets, … / contrast_*) | **USE** as a report-only false-positive check for "fine" |
| **SimpleSafetyTests** | Vidgen et al. 2023 | github.com/bertiev/SimpleSafetyTests | **CC BY 4.0** (LICENSE file; README "Licence is CC-BY"; commit `d7aee9a`) | 100 hand-written prompts in 5 harm areas, 20 on **suicide, self-harm, eating disorders** | none | 100 | harm area | **USE**, report-only (too small to train) |
| **Anthropic HH-RLHF**: harmless-base and red-team-attempts | Anthropic | github.com/anthropics/hh-rlhf (archived 2025-06) / HF Anthropic/hh-rlhf rev `09be8c5` | **MIT** (repo LICENSE, commit `c72f5ce`; HF card) | Human turns written by paid red-teamers (MTurk and Upwork); assistant turns are model output | `red_team_member_id` (pseudonymous) | ~39k red-team transcripts; harmless-base ~42k | no category labels except `tags` on 1,000 items and a free-text `task_description` | **USE the human turns only**, as a text source. It is labelled through Aegis (below). Never use the assistant turns (LLM-generated) |
| **Aegis 1.0** (Nemotron Content Safety V1) | NVIDIA (Ghosh et al. 2024) | HF nvidia/Aegis-AI-Content-Safety-Dataset-1.0 rev `bd96d86` | **CC BY 4.0** (card; NVIDIA is the publisher, so the card is the source). Prompts come from HH-RLHF (MIT) | Prompts are HH-RLHF human turns; responses are **Mistral-7B generated**; 12 NVIDIA annotators (US), volunteer basis, adult-content acknowledgement signed | little (the card says PII was removed) | 11,997 rows; **3,458 prompt-only rows (`text_type=user_message`)** | 13 categories + Safe / Needs Caution. Majority label over the prompt rows: Safe 1,232; Criminal Planning 582; Needs Caution 573; **Hate 272; Harassment 184; Sexual 132; Violence 75; Threat 10; Suicide/Self-harm 12**; Profanity 63; PII 148; … | **USE, `user_message` rows only** (human text, human labels). Drop every row that contains model text |
| **Aegis 2.0** (Nemotron Content Safety V2) | NVIDIA (Ghosh et al. 2025) | HF nvidia/Aegis-AI-Content-Safety-Dataset-2.0 rev `d86bb8b` | **CC BY 4.0** (card) | Prompts from HH-RLHF (MIT), **DAN** jailbreaks (scraped from Reddit/Discord/websites) and **AART** (AI-assisted, i.e. LLM-generated red-teaming). Responses come from Mistral. Response labels are partly an **LLM jury**. Self-harm rows from a **Reddit** Suicide-Watch Kaggle set are REDACTED | as above | 33,416 (train 30,007) | `prompt_label` (always human); `violated_categories` (same taxonomy). Test+val: 91 self-harm rows, of which 32 are REDACTED Reddit | **CAUTION**. Use only prompt-only rows whose prompt **exactly matches an HH-RLHF human turn**; there is no source column. That filter drops DAN, AART and the Reddit rows. `violated_categories` on prompt-only rows describes the prompt |
| **ConvAbuse** | Cercas Curry, Abercrombie, Rieser (EMNLP 2021) | github.com/amandacurry/convabuse | **CC BY 4.0** (LICENSE file; commit `c0a9469`) | Real user turns to two chatbots: **CarbonBot** on Facebook Messenger (1,515 rows, 2019–20; users told that conversations are recorded for research; "in accordance with GDPR") and **ELIZA** at the Jožef Stefan Institute (2,670 rows, 2002–07; the paper says "it is unclear how user consent was obtained"). The Alana (Alexa) data is not released | anonymised per the paper; free-text user turns | 4,185 | abuse level (1 not / 0 ambiguous / −1 mild / −2 strong / −3 very strong); type incl. **sex_harassment**, racist, sexism, homophobic…; target; directness | **CAUTION**: implied consent on ELIZA. The CarbonBot subset is cleaner. Useful as an out-of-domain test (human-to-machine abuse) |
| OpenAI moderation-api-release | OpenAI (Markov et al. 2022) | github.com/openai/moderation-api-release | **MIT** (LICENSE file; commit `f4ab51b`) | 1,680 samples "containing only samples from public data" (paper §4.2). The paper's public data = "academic datasets and Web data (Common Crawl)"; it also used synthetic data in training. **The per-sample source is not given** | the paper says there is no production data | 1,680 | S, H, V, HR, **SH**, S3, H2, V2 (binary; missing = unknown) | **CAUTION**: text provenance unknowable per row (it could include Twitter-derived academic sets or synthetic text). At most a report-only test, never training |
| Measuring Hate Speech | Kennedy et al., UC Berkeley D-Lab | HF ucberkeley-dlab/measuring-hate-speech rev `5468f6e` | CC BY 4.0 (card) | **YouTube, Twitter and Reddit comments** (arXiv 2009.10277 abstract); 11,143 US MTurk raters | social-media text | 136k annotations over ~50k comments | hate-speech score (IRT), insult, humiliate, dehumanize, violence, respect… | **NO**. Twitter/YouTube text; CC BY on the labels cannot cure it |
| Gab Hate Corpus | Kennedy et al. 2022 | osf.io/edua3 | CC BY 4.0 (OSF licence field, fetched today) | 27,665 **Gab** posts, ≥3 trained annotators | real users' hateful posts | 27,665 | hate (HD/CV/VO), targets | **NO**. Scraped hate by real users, same reasoning as the ledger's DROP of Stormfront |
| ETHOS | Mollas et al. 2020 | github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset | GPL-3.0 (ledger) | YouTube and Reddit | yes | ~1k | hate, multi-label incl. violence | **NO** (ledger DROP) |
| CAD (Contextual Abuse Dataset) | Vidgen, Nguyen et al. 2021 | zenodo.org/records/4881008 | CC BY 4.0 (Zenodo record) | **Reddit** threads, expert annotators | Reddit text | ~25k entries (paper figure, not rechecked) | identity-directed / affiliation-directed / person-directed abuse, counter-speech, non-hateful slurs | **CAUTION → NO for training**. Reddit's 2023 data terms. A very good taxonomy, though (person- vs identity-directed) |
| Social Bias Frames | Sap et al. 2020 | HF allenai/social_bias_frames | CC BY 4.0 (ledger) | Reddit, Twitter, Gab, Stormfront | yes | 150k | offensive, intent, group | **NO** (ledger REVIEW; the sources fail) |
| ToxiGen | Hartvigsen et al. (Microsoft) 2022 | HF toxigen/toxigen-data; github.com/microsoft/TOXIGEN | GitHub licence = NOASSERTION; the HF access form gives **no licence** stated | **All text generated by GPT-3** (card) | none | 274k | toxicity 1–5, target group | **NO** (synthetic; licence unverified). See the synthetic list |
| BeaverTails | PKU-Alignment | HF PKU-Alignment/BeaverTails | **CC BY-NC 4.0** (card) | Prompts + **LLM responses** | — | 364k | 14 harm categories | **NO** (NC, and LLM text) |
| Real Toxicity Prompts | Gehman et al. (AllenAI) | github.com/allenai/real-toxicity-prompts | Apache-2.0 per the HF card (**unverified**: GitHub rate-limited) | OpenWebText sentences (web pages linked from Reddit) + Perspective API scores (machine labels) | web text | 100k | Perspective scores | **NO**. Machine labels, and copyrighted web text |
| Self-harm / suicide sets (UMD Reddit Suicidality, CLPsych, Suicide-Watch Kaggle, SWMH, Dreaddit) | various | — | DUA / research-only / none | **Reddit** | highly sensitive health data | — | — | **NO**. No clean human-written self-harm corpus was found. The UMD set needs a data use agreement; the Kaggle Suicide-Watch set is scraped Reddit (Aegis 2.0 itself redacts it) |
| OLID, HatEval, Davidson, HateXplain, Implicit Hate, TweetEval-offensive | various | — | various (MIT for some) | **Twitter** (+Gab) | — | — | — | **NO** (platform), not re-verified individually |

### Synthetic (listed separately; never test data)

| name | licence | nature | verdict |
|---|---|---|---|
| ToxiGen | no stated licence (access form) | GPT-3-generated implicit hate | **NO** |
| Aegis 2.0 AART rows | CC BY 4.0 | LLM-assisted red-team prompts | a training supplement at most; not recommended |
| Do-Not-Answer | Apache-2.0 per the card (not verified) | GPT-4-generated questions | not needed |

---

## 2. Harm: proposed taxonomy and mapping

**Options (5): `hate` / `harassment` / `threat` / `sexual` / `fine`.**

- `hate`: attacks a group, or a person as a member of a protected group.
- `harassment`: insults or attacks a person or a non-protected target.
- `threat`: threatens or incites violence.
- `sexual`: explicit sexual content or sexual harassment.
- `fine`: none of these.

When a text carries several labels, one option is picked by precedence, written now: **threat > sexual > hate
> harassment**.

**Self-harm is deferred, not in the taxonomy.** Clean human-written self-harm text is scarce: 12 prompt rows
in Aegis 1.0, a few dozen human-sourced rows in the Aegis 2.0 test/val (and perhaps a few hundred in train,
**unverified** because the train file is 22 MB and was not fetched), and 20 SimpleSafetyTests prompts. Every
large source is Reddit. A self-harm detector trained on a few hundred request-style prompts, with a false
negative that costs a life, should not ship under the Bosco name. Revisit if a clean source appears, or if
Nick decides on commissioned, consented writing.

**Profanity/obscenity is not harm here.** Civil Comments `obscene` and Wikipedia `obscene` belong to **tone**
(rude), not to a moderation category. Rows whose only label is obscene are dropped from harm.

**Mapping (write it into the pre-registration before any embedding is scored):**

| source | hate | harassment | threat | sexual | fine | drop |
|---|---|---|---|---|---|---|
| Civil Comments | toxicity ≥ 0.5 and identity_attack ≥ 0.5 | toxicity ≥ 0.5 and insult ≥ 0.5 | threat ≥ 0.5 | sexual_explicit ≥ 0.5 | toxicity = 0 (sample; 1.4M available) | 0 < toxicity < 0.5; obscene-only |
| Wikipedia Toxicity Subtypes | identity_hate | insult | threat | — | toxic = 0 | toxic without a mapped subtype; obscene-only |
| Wikipedia Detox Attacks | — | attack ≥ 0.5 (mean of raters) | — | — | attack = 0 | the rest; dedupe with Subtypes by rev_id |
| DynaHate | hate, type ∈ {Animosity, Derogation, Dehumanization, Support} | — | hate, type = Threatening | — | nothate (see note) | notgiven type → hate |
| HatemojiBuild | hateful = 1 | — | — | — | hateful = 0 | — |
| Aegis 1.0/2.0 (prompt-only, HH-matched) | Hate/Identity Hate | Harassment | Threat, Violence | Sexual | Safe | Needs Caution, Criminal Planning, PII, weapons, substances, Profanity, Other, Suicide/Self-harm, Sexual (minor) |
| ConvAbuse (report-only) | abuse ≤ −1 and type ∈ {racist, sexism, homophobic, transphobic, ableism} | abuse ≤ −1 and type = intellectual or no type | — | abuse ≤ −1 and type = sex_harassment | abuse = 1 (majority) | ambiguous (0) |
| HateCheck (report-only) | all `*_h` except threat | `target_indiv_nh`, `target_group_nh` | `threat_dir_h`, `threat_norm_h` | — | `ident_*_nh`, `counter_*_nh`, `negate_neg_nh`, `slur_reclaimed_nh`, `slur_homonym_nh`, `profanity_nh` | `target_obj_nh` |

Notes on the mapping:
- The DynaHate `nothate` rows are adversarial contrasts. Some are rude but not hateful, so `fine` is imperfect
  there. The pre-registration should state this, or restrict DynaHate to `hate`/`threat` examples.
- Aegis "Violence" is mostly *requests* ("how do I hurt…"), not threats. Mapping it to `threat` is a stated
  choice. The option could be named **`violence`** (threats and incitement) to fit both sources.

**Disputed labels: yes, for the whole specialist** (bar = 0.9 × encoder ceiling), for three reasons:
- Gate 2's hate specialist already needed the disputed bar (ceiling ≈ 0.73).
- Civil Comments and Wikipedia labels are fractions of about 10 raters.
- The hate/harassment boundary is exactly where raters split.

`threat` and `sexual` are individually less disputed, but the bar is set per specialist.

**Held-out source for the breadth test: Aegis prompt-only rows (HH-RLHF prompts, NVIDIA human labels).**
- It is the only clean source that covers **all five options**: hate 272, harassment 184, sexual 132,
  violence+threat 85 and safe 1,232 in v1's prompts alone, more with the v2 HH-matched rows.
- Its register is different (requests typed to a chatbot, not comments), so it is a real breadth test.
- The training pool is then Civil Comments + Wikipedia (Subtypes + Attacks) + DynaHate + HatemojiBuild.
- The **alternative** is holding out Wikipedia Toxicity Subtypes, which covers four options (no sexual). It
  is a weaker test, because Wikipedia and Civil Comments are both comment threads.
- Report-only, not graded: HateCheck by functionality, XSTest (the safe half should be `fine`), ConvAbuse
  (human→bot abuse) and SimpleSafetyTests.

**Leakage checks to pre-register:**
- **HH-RLHF prompts recur across Aegis 1.0 and 2.0.** Dedupe on exact and near matches (MinHash or
  embedding cosine > 0.95), and put every copy on one side.
- **Wikipedia Toxicity Subtypes and Detox Attacks share comments.** Dedupe by rev_id and by text.
- **DynaHate rounds 3–4 are adapted from real posts.** Near-duplicate check against Civil Comments and
  Wikipedia.
- **HateCheck templates** must never be split across train and test.
- **HatemojiCheck vs HateCheck:** both are templated suites by the same group. Check near-duplicates if
  either is ever trained on.

**If Civil Comments is not cleared:**
- `sexual` then rests on Aegis alone (≈132 + v2 rows), which is also the proposed hold-out.
- In that case either drop `sexual` (4 options, hold out Aegis), or keep it and hold out Wikipedia for
  the other four with sexual scored report-only.
- This choice must be made before the run.

---

## 3. Candidate datasets: tone

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **Stanford Politeness, Stack Exchange** | Danescu-Niculescu-Mizil et al. / ConvoKit | convokit.cornell.edu/documentation/stack_politeness.html | CC BY 4.0 (ConvoKit); SE text CC BY-SA (already on the ledger, fetched 2026-09-26) | Real requests from SE, MTurk-rated (5 raters) | usernames in metadata (not read) | 6,603 | normalised score; binary 1 / 0 / −1 = top quartile / middle / bottom quartile | **USE** (attribute; BY-SA text) |
| **Stanford Politeness, Wikipedia** | same | convokit.cornell.edu/documentation/wiki_politeness.html | "ConvoKit's Stanford Politeness Corpus is governed by the CC BY license v4.0" (docs page, fetched today); Wikipedia text CC BY-SA | Real requests from Wikipedia talk pages, MTurk | usernames | 4,353 | same | **USE** (BY-SA) |
| **Wikipedia Detox: Aggression** | Wulczyn et al. | figshare article 4267550 (API: "Wikipedia Talk Labels: Aggression", licence **CC0**); meta Detox page | CC0 labels (figshare API, fetched today); text CC BY-SA (Detox page) | Talk-page comments, about 10 CrowdFlower raters each | usernames in text; worker demographics file (do not read) | ~100k comments (58 MB file, not fetched) | **aggression score from "very aggressive (−3), to neutral (0), to very friendly (3)"** | **USE (BY-SA)**. The key source: it spans the whole tone axis, friendly to hostile |
| **Wikipedia Detox: Toxicity** | same | figshare (Wikipedia Talk Labels: Toxicity) | CC0 labels / BY-SA text (Detox page) | same | same | ~160k | **"very toxic (−2) … neutral (0) … very healthy (2)"** | **USE (BY-SA)**, secondary to Aggression; "healthy" is not the same as polite |
| Civil Comments | Jigsaw | — | CC0 (TFDS) | news comments | — | 2M | toxicity, insult, obscene | **CAUTION** (REVIEW). The hostile and rude end only (no polite label) |
| Wikipedia Toxicity Subtypes | Jigsaw | — | labels CC0 / text BY-SA | — | — | 160k | obscene, insult, toxic | **USE (BY-SA)**. Rude (obscene, mild insult) and hostile ends |
| Conversations Gone Awry (Wikipedia) | Zhang, Chang et al. / ConvoKit | convokit.cornell.edu/documentation/awry.html | **Unverified**: the docs page states no licence; the ConvoKit repo is MIT (code); Wikipedia text is BY-SA | Talk-page conversations; 3 crowd annotators + internal check | usernames, timestamps | 4,188 conversations / 30,021 comments | comment_has_personal_attack | **CAUTION** (licence unstated). Adds nothing beyond Detox |
| ConvAbuse | see §1 | — | CC BY 4.0 | human→chatbot | — | 4,185 | abuse −3…1 | **CAUTION**. A good out-of-domain test for the hostile end |
| Pavlick & Tetreault formality corpus | Pavlick & Tetreault 2016 (Penn / Yahoo) | original tarball at upenn.edu/~nlp/resources/formality-corpus.tgz → **404** (2026-09-27); HF mirror osyvokon/pavlick-formality-scores says CC BY 3.0 | **Unverified at source** | Sentences from **Yahoo! Answers** (4,977), Technorati blogs (1,821), the Jeb Bush email archive (1,701), and 20 news outlets (CNN, Reuters, BBC, NYT…) (2,775). MTurk ratings −3…3 | the emails are real people's correspondence | 11,274 | formality score | **NO**. The licence is unverified, and the Yahoo Answers and news text is not the mirror's to license |
| GYAFC | Rao & Tetreault 2018 (Grammarly) | github.com/raosudha89/GYAFC-corpus | Access by email only after obtaining **Yahoo L6 (Webscope, research-only)** | Yahoo Answers | — | 110k pairs | formal/informal | **NO** |
| GoEmotions (for angry/calm) | Google | — | Apache-2.0 on Reddit text | Reddit | — | 58k | 27 emotions | **CAUTION** (as in free-datasets.md). Not proposed |
| DailyDialog, EmpatheticDialogues | — | — | NC | — | — | — | — | **NO** |

---

## 4. Tone: proposed taxonomy and mapping

**Options (4): `warm` / `neutral` / `curt` / `hostile`**, one axis from friendly to aggressive.

- `warm`: polite, friendly, appreciative.
- `neutral`: matter-of-fact.
- `curt`: impolite, brusque, mildly rude or profane, but not attacking anyone.
- `hostile`: aggressive, insulting, attacking.

Formality (formal/informal) is **not supported**: no clean labelled data (Pavlick and GYAFC both fail), and
gate 3's `plain` preflight showed that style barely survives the meaning encoder. Calm/angry has no clean
emotion data either.

**Warning before any gate.** Gate 2's politeness passed its disputed bar by 0.0006, with an encoder ceiling
of 0.666 on binary. A 4-way tone task will have a low ceiling. **Run a preflight first**: logistic regression
on the 768-dim embeddings, pool → held-out source, as gate 3 did for `plain`. Stop if the ceiling is near
chance.

**Mapping (thresholds on the mean rater score, written before scoring):**

| source | warm | neutral | curt | hostile | drop |
|---|---|---|---|---|---|
| Wikipedia Detox Aggression (−3…3) | mean ≥ +1.0 | −0.3 ≤ mean ≤ +0.3 | −1.5 < mean ≤ −0.7 | mean ≤ −1.5 | the gaps between bands |
| Wikipedia politeness (ConvoKit binary) | 1 (top quartile) | 0 | −1 (bottom quartile) | — | — |
| Stack Exchange politeness | 1 | 0 | −1 | — | — |
| Wikipedia Toxicity Subtypes | — | toxic = 0 (sampled) | obscene only (no insult/threat/identity_hate) | toxic and (insult or threat or identity_hate) | other |
| Civil Comments (if cleared) | — | toxicity = 0 | obscene ≥ 0.5 and insult < 0.2 | toxicity ≥ 0.66 and insult ≥ 0.5 | the rest |
| ConvAbuse (report-only) | — | not abusive (majority) | mild (−1) | strong / very strong (−2, −3) | ambiguous |

**Disputed labels: yes** (bar = 0.9 × ceiling). Politeness and aggression are both mean-of-raters scales
with known disagreement, as gate 2 found.

**Held-out source for the breadth test: Stack Exchange politeness** (6,603).
- It is the only tone source that is neither Wikipedia nor news comments.
- It covers warm / neutral / curt; balanced accuracy is taken over the 3 options present.
- The `hostile` end is tested report-only on **ConvAbuse**, a different platform (chatbot users).
- Training pool: Detox Aggression + Wikipedia politeness + Wikipedia Toxicity Subtypes (+ Civil Comments if
  cleared).
- Caveat: the pool is Wikipedia-heavy. Without Civil Comments, every training row is Wikipedia talk, and
  the breadth test is exactly the check on whether that generalises.

**Leakage checks:**
- **Wikipedia politeness, Detox Aggression, Detox Toxicity and Toxicity Subtypes can share talk-page
  comments.** Dedupe by rev_id and by text across all four.
- The SE hold-out shares no text with Wikipedia. Still run a near-duplicate check, since SE users may paste
  boilerplate.

---

## 5. Recommendation

**Harm.** Build a 5-option specialist: `hate` / `harassment` / `threat` (or `violence`) / `sexual` / `fine`,
on the disputed bar.
- Train: Civil Comments subtypes + Wikipedia Toxicity Subtypes + Detox Personal Attacks + DynaHate +
  HatemojiBuild.
- Hold out Aegis 1.0/2.0 prompt-only rows, filtered to HH-RLHF human prompts (CC BY 4.0 over MIT text,
  human-labelled), which cover all five options in a different register.
- Report-only: HateCheck, XSTest, SimpleSafetyTests and ConvAbuse.
- **Defer self-harm.** No clean corpus exists; all the large ones are Reddit.
- Everything hinges on clearing Civil Comments' REVIEW. Without it, `sexual` loses its only large source.
- Treat all Wikipedia text as CC BY-SA, whatever TFDS says.

**Tone.** Build a 4-option specialist, `warm` / `neutral` / `curt` / `hostile`, on the disputed bar. Its
backbone is Wikipedia Detox Aggression, whose −3…+3 scale runs from very aggressive to very friendly (CC0
labels, BY-SA text).
- Add ConvoKit Wikipedia politeness and Wikipedia Toxicity Subtypes (and Civil Comments if cleared).
- Hold out Stack Exchange politeness; test the hostile end report-only on ConvAbuse.
- Drop formality: Pavlick's licence is unverified (the source 404s) and its text is Yahoo Answers and
  news; GYAFC is Yahoo-restricted.
- Preflight the encoder ceiling before pre-registering: politeness only just passed in gate 2, and `plain`
  failed.

## Sources checked (2026-09-27)

- GitHub licence API / LICENSE files and HEAD commits:
  - hatecheck-data `3490854` CC-BY-4.0
  - xstest `d7bb5bd` CC-BY-4.0
  - SimpleSafetyTests `d7aee9a` CC-BY-4.0
  - convabuse `c0a9469` CC-BY-4.0
  - moderation-api-release `f4ab51b` MIT
  - hh-rlhf `c72f5ce` MIT
  - TOXIGEN NOASSERTION
  - wiki-detox NOASSERTION (code)
  - ConvoKit MIT (code)
- HF cards and revisions:
  - Aegis 1.0 `bd96d86`, Aegis 2.0 `d86bb8b` (CC BY 4.0)
  - hh-rlhf `09be8c5` (MIT)
  - measuring-hate-speech `5468f6e` (CC BY 4.0)
  - toxigen-data (form, GPT-3)
  - BeaverTails (CC BY-NC 4.0)
  - pavlick-formality-scores (CC BY 3.0, mirror only)
- Other pages:
  - OSF edua3: Gab Hate Corpus, CC BY 4.0
  - Zenodo 4881008: CAD, CC BY 4.0, Reddit
  - figshare API 4267550: Aggression, CC0
  - meta.wikimedia.org Research:Detox/Data_Release: labels CC0, text CC BY-SA, CrowdFlower
  - TFDS catalog pages for civil_comments and wikipedia_toxicity_subtypes
  - ConvoKit docs pages: wiki_politeness, awry
- Papers:
  - arXiv 2208.03274 (OpenAI moderation, §4.2 and §7.4)
  - arXiv 2009.10277 abstract (Measuring Hate Speech platforms)
  - ACL 2021.emnlp-main.587 (ConvAbuse §3, §6)
