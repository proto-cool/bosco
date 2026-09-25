# A4 broad results

Pre-registration: `docs/A4-BROAD.md`. Balanced accuracy.

Near-duplicates of training dropped: {'test': 526, 'cold': 1}.

Start (fly_init_keep): gain 2.0, threshold 0.05, similarity kept 0.555, KC 0.050.

## Taught kinds (test)

| kind | labels | chance | nose (1,024) | nose (bi46) | plain net | real brain | layered brain |
|---|---|---|---|---|---|---|---|
| AmbigNQ-clarifying-question | 2 | 0.50 | 0.48 | 0.53 | 0.50 | 0.47 | 0.50 |
| HatemojiBuild | 2 | 0.50 | 0.56 | 0.62 | 0.59 | 0.55 | 0.60 |
| ade_corpus_v2/Ade_corpus_v2_classification | 2 | 0.50 | 0.54 | 0.59 | 0.52 | 0.47 | 0.51 |
| citation_intent | 6 | 0.17 | 0.27 | 0.26 | 0.14 | 0.20 | 0.15 |
| crowdflower/airline-sentiment | 3 | 0.33 | 0.62 | 0.64 | 0.71 | 0.69 | 0.70 |
| crowdflower/corporate-messaging | 4 | 0.25 | 0.34 | 0.26 | 0.31 | 0.54 | 0.33 |
| crowdflower/political-media-message | 9 | 0.11 | 0.16 | 0.15 | 0.14 | 0.16 | 0.17 |
| crowdflower/sentiment_nuclear_power | 3 | 0.33 | 0.31 | 0.28 | 0.40 | 0.33 | 0.33 |
| dbpedia_14/dbpedia_14 | 14 | 0.07 | 0.93 | 0.64 | 0.46 | 0.45 | 0.39 |
| dnd_style_intents | 17 | 0.06 | 0.36 | 0.30 | 0.24 | 0.19 | 0.21 |
| dynahate | 2 | 0.50 | 0.56 | 0.52 | 0.54 | 0.58 | 0.55 |
| dynasent/dynabench.dynasent.r1.all/r1 | 3 | 0.33 | 0.54 | 0.50 | 0.50 | 0.50 | 0.55 |
| dynasent/dynabench.dynasent.r2.all/r2 | 3 | 0.33 | 0.60 | 0.58 | 0.55 | 0.61 | 0.62 |
| emo/emo2019 | 4 | 0.25 | 0.62 | 0.57 | 0.58 | 0.57 | 0.54 |
| ethics/commonsense | 2 | 0.50 | 0.56 | 0.53 | 0.53 | 0.55 | 0.51 |
| ethos/binary | 2 | 0.50 | 0.61 | 0.68 | 0.56 | 0.63 | 0.67 |
| glue/cola | 2 | 0.50 | 0.50 | 0.55 | 0.53 | 0.49 | 0.50 |
| hate_speech18 | 3 | 0.33 | 0.39 | 0.35 | 0.42 | 0.31 | 0.36 |
| hate_speech_offensive | 3 | 0.33 | 0.38 | 0.36 | 0.36 | 0.35 | 0.34 |
| hope_edi/english | 2 | 0.50 | 0.65 | 0.61 | 0.57 | 0.49 | 0.50 |
| hyperpartisan_news | 2 | 0.50 | 0.45 | 0.53 | 0.51 | 0.65 | 0.60 |
| implicit-hate-stg1 | 3 | 0.33 | 0.34 | 0.32 | 0.35 | 0.42 | 0.37 |
| insincere-questions | 2 | 0.50 | 0.58 | 0.67 | 0.63 | 0.50 | 0.53 |
| lex_glue/ledgar | 93 | 0.01 | 0.34 | 0.20 | 0.16 | 0.04 | 0.05 |
| logical-fallacy | 13 | 0.08 | 0.18 | 0.19 | 0.12 | 0.14 | 0.13 |
| open_question_type | 10 | 0.10 | 0.33 | 0.23 | 0.19 | 0.12 | 0.15 |
| poem_sentiment | 4 | 0.25 | 0.42 | 0.42 | 0.39 | 0.45 | 0.32 |
| pragmeval/mrda | 42 | 0.02 | 0.05 | 0.04 | 0.09 | 0.04 | 0.05 |
| pragmeval/switchboard | 34 | 0.03 | 0.10 | 0.04 | 0.05 | 0.08 | 0.05 |
| pragmeval/verifiability | 3 | 0.33 | 0.46 | 0.43 | 0.32 | 0.34 | 0.36 |
| scicite | 3 | 0.33 | 0.54 | 0.50 | 0.40 | 0.45 | 0.36 |
| scruples | 2 | 0.50 | 0.54 | 0.54 | 0.49 | 0.51 | 0.52 |
| silicone/dyda_e | 5 | 0.20 | 0.45 | 0.37 | 0.19 | 0.21 | 0.20 |
| silicone/meld_e | 7 | 0.14 | 0.25 | 0.29 | 0.27 | 0.22 | 0.25 |
| silicone/meld_s | 3 | 0.33 | 0.47 | 0.53 | 0.58 | 0.54 | 0.52 |
| silicone/sem | 3 | 0.33 | 0.58 | 0.57 | 0.57 | 0.58 | 0.64 |
| snips_built_in_intents | 10 | 0.10 | 0.69 | 0.61 | 0.52 | 0.33 | 0.24 |
| subjectivity | 2 | 0.50 | 0.52 | 0.50 | 0.50 | 0.53 | 0.59 |
| tweet_eval/hate | 2 | 0.50 | 0.56 | 0.53 | 0.50 | 0.53 | 0.60 |
| tweet_eval/irony | 2 | 0.50 | 0.51 | 0.53 | 0.50 | 0.54 | 0.52 |
| tweet_eval/offensive | 2 | 0.50 | 0.52 | 0.52 | 0.54 | 0.63 | 0.66 |
| tweet_eval/sentiment | 3 | 0.33 | 0.55 | 0.53 | 0.53 | 0.52 | 0.51 |
| tweets_hate_speech_detection | 2 | 0.50 | 0.63 | 0.71 | 0.61 | 0.56 | 0.65 |
| twitter-financial-news-sentiment | 3 | 0.33 | 0.61 | 0.62 | 0.59 | 0.36 | 0.50 |
| **macro** | | **0.322** | **0.469** | **0.453** | **0.426** | **0.419** | **0.418** |

## Cold (BTZSC) by tier

| task | tier | labels | chance | nose (1,024) | nose (bi46) | plain net | real brain | layered brain |
|---|---|---|---|---|---|---|---|---|
| amazonpolarity | 1 | 2 | 0.50 | 0.98 | 0.95 | 0.96 | 0.96 | 0.94 |
| appreviews | 1 | 2 | 0.50 | 0.94 | 0.93 | 0.94 | 0.90 | 0.93 |
| imdb | 1 | 2 | 0.50 | 0.88 | 0.85 | 0.83 | 0.87 | 0.88 |
| rottentomatoes | 1 | 2 | 0.50 | 0.82 | 0.81 | 0.82 | 0.82 | 0.84 |
| yelpreviews | 1 | 2 | 0.50 | 0.94 | 0.96 | 0.98 | 0.95 | 0.91 |
| financialphrasebank | 1 | 3 | 0.33 | 0.70 | 0.65 | 0.74 | 0.63 | 0.56 |
| biasframes_offensive | 1 | 2 | 0.50 | 0.47 | 0.48 | 0.48 | 0.67 | 0.67 |
| emotiondair | 2 | 6 | 0.17 | 0.50 | 0.37 | 0.38 | 0.29 | 0.32 |
| empathetic | 2 | 32 | 0.03 | 0.41 | 0.26 | 0.11 | 0.06 | 0.08 |
| agnews | 2 | 4 | 0.25 | 0.63 | 0.64 | 0.51 | 0.38 | 0.36 |
| yahootopics | 2 | 10 | 0.10 | 0.48 | 0.47 | 0.37 | 0.21 | 0.19 |
| banking77 | 2 | 77 | 0.01 | 0.59 | 0.33 | 0.24 | 0.03 | 0.07 |
| massive | 2 | 59 | 0.02 | 0.59 | 0.41 | 0.31 | 0.03 | 0.05 |
| wikitoxic_insult | 2 | 2 | 0.50 | 0.66 | 0.70 | 0.74 | 0.73 | 0.77 |
| wikitoxic_obscene | 2 | 2 | 0.50 | 0.71 | 0.69 | 0.70 | 0.78 | 0.79 |
| wikitoxic_threat | 2 | 2 | 0.50 | 0.68 | 0.75 | 0.95 | 0.71 | 0.93 |
| wikitoxic_toxicaggregated | 2 | 2 | 0.50 | 0.65 | 0.66 | 0.57 | 0.60 | 0.71 |
| biasframes_intent | 2 | 2 | 0.50 | 0.38 | 0.41 | 0.44 | 0.54 | 0.53 |
| biasframes_sex | 3 | 2 | 0.50 | 0.45 | 0.54 | 0.48 | 0.51 | 0.59 |
| capsotu | 3 | 21 | 0.05 | 0.53 | 0.32 | 0.15 | 0.08 | 0.20 |
| manifesto | 3 | 56 | 0.02 | 0.23 | 0.09 | 0.09 | 0.03 | 0.02 |
| trueteacher | 3 | 2 | 0.50 | 0.46 | 0.46 | 0.48 | 0.45 | 0.50 |
| **tier 1 macro** | | | **0.476** | **0.820** | **0.805** | **0.819** | **0.828** | **0.819** |
| **tier 2 macro** | | | **0.280** | **0.571** | **0.517** | **0.483** | **0.396** | **0.438** |
| **tier 3 macro** | | | **0.266** | **0.417** | **0.353** | **0.300** | **0.267** | **0.328** |

Validation (taught kinds): real 0.429, layered 0.418, plain net 0.444

## Decision

- **G1 (carries the catalogue): real 0.419 vs plain net 0.426 − 0.05 and chance 0.322 + 0.15: FAIL**
- **G2 (taught labels carry to new data): tier 1 real 0.828 vs nose alone 0.820: PASS**

## Replay wobble (amendment 3)

The real brain's 150-batch preflight, rerun on the same 3080: an identical start
(similarity kept 0.5554391, the first 10 losses equal to 1e-7), then drift by batch 150
(loss 1.485 vs 1.498; picks 0.45 vs 0.44 of 300). GPU scatter-adds do not repeat bit
for bit. Differences of about 0.01 between arms are inside what a replay moves.

## Reading (after the numbers; the rules are unchanged)

- **G1 fails, on the chance bar.** The real brain carries the 44 taught kinds at 0.419
  macro balanced accuracy. That clears the plain-net bar (0.426 − 0.05) but not
  chance + 0.15 (0.472). Real (0.419), layered (0.418) and plain net (0.426) are
  within replay noise of each other.
- **No trained model beats his untrained nose, even on the kinds it was taught.** Nose
  alone scores 0.469 on all 1,024 numbers, and 0.453 through the same 46 channels the
  brain sees. Training on about 41,000 examples of these very kinds did not beat plain
  matching. This is the core result. It holds for the plain net as well, so it is about
  this task setup (one softmax over label words, balanced accuracy on skewed kinds),
  not the fly.
  - A likely part of it: trained models learn which label is common, and balanced
    accuracy punishes that; cosine has no prior. That is a reason, not an excuse: the
    metric was fixed in advance, and it is the right one for a product.
  - Where he loses most: many-option kinds (ledgar 93 labels, 0.04 vs nose 0.34; snips
    0.33 vs 0.69; dbpedia 0.45 vs 0.93).
  - Where he gains on the nose: some binary and few-label kinds (corporate-messaging
    0.54 vs 0.34; hyperpartisan 0.65 vs 0.45; tweet_eval/offensive 0.63 vs 0.52).
- **G2 passes, narrowly.** On new data with taught label words (tier 1: sentiment,
  offensive), real 0.828 against nose alone 0.820. Everything sits between 0.82 and
  0.83 there, so it is a tie with his nose, not a gain. The one clear gain is
  biasframes_offensive: 0.67 for both brains against 0.47 for the nose.
- **Tiers 2 and 3** (taught concept with new label words, and new concepts): every
  trained model falls below the nose, and most on many-option tasks (banking77 0.03,
  massive 0.03 vs nose 0.59). This is the A4 pilot's finding again.
- **By the rule:** the broad catalogue is not the public v1. What he carries: binary
  and few-label judgements (sentiment, offensive, toxic, hate), at about his nose's
  level, with an occasional gain; not many-option classification.
