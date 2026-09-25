# A4 pilot results

Pre-registration: `docs/A4-PILOT.md`. Cold accuracy on 22 BTZSC tasks never trained on.

Training kinds: AmbigNQ-clarifying-question, HatemojiBuild, I2D2, SDOH-NLI, SIGA-nli, ScienceQA_text_only, avicenna, babi_nli/basic-deduction, babi_nli/lists-sets, babi_nli/path-finding, bigbench/dyck_languages, clcd-english, cnli, counterfactually-augmented-snli, crowdflower/airline-sentiment, dbpedia_14/dbpedia_14, dnc, glue/cola, glue/qnli, hope_edi/english, idioms-nli, implicit-hate-stg1, imppres/implicature_connectives/prag, imppres/implicature_numerals_10_100/log, imppres/presupposition_both_presupposition/presupposition, insincere-questions, lexical_relation_classification/EVALution, lexical_relation_classification/K&H+N, logical-entailment, natural-language-satisfiability, nli4ct_semeval2024, open_question_type, recast_white/dpr, robust_nli/ST_NE, robust_nli_li_ts, scinli, silicone/meld_e, silicone/meld_s, stepgame, syntactic-augmentation-nli.
Cold items: 4400 (0 near-duplicates of training dropped of 4400).

| cold task | options | chance | nose alone (words) | nose alone (sentences) | plain net | real brain | layered brain |
|---|---|---|---|---|---|---|---|
| agnews | 4 | 0.25 | 0.68 | 0.84 | 0.28 | 0.36 | 0.34 |
| amazonpolarity | 2 | 0.50 | 0.98 | 0.97 | 0.94 | 0.89 | 0.95 |
| appreviews | 2 | 0.50 | 0.94 | 0.95 | 0.90 | 0.90 | 0.91 |
| banking77 | 77 | 0.01 | 0.62 | 0.59 | 0.02 | 0.02 | 0.01 |
| biasframes_intent | 2 | 0.50 | 0.45 | 0.55 | 0.48 | 0.48 | 0.48 |
| biasframes_offensive | 2 | 0.50 | 0.49 | 0.55 | 0.47 | 0.47 | 0.55 |
| biasframes_sex | 2 | 0.50 | 0.34 | 0.44 | 0.82 | 0.96 | 0.95 |
| capsotu | 21 | 0.05 | 0.54 | 0.54 | 0.20 | 0.07 | 0.18 |
| emotiondair | 6 | 0.17 | 0.53 | 0.57 | 0.34 | 0.36 | 0.39 |
| empathetic | 32 | 0.03 | 0.41 | 0.45 | 0.04 | 0.04 | 0.02 |
| financialphrasebank | 3 | 0.33 | 0.46 | 0.42 | 0.74 | 0.66 | 0.63 |
| imdb | 2 | 0.50 | 0.88 | 0.89 | 0.87 | 0.80 | 0.85 |
| manifesto | 56 | 0.02 | 0.28 | 0.34 | 0.03 | 0.01 | 0.10 |
| massive | 59 | 0.02 | 0.53 | 0.49 | 0.01 | 0.01 | 0.01 |
| rottentomatoes | 2 | 0.50 | 0.82 | 0.83 | 0.86 | 0.77 | 0.85 |
| trueteacher | 2 | 0.50 | 0.48 | 0.49 | 0.52 | 0.54 | 0.46 |
| wikitoxic_insult | 2 | 0.50 | 0.70 | 0.67 | 0.53 | 0.56 | 0.56 |
| wikitoxic_obscene | 2 | 0.50 | 0.75 | 0.68 | 0.59 | 0.57 | 0.55 |
| wikitoxic_threat | 2 | 0.50 | 0.54 | 0.42 | 0.96 | 0.96 | 0.96 |
| wikitoxic_toxicaggregated | 2 | 0.50 | 0.70 | 0.69 | 0.46 | 0.52 | 0.55 |
| yahootopics | 10 | 0.10 | 0.43 | 0.52 | 0.16 | 0.14 | 0.09 |
| yelpreviews | 2 | 0.50 | 0.94 | 0.97 | 0.97 | 0.97 | 0.98 |
| **macro** | | **0.340** | **0.613** | **0.631** | **0.510** | **0.503** | **0.516** |

Validation on the 40 training kinds: real 0.540, layered 0.557, plain net 0.602

## Decision

- **G1 (generalises): real 0.503 vs nose alone 0.631 + 0.03 and chance 0.340 + 0.10: FAIL**

## Reading (after the numbers; the rules are unchanged)

- **G1 fails, clearly.** Trained on 40 question kinds, the real brain answers
  the 22 unseen kinds at 0.503 on average. His **untrained nose alone** gets
  0.631, and chance is 0.340. The layered brain (0.516) and the plain network
  (0.510) do the same: this is about how the training was set up, not the fly.
- **Training on these 40 kinds made every trained model worse at new kinds
  than no training at all.** The worst losses are the many-option tasks
  (banking77, 77 options: nose 0.62, brains 0.01–0.02; massive, 59: 0.53 vs
  0.01; capsotu, manifesto, empathetic). The 40 kinds were mostly two- or
  three-option reasoning tasks (NLI, logic, entailment), so the trained models
  learned to weigh a few familiar label words, not to match an item's meaning
  to an option's meaning.
- **Some wins are imbalance, not understanding.** The cold items were a random
  200 per task and are scored by plain accuracy. On skewed tasks
  (biasframes_sex, wikitoxic_threat: 0.96), always picking the majority label
  looks good. The nose alone does not get that boost, so it is, if anything,
  understated.
- Even on the training kinds' own validation, the brains reach only
  0.540–0.557 (plain net 0.602). Many of those kinds are two-text reasoning
  tasks, which this nose cannot carry (the passage check).
- **By the rule, the full A4 is not run as planned.** Bosco answers the kinds
  of question he is trained on. Answering unseen kinds cold is not something
  this design does; his nose alone does it better.
