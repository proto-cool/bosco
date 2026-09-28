# Free datasets for the specialists (mood, junk, urgent/complaint), checked 2026-09-25

Research only; nothing fetched or trained. The question: which free datasets may train
small text classifiers whose **weights are redistributed** (Bosco is self-hostable and
Halteres is a SaaS)? That needs a licence that allows commercial use and redistribution
of derivatives. CC0, CC BY, CC BY-SA, Apache-2.0, MIT, CDLA-Permissive and US-government
public domain pass. CC BY-NC, research-only terms, no stated licence, text taken from
platforms whose terms forbid reuse (Twitter/X, Yelp, Amazon, IMDb), and LLM-generated
sets do not. Reddit is flagged as CAUTION. Synthetic sets are listed separately.

Verdicts: **USE** means the licence is verified at source and the provenance is clean.
**CAUTION** means usable only with the stated risk accepted. **NO** means do not use.
"Unverified" means I could not confirm the licence at the original source. None of
this is legal advice. Where a CC BY-SA text ends up in weights, whether the weights are
an "adapted work" is unsettled. The conservative reading is to attribute and to keep the
data itself under BY-SA if it is redistributed.

The existing ledger for the tasks already in use (dynahate, hatemoji, civil_comments,
MASSIVE, CLINC) is `docs/clean-data.md` / `docs/audit-2026-09-25/provenance.md`.

## 1. Mood / sentiment / emotion

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **DynaSent round 2** | Potts, Wu, Geiger, Kiela (Stanford / FAIR) | github.com/cgpotts/dynasent | CC BY 4.0, verbatim in the repo README | Crowd-written on Dynabench (MTurk) to fool a model. The `has_prompt=True` rows were written after the worker saw a Yelp sentence | none expected (original sentences) | ~13k train, 720 dev, 720 test (paper figures, not rechecked) | positive / negative / neutral (+ mixed at validation) | **USE**. For extra caution drop the `has_prompt` rows: the text is new, but it was written from a Yelp prompt |
| DynaSent round 1 | same | same | CC BY 4.0 on the annotations | Sentences **taken from the Yelp Academic Dataset** | low | ~80k train, 3.6k dev/test (paper) | same | **NO**. The Yelp Dataset Terms of Use are "academic purposes only", with no commercial use (yelp-dataset-agreement PDF, 2018/2020/2023 versions). The CC BY licence on the labels cannot relicense Yelp's text |
| poem_sentiment | Google Research | huggingface.co/datasets/google-research-datasets/poem_sentiment | CC BY 4.0 (HF card) | Verses from Project Gutenberg poems (public domain), labelled by Google | none | 1,101 | negative / positive / no_impact / mixed | **USE**, but it is tiny and in the wrong register (poetry). A sanity set, not a training set |
| Stanford Politeness Corpus (Stack Exchange + Wikipedia) | Danescu-Niculescu-Mizil et al. / Cornell ConvoKit | convokit.cornell.edu/documentation/stack_politeness.html | CC BY 4.0 (ConvoKit docs). The underlying SE and Wikipedia text is CC BY-SA | Real requests from SE and Wikipedia talk pages, MTurk-rated | usernames / ids in metadata | 6,603 SE (+ ~4.3k Wikipedia) | polite / neutral / impolite + score | **USE** (attribute; treat the text as BY-SA). This is tone, not sentiment, but it helps "complaint/rude" |
| EmoBank | Buechel & Hahn (JULIE Lab) | github.com/JULIELab/EmoBank | CC BY-SA 4.0 (repo) | MASC corpus sentences (fiction, letters, news, blogs) + SemEval-2007 news headlines | none expected | ~10k sentences | Valence-Arousal-Dominance (writer and reader); Ekman subset | **CAUTION**. The licence is fine (BY-SA). Upstream: SemEval-2007 headlines are news-agency text. MASC is an open subcorpus. Continuous VAD labels, not classes |
| GoEmotions | Google Research | huggingface.co/datasets/google-research-datasets/go_emotions | Apache-2.0 (HF card) | **Reddit comments**, scraped; labelled by raters in India | Reddit usernames retained (card says identities may be discoverable) | 58k (54.3k simplified) | 27 emotions + neutral, multi-label | **CAUTION**. Google's Apache licence covers its release, not Reddit users' text. Reddit's 2023 data terms restrict commercial and model-training use. Clear it with a lawyer before shipping weights trained on it |
| XED (English) | Helsinki-NLP | github.com/Helsinki-NLP/XED | CC BY 4.0 (repo) | **Movie subtitle lines from OPUS/OpenSubtitles** | none | ~17.5k unique + 6.4k neutral | Plutchik 8 + neutral, multi-label | **CAUTION**. The subtitles are copyrighted film dialogue, and OpenSubtitles' own terms are unclear. The CC BY covers the annotations |
| BRIGHTER / SemEval-2025 Task 11 | Muhammad et al. | huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories | CC BY 4.0 on the HF card, **but** the paper (arxiv.org/html/2502.11926) says the data may not be used "for commercial purposes or by state actors in high-risk applications" without creator approval | English = AskReddit personal narratives; German = Reddit; Spanish = YouTube comments | social-media text | eng: 2,764 train / 230 dev / 5,528 test | anger, disgust, fear, joy, sadness, surprise (+ neutral) | **NO**. The card and the paper conflict, and the stricter statement is non-commercial. The sources are Reddit and YouTube |
| Financial PhraseBank | Malo et al. (Aalto) | huggingface.co/datasets/takala/financial_phrasebank | **CC BY-NC-SA 3.0** (HF card) | LexisNexis news sentences | none | 4,840 | pos / neg / neutral | **NO**. Non-commercial |
| SST-2 / SST-5 | Socher et al. (Stanford) | huggingface.co/datasets/stanfordnlp/sst2 | **"Unknown"** on the HF card | Rotten Tomatoes review snippets | none | 11,855 sentences | binary / 5-way | **NO**. No licence, and the text is scraped review content |
| dair-ai / CARER "emotion" (also mteb/emotion) | Saravia et al. | huggingface.co/datasets/mteb/emotion | "Unknown" on the card | **Twitter** | tweets | 20k | 6 emotions | **NO**. Twitter, and no licence |
| UCI Sentiment Labelled Sentences | Kotzias et al. | archive.ics.uci.edu/dataset/331 | CC BY 4.0 on UCI | Amazon, IMDb and Yelp reviews | none | 3,000 | pos / neg | **NO**. The source platforms forbid reuse, so UCI's CC BY cannot cure it |
| IberaSoft/ecommerce-reviews-sentiment | IberaSoft (HF) | huggingface.co/datasets/IberaSoft/ecommerce-reviews-sentiment | CC BY 4.0 claimed | By its own card: Amazon and Yelp API reviews + G2/Capterra/TrustRadius | reviewer text | 20k | neg / neutral / pos | **NO**. It relicenses platform content it has no right to relicense |
| TweetEval, Sentiment140, SemEval tweet tasks, Amazon/Yelp/IMDb polarity, EmpatheticDialogues (CC BY-NC), DailyDialog (CC BY-NC-SA), MELD (Friends) | various | — | — | — | — | — | — | **NO**, not individually re-verified. They fail on platform (Twitter/Amazon/Yelp/IMDb/TV) or on an NC licence |

## 2. Junk / spam

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **SMS Spam Collection v.1** | Almeida & Gómez Hidalgo | archive.ics.uci.edu/dataset/228/sms+spam+collection | **CC BY 4.0** on UCI. The original readme says "© Almeida & Gómez Hidalgo… provided AS IS" and asks for a citation and a notice of use | Grumbletext (UK forum where users posted spam they received) 425 spam; NUS SMS Corpus 3,375 ham; Caroline Tag thesis 450 ham; SMS Spam Corpus v0.1 Big 1,002 ham / 322 spam | real SMS. Some ham holds first names or phone-ish tokens | 5,574 (747 spam) | ham / spam | **USE, with a note**. The licence is clean at UCI. The upstream parts came from "free or free for research sources", and the NUS SMS Corpus's own CC variant was unverified (site down). It is the best-documented short-message spam set. Note that the HF mirror `ucirvine/sms_spam` says "Unknown": cite UCI, not HF |
| Enron-Spam (Metsis, Androutsopoulos, Paliouras 2006) | AUEB / NCSR | aueb.gr/users/ion/data/enron-spam (certificate error when fetched); mirrors: github.com/MWiechmann/enron_spam_data, HF SetFit/enron_spam | **Unverified. No licence stated** in the readme (research.cs.wisc.edu mirror) or on the HF card | Ham from six Enron employees' mailboxes (Enron corpus, released through FERC). Spam from SpamAssassin, the Honeypot project, Bruce Guenter's spam trap and the authors' own | **yes**: real employees' names, email addresses, private content | 33,716 (17,171 spam) | ham / spam | **CAUTION → effectively NO for shipped weights** until a licence is found. The Enron text is a public record but unlicensed, and it carries heavy personal data |
| SpamAssassin public corpus | Apache SpamAssassin project | spamassassin.apache.org/old/publiccorpus/readme.html | **No licence.** The readme says "Copyright for the text in the messages remains with the original senders" | Public fora, newsletters, the maintainer's own mail, spam traps | email headers; some addresses obfuscated | 6,047 (1,897 spam) | spam / easy_ham / hard_ham | **CAUTION**. It is widely used, but nothing grants rights. It is good as a held-out evaluation set that is never redistributed |
| Ling-Spam | Androutsopoulos et al. | aueb.gr/users/ion/data/lingspam_public.tar.gz | Terms: acknowledge the corpus in published work and notify the creator. No commercial grant (per secondary sources, not seen at source) | Linguist mailing-list posts + the author's spam | list posters' names | 2,893 (481 spam) | ham / spam | **NO**. Research terms, unverified, and the ham is narrow (linguistics) |
| TREC 2005–2007 public spam corpora | Cormack (Waterloo) / NIST | cormack.uwaterloo.ca/treccorpus07 (404 on 2026-09-25) | **Unverified**. Historically released under a research usage agreement | Real mail stream | yes | 75k (2007) | spam / ham | **NO** (unverified, research terms) |
| UCI YouTube Spam Collection | Alberto & Lochter | archive.ics.uci.edu/dataset/380 | CC BY 4.0 on UCI | **YouTube comments** scraped from 5 music videos | usernames | 1,956 | spam / ham | **CAUTION**. YouTube ToS on scraped comments. It is also small and topically odd |
| Deysi/spam-detection-dataset | "Deysi" (HF user) | huggingface.co/datasets/Deysi/spam-detection-dataset | Apache-2.0 (card) | **Provenance not stated**: looks like forum and social posts | unknown | 10,900 | spam / not_spam | **CAUTION**. The licence is claimed but the source is unknown, so it is not verifiable |

## 3. Urgent / complaint

No clean, permissively licensed **human-written "urgent vs not"** dataset turned up.
The support-ticket sets that carry urgency labels are synthetic and/or non-commercial,
e.g. `Tobi-Bueck/customer-support-tickets`: CC BY-NC 4.0 per its HF card, synthetic.
Complaint text is available from US-government sources. Urgency will need a proxy (below) or our own labels.

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **CFPB Consumer Complaint Database** (narratives) | US Consumer Financial Protection Bureau | consumerfinance.gov/data-research/consumer-complaints/ ; catalog.data.gov/dataset/consumer-complaint-database | The CFPB page says "All complaint data we publish is freely available for anyone to use, analyze, and build on". data.gov lists access "public" but **no licence field**. The HF loader card `CFPB/consumer-finance-complaints` says CC0 | Consumer-written narratives, published only with **opt-in consent** and scrubbed per the CFPB Narrative Scrubbing Standard (2015) | scrubbed (XXXX masks); ZIP3 and state kept | Millions of complaints; narratives on a large subset (>1M historically, not rechecked) | product, sub-product, **issue, sub-issue**, company response, timely y/n | **CAUTION → likely USE**. Two risks. (a) The narratives are consumers' own writing, not federal works, so 17 USC 105 public domain does not strictly cover them. CFPB now calls the published narratives "public domain for FOIA purposes". (b) **On 2026-08-14 CFPB stopped publishing narratives** (CFPB newsroom; ABA Banking Journal). Past narratives are moving to the CFPB FOIA Reading Room, and the live API sample I fetched had no narrative field. The HF entry is a loader script (26.5 kB) that pulls live data, so it may no longer return narratives. **If wanted, snapshot the narratives now** from the FOIA Reading Room or an existing archived CSV, and record the hash. Every text is a complaint, so it gives complaint *categories* (issue) but no negatives |
| **NHTSA ODI vehicle complaints** | US DOT / NHTSA | static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip ; catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-complaints | **US Public Domain** (`usa.gov/publicdomain/label/1.0`) on data.gov | Owner-written complaint text (`CDESCR`, ≤2,048 chars) since 1995 | the text may name people. The structured fields include city/state, VIN and dealer contacts; a vehicle-operator-name field was added 2026-04 (drop the structured PII) | 2M+ complaints (not counted) | component, crash / fire / injury flags (a usable **severity/urgency proxy**: crash=Y, fire=Y, injuries>0) | **USE** (drop the PII columns). Same caveat as CFPB on consumer authorship, but the dataset licence is an explicit US PD mark. Domain: vehicles only |
| BANKING77 | PolyAI | huggingface.co/datasets/PolyAI/banking77 | CC BY 4.0 (HF card) | Real-style banking customer queries (not synthetic per paper) | none | 13,083 | 77 intents, incl. lost_or_stolen_card, compromised_card, card_swallowed, transaction_charged_twice, card_payment_not_recognised, failed_transfer, pin_blocked, lost_or_stolen_phone | **USE**. It can be mapped to urgent-vs-routine and problem-vs-request, but the mapping must be written down **before** looking at results |
| CLINC150 (oos-eval) | Clinc / Larson et al. | github.com/clinc/oos-eval (LICENSE = CC BY 3.0) ; HF clinc/clinc_oos | **CC BY 3.0** (repo LICENSE, verified) | Crowd-written paraphrases / scenario responses | none | 23,700 (150 intents + OOS) | incl. report_fraud, report_lost_card, damaged_card, freeze_account, account_blocked, card_declined, lost_luggage | **USE** (already in the pilot). The same proxy mapping applies |
| MASSIVE (en-US) | Amazon | huggingface.co/datasets/AmazonScience/massive | CC BY 4.0 (HF card) | SLURP utterances, human-localised on MTurk | none | 11.5k / 2k / 3k | 60 voice-assistant intents; **no complaint or urgent intent** | USE for intents, **not useful** for this task |
| Stanford Politeness (see §1) | — | — | CC BY 4.0 | — | — | — | impolite / neutral / polite | **USE** as a tone signal for "complaint" |
| Jigsaw Toxic Comment Classification (Wikipedia talk) | Jigsaw / Google | Kaggle jigsaw-toxic-comment-classification-challenge | Dataset CC0; comment text CC BY-SA 3.0 (Wikipedia) (from secondary sources and the Kaggle mirror, not the competition rules page) | Wikipedia talk comments | usernames sometimes in the text | ~160k train | toxic, severe_toxic, obscene, threat, insult, identity_hate | **USE (BY-SA)** for hostile tone, not for complaint as such |
| Jigsaw Civil Comments | Jigsaw | huggingface.co/datasets/google/civil_comments | CC0 1.0 (card; TFDS) | Civil Comments platform comments (news sites) | low | ~2M | toxicity + 6 subtypes | already on the ledger (REVIEW). Tone only |

## Synthetic / LLM-generated (listed separately, per the non-negotiables)

| name | licence | nature | verdict |
|---|---|---|---|
| Bitext customer-support (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`) | **CDLA-Sharing-1.0** (HF card) | "Hybrid synthetic": NLG-generated from seed texts curated by linguists. 26,872 rows, 27 intents incl. `complaint`, `review`; tags for politeness, colloquial, offensive and typos | **Licence OK** (copyleft on the *data*; CDLA-Sharing puts no obligation on model outputs or trained weights). **Synthetic**, so do not use as the test set. At most a training supplement, and it must be flagged in any gate report |
| Tobi-Bueck/customer-support-tickets | **CC BY-NC 4.0** (HF card) | Synthetic tickets with priority/urgency | **NO** (NC) |

## Short answer

- **Mood:** DynaSent round 2 (CC BY 4.0, crowd-written) is the only clean, real-text,
  general-domain 3-way sentiment set found. Add the Stanford Politeness corpus (CC BY 4.0 / BY-SA)
  for tone. GoEmotions is the only large emotion set and is CAUTION (Reddit).
- **Junk:** the SMS Spam Collection (CC BY 4.0 at UCI) is the one spam set with a stated
  permissive licence. Every email spam corpus (Enron-Spam, SpamAssassin, Ling-Spam,
  TREC) is unlicensed or research-only.
- **Urgent/complaint:** NHTSA complaints (explicit US PD mark; crash/fire/injury flags
  as a severity proxy) and CFPB narratives (free to use, consented and scrubbed, but no
  formal licence, and **publication ended 2026-08-14**, so snapshot now). BANKING77 and
  CLINC150 give problem/urgent intents via a pre-written mapping. No clean human-labelled
  "urgent vs not" set exists, so that needs our own labels.
