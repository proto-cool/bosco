# Training data for the topic, language and danger specialists (gate 4 research), checked 2026-09-27

Research only. Nothing was trained, and nothing was fetched into `data/`. What was run:
- small API calls for counts: Wikinews category sizes, openFDA class counts, Stack Exchange tag counts, and archive.org metadata for the SE dump;
- the nomic tokenizer, over the local MASSIVE 1.1 tarball (`data/raw/clean/massive/`), to count unknown tokens per locale. This was tokenisation only, with no embedding and no fitting.

Every request carried the User-Agent `bosco-research/0.1 (https://bosco.systems)`. None of this is legal advice.

This builds on:
- `docs/free-datasets.md`
- `docs/clean-data.md`
- `docs/wiki-data.md`
- `docs/specialist-gate-3-results.md`
- `docs/research-gate4/harm-tone.md`

Verdicts:
- **USE** means the licence is verified at source and the provenance is clean.
- **CAUTION** means usable only with a stated risk accepted.
- **NO** means do not use.
- "Unverified" means I could not confirm the licence at the original source.

**Rules that apply to all three fields:**
1. **Every training source contributes both (all) labels where it can.** A source that brings only one label teaches "which source is this" instead of the field. Where a source is one-label (CPSC recalls, OSHA), it is paired with a same-register source that brings the other label, and this is written down before any numbers.
2. **Text cutoff 2022-11-01** for user-written text (Stack Exchange, Tatoeba, Wikipedia, Wikinews), as in `wiki-data.md`. Government registers (FDA, CPSC, NWS) are formulaic and staff-written, so the cutoff is optional there. It is simplest to apply it everywhere.
3. **Split by the natural unit:** QID, SE question id, FDA `event_id`, NHTSA campaign or ODI number, Wikinews page id. Never split by row.
4. **Breadth test:** one pre-registered held-out source, with the 0.80 bar on it. Also report leave-one-source-out: train on all sources but X, test on X, for every X. Report it but do not gate on it.

---

## 1. Topic

### 1.1 Candidate datasets

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **Wikinews (English)** | Wikinews volunteers / WMF | en.wikinews.org; dumps.wikimedia.org/enwikinews | **CC BY 2.5** for articles published 2005-09-25 to 2024-12-15. CC BY 4.0 after that. Public domain before 2005-09-25. Verified at en.wikinews.org/wiki/Wikinews:Copyright on 2026-09-27 | Volunteer-written news. **Closed and read-only since 2026-05-04** (WMF decision; en.wikinews.org/wiki/Wikinews:Closure_of_Wikinews), so it is now a frozen corpus | Names public figures, and in crime and accident stories also private people (victims, suspects). Scrub names, or drop the Crime_and_law articles that name individuals | 22,222 published articles (Category:Published, API 2026-09-27) | Categories. Counts from the API: Politics_and_conflicts 7,850; Crime_and_law 4,552; Disasters_and_accidents 2,817; Sports 2,462; Economy_and_business 2,365; Culture_and_entertainment 2,214; Science_and_technology 2,106; Health 1,254; Transport 1,052; Internet 858; Obituaries 836; Aviation 803; Football_(soccer) 774; Religion 685; Environment 632; Space 565; Weather 534; Computing 522; Music 468; Education 360; Film 266; Military 248; Media 237; Food 143; Games 126; Agriculture 46. Articles carry several categories | **USE** (attribute; CC BY). **Best held-out source**: it is news, a register none of the training sources has. Food, family, home and travel are thin or missing |
| **Stack Exchange data dump** (per-site) | Stack Exchange Inc. / users | archive.org/details/stackexchange (dump 2024-04-02) | Content **CC BY-SA 4.0** (2.5/3.0 for older posts), per the archive.org item description, checked 2026-09-27. **Caveat:** since July 2024 new dumps sit behind a login, with a click-through saying the file is not for LLM training and that SE may refuse future downloads (devclass.com 2024-07-30; search.feep.dev 2025-02-20). The archive.org 2024-04 copy predates that and has no click-through. Critics say the new terms conflict with CC BY-SA §2(a)(5) | Q&A written by users | User display names and ids. They are needed for attribution (SE asks for author name and link), and are never put in the text. Questions can hold personal anecdotes | Per-site 7z (archive.org metadata), for example: cooking 88 MB (28,188 questions), travel 156, law 104, politics 115, money 133, health 22, sports 17, music 107, movies 85, gardening 44, diy 228, mechanics 66, parenting 46, pets 27, christianity 125, islam 55, judaism 139, hinduism 72, buddhism 52, history 94, biology 83, physics 729, chemistry 126, earthscience 23, sustainability 10, academia 199, workplace 201, interpersonal 36, outdoors 31, bicycles 82, fitness 35, astronomy 51, economics 44, space 80, aviation 107, expatriates 18, boardgames 45, gaming 241, scifi 338, literature 28, philosophy 127, photo 114, woodworking 15, homebrew 17, coffee 5, lifehacks 14, genealogy 15 | The site is the topic (pre-written map below). Tags refine it | **CAUTION → likely USE**. The content licence is clean and verified. The dump-terms dispute is the accepted risk: we take the archive.org 2024-04 copy, not the gated one, and we do not train an LLM. Keep posts created before 2022-11-01. Carry the attribution manifest (post URL and author) |
| **English Wikipedia via our pipeline, labelled by Vital Articles level 5** | WP editors; list by WikiProject Vital Articles | en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5 | Text CC BY-SA 4.0 (as `wiki-data.md`). The list pages are CC BY-SA | Curated list of 49,802 articles in 11 sections with finer sub-lists: People 14,177; History 3,273; Geography 5,180; Arts 3,822; Everyday life 2,460 (home, clothing, food and drink, family, stages of life…); Philosophy and religion 1,549; Society and social sciences 4,399 (social studies; politics and economics; culture); Biology and health 5,698; Physical sciences 4,765; Technology 3,268; Mathematics 1,211 | People: keep only persons born before 1900 who have died (the gate-3 rule). Otherwise as `wiki-data.md` | ~50k candidates. After the pre-2022 text fetch and cleaning, expect ~40k | The sub-list heading is the topic (map below) | **USE**. The list membership is today's; the text is the pre-2022-11 revision. This fixes the `wiki-data.md` "subject: skipped" problem: an article sits in exactly one sub-list, so no precedence rule is needed |
| Wikipedia via WikiProject → ORES *articletopic* taxonomy | Wikimedia (drafttopic) | mediawiki.org/wiki/ORES/Articletopic | Page CC BY-SA. The mapping ships in the drafttopic repo (licence not checked) | WikiProject tags on talk pages, mapped to 64 topics (Culture / Geography / History and Society / STEM) | as above | millions | 64 topics, multi-label | **CAUTION**. Use it as a *reference taxonomy*, not a label source. Labels are current talk-page state and multi-label (`wiki-data.md` §subject). Vital Articles is cleaner |
| **arXiv metadata (titles + abstracts)** | arXiv / Cornell | info.arxiv.org/help/api/tou.html; OAI-PMH or Kaggle `Cornell-University/arxiv` | **CC0 1.0**, and it explicitly covers "title, abstract, authors, identifiers, and classification terms". Verified at info.arxiv.org 2026-09-27. The Kaggle page did not render, so the mirror is unverified; use OAI-PMH | Author-written abstracts | Author names (not needed; drop) | ~2.5M | Primary category: cs, math, physics, q-bio, q-fin, stat, econ, eess | **USE**, but only for science, technology and maths, and only abstracts before 2022-11. Cap its share so it does not swamp the pool |
| **MASSIVE 1.1 (en-US), scenarios** | Amazon | already local (`clean-data.md`) | CC BY 4.0 (verified, ledger) | Crowd-localised voice-assistant utterances | `worker_id`, dropped | en-US 16.5k. Scenarios: calendar 2,370, play 2,024, qa 1,685, email 1,381, iot 1,107, general 963, weather 855, transport 805, lists 793, news 709, recommendation 596, datetime 578, social 565, alarm 550, music 469, audio 387, takeaway 358, cooking 326 | 18 scenarios | **USE** for chat register, for the few scenarios that map: cooking and takeaway → food; transport → travel or vehicles; weather → environment; music and play → arts. Drop the rest (assistant commands with no subject) |
| **Schema-Guided Dialogue (SGD)** | Google Research | github.com/google-research-datasets/dstc8-schema-guided-dialogue | **CC BY-SA 4.0** (repo README, verified 2026-09-27) | A dialogue simulator made outlines; paid crowd-workers paraphrased them. Not LLM-generated | none expected | >20k dialogues, 20 domains | Domain or service: Restaurants, Flights, Hotels, Travel, Buses, Trains, RentalCars, Movies, Music, Media, Events, Banks, Payment, Homes, Doctors/Dentists, Weather, … | **USE** as user turns for chat register. Its origin is template-seeded, so say so on the card |
| Wikivoyage / Wikibooks (Cookbook) | WMF projects | en.wikivoyage.org, en.wikibooks.org/wiki/Cookbook | CC BY-SA 4.0 (Wikimedia terms). Footers not rechecked separately | Volunteer-written | low | Wikivoyage ~30k pages; Cookbook some thousands | Travel pages, recipes | **USE** to fill travel and food with non-encyclopaedic, how-to text. Pre-2022-11 revisions. *Leakage:* FLORES and SIB-200 sentences were drawn from Wikinews, Wikivoyage and Wikibooks. If SIB-200 is a test set, drop any training passage that contains a SIB sentence |
| **SIB-200 (English slice)** | Adelani et al. | github.com/dadelani/sib-200; HF Davlan/sib200 | Repo LICENSE **Apache-2.0**; HF card **CC BY-SA 4.0** (the FLORES sentences). Both seen 2026-09-27 | Topic labels on FLORES-200 sentences | none | 1,004 sentences (701/99/204) | science/technology, travel, politics, sports, health, entertainment, geography | **USE as a second held-out test only** (FLORES says it must not be used for training). Its 7 labels map to a subset of ours |
| MedlinePlus health-topic summaries | US NLM | medlineplus.gov/xml.html | **Public domain** for health-topic summaries. The **A.D.A.M. encyclopedia and ASHP drug monographs are copyrighted**. Verified at medlineplus.gov/about/using/usingcontent 2026-09-27 | NLM staff | none | ~1,000 topics (EN) | health (single topic) | **USE** (summaries only) as extra health text. It is one-topic, so it cannot be a held-out source |
| Project Gutenberg catalogue (LoCC / subjects) | PGLAF | gutenberg.org/cache/epub/feeds/pg_catalog.csv | Texts are PD **in the US**; PG trademark licence applies to redistribution with the header. Not rechecked | Digitised old books | none | 70k+ books | LoC class, subjects, bookshelves | **CAUTION**. The same issue as MultiNLI fiction: authors who died less than 70 years ago are in copyright in the EU and the UK (we serve from France). The register is 19th-century. Low priority |
| Yahoo Answers Topics | Zhang et al., from Yahoo Webscope | HF community-datasets/yahoo_answers_topics | **"unknown"** on the HF card (2026-09-27). Webscope is research-only (not seen at source) | Scraped Q&A | usernames | 1.4M | 10 topics | **NO** |
| AG News | Gulli (ComeToMyHead) | groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html | The page says it is for "research … and any other non-commercial activity". Verified 2026-09-27 | News-agency text | — | 1M | 4 | **NO** |
| BBC News (Greene & Cunningham) | UCD | mlg.ucd.ie/datasets/bbc.html | "non-commercial and research purposes only … copyright … owned by the BBC". Verified 2026-09-27 | BBC articles | — | 2,225 | 5 | **NO** |
| 20 Newsgroups | Lang | qwone.com/~jason/20Newsgroups | No licence stated (not rechecked); Usenet posts belong to their authors | Usenet | emails and names | 18,846 | 20 | **NO** |
| HuffPost "News Category" and other scraped news sets | various | Kaggle | CC BY claimed by the uploader over scraped publisher text | — | — | — | — | **NO** (the uploader cannot relicense a publisher's text) |
| DBpedia-14 | — | local (`clean-data.md`) | CC BY-SA / GFDL | — | — | — | *Entity type*, not topic | Not a topic source. `kind` already covers entity type |

### 1.2 Proposed taxonomy (20 options, written before any data is pulled)

| # | option | what counts |
|---|---|---|
| 1 | politics & government | elections, parties, legislation, diplomacy, public policy |
| 2 | war & conflict | armed conflict, military, terrorism as conflict |
| 3 | law & crime | courts, crime, policing, legal questions |
| 4 | business & economy | companies, markets, trade, macro-economics |
| 5 | personal money & work | personal finance, jobs, careers, workplace |
| 6 | science | natural sciences and maths, space |
| 7 | technology & computing | computers, internet, software, engineering, gadgets |
| 8 | health & medicine | disease, treatment, fitness-as-health, nutrition science |
| 9 | sport | sports, athletes (dead only, per the gate-3 rule), competitions |
| 10 | arts & entertainment | film, TV, music, books, games, visual arts, celebrities |
| 11 | food & drink | cooking, recipes, dishes, beverages, restaurants |
| 12 | travel & places | travel, tourism, geography of places |
| 13 | education | schools, universities, learning, academia |
| 14 | religion & belief | religions, faith, philosophy of religion |
| 15 | environment & nature | climate, weather, ecology, animals, plants |
| 16 | history | past events, historical periods |
| 17 | family & relationships | parenting, marriage, friendship, interpersonal |
| 18 | home & garden | housing, DIY, household, gardening |
| 19 | vehicles & transport | cars, aviation, rail, cycling, public transport |
| 20 | society & culture | social issues, media, language, customs (the named catch-all for social science) |

No "other" option. The card must say that text about none of these gets the nearest topic.

**Mapping plan (sketch; the full table is fixed in the pre-registration before any data is pulled):**
- **Wikinews:** Politics_and_conflicts → 1 (Military → 2); Crime_and_law → 3; Economy_and_business → 4; Science_and_technology → 6 or 7 (Computing, Internet → 7; Space → 6); Health → 8; Sports and Football → 9; Culture_and_entertainment, Music, Film, Games → 10; Food → 11; Education → 12; Religion → 14; Environment, Weather → 15; Transport, Aviation → 19; Media → 20.
  - **Keep only articles whose categories map to exactly one option.** Drop Obituaries and Disasters_and_accidents (not a topic here; danger uses them).
- **Stack Exchange:**
  - politics → 1; law → 3; economics → 4; money, workplace, freelancing → 5;
  - physics, chemistry, biology, astronomy, earthscience, space → 6 (physics sampled; math.SE left out as formula-heavy);
  - superuser (sampled), engineering, photo → 7; health, fitness → 8; sports → 9;
  - movies, music, scifi, literature, gaming, boardgames, writers → 10;
  - cooking, coffee, homebrew → 11; travel, expatriates → 12;
  - academia → 13; christianity, islam, judaism, hinduism, buddhism → 14;
  - sustainability, gardening (?), pets → 15 (gardening goes to 18, pets to 15; fix now);
  - history, hsm → 16; parenting, interpersonal → 17; diy, woodworking, lifehacks → 18;
  - mechanics, bicycles, aviation → 19; philosophy, english/ell → 20.
  - **SE gives no questions for:** 2.
- **Wikipedia Vital Articles L5:**
  - **By section:** History → 16 (the military-history sub-list → 2); Geography → 12 (physical geography sub-lists → 15); Arts → 10; Philosophy and religion → 14 (the philosophy sub-list → 20).
  - **Everyday life:** home → 18, food and drink → 11, family → 17, sports and recreation → 9.
  - **Society:** politics → 1, economics → 4, law → 3, education → 13, media and language → 20.
  - **Science and technology:** Biology and health → 8 (medicine sub-lists) or 15 (organisms and ecology); Physical sciences and Mathematics → 6; Technology → 7 (transport sub-list → 19).
  - **People:** by occupation sub-list, dead pre-1900 people only.
- **MASSIVE and SGD:**
  - MASSIVE: as in the table above.
  - SGD: Restaurants → 11; Flights, Hotels, Travel, Buses, Trains, RentalCars → 12 or 19 (fix before data: trips → 12, vehicles → 19); Movies, Music, Media, Events → 10; Banks, Payment → 5; Homes → 18; Doctors, Dentists → 8; Weather → 15.
- **arXiv:** everything except cs → 6; cs → 7; q-fin and econ → 4. Cap at 10% of the pool.

**Pool and held-out source.**
- **Train:** Wikipedia (Vital Articles), Stack Exchange, MASSIVE, SGD, Wikivoyage/Wikibooks, arXiv and MedlinePlus. This covers encyclopaedic, question, chat and how-to registers.
- **Held-out test (gated): Wikinews**, news register, pre-2022-11 and single-mapped. Balance it per option; options with fewer than 50 test articles are reported, not scored. That is likely 11, 17, 18 and 12 (Wikinews has no Travel category), so the gated score covers about 15 of the 20 options. **Say this plainly on the card.**
- **Second test (report only):** SIB-200 English, 7 options.
- **Leave-one-source-out (report only):** Stack Exchange held out is the one that tests "questions".

---

## 2. Language

### 2.1 What the encoder can see (measured 2026-09-27)

`nomic-embed-text-v1.5`:
- The model card tags it English only. Nomic's own v2 card calls v1 "English-only".
- It uses the **`bert-base-uncased` WordPiece tokenizer** (BertTokenizer, 30,522 entries, lower-casing).
- It therefore **strips accents**: "schön" becomes "schon" and "trời đẹp" becomes "troi đep". Language cues from diacritics are partly lost.
- Any word that holds a character outside the vocabulary becomes a single `[UNK]`.

The tokenizer was run over every MASSIVE 1.1 **test** utterance per locale (tokenisation only):

| unknown-token share | locales |
|---|---|
| **0%** | all Latin-script locales (af, az, ca, cy, da, de, en, es, fi, fr, hu, id, is, it, jv, lv, ms, nb, nl, pl, pt, ro, sl, sq, sv, sw, tl, tr, vi), and ru, el, he, ar, fa |
| 1–5% | ur 0.9%, bn 1.3%, mn 3.3%, ko 4.0%, hi 5.3% |
| 20–26% | hy 20%, ja 20%, ka 22%, ta 26% (17–27% of their utterances are at least half `[UNK]`) |
| **48–98% (unusable)** | th 48%, zh-TW 66%, zh-CN 68%, kn 90%, km 95%, my 97%, am 98%, te 98%, ml 98% (62–99.8% of utterances are at least half `[UNK]`) |

**What this means:**
- **Script is visible for the others.** Cyrillic, Greek, Arabic, Hebrew and Bengali letters survive as single characters. The encoder will likely separate scripts, but separating languages *within* a script depends on English-trained representations of foreign word pieces. Examples: ar/fa/ur, id/ms, da/nb, es/pt/ca/gl, hr/sr/bs, cs/sk.
- **No embeddings were computed.** Measure a logistic ceiling on the encoder before pre-registering, as the gate-3 preflight did.
- **For the unusable scripts** (Thai, Chinese, Khmer, Burmese, Amharic, Telugu, Kannada, Malayalam) the encoder sees almost nothing. Most such texts become a run of `[UNK]` and are indistinguishable from one another.
- **The honest options:**
  - leave these languages out and say so;
  - or map a mostly-`[UNK]` input to "unsupported script" with a rule outside the brain. But that rule would be the decider, which the non-negotiables forbid;
  - or change the encoder. `nomic-embed-text-v2-moe` is Apache-2.0 with ~100 languages on an XLM-R base (HF card, 2026-09-27). That is Nick's decision, not something this doc changes. It would re-open every shipped specialist.

### 2.2 Candidate datasets

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **MASSIVE 1.1 (all locales)** | Amazon | already local: `data/raw/clean/massive/amazon-massive-dataset-1.1.tar.gz` | **CC BY 4.0** (tarball LICENSE; HF card lists 52 locales). Verified | Professional/MTurk localisation of the same 16.5k utterances into each locale | `worker_id`, dropped | ~16.5k per locale × 52 | locale | **USE**. Chat register. The content is parallel across languages, so topic cannot give the language away (good). Split by utterance `id`, so a parallel set stays in one split |
| **Tatoeba** | Tatoeba contributors | tatoeba.org/en/downloads | Default **CC BY 2.0 FR**, plus a separate **CC0** export for part of the sentences. Verified at tatoeba.org/en/terms_of_use and /downloads, 2026-09-27. Contributors may attach NC terms to *audio*, not text | Volunteer-written sentences and translations | Usernames (needed for attribution) | Millions of sentences in 400+ languages; large for ~50 | language code (contributor-set; some errors) | **USE**. Attribute by sentence id and username, or take only the CC0 export. Keep sentences added before 2022-11 (date field: unverified whether it is exported). Very short sentences |
| **Wikipedia in many languages, via our pipeline** | WP editors | xx.wikipedia.org; Wikidata sitelinks | CC BY-SA 4.0 (as `wiki-data.md`) | Pick QIDs with sitelinks in the target languages; take the pre-2022-11 lead in each language | as `wiki-data.md` (no living people) | as many as fetched | the wiki's language | **USE**. Encyclopaedic register. **Split by QID across all languages**, so one subject is in one split. Watch out for (a) bot stubs (sv/ceb/war lsjbot, templated village stubs), which should be capped per template; (b) Content-Translation copies of the English lead; (c) English quotes and names inside non-English leads |
| **FLORES+** (OLDI) | Meta, then Open Language Data Initiative | huggingface.co/datasets/openlanguagedata/flores_plus | **CC BY-SA 4.0**. **Gated**, with the stated purpose: "It should not be used as training data." Verified on the HF card, 2026-09-27. The GitHub repo is archived and points to HF | Professional translations of English sentences from Wikinews, Wikijunior and Wikivoyage | none | dev 997 + devtest 1,012 sentences × 230 varieties | language | **USE as the held-out test only**, which is what it asks for. Record the gate acceptance with the account and date |
| **WiLI-2018** | Thoma | zenodo.org/records/841984 | **ODbL 1.0** (Zenodo record, verified). Text is Wikipedia CC BY-SA | Random Wikipedia paragraphs | low | 235k paragraphs, 235 languages (1,000 each) | language | **CAUTION → USE**. Pre-LLM (2018). ODbL asks for a notice on "produced works" and share-alike on derived *databases*. Redundant with our own multilingual Wikipedia fetch; useful as a cheap second Wikipedia source |
| Common Voice sentence corpus (text only) | Mozilla / communities | github.com/common-voice/common-voice (`server/data/<lang>`) | The project states that sentences are **CC0** (community playbook, common-voice.github.io/community-playbook/sub_pages/text.html). The repo LICENSE is MPL-2.0 for the code; **no per-folder CC0 file was seen** | Sentence Collector (volunteers, 2-of-3 review) plus a Wikipedia extractor | low | 100+ languages; 5k+ sentences per enabled language | language | **CAUTION**. The Wikipedia-extracted part rests on Mozilla's legal view of short excerpts, not on a licence. Prefer Sentence-Collector sentences. Good read-aloud register |
| SIB-200 | see §1 | — | Apache-2.0 / CC BY-SA 4.0 | FLORES sentences | — | 205 languages | topic | Same sentences as FLORES; not an extra source for language |
| Europarl v7 | Koehn | statmt.org/europarl | "We are not aware of any copyright restrictions" (statmt.org, verified). **No explicit licence** | EU Parliament proceedings | Speaker names of MEPs (drop the speaker tags) | 21 EU languages, ~2M sentences each | language | **CAUTION**. Rests on the EP's reuse notice, not checked here. Parliamentary register |
| Universal Dependencies 2.18 | UD community | universaldependencies.org | **Per treebank**: CC BY-SA, CC BY, GPL, **and CC BY-NC-SA** (verified that the mix exists) | varied | varied | 200+ treebanks, 150+ languages | language | **CAUTION**. Only treebanks whose own LICENSE is BY or BY-SA, checked one by one. Little gain over the above |
| OPUS (general) | Tiedemann | opus.nlpl.eu | **Per corpus.** OpenSubtitles and others are unclear or restricted | — | — | — | — | **NO as a block**; decide per corpus (Tatoeba-in-OPUS is Tatoeba) |
| papluca/language-identification and similar HF sets | — | — | built from the Amazon multilingual reviews corpus | — | — | — | — | **NO** (Amazon) |

### 2.3 Proposal

**Language set.** Take the languages the encoder can see (0–5% `[UNK]` above) that also appear in MASSIVE, Tatoeba, Wikipedia and FLORES:
- **Western European:** en, de, fr, es, pt, it, nl, ca.
- **Nordic:** sv, da, nb, fi, is.
- **Central and Eastern European, Latin script:** pl, ro, hu, lv, sl, sq.
- **Other scripts:** ru, el, he, ar, fa, ur, hi, bn, ko, mn.
- **Rest of the world:** tr, az, id, ms, vi, tl, sw, cy, af, jv.

That is **40 languages**. Two tiers:
- **Gated set (~30):** drop the likely confusable pairs jv, ms, ur, mn, az and af unless the preflight ceiling shows they separate. id/ms and da/nb are the known hard pairs.
- **Out, and said on the card:** zh, ja, th, km, my, am, te, kn, ml, ta, ka, hy (the tokenizer loses them).

Add cs, uk, hr, sk, bg, lt, et if the preflight shows Latin and Cyrillic neighbours separate. They are in Tatoeba, Wikipedia and FLORES but not in MASSIVE, so their training would have only two sources.

**Pool.**
- **Train:** MASSIVE (chat), Tatoeba (short sentences) and multilingual Wikipedia leads (encyclopaedic), balanced per language and per source. Optionally Common Voice (Sentence Collector part).
- **Held-out test (gated): FLORES+ devtest**, professional translations of news and travel text. That is a register none of the training sources is.
- **Report only:** leave-one-source-out, and dev as a sanity check.
- **Leakage check:** FLORES sentences are translations of Wikinews, Wikivoyage and Wikibooks sentences, not of Wikipedia. Still, an exact and near-duplicate check against the training pool is needed, because Tatoeba contributors sometimes paste in public sentences.

---

## 3. Danger

### 3.1 Definition (proposed; fix in the pre-registration)

**"Is this text about a risk of physical harm to people?"** Harm means injury, illness, poisoning or death. This covers:
- hazards and unsafe conditions;
- accidents and disasters;
- dangerous products, recalls and contamination;
- toxic substances;
- infectious or serious disease;
- weapons and violence as physical danger;
- severe weather warnings;
- questions asking whether something is safe.

**Options: dangerous / not dangerous** (2), the same as gate 3, so the shipped `danger` has a broader successor.

**Out of scope:**
- **Money scams and fraud:** a different harm. It could become a separate `scam` specialist later: FTC consumer alerts are US-government text, but FTC's copyright page returned 404 and is unverified.
- **Hateful or toxic language:** that is `hate` and harm.
- **Emotional harm.**

**Later, a 4–5 way "danger kind" specialist** (health/disease; accident/disaster; product/substance; weather; weapon/violence) could reuse the same pool. It is not proposed now.

**The label is "about danger", not "is itself harmful".** Instructions for making weapons would be positive here *and* are a harm-specialist question. The card says so.

### 3.2 Candidate datasets

| name | maker | URL | licence (where verified) | collection | personal data | size | labels | verdict |
|---|---|---|---|---|---|---|---|---|
| **Our Wikipedia/Wikidata danger set** | us | `data/raw/clean/wiki/danger` | Wikidata CC0; text CC BY-SA (`wiki-data.md`) | our rule over Wikidata classes | none (no humans) | 7,004 (≈50/50) | danger 1/0 | **USE**. Encyclopaedic; answer-in-the-lead bias noted in gate 3 |
| **openFDA enforcement reports (food, drug, device)** | US FDA | open.fda.gov/apis/food/enforcement ; /drug/ ; /device/ | **CC0 1.0 / public domain** for openFDA data. The exception is GMDN device-nomenclature content, which must not be used; we do not need those fields. Verified at open.fda.gov/license, 2026-09-27 | FDA Recall Enterprise System, 2004 on; FDA-staff-classified | Company names only | Counts from the API, 2026-09-27: **food** Class I 12,926 / II 14,729 / III 1,760; **drug** I 1,747 / II 14,511 / III 1,715; **device** I 3,632 / II 35,307 / III 1,029 | `classification` | **USE**. The strongest *hard-negative* source. **Class I** means "reasonable probability of serious adverse health consequences or death"; **Class III** means "not likely to cause adverse health consequences". Both are the same register and often the same wording ("undeclared egg", Class I, against "sesame … declared in the statement of identity", Class III). **Class I → 1, Class III → 0; drop Class II.** Text = `reason_for_recall` (+ `product_description`). Dedup and split by `event_id`, because one event has many near-identical rows |
| **CPSC Recalls API** | US CPSC | saferproducts.gov/RestWebServices/Recall ; catalog.data.gov/dataset/recalls-api | **US Public Domain** (`usa.gov/publicdomain/label/1.0`) on data.gov, verified 2026-09-27 | CPSC recall notices | Firm names; consumer contact numbers (drop) | Thousands (the API was flaky when probed; not counted) | `Hazards`, `Injuries`, `Description`, `Title` | **USE**, positives only. All recalls are hazards. Pair them with FDA Class III as the same-register negatives. Do not let "recall" alone mean danger |
| **NHTSA vehicle recalls (FLAT_RCL)** | US DOT / NHTSA | www-odi.nhtsa.dot.gov/downloads/folders/Recalls/FLAT_RCL.zip ; catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls | **US Public Domain** mark on data.gov, verified 2026-09-27 | Manufacturer defect reports (Part 573) | none in the text | ~25k+ campaigns (not counted) | Defect / consequence / remedy summaries | **USE**, positives only (every one is safety-related). The `CONSEQUENCE` summary is the danger text |
| **NHTSA ODI complaints (FLAT_CMPL)** | US DOT / NHTSA | static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip | US Public Domain mark (`free-datasets.md`, verified 2026-09-25) | Owner-written complaints | The text can name people; the structured fields hold VIN, city and names (drop them, and scrub the text) | 2M+ | crash / fire / injured / deaths flags | **USE with a noisy-label note.** Flagged (crash=Y, fire=Y, injuries>0 or deaths>0) → 1. Complaints with no flag *and* in non-safety components (e.g. paint, electrical accessories, rather than brakes, steering or airbags; the list fixed before data) → 0. Unflagged brake or steering complaints are dropped as ambiguous. Consumer register; a good **second held-out** |
| **OSHA Severe Injury Reports** | US DOL / OSHA | osha.gov/severe-injury-reports ("Download the full SIR data set", 2015-01 to 2025-11) | **No licence statement on the page**. Likely public domain as a federal work (17 USC 105): the narratives are OSHA-processed employer reports. **Unverified** | Employer reports of amputation, hospitalisation or eye loss | Establishment names and addresses. Employees are usually not named, but check | ~100k+ (not counted) | all positive; nature and body part | **CAUTION**. Positives only, with no same-register negatives, so report-only (e.g. the recall rate on held-out positives), not in the gated pool |
| **NWS text products (warnings vs routine forecasts)** | NOAA / NWS; archived by Iowa Environmental Mesonet | mesonet.agron.iastate.edu (AFOS archive, 1996 on) ; weather.gov/disclaimer | NWS content is **public domain** ("may be used without charge for any lawful purpose"; weather.gov/disclaimer, verified 2026-09-27). **IEM's own terms are unverified** (disclaimer page not read) | Forecaster-written | none | very large | product type | **USE (NWS) / check IEM terms.** Positives are Tornado, Flash Flood, Severe Thunderstorm, Hurricane, Blizzard, Excessive Heat and Tsunami *Warnings*. Negatives are Zone Forecast Products and Area Forecast Discussions on quiet days. **Beware the ALL-CAPS register and product headers**: strip the headers and lowercase, or style will give the label away |
| **Stack Exchange safety-tagged questions** | SE users | archive.org dump (§1) | CC BY-SA 4.0 (§1; same dump-terms caveat) | User questions | usernames (attribution only) | Tag counts from the API, 2026-09-27: cooking `food-safety` 2,446; diy `safety` 823; outdoors `safety` 632; travel `safety` 651; bicycles `safety` 555; chemistry `safety` 539; parenting `safety` 272; mechanics `safety` 199; gardening `safety` 67. Total ≈ 6,100 | 1 = has a safety tag. 0 = same site, no safety tag, sampled to match | **CAUTION → likely USE.** The only *question-register* danger source. Tag noise: some untagged electrical or chemistry questions are about danger. Report a hand check of 100 negatives before the gate |
| **Wikinews Disasters_and_accidents (+ Weather, Fires, Earthquakes)** | Wikinews | §1 | CC BY 2.5 (verified) | News | news names (§1) | 2,817 positives (Fires 364, Earthquakes 299, Weather 534, Product_recalls 28) | 1 = in Disasters_and_accidents. 0 = in Sports, Culture_and_entertainment, Economy_and_business, Science_and_technology or Education, and in **none** of Disasters, Crime_and_law, Health, Military or Politics_and_conflicts | **USE**, news register with both labels. **Proposed held-out test.** Crime, war and health news are ambiguous and are left out of both classes; this is written down now |
| MedlinePlus health topics / CDC pages | NLM / CDC | §1 ; cdc.gov/other/agencymaterials.html | PD summaries (verified); CDC "most … public domain" with contractor exceptions (verified 2026-09-27) | Agency staff | none | ~1k / many | none directly | **CAUTION.** "Is a disease page dangerous?" is the gate-3 debatable-label problem. Leave it out, or use only CDC outbreak and travel notices as positives |
| OpenFEMA | FEMA | fema.gov/about/openfema/terms-conditions | Not a licence: terms that require a set disclaimer, ban re-identification, and include an **indemnity clause** (verified 2026-09-27) | Mostly structured, little text | — | — | — | **NO / not needed.** Little text, and it adds terms |
| CFPB complaints | — | `free-datasets.md` | — | — | — | — | — | Not danger (financial) |

### 3.3 Proposal

**The pool.** Each register brings both labels:

| register | source | label 1 | label 0 |
|---|---|---|---|
| encyclopaedic | Wikipedia/Wikidata danger set | dangerous items | the other items |
| recall | FDA enforcement reports | Class I | Class III |
| recall | CPSC and NHTSA recalls | every recall (positives only) | balanced by FDA Class III |
| forecaster | NWS | warnings | routine forecasts |
| question | Stack Exchange | safety-tagged | untagged, same sites |

**Tests:**
- **Held-out test (gated): Wikinews.** Disasters and accidents against clearly non-danger news, pre-2022-11, balanced.
- **Second held-out (report only): NHTSA complaints**, consumer register, flags as labels.
- **Report only:** leave-one-source-out and the OSHA positives-only recall.

**Honest limits for the card:**
- A news event is labelled by its Wikinews category, not by a reader.
- The FDA classes are the FDA's health-hazard judgement of the product, not of the text.
- "Safety" tags are what an asker chose.
- The v1.5 encoder is English-only, so danger is English-only.

---

## 4. Short recommendations

- **Topic:**
  - **Taxonomy:** the 20-option one in §1.2.
  - **Train:** Wikipedia (labels from Vital Articles level 5, which fixes the "subject: skipped" problem) + Stack Exchange per-site dump (archive.org 2024-04, CC BY-SA; accept the dump-terms dispute as CAUTION) + MASSIVE/SGD chat + Wikivoyage/Wikibooks + capped arXiv (CC0).
  - **Test:** gate on **Wikinews** (CC BY 2.5, frozen since 2026-05). About 15 options will have enough test articles. SIB-200 English is a report-only second test.
  - **Excluded:** AG News, BBC, Yahoo Answers and 20NG are NO.
- **Language:**
  - **Train:** MASSIVE (local, CC BY 4.0) + Tatoeba (CC BY 2.0 FR, or its CC0 export) + multilingual Wikipedia via the pipeline, split by QID.
  - **Test:** gate on **FLORES+ devtest** (CC BY-SA, which asks to be test-only).
  - **Scope:** limit to ~30–40 Latin, Cyrillic, Greek, Arabic, Hebrew and Indic-lite languages. **The v1.5 tokenizer turns Chinese, Thai, Khmer, Burmese, Amharic, Telugu, Kannada and Malayalam text into mostly `[UNK]`, and strips accents.** Run an encoder-ceiling preflight before pre-registering. If Nick wants those languages, the fix is an encoder change (nomic v2-moe, Apache-2.0), not more data.
- **Danger:**
  - **Definition:** "about a risk of physical harm to people", 2 options. Scams and fraud are out of scope.
  - **Train:** the Wikipedia danger set + **FDA enforcement Class I vs Class III** (CC0, the best hard negatives) + CPSC and NHTSA recalls as positives + NWS warnings vs routine forecasts + SE safety tags.
  - **Test:** gate on **Wikinews Disasters_and_accidents vs clearly non-danger news**. NHTSA complaints are a report-only second test.
  - **Report only:** OSHA (licence statement missing) as a positives-only check.
