# Gate 4 data: topic, language, danger

Data for three gate-4 specialists (decision 29: each is a generalist in its field). The plan is
`docs/research-gate4/topic-language-danger.md`. The build is `scripts/v1_gate4_topic_language_danger.py`
(`fetch`, `build`, `embed`, `ceilings`). The provenance ledger is `docs/gate4-data-manifest-topic-language-danger.json`.

Part 1 (taxonomies, definitions, source mappings, caps, splits) was written on 2026-09-27 **before any item
was embedded and before any model score was computed**. Part 2 records what the build produced, and part 3 the
ceilings. Nothing in part 1 was changed after the numbers; later notes are marked as such.

Nothing here trains the fly. No LLM-generated text is used. No source is NC.

---

# Part 1: pre-registration (written before any score)

## Rules for all three fields

- **Sources:** only those with a USE (or CAUTION accepted by Nick) verdict in the research doc. Each licence is
  re-checked at its source on the day of download; the manifest records where.
- **Cutoff:** user-written text (Wikipedia, Wikinews, Stack Exchange, Tatoeba, arXiv metadata) must be
  from on or before 2022-11-01. Government registers are formulaic, staff-written text: the cutoff is applied
  where the data carries a date (FDA, CPSC, NHTSA), and NWS is the one exception (its API serves only recent
  products; see danger).
- **Roles:** each field has a **pool** (train/val) and a **held-out** source that it never trains on.
  - Pool → `train` 90% / `val` 10%, split by the natural unit through a seeded hash (`sha256("20260927:<unit>")`):
    QID, Stack Exchange `site/question id`, arXiv id, SGD dialogue id, MASSIVE utterance id (MASSIVE's own
    partitions), Tatoeba sentence id, FDA `event_id`, CPSC `RecallID`, NHTSA `CAMPNO`, NWS product id.
    The same unit gets the same split in every field.
  - Held-out → `dev` 20% / `test` 80% by the same kind of hash (Wikinews page id; SIB-200 `index_id`, so one
    parallel sentence is in one split in every language). **`test` is sealed**: it is counted, embedded (for the
    leakage check only) and never scored.
- **Leakage (after embedding):**
  1. `val` items whose cosine to any `train` item of the same field is > 0.95 are dropped;
  2. pool items whose cosine to any held-out item (dev or test) of the same field is > 0.95 are dropped;
  3. language: any pool text that contains a SIB-200 sentence (any language, normalised) is dropped;
  4. exact duplicates (normalised text) are kept once, in the split of their first unit, before embedding.
  Counts are reported.
- **Personal data:**
  - Wikinews: person names are scrubbed in **every** Wikinews text (not only crime), so the scrub is not a
    class cue. Rule: a run of capitalised words that starts with a given name from Wikidata's given-name items
    (CC0), or follows Mr/Mrs/Ms/Dr/Sir/Judge/President/etc., becomes `<name>`; a surname seen in such a run is
    also scrubbed where it recurs alone in the same article.
  - Stack Exchange: usernames and user ids are never read into items; they go only into the attribution
    manifest (post URL). Emails, URLs and phone numbers are scrubbed in all texts of all sources (`<email>`,
    `<url>`, `<number>`).
  - NHTSA: only `CAMPNO`, `COMPNAME`, `RCDATE`, `DESC_DEFECT`, `CONEQUENCE_DEFECT` are read. The manufacturer
    and owner-contact columns are never read.
  - CPSC: `ConsumerContact` is never read.
  - MASSIVE: `worker_id` is never read.
  - Tatoeba: usernames go only into the attribution manifest.
  - Wikipedia: no item that is a human (P31 = Q5) is used in topic or language; the Vital Articles "People"
    sub-lists are not used at all.
- **Text length:** at most 120 words (Wikipedia/Wikinews: cut at a sentence boundary as in `wiki-data.md`;
  others: first 120 words).
- **Designs** (decision 27): topic (20 options) and language (38 options) → **B**; danger (2) → **A**.

## Topic

**Question:** what is this text about? **20 options**, the research doc's taxonomy, unchanged:

| # | option (as served) | what counts |
|---|---|---|
| 1 | politics and government | elections, parties, legislation, diplomacy, public policy, government bodies |
| 2 | war and conflict | armed conflict, military, weapons, terrorism as conflict |
| 3 | law and crime | courts, crime, policing, legal questions |
| 4 | business and economy | companies, markets, trade, macro-economics |
| 5 | personal money and work | personal finance, jobs, careers, workplace |
| 6 | science | natural sciences, maths, space science |
| 7 | technology and computing | computers, internet, software, engineering, machines, energy technology |
| 8 | health and medicine | disease, treatment, anatomy, drugs, fitness as health |
| 9 | sport | sports, competitions, sports organisations |
| 10 | arts and entertainment | film, TV, music, literature, games, visual arts, architecture |
| 11 | food and drink | cooking, recipes, dishes, beverages, restaurants |
| 12 | travel and places | travel, tourism, named places (cities, countries, rivers, islands, mountains) |
| 13 | education | schools, universities, learning, academia |
| 14 | religion and belief | religions, faith, mythology |
| 15 | environment and nature | climate, weather, ecology, animals, plants |
| 16 | history | past events, periods, historical places and states |
| 17 | family and relationships | family, marriage, parenting, friendship, interpersonal |
| 18 | home and garden | housing, household, DIY, gardening |
| 19 | vehicles and transport | cars, aviation, rail, cycling, shipping, public transport |
| 20 | society and culture | social issues, media, language, customs, philosophy (the named catch-all) |

No "other" option: text about none of these gets the nearest topic (card note).

### Pool sources and mappings

**English Wikipedia, labelled by Vital Articles level 5** (list pages as of 2026-09-27, stored with their revids;
text = lead of the last revision on or before 2022-11-01, cleaned as in `wiki-data.md`). The topic comes from
the sub-list page and its level-2 (`==`) heading; a level-3 (`===`) override is named where used. Unlisted
headings, and every heading marked *drop*, are not used.

| VA page | heading | option |
|---|---|---|
| Arts/Audiovisual arts | all (architecture, venues, music, performing, visual arts) | 10 |
| Arts/Narrative arts | all (literature, theatre, film and TV, fictional characters) | 10 |
| Biology and health sciences/Animals | all | 15 |
| Biology and health sciences/Plants | all | 15 |
| Biology and health sciences/Biology | Ecology → 15; every other heading → 6 | 6 / 15 |
| Biology and health sciences/Health | all | 8 |
| Everyday life | Home living, Household items → 18; Cooking, food and drink → 11; Family and kinship, Stages of life → 17; Clothing and fashion, Sexuality and gender, General → *drop* | 18 / 11 / 17 |
| Everyday life/Sports, games and recreation | Sports, Sports organizations → 9; Games → 10; Entertainment → *drop* (toys, parks, tourism mixed) | 9 / 10 |
| Geography/Cities | all except Urban studies and planning (*drop*) | 12 |
| Geography/Regions and countries | all | 12 |
| Geography/Physical | Vegetation features → 15; every other heading (named seas, rivers, lakes, islands, mountains, parks) → 12 | 12 / 15 |
| Geography/Technical | *drop* | — |
| History | all | 16 |
| Mathematics | all | 6 |
| People/* | *drop* (biographies; living-people rule) | — |
| Philosophy and religion | Philosophy → 20; Religion and spirituality, Abrahamic religions, Eastern religions, Other religions, Mythology → 14 | 20 / 14 |
| Physical sciences/Astronomy, Basics and measurement, Chemistry, Physics | all | 6 |
| Physical sciences/Earth science | Air (weather, climate) → 15; every other heading → 6 | 6 / 15 |
| Society and social sciences/Culture | Education → 13; Communication, Culture, Ethnology and anthropology, Language, Journalism and mass media → 20 | 13 / 20 |
| Society and social sciences/Politics and economics | Business and economics → 4 (level-3 Employment and Common trades and professions → 5); Companies → 4; Law → 3; Organizations, Politics and government → 1; War and military → 2 | 4 / 5 / 3 / 1 / 2 |
| Society and social sciences/Social studies | Society, Sociology → 20; Psychology, Basics → *drop* | 20 |
| Technology | all (general, energy, industry, infrastructure, machinery) | 7 |
| Technology/Agriculture | Biotechnology → 7; Medical technology → 8; Agriculture → *drop* | 7 / 8 |
| Technology/Computing and communication | all | 7 |
| Technology/Optical, navigation and astronomical | all | 7 |
| Technology/Transportation | all | 19 |
| Technology/Weapons | all | 2 |

Deviation from the research sketch, stated now: Geography/Physical goes to 12 (its items are named places),
not 15; only its vegetation heading goes to 15. Items that are humans (Wikidata P31 = Q5) are dropped wherever
they are listed. Candidates per option are drawn round-robin over their (page, heading) groups, seed 20260927.

**Stack Exchange** (archive.org item `stackexchange`, the 2024-04 dump; questions only, `CreationDate` before
2022-11-01, not closed, score ≥ 0; text = title + body with HTML removed). The site is the topic:

| option | sites |
|---|---|
| 1 | politics |
| 3 | law |
| 4 | economics |
| 5 | money, freelancing |
| 6 | astronomy, biology, chemistry, earthscience |
| 7 | softwarerecs, hardwarerecs, webapps, engineering |
| 8 | health, fitness |
| 9 | sports, martialarts |
| 10 | movies, literature, boardgames |
| 11 | cooking, coffee, homebrew |
| 12 | travel, expatriates |
| 13 | academia |
| 14 | christianity, islam, buddhism |
| 15 | sustainability, pets |
| 16 | history, hsm |
| 17 | parenting, interpersonal |
| 18 | diy, gardening, woodworking, lifehacks, crafts |
| 19 | mechanics, bicycles |
| 20 | philosophy, linguistics |

Stack Exchange gives nothing for 2. `outdoors` is downloaded for danger only.

**MASSIVE 1.1 en-US** (local; its train and dev partitions → train and val): scenario `cooking`, `takeaway` → 11;
`transport` → 19; `weather` → 15; `music`, `play` → 10. Other scenarios are assistant commands with no subject
and are not used.

**Schema-Guided Dialogue** (GitHub, CC BY-SA 4.0; the first user turn of ≥ 4 words of each train dialogue that
uses services of one domain only): Restaurants → 11; Flights, Hotels, Travel, Buses, Trains → 12;
RentalCars, RideSharing → 19; Movies, Music, Media, Events → 10; Banks, Payment → 5; Homes → 18;
Doctors, Dentists → 8; Weather → 15. Others (Calendar, Alarm, Messaging, Services) not used.

**arXiv metadata** (OAI-PMH, CC0 metadata; records whose OAI datestamp is in 2021, i.e. last changed before the
cutoff; text = title + abstract, authors never read): `cs`, `eess` → 7; `q-fin`, `econ` → 4; `math`,
`physics` (all physics archives), `q-bio`, `stat` → 6.

**Not used** (named so the card can say so): Wikivoyage, Wikibooks, MedlinePlus (scope; Wikivoyage/Wikibooks
would also need a FLORES check), Vital Articles People, SIB-200 as a topic test (the language held-out uses it).

### Topic caps (per option, per source, pool)

Wikipedia 600, Stack Exchange 500 (round-robin over the option's sites), MASSIVE 250, SGD 250,
arXiv 300 for options 6 and 7 and 200 for 4. Pool total at most ~20k train.

### Topic held-out: Wikinews English (CC BY 2.5)

Text: the last revision on or before 2022-11-01 (articles published later are dropped), body before the
first section heading, `{{w|…}}` links rendered, date line removed, names scrubbed, ≤ 120 words. Categories are
read from that revision's wikitext.

| Wikinews category | option |
|---|---|
| Politics_and_conflicts | 1 |
| Military | 2 |
| Crime_and_law | 3 |
| Economy_and_business | 4 |
| Science_and_technology, Space | 6 |
| Computing, Internet | 7 |
| Health | 8 |
| Sports, Football_(soccer) | 9 |
| Culture_and_entertainment, Music, Film, Games | 10 |
| Food | 11 |
| Education | 13 |
| Religion | 14 |
| Environment, Weather | 15 |
| Transport, Aviation | 19 |
| Media | 20 |

Precedence (subtopic wins over its parent, fixed now): Military present → Politics_and_conflicts is ignored;
Computing, Internet or Space present → Science_and_technology is ignored. Then **an article is kept only if its
categories map to exactly one option**, and articles in Disasters_and_accidents or Obituaries are dropped.
Cap 250 per option. **Options with fewer than 50 held-out articles are excluded from the held-out and named**
(expected: 5, 12, 16, 17, 18 have no category at all; 11 and 2 may fall short).

## Language

**Question:** which language is this text in? **Options: the languages in the table below whose
unknown-token rate under the pinned nomic tokenizer is below 5%** (decision 32: 30–40 languages).

Candidates (39): every MASSIVE locale in a script the research doc measured at ≤ 5.3% `[UNK]`. The rate is
re-measured in the build on MASSIVE test utterances plus up to 500 Tatoeba sentences per language
(tokenisation only, no model); a language at ≥ 5% is excluded (Hindi, 5.3% in the research, is the expected
exclusion, leaving 38).

| option | MASSIVE | Tatoeba | Wikipedia | SIB-200 |
|---|---|---|---|---|
| Afrikaans | af-ZA | afr | af | afr_Latn |
| Arabic | ar-SA | ara | ar | arb_Arab |
| Azerbaijani | az-AZ | aze | az | azj_Latn |
| Bengali | bn-BD | ben | bn | ben_Beng |
| Catalan | ca-ES | cat | ca | cat_Latn |
| Welsh | cy-GB | cym | cy | cym_Latn |
| Danish | da-DK | dan | da | dan_Latn |
| German | de-DE | deu | de | deu_Latn |
| Greek | el-GR | ell | el | ell_Grek |
| English | en-US | eng | en | eng_Latn |
| Spanish | es-ES | spa | es | spa_Latn |
| Persian | fa-IR | pes | fa | pes_Arab |
| Finnish | fi-FI | fin | fi | fin_Latn |
| French | fr-FR | fra | fr | fra_Latn |
| Hebrew | he-IL | heb | he | heb_Hebr |
| Hindi | hi-IN | hin | hi | hin_Deva |
| Hungarian | hu-HU | hun | hu | hun_Latn |
| Indonesian | id-ID | ind | id | ind_Latn |
| Icelandic | is-IS | isl | is | isl_Latn |
| Italian | it-IT | ita | it | ita_Latn |
| Javanese | jv-ID | jav | jv | jav_Latn |
| Korean | ko-KR | kor | ko | kor_Hang |
| Latvian | lv-LV | lvs | lv | lvs_Latn |
| Mongolian | mn-MN | mon | mn | khk_Cyrl |
| Malay | ms-MY | zsm | ms | zsm_Latn |
| Norwegian Bokmål | nb-NO | nob | no | nob_Latn |
| Dutch | nl-NL | nld | nl | nld_Latn |
| Polish | pl-PL | pol | pl | pol_Latn |
| Portuguese | pt-PT | por | pt | por_Latn |
| Romanian | ro-RO | ron | ro | ron_Latn |
| Russian | ru-RU | rus | ru | rus_Cyrl |
| Slovenian | sl-SL | slv | sl | slv_Latn |
| Albanian | sq-AL | sqi | sq | als_Latn |
| Swedish | sv-SE | swe | sv | swe_Latn |
| Swahili | sw-KE | swh | sw | swh_Latn |
| Tagalog | tl-PH | tgl | tl | tgl_Latn |
| Turkish | tr-TR | tur | tr | tur_Latn |
| Urdu | ur-PK | urd | ur | urd_Arab |
| Vietnamese | vi-VN | vie | vi | vie_Latn |

Out of scope, named on the card: Chinese, Japanese, Thai, Khmer, Burmese, Amharic, Telugu, Kannada,
Malayalam, Tamil, Georgian, Armenian (the tokenizer loses them), and every language not in MASSIVE (not
measured here).

**Pool** (three registers, each brings every label):
- **MASSIVE 1.1** (chat; train partition → train, dev → val);
- **Tatoeba** (short sentences; per-language `sentences_detailed` exports, CC BY 2.0 FR; sentences added before
  2022-11-01 or undated; attribution = sentence id + username, kept only in the manifest);
- **Wikipedia leads** in each language (encyclopaedic; Wikidata items with ≥ 10 sitelinks, not humans, drawn by
  Wikidata's random sampler; the last revision on or before 2022-11-01; split by QID).

**Caps:** per language per source, 200 train and 25 val. Pool train ≤ 38 × 600 = 22,800.

**Held-out: FLORES+ was the plan, but it is gated on Hugging Face** (`gated: auto`; an anonymous request for
`devtest/fra_Latn.parquet` returned HTTP 401 on 2026-09-27). Per the brief, the gate was not worked around.
**Fallback: SIB-200** (`Davlan/sib200`, ungated, CC BY-SA 4.0), which carries the same FLORES-200 sentences
(all 1,004 per language, its train+dev+test). Test-only role, as FLORES asks. Split dev/test by `index_id`.

## Danger

**Question:** "Is this text about a risk of physical harm to people?" Options: **dangerous / not dangerous**
(same served options as gate 3). Harm = injury, illness, poisoning or death: hazards and unsafe conditions,
accidents and disasters, dangerous products and contamination, toxic substances, serious or infectious
disease, weapons and violence as physical danger, severe-weather warnings, questions about whether something
is safe. Out of scope: scams and fraud, hateful language, emotional harm. The label is "about danger", not
"is itself harmful".

### Pool sources (every register brings both labels)

| register | source | dangerous (1) | not dangerous (0) | cap per class |
|---|---|---|---|---|
| encyclopaedic | our Wikipedia/Wikidata danger set (`data/raw/clean/wiki/danger`) | its positives | its negatives | all of its gate-3 **train** and **val**; its gate-3 **test** stays sealed and is not used |
| recall | openFDA enforcement (food, drug, device) | Class I | Class III | Class III: all events (≤ 2,500); Class I: half the Class III count |
| recall | CPSC recalls + NHTSA recalls (FLAT_RCL_POST_2010) | every recall | (FDA Class III above) | CPSC and NHTSA: a quarter of the Class III count each |
| forecaster | NWS via api.weather.gov (public domain) | Tornado, Severe Thunderstorm, Flash Flood warnings | Zone Forecast Products with no warning, watch or advisory | ≤ 500 each |
| question | Stack Exchange (2024-04 dump, pre-2022-11 questions) | questions tagged `safety` or `food-safety` on cooking, diy, outdoors, travel, bicycles, chemistry, parenting, mechanics, gardening | untagged questions, same site, same number per site | ≤ 400 positives per site |

- FDA: one item per `event_id` (rows of an event are near-identical); events whose rows disagree on class are
  dropped; Class II is never used. Text = `reason_for_recall` + the first 30 words of `product_description`.
  Recall initiated before 2022-11-01.
- CPSC: text = `Title` + the `Hazards` descriptions; recall date before 2022-11-01.
- NHTSA: text = defect summary + consequence summary; one item per `CAMPNO`; report received before 2022-11-01.
- NWS: product headers, UGC/zone codes, timestamps, lat/lon and tag lines are stripped; the first 80 words of
  the body. **Exception to the cutoff:** the API serves only recent products (2026-09); this is forecaster-
  written government text, which the research doc allows without the cutoff. Split by product id.
- SE: positives and negatives are sampled per site to the same count, so the site is not a cue.

### Danger held-out: Wikinews (CC BY 2.5)

- **dangerous:** in Disasters_and_accidents and in none of Crime_and_law, Politics_and_conflicts, Military,
  Health.
- **not dangerous:** in at least one of Sports, Football_(soccer), Culture_and_entertainment, Music, Film,
  Games, Economy_and_business, Science_and_technology, Computing, Internet, Space, Education; and in none of
  Disasters_and_accidents, Crime_and_law, Politics_and_conflicts, Military, Health, Weather, Environment,
  Transport, Aviation.
- Crime, war and health are left out of both classes (ambiguous), as written in the research doc; weather,
  environment and transport are also kept out of the negatives (accidents and storms), fixed now.
- Cap 400 per class. Text as for topic (pre-2022-11 revision, names scrubbed).

## Ceilings (report only; how they will be computed)

On pool `val` and held-out `dev` only; never on `test`.
- Logistic regression (scikit-learn, `lbfgs`, max_iter 2000, balanced class weights) on the 768-dim nomic
  embedding X, with C ∈ {0.1, 1, 10} chosen on pool val balanced accuracy → balanced accuracy on pool val and
  on held-out dev; the same through the family antenna Z (46-dim).
- Nearest prototype (class means of train, cosine) on held-out dev, in X and in Z.
- Per-class recall on held-out dev (logistic, X).
- Written into part 3 and `runs/gate4-ceilings/topic-language-danger.json`.

---

# Part 2: what was built (2026-09-27)

Commands, in order (all logs in `runs/gate4-ceilings/logs/`):

    uv run python scripts/v1_gate4_topic_language_danger.py fetch licences fda cpsc nhtsa tatoeba sib arxiv nws se
    uv run python scripts/v1_gate4_topic_language_danger.py fetch names wikinews wikilang va
    uv run --with "transformers==4.46.3" --with mwparserfromhell python scripts/v1_gate4_topic_language_danger.py build
    uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops python scripts/v1_gate4_topic_language_danger.py embed
    uv run --with scikit-learn python scripts/v1_gate4_topic_language_danger.py ceilings
    uv run python scripts/v1_gate4_topic_language_danger.py manifest

Outputs: `data/cache/v1-gate4-topic-language-danger/` holds `items.json` (gate 2/3 schema plus `source`, `unit`,
`url`; sha256 `b274a8e3…c72c`), `emb.npz` (X, L, Z, ZL; family antenna `service/families/v1`; sha256 `ea2eafa6…daf`),
`BUILD.json` (every count and drop), `attribution/*.jsonl` (Wikipedia and Wikinews revision URLs, Stack Exchange post
URL and author, Tatoeba sentence URL and username, arXiv abs URL). Raw files: `data/raw/clean/<source>/` with a
`FETCH.json` each; licence pages as fetched in `data/raw/clean/_licences/`. Ledger:
`docs/gate4-data-manifest-topic-language-danger.json`.

## Sources and licences (each re-checked at source on 2026-09-27; the page is saved)

| source | licence | where verified | used for |
|---|---|---|---|
| Wikipedia (en; 38 languages) | CC BY-SA 4.0 | en.wikipedia.org/wiki/Wikipedia:Copyrights | topic pool, language pool |
| Vital Articles L5 list pages | CC BY-SA 4.0 | same | topic labels |
| Wikidata | CC0 | wikidata.org/wiki/Wikidata:Copyright | QIDs, humans check, random samples, given names |
| Wikinews (en) | CC BY 2.5 | en.wikinews.org/wiki/Wikinews:Copyright | topic and danger held-out |
| Stack Exchange 2024-04 dump (archive.org) | CC BY-SA 4.0 | archive.org item description | topic pool, danger pool |
| arXiv metadata (OAI-PMH) | CC0 | info.arxiv.org/help/api/tou.html | topic pool |
| MASSIVE 1.1 | CC BY 4.0 | local ledger (`docs/clean-data-manifest.json`) | topic pool, language pool |
| Schema-Guided Dialogue | CC BY-SA 4.0 | repo LICENSE.txt | topic pool |
| openFDA enforcement | CC0 | open.fda.gov/license | danger pool |
| CPSC Recalls API | US public domain | catalog.data.gov (usa.gov/publicdomain/label/1.0) | danger pool |
| NHTSA FLAT_RCL_POST_2010 | US public domain | catalog.data.gov (same label) | danger pool |
| NWS text products (api.weather.gov) | public domain | weather.gov/disclaimer | danger pool |
| Tatoeba | CC BY 2.0 FR | tatoeba.org/en/terms_of_use | language pool |
| SIB-200 | CC BY-SA 4.0 | HF card (Davlan/sib200) | language held-out |
| our gate-3 danger set | text CC BY-SA 4.0, labels CC0 | `docs/wiki-data.md` | danger pool |

No NC source. No LLM-generated text: user-written text is from before 2022-11-01; the NWS warnings and forecasts
(2026-09) are the stated exception (forecaster-written government text).

## Counts (after exact-duplicate and near-duplicate drops)

| field | options | train | val | held-out dev | held-out test (sealed) |
|---|---|---|---|---|---|
| topic | 20 (held-out covers 13) | 20,213 | 2,186 | 532 | 2,056 |
| language | 38 | 20,880 | 2,329 | 6,954 | 31,197 |
| danger | 2 | 15,005 | 1,334 | 158 | 642 |

**By source (train / val, or dev / test for held-out):**
- topic: Wikipedia VA 8,875 / 1,002; Stack Exchange 8,563 / 937; SGD 1,165 / 81; MASSIVE 896 / 84; arXiv 714 / 82;
  Wikinews dev 532 / test 2,056.
- language: MASSIVE 7,507 / 859; Tatoeba 7,525 / 886; Wikipedia 5,848 / 584; SIB-200 dev 6,954 / test 31,197
  (183 dev and 821 test sentences per language).
- danger: Wikipedia danger set 5,549 / 358; Stack Exchange 5,098 / 596; openFDA 2,602 / 282; NWS 890 / 6;
  CPSC 436 / 44; NHTSA 430 / 48; Wikinews dev 158 (77 dangerous, 81 not) / test 642 (323, 319).

**Topic train per option:** arts 1,437; food 1,408; vehicles 1,314; science 1,258; technology 1,246;
environment 1,211; travel 1,182; business 1,172; history 1,006; religion 998; health 992; society 985; sport 979;
politics 949; law 926; home 759; education 726; money/work 682; family 554; war 429.

**Topic held-out, options excluded (fewer than 50 single-mapped Wikinews articles):** food and drink (7), society and
culture (24), and the five with no Wikinews category at all: personal money and work, travel and places, history,
family and relationships, home and garden. The held-out covers the other 13 options. Education (66), vehicles (85),
war (91) and religion (96) are thin: 11–22 dev items each.

**Language:** 38 options. Hindi was excluded by the pre-registered rule (5.22% `[UNK]` on 2,974 MASSIVE test
utterances + 500 Tatoeba sentences). Highest kept: Korean 4.40%, Mongolian 3.38%, Urdu 1.40%, Bengali 1.26%; all others
below 0.1%. MASSIVE and Tatoeba reach their caps (200/25) for every language; Wikipedia falls short where leads are
stubs or were created after the cutoff (train 46 for Bengali, 68 Urdu, 103 Mongolian, 113 Swahili, up to 200).

## Drops and leakage

- Exact duplicates (normalised text; a pool item equal to a held-out item loses): topic 163, language 43, danger 32.
- `val` near `train` (cosine > 0.95): topic 68, language 189, danger 99. **NWS lost 94 of its 100 val items here**:
  warnings of one type are templated, so NWS val is 6 items and its val score means little.
- Pool near held-out (cosine > 0.95): topic 0, danger 0, language 330 (Wikipedia 151, MASSIVE 94, Tatoeba 85; short
  generic sentences near FLORES sentences).
- Pool text containing a SIB-200 sentence (same language, and English in any pool): 0.
- Topic VA: 146 leads not clean, 8 articles created after the cutoff, 2 titles without a QID; 0 humans found among the
  planned VA items (the People lists were not used).
- Wikinews: 4,984 articles fetched; 101 had no revision before the cutoff, 129 no clean text; 4,754 usable. The name
  scrub replaced 8,095 spans; 2,087 of the 3,388 Wikinews items used contain `<name>`. The scrub is heuristic and
  over-scrubs some places and organisations named after people (applied the same way to every class).
- openFDA: 211 events with mixed classes dropped; 1,932 Class III events available (below the 2,500 cap), so Class I
  966, CPSC 483 and NHTSA 483 (before splitting).
- NWS: 339 zone forecasts with a warning, watch, advisory, statement or hazard mention dropped from the negatives.
- The gate-3 danger `test` split (1,092) is not used anywhere.

## Deviations from part 1 and incidents (reported, not hidden)

1. **FLORES+ is gated** (HF `gated: auto`, anonymous HTTP 401); not worked around. SIB-200 is the language held-out.
2. **SGD** was not downloaded by this script: the intent/finance fetcher had already placed the repo in
   `data/raw/clean/sgd/` with its own ledger (pinned commit). This script only reads its `train/` files. An early
   attempt by this script to download into that directory stopped before changing anything (its `LICENSE.txt` hash
   is unchanged); the script now refuses to write into a directory whose ledger it does not own.
3. **Wikimedia rate limits.** For about 12 minutes five fetch streams ran at once (Vital Articles, Wikinews, three
   language groups) and drew HTTP 429s. They were stopped and replaced by one serial stream (≥ 0.25 s between
   requests, maxlag=5, backing off on 429; 60-odd further 429s over ~2 h, each waited out). Total: ~4,900 Wikinews,
   ~10,300 multilingual and ~10,500 English revision requests, plus list/pageprops calls.
4. The given-name list for the scrub came from a second, fuller Wikidata query (≥ 1 sitelink, 30,586 names) after the
   first (≥ 3 sitelinks) returned truncated JSON; both queries are cached.
5. Three business phone numbers (FDA) and one agency email (NWS boilerplate) escaped the first scrub; the regex and
   the NWS path were fixed, the build re-run, and only the changed texts re-embedded (the embed stage now keeps an
   exact-text cache, `emb_cache.npz`). Final items: 0 emails, 0 phone numbers by the check regex.
6. Two filters not spelled out in part 1: Tatoeba sentences must also be *last modified* before the cutoff and have
   at least 3 words; multilingual Wikipedia leads need at least 12 words (25 for English Vital Articles, as in
   `wiki-data.md`).

---

# Part 3: ceilings (report only; computed after part 1 and part 2 were fixed)

Logistic regression (balanced class weights) with C chosen on pool val; nearest prototype = train class means
(cosine in X, Euclidean in Z). Balanced accuracy. `runs/gate4-ceilings/topic-language-danger.json`. **The sealed
test splits were not scored.**

| field | space | C | pool val | held-out dev | prototype, held-out dev |
|---|---|---|---|---|---|
| topic | X (768) | 10 | 0.843 | **0.508** | 0.531 |
| topic | Z (46) | 10 | 0.751 | 0.509 | 0.504 |
| language | X | 10 | 0.898 | 0.943 | 0.914 |
| language | Z | 10 | 0.746 | 0.787 | 0.751 |
| danger | X | 1 | 0.862 | 0.862 | 0.851 |
| danger | Z | 10 | 0.800 | 0.806 | 0.842 |

C = 10 is the top of the grid for most rows; a larger C might do slightly better. Not explored (the grid was fixed).

**Pool val by source (X):** topic: SGD 0.957, MASSIVE 0.947, arXiv 0.852, Stack Exchange 0.834, Wikipedia 0.819.
Language: Wikipedia 0.952, Tatoeba 0.895, MASSIVE 0.867. Danger: Wikipedia 0.969, Stack Exchange 0.809, openFDA
0.775, NWS 1.000 (6 items).

**Held-out dev recall by class (X):**
- topic: sport 0.84, politics 0.71, religion 0.71, law 0.57, education 0.55, technology 0.53, business 0.48, arts 0.47,
  environment 0.44, vehicles 0.40, health 0.37, war 0.32, **science 0.22**.
- language: 1.00 or ≥ 0.92 for 31 languages; the weak ones are the known close pairs: **Malay 0.55 / Indonesian 0.61,
  Danish 0.69 / Norwegian Bokmål 0.64, Afrikaans 0.79** (Dutch 0.93).
- danger: dangerous 0.91, not dangerous 0.81.

## What the ceilings say (for Nick)

1. **Topic fails its breadth test on the encoder alone.** 0.84 on its own sources, 0.51 on Wikinews news. News is a
   register none of the pool sources has, and Wikinews categories are "what section is this in" (a science-and-
   technology story about a company reads as business). No specialist trained on this pool can reach 0.80 on this
   held-out through this encoder. Options: add a clean news-register source to the pool (the only clean one found is
   Wikinews itself; using part of it for training would need a different held-out, e.g. SIB-200 English's 7 topics),
   or accept a narrower claim (not "a generalist in topic" on news). Needs a decision before a gate is written.
2. **Language is strong in X and loses ~15 points through the antenna** (0.943 → 0.787 on dev). The 46-dim family
   antenna is the bottleneck for a 38-way task. The close pairs (ms/id, da/nb, af/nl) would stay hard in any case;
   the research doc suggested gating on ~30 languages without them.
3. **Danger passes 0.80 on held-out dev in both spaces** (0.862 X, 0.806 Z; prototype 0.842 in Z), with a thin margin
   in Z. NWS separates perfectly because warnings and forecasts differ in form, so NWS teaches little; FDA Class I vs
   III is the hard part (0.775).
4. The Stack Exchange download terms (decision 31) and the Wikinews name scrub go on the cards of any specialist
   trained or tested on them.
