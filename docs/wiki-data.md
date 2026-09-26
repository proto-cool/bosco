# Wikipedia and Wikidata datasets (kind, food, danger, plain), fetched 2026-09-26

This covers the Wikipedia/Wikidata item of `docs/audit-2026-09-25/provenance.md` §7 and decision 10 of `docs/DECISIONS-2026-09-25.md`:
- The **labels** come from Wikidata, which is CC0.
- The **texts** are lead paragraphs of English Wikipedia and Simple English Wikipedia articles, which are CC BY-SA 4.0.
- Each text is taken **from the last revision at or before 2022-11-01**. This follows the ledger's REVIEW recommendation to keep text written by LLMs out.

Nothing here is legal advice.

- **Build:** `scripts/v1_gate3_fetch.py`, run in three stages: `select`, then `text`, then `build`.
- **Outputs:** `data/raw/clean/wiki/<task>/{train,val,test}.jsonl`. The directory is gitignored.
- **Record:** `data/raw/clean/wiki/SUMMARY.json` holds the counts, the drops and the SHA-256 of every file.
- **Cache:** every SPARQL result is kept, with its query and fetch time, under `data/raw/clean/wiki/_cache/sparql/`. Every fetched revision is kept in `_cache/revs_{en,simple}.jsonl`.
- **Replay:** `build` runs offline from the cache and is byte-for-byte reproducible. It was rerun after a reformat and gave identical hashes.

## Table

| task | question | train | val | test | total | balance |
|---|---|---|---|---|---|---|
| kind | what kind of thing is this about (7 classes) | 5,597 | 377 | 1,088 | 7,062 | 1,003–1,012 per class |
| food | is this about food or drink? | 5,080 | 308 | 936 | 6,324 | 3,149 yes / 3,175 no |
| danger | is this about something dangerous to people? | 5,549 | 363 | 1,092 | 7,004 | 3,488 yes / 3,516 no |
| plain | is this the plain (Simple English) version? | 3,984 | 250 | 862 | 5,096 (2,548 pairs) | exactly 1:1 |
| subject | broad field of the article | — | — | — | skipped | see below |

**Per-split labels:**
- **kind, test:** creative_work 153, event 150, organisation 167, person 140, place 168, product_technology 143, species 167.
- **food, val:** 172 no / 136 yes. Its other splits are within 1% of 50/50.
- **danger:** all splits are within 2% of 50/50.

**Length:** texts average 71–80 words and are capped at 120.

**Revision timestamps:**
- The newest revision in every task is from 2022-10-31. None is later than the cutoff.
- The oldest is from 2013. These are pages not edited between then and the cutoff.

## Record format (one JSON object per line)

**Fields:**
- `task`, `qid` (Wikidata item) and `label`. `label` is a class name for kind, and 1 or 0 for the others.
- `text`, and `wiki`, which is `enwiki` or `simplewiki`.
- `title`, `pageid`, `revid` and `rev_timestamp`.
- `url`, which is `https://en.wikipedia.org/w/index.php?oldid=<revid>`, the attribution link to the exact revision used.
- `split`.
- `source_class`, the Wikidata class or group that selected the item.
- For food and danger negatives, `neg_kind`.
- For plain, `pair_id` (the QID) and `side` (`plain` or `complex`).

## Method

**1. Selection (Wikidata Query Service).** Requests carried the User-Agent `Bosco-research/0.1 (...; nduncan@fastmail.com)`. They were sent at most once a second, and the client backed off on 429, 500 and 504.
- Candidates must have an English Wikipedia sitelink.
- Kind candidates also need at least 5 sitelinks: at least 8 for persons, and at least 3 for food and danger positives.
- Wikimedia pages are excluded: disambiguation, list, category and template pages.
- Large classes were sampled with Blazegraph's random sampler (`bd:sample`), so a fresh `select` draws a different, equally valid sample. The cached result is the record of the sample actually used.
- The script then picks candidates deterministically with seed 20260926. For kind it takes them round-robin over subclasses, so no single subclass dominates a class.

**2. Text (MediaWiki API).**
- **The request:** `prop=revisions&rvstart=2022-11-01T00:00:00Z&rvdir=older&rvlimit=1&rvsection=0&rvslots=main&maxlag=5`, one title per request. The API allows `rvstart` only for a single page, so these requests cannot be batched.
- **Workers:** two, then four once throughput was measured. They retry on maxlag and on Retry-After.
- **Volume:** 22,028 English and 4,200 Simple English requests, between 15:45 and 17:50 MDT on 2026-09-26.
- **Dropped articles:**
  - Articles created after the cutoff have no revision and are dropped (`no_revision_before_cutoff`).
  - So are titles that were a redirect or disambiguation page at the time (`no_clean_lead`).

**3. Cleaning.**
- The wikitext is parsed with mwparserfromhell.
- **Removed:** infoboxes, tables, files, references, comments and hatnotes.
- **Rendered as text:** a small set of inline templates, including `convert`, `lang`, `nowrap`, dates, `ship` and `ill`.
- **Dropped:** every other template. A parenthetical that contained a dropped template is removed whole. This clears most pronunciations, native-script names and lifespans.
- **Filtered:** leads shorter than 25 words (10 for Simple English) are dropped, as are leads with leftover markup.
- **Trimmed:** texts are cut at a sentence boundary to at most 120 words.
- **Deduplicated:** a page, a QID or an identical text appears only once per task and side. Two QIDs can resolve to one page through a redirect.
- **Kept:** the article title is left in the text wherever the lead starts with it.

**4. Splits.**
- A text's split comes from a seeded hash of its QID: `sha256("20260926:<qid>")`. The cut points are 80/5/15.
- **Across tasks:** a subject lands in the same split in every task it appears in.
- **Within a task:** no QID appears in two splits. This was checked for all four tasks.
- **Plain pairs:** both sides of a pair share a QID, so they are always in the same split.

## Living people (rule)

- **In no task but kind:** no item with P31 = Q5 (human) appears in food, danger or plain.
- **Kind, class `person`:** a human must have a date of birth (P569) before 1900 **and** a date of death (P570). Food and danger reuse some of these as negatives.
- **Plain:** excludes humans altogether.
- **Not screened:** non-biography articles, such as films, organisations and events, can name living people. They were not screened for this.

## Class mappings (Wikidata QIDs)

### kind

Each item is a direct instance (P31) of one of the listed classes, with two exceptions: person uses the birth and death rule above, and species uses taxon rank (P105) = species (Q7432).

| class | Wikidata classes |
|---|---|
| person | Q5 human, born before 1900, with a date of death |
| place | Q515 city, Q3957 town, Q532 village, Q8502 mountain, Q4022 river, Q23397 lake, Q23442 island |
| organisation | Q4830453 business, Q7278 political party, Q3918 university, Q476028 association football club, Q163740 nonprofit organization, Q327333 government agency, Q18127 record label |
| creative_work | Q11424 film, Q482994 album, Q3305213 painting, Q5398426 television series, Q7725634 literary work, Q860861 sculpture |
| event | Q178561 battle, Q198 war, Q188055 siege, Q1076105 general election, Q124757 riot, Q10931 revolution, Q273120 protest, Q45382 coup d'état, Q132241 festival, Q7944 earthquake, Q8092 tropical cyclone |
| species | P105 = Q7432 (any taxon of species rank; animals, plants, fungi and so on) |
| product_technology | Q3231690 car model, Q15056995 aircraft model, Q9143 programming language, Q9135 operating system, Q6368 web browser, Q19723451 smartphone model, Q19832486 locomotive class, Q20888659 camera model, Q7397 software. Q8076 video game console was also tried and returned nothing. |

**Left out of kind:**
- **Novels and plays:** Q8261 novel and Q25379 play return nothing as direct P31. Wikidata models them as literary work plus form of creative work (P7937), so "literary work" carries them.
- **Musical groups:** left out as borderline between organisation and creative work.
- **Weapons:** left out of products.

**Class sizes:** after cleaning, the classes were cut to the smallest one, person with 1,012.
- **Species:** needed 2,600 candidates to reach that number, because many species leads are one-line stubs.
- **General elections:** give only 34 texts. Other event subclasses give 49–63 each.

### food

**Positive rule:**
- The item is an instance of (P31), or is itself, a class in the P279* subclass tree under Q2095 (food) or Q40050 (drink). The two trees hold 31,785 classes.
- The item is not a taxon, not a human, and not in any danger group.

**Result:** 12,789 candidates, of which 3,600 were fetched and 3,149 kept. A random look at 80 candidates found dishes, cheeses, drinks, snacks and brands (such as Wrigley's Spearmint and Finlandia Vodka), and one oddity, the enzyme "Sucrose α-glucosidase".

**Negatives:**
- **Pool:** round-robin over the kind pool: person, place, organisation, creative_work, event and product_technology, and their subclasses, with 526–531 each.
- **Food excluded:** anything in the food tree.
- **Species excluded:** from food negatives entirely, because edible species (cod, cabbage) would be mislabelled near-misses.

### danger

Positives come in five groups. An item is taken for the first group it fits, and items in the food tree are excluded.

| group | rule | kept |
|---|---|---|
| disease | classes in the P279* tree under Q18123741 infectious disease **that carry an ICD-10 (P494), ICD-9 (P493) or ICD-11 (P7807) code**; this drops bee, fish and pig diseases | 526 |
| toxic | Q184651 toxin, Q40867 poison, Q21973549 chemical weapon or Q2612896 nerve agent (as instance or P279* subclass), or P2868 "subject has role" = Q187661 carcinogen, Q21074597 occupational carcinogen or Q40867 poison; **minus anything with an ATC code (P267)**, i.e. medicines such as tamoxifen | 502 |
| weapon | Q728 weapon and its P279* subclasses (weapon types), their instances, and instances of Q15142894 weapon model and its subclasses | 961 |
| venomous | species (P105 = Q7432) whose P171+ parent-taxon chain reaches Q186554 Elapidae, Q163656 Viperidae, Q19125 Scorpiones or Q273179 Cubozoa (box jellyfish) | 556 |
| disaster | instances of a P279* subclass of Q8065 natural disaster (earthquakes, floods, cyclones, eruptions, landslides and so on) | 943 |

**Negatives:**
- **Pool:** round-robin over the kind pool: person, place, organisation, creative_work and product_technology, with 698–706 each.
- **Danger excluded:** anything in any danger group.
- **Near-misses kept out of negatives:**
  - all events, because battles, wars, riots and massacres are dangerous in the ordinary sense;
  - earthquakes and tropical cyclones;
  - aircraft models, which include military aircraft;
  - all species, which include predators and poisonous plants and fungi that are not in the venomous list.

### plain

- **Sample:** a random sample of 30,000 Simple English Wikipedia sitelinks, drawn by Wikidata's `bd:sample`, with an English sitelink on the same item. Humans and Wikimedia pages are excluded. This gave 15,561 pairs, and 4,200 were fetched.
- **Revisions:** both sides use their last revision before 2022-11-01.
- **Minimum length:** the Simple English lead needs at least 10 words and the English lead at least 25.
- **Near-copies:** 67 pairs are dropped because the Simple English lead is a near-copy of the English one (word-set Jaccard > 0.8).
- **Result:** 2,548 pairs kept.
- **Labels:** 1 = plain (Simple English), 0 = complex (English).

## subject: skipped

**Why:** no clean labels exist without heavy heuristics. Wikidata has no general "field" property for arbitrary articles:
- P921 (main subject) is on works;
- P101 (field of work) is on people and organisations;
- P2578 (studies) is on the disciplines themselves.

**The closest signal:** English Wikipedia's WikiProject assessments (`prop=pageassessments`). But they have three problems:
- they are talk-page metadata in their **current** state, not their state in 2022;
- most articles belong to several projects, so a precedence rule would have to be invented;
- their project scopes overlap (History against Military history against Politics).

**Cost:** it would also have needed about 8,000 more single-page revision fetches beyond the budget. It is doable later with the WikiProject route if an "exactly one of the target projects" rule is accepted.

## Licences and attribution

| part | licence | what it requires |
|---|---|---|
| Wikidata labels, classes, QIDs | CC0 1.0 | nothing |
| English Wikipedia text (kind, food, danger, and the `complex` side of plain) | CC BY-SA 4.0 (also GFDL for older text) | attribution: the `url` field links the exact revision, whose history lists the authors; share-alike on any redistributed derivative of the text |
| Simple English Wikipedia text (the `plain` side of plain) | CC BY-SA 4.0 (also GFDL) | same |

As in the ledger:
- **Do not redistribute the texts with the weights.** If they are ever shipped, ship them under CC BY-SA with the attribution manifest, which is the `url` and `pageid` of each row.
- **Whether weights trained on them are an "adaptation" is unsettled.**

## Caveats and things to double-check

1. **Text is pre-2022, not current.** This was deliberate. But the cutoff is on the revision, not the article:
   - an article created in 2012 carries its October 2022 lead;
   - an article created after the cutoff is dropped. This is why 2023–2025 disasters are missing from danger.
2. **Labels are Wikidata's, which is volunteer data.** Spot checks look clean, but with residual noise:
   - **food:** an enzyme was pulled in through the food tree.
   - **toxic:** includes dyes and pesticides flagged as carcinogens (trypan blue, acetochlor).
   - **weapon:** includes ancient weapon types (quarterstaff) and ammunition calibres ("8 mm caliber").
   - **disease:** includes mild infections (pityriasis rosea) and "infestation".

   "Dangerous to people" is our rule, not Wikidata's.
3. **Label cues in the text are strong.**
   - **Wording:** "is a species of venomous snake", "is a dish", "earthquake" and years in titles all give the answer away. That fits "does this label fit this text", but it makes the tasks easier than real-world input.
   - **Negatives:** kind-derived negatives are encyclopaedic in style, like the positives, so style alone does not separate them.
4. **Plain has a length shortcut.**
   - **The gap:** English leads average 84 words (44% are cut at the cap) and Simple English leads average 57.
   - **The baseline:** a word-count threshold alone scores **0.65** on test.
   - **Report it next to the model's result.** If needed, length-match each pair: trim the English side to its Simple partner's length. The script does not do this.
   - **Many texts are about places:** the Simple English sample is heavy on French communes and US towns, because it is a random sample of Simple English articles.
5. **Kind uses templated texts.** Villages, species and football clubs have formulaic leads ("X is a village in Y"). These are near-duplicates across splits but different subjects, and no QID or text is shared.
6. **Class balance inside kind is by subclass round-robin.**
   - **Events:** battles are 60 of about 1,000, not 5,853 of 12,000.
   - **General elections:** only 34, because most election items are instances of subclasses.
7. **Cleaning is heuristic.**
   - **Artifacts:** about 8 of 25,486 rows keep a small artifact, such as an empty "(also referred to as ...)" or "The.480 Ruger".
   - **Dropped templates:** dropped templates outside parentheses can leave a gap. For example, a coordinate after "located at" is handled, but other cases may not be.
8. **The same QID appears in several tasks with the same split.** For example, a 19th-century person is a kind example and a food negative. Different tasks never leak into each other's test sets, because the hash is shared.
9. **The Wikidata query service is not a dated snapshot.** Class trees and sitelinks are as of 2026-09-26, even though the text is from before 2022-11. Each file in `_cache/sparql/` records its query and fetch time.
