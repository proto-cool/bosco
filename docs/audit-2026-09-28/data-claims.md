# Audit 2026-09-28: data provenance and public claims

Read-only adversarial audit (branch `v2`, HEAD `cc31052`). Nothing was trained, committed or changed except this file.
Sealed gate-4/4b tests were not read. Only the text fields of `data/cache/*/items.json` were scanned for PII
patterns; no labels were printed. Licences were re-checked at their original sources on 2026-09-28 (User-Agent
`bosco-research/0.1 (https://bosco.systems)`). None of this is legal advice.

**State at audit time.** Commit `cc31052` (Nick, 06:50 today) deleted the registry cards, the results docs and all
trained weights. **No specialist is shipped now.** The eight cards audited here (topic, junk, support, hate,
politeness, kind, food, danger; `git show f1b35fc:registry/...`) are the last shipped set, and every rebuilt
specialist will inherit whatever they got wrong. Gate 4 and 4b data (built, halted) is audited as "trained".
`docs/SERVICE.md` ("8 as of 2026-09-27") and `docs/REGISTRY.md` ("Shipped (2026-09-27)") still describe the deleted
set; see C9.

Severity: **S1** means a public claim is false, or a licence or ethics rule is broken in a shipped or trained
artefact. **S2** means a material gap that must close before launch or self-hosting. **S3** is hygiene.

---

## Part 1. Claims (pitch lines and public docs)

### C1 (S1). "a real fruit fly's brain, all 50,000 mapped neurons" / "his full brain"
- **Facts.**
  - MaleCNS v1.0 has **165,122 traced bodies** (`docs/phase0-coverage.md`).
  - Family v1 simulates **50,140** of them (`service/families/v1/family.json` as exported): the central brain,
    ascending and descending neurons, plus about 9,201 visual projection neurons.
  - The optic lobes and the ventral nerve cord are out, and decision 39 keeps them out.
  - Connections are **cut at 5 synapses** (every KC→MBON connection is kept).
  - It is a **rate model**, not spiking (decision 5).
- So it is about 30% of the mapped neurons, not all of them, and not his full brain. Decision 39 itself says
  "'His brain' means the 50,140-neuron central brain ... and the cards say so." No card said so.
- **Honest wording:** "a computer model of a real male fruit fly's central brain: 50,140 of the ~165,000 neurons
  mapped in the MaleCNS connectome, wired as mapped (connections of 5+ synapses)."
- **Also stale:** `docs/ASK-BOSCO-PRIORITIES.md` §2 and `docs/SERVICE-R10.md` l.26 still say the model "grows to
  about 144,000 when the optic lobes go in". Decision 39 withdrew that. `docs/HANDOFF-HALTERES.md` says "about
  200 ms of fly time"; the model runs 80 × 5 ms = 400 ms.

### C2 (S1). "the fly decides" / "a real fly brain does the deciding" / "A real fly connectome ... makes every decision"
Sources: `docs/BRIEF.md`, `docs/ASK-BOSCO-PRIORITIES.md` §5, `docs/HANDOFF-HALTERES.md`.

Evidence from the deleted results docs (`git show f1b35fc:docs/...`):
- **A plain logistic probe on the encoder's 768 numbers matches or beats the fly on every shipped specialist.**

  | specialist | fly | plain probe |
  |---|---|---|
  | topic | 0.884 | 0.983 |
  | kind | 0.940 | 0.991 |
  | support | 0.829 | 0.884 |
  | hate | 0.694 | 0.711 |
  | politeness | 0.600 | 0.661 |
  | junk | 0.906 | 0.908 |
  | food | 0.991 | 0.995 |
  | danger | 0.974 | 0.993 |

  In gate 3 the prototype matcher alone reaches 0.959–0.986. The gate-3 reading says "the smell carries most of
  the answer ... the fly's part is steering it to the right descending neurons."
- **The real wiring does not beat a same-size layered control** (specialist pilot: topic 0.80 vs 0.82, hate 0.63
  vs 0.59). Its own reading: "the fly's *kind* of brain carries these tasks, not its exact map."
- **The fly's learning sites and per-type parameters are trained by gradient descent on task labels.**
- **The final answer is computed by the simulated circuit,** so "decides" is literally true of the last step. As
  a statement of where the competence comes from, it misleads.
- **Honest wording:** "Every answer is read from his simulated descending neurons: the moonwalker backing away,
  the feeding neurons leaning in. An encoder translates the text into a smell, and his learning synapses were
  trained by gradient descent. A plain classifier on the same encoder is as accurate, and a same-size scrambled
  brain learns about as well. What is his is how the decision forms, and you can watch it."

### C3 (S1). The controls footnote is promised but on no card
- Decision 21, `ASK-BOSCO-PRIORITIES.md` §3 ("the controls as a footnote: 'a scrambled copy of his wiring learns
  this about as well'") and the gate-3 reading ("Every card says this") all promise it.
- None of the 8 cards carried the control result or the plain-probe gap. Each card's only accuracy context was
  `report_only` numbers with no explanation.
- The gate-3 cards also did not say that the smell alone carries most of the answer.
- `scripts/export_family.py` has no field for either, so rebuilt cards will repeat the gap unless it is added.

### C4 (S1). "learns only from human-written, openly licensed data"
This is false at three levels.
1. **The encoder.** The encoder in every answer (nomic-embed-text-v1.5) was trained by Nomic on about 235M scraped
   pairs:
   - pretraining (arXiv 2402.01613 Table 7): Reddit 28%, PAQ, Amazon reviews, Yahoo Answers via Kaggle, Quora,
     CNN, and more;
   - fine-tuning: **MS MARCO**, which is "non-commercial research purposes only".

   The weights are Apache 2.0, but the data is not "openly licensed". This is known internally (`encoders.py`
   docstring; `audit-2026-09-25/provenance.md` §5 rates it REVIEW, "still not clean"). It appears on no card: each
   card's encoder block carries only `"licence": "apache-2.0"`. Decision 14 ("auditable first") needs that
   disclosure on the card.
2. **The specialists' data** is not all openly licensed or human-written:
   - **SMS Spam (junk):** the upstream parts carry no licence (P1);
   - **NWS text (gate-4 danger pool):** machine-formatted from forecast grids, not written by forecasters (P6);
   - **CFPB narratives (finance held-out):** have no licence at all, only a public-domain statement "for FOIA
     purposes", on text written by consumers (P9).
3. **The Aegis/HH-RLHF held-out source** sits in LLM-generated conversations, and its cards say "intended for
   research purposes". Only the human turns are used, but that is a qualification the pitch line hides (P7).

**Honest wording:** "His specialists learn from openly licensed datasets with recorded provenance, written by people
before ChatGPT (with stated exceptions), and never from LLM output. His translator, Nomic's open encoder, was
trained by Nomic on published web data that is not all openly licensed; we are building a clean replacement."

### C5 (S2). "you watch the actual cells fire"
- It is a rate model, so nothing spikes.
- Each dot in the flat views averages about 16 neurons.
- Activity is shown relative to the model's own rest, with a display afterglow (×0.9 per step).
- The wave view reorders neurons by response time.
- `BRAIN-VIEWS.md` has the right label ("a computer model ... Green means firing above his resting rate; it is not
  measured calcium"). The pitch line drops it, and "calcium imaging palette" invites the misreading.
- **Honest wording:** "you watch his real cell types (named from the connectome) light up in the simulation as the
  decision forms."

### C6 (S2). "no LLM anywhere"
- **True in the generative sense:** nothing generates text, and no LLM decides.
- **But:**
  - the encoder is a 137M-parameter transformer;
  - the harm held-out sets come from LLM chat logs (P7);
  - "the only ML outside the brain" (HANDOFF) ignores the learned 46-component antenna whitening, the temperature
    calibration, and the gradient-trained brain parameters.
- **Keep the line,** add "no generative model, and no LLM-written training data", and drop "the only ML".

### C7 (S2). Stale architecture in public-facing docs
- **`docs/BRIEF.md`** still says:
  - "CLIP (his nose)" and "52 glomeruli";
  - the answer is read from the MBONs;
  - "Not arithmetic on his synapse weights";
  - "trained ... never taught through the API".
- Now the encoder is nomic, the antenna is 46 channels, the read is from the DNs, and training is by gradient
  descent.
- **`docs/HANDOFF-HALTERES.md`** says "The encoders understand, the fly decides" and gives 200 ms.
- **Fix:** mark BRIEF as superseded, or rewrite it.

### C8 (S2). The card licence strings point at nothing
- Junk, support and hate carry "Bosco specialist licence (the self-host bundle's LICENSE)"
  (`scripts/export_family.py` l.39). **No such LICENSE exists** anywhere in the repo.
- There is no NOTICE file and no bundle builder (see L1).

### C9 (S3). The docs describe a deleted fleet
- `SERVICE.md`, `REGISTRY.md` and `ASK-BOSCO-PRIORITIES.md` §3 describe 8 shipped specialists. The registry and the
  weights were deleted in `cc31052`.

---

## Part 2. Licences (re-verified at source)

| # | dataset | where used | licence at source | verdict |
|---|---|---|---|---|
| P1 | **SMS Spam v.1** | junk (shipped) | UCI: CC BY 4.0. The bundled readme: © Almeida & Gómez Hidalgo, "collected from **free or free for research** sources". Upstream: Grumbletext forum posts (**no licence**); Tagg 2009 thesis (eTheses page, **no licence**); SMS Spam Corpus v0.1 Big (site unreachable); NUS SMS Corpus (ScholarBank: CC BY 4.0) | **S1.** The 2026-09-25 ledger said "**DROP** as is (REVIEW only if scrubbed and the consent trail is accepted)". `free-datasets.md` changed it to "USE, with a note" and **no decision records Nick accepting it**. UCI's CC BY is an aggregator relicensing third-party texts. Real private messages; scrub incomplete (P-II1) |
| P2 | DynaHate v0.2.3 | hate (shipped), harm pool | README: "licenced under CC-BY 4.0". **No LICENSE file** (GitHub API `license: None`) | S3: acceptable, but record the README commit as the licence evidence |
| P3 | MASSIVE 1.1 | support (shipped), intent/topic/language pools | NOTICE: CC BY 4.0. SLURP text CC BY 4.0 (its **audio is CC BY-NC**; not used) | OK |
| P4 | DBpedia-14 | topic (shipped) | readme: CC BY-SA (no version) + GFDL | OK as BY-SA 4.0 via the later-version clause. The card says "3.0"; the readme gives no version |
| P5 | SE Politeness (ConvoKit) | politeness (shipped), tone held-out | ConvoKit: CC BY 4.0; SE text from about 2013: CC BY-SA 2.5/3.0 | OK as BY-SA 4.0. `@username` mentions stay in the training text (P-II3) |
| P6 | **NWS products** | gate-4 danger pool | public domain | **S2.** Licence fine. But the docs call it "forecaster-written government text" to justify the **only** exception to the 2022-11 cutoff. Zone Forecast Products are generated by NWS formatter software from forecaster-edited grids; warnings are template-built. Not LLM output, but not human writing either. Decision 33 allows relaxing the cutoff only for "supervised human writing under a no-AI-tools rule, written down before use". NWS was not taken to Nick as such. Low impact: NWS separates perfectly, and its val set is 6 items |
| P7 | **Aegis 1.0/2.0 + HH-RLHF** | harm held-out (the sealed test for 4 harm experts) | Aegis: CC BY 4.0, but both cards say "intended for research purposes ... not intended for training dialogue agents". HH-RLHF: MIT, "intended for research purposes" | **S2.** The code keeps only prompt rows that exactly match an HH-RLHF human turn (`v1_gate4_harm_tone.py` l.270–335). The text is human-written. But in multi-turn HH, human turns reply to 52B-model output, and **Aegis 2.0 labels are partly from an "LLM jury"** (Mixtral-8x22B, Mistral-NeMo, Gemma-2; `response_label_source`). The code does not filter on label source. Prompt labels may be human; this is unverified per row. Test-only use, so the effect is on the evaluation's integrity, not the weights. Filter to human-labelled rows, or state it |
| P8 | Stack Exchange (Money SE, SE sites) | finance/topic/danger pools | CC BY-SA 2.5/3.0/4.0 by post date. The 2024-04 archive.org dump predates the gated download (since July 2024 users tick "not for LLM training") | OK under decision 31, provided the 2024-04 archive.org copy is the one used (it is: `stackexchange_20240402`). The card note is still owed |
| P9 | CFPB narratives | finance held-out only | no licence. The site: "freely available for anyone to use"; CFPB treats published narratives as "public domain for FOIA purposes". The text is written by consumers (17 USC 105 does not strictly cover it); opt-in; scrubbed by CFPB | S3 (test-only). Do not call it "public domain" or "openly licensed" on a card; say "published by the CFPB with consumer consent, used for testing only" |
| P10 | SGD | intent/finance/topic pools | CC BY-SA 4.0. Utterances are **crowd paraphrases of simulator template outlines** (arXiv 1909.05855) | OK (human-written, 2019). Makes those specialists BY-SA (noted in the doc) |
| P11 | CLINC150 | intent/finance pools | **CC BY 3.0** Unported | OK. Cards must say 3.0 |
| P12 | BANKING77, NLU++ | intent/finance pools | CC BY 4.0 (repo LICENSE) | OK |
| P13 | ABCD | intent pool | MIT | OK (crowd role-play) |
| P14 | MultiDoGO | intent held-out, finance pool | CDLA-Permissive-1.0 | OK |
| P15 | Wikinews | topic/danger held-out | CC BY 2.5 (to 2024-12-15), CC BY 4.0 after | OK |
| P16 | SIB-200 / FLORES-200 | language held-out | CC BY-SA 4.0 (HF card); repo code Apache 2.0; FLORES asks not to train on it | OK (test-only) |
| P17 | Tatoeba | language pool | CC BY 2.0 FR (some CC0). Audio has other terms, including NC; not used | OK |
| P18 | arXiv metadata | topic pool | CC0 1.0 for metadata, abstracts included (info.arxiv.org/help/api/tou.html) | OK |
| P19 | openFDA | danger pool | CC0, **except GMDN terms**, which need a licence "for commercial applications or AI training" | OK: the builder excludes GMDN fields (manifest note). Keep it that way |
| P20 | CPSC, NHTSA | danger pool | data.gov: US public-domain label. The CPSC API page states no terms | OK / S3 (record the data.gov evidence, as done) |
| P21 | Civil Comments | harm/tone pools | CC0 (TFDS; no Jigsaw page) | Accepted (decision 30); both ethics gaps go on every card |
| P22 | Wikipedia Detox | harm pool, 4b held-out | labels CC0 (figshare); talk-page text CC BY-SA, which Wikimedia's CC0 cannot waive | OK as BY-SA (noted in the harm-tone doc) |
| P23 | **ConvAbuse** | 4b development (dev evaluation) | CC BY 4.0 (repo LICENSE) | **S2 process.** Its own FETCH.json says "CAUTION ... needs Nick's decision before any gate use". It was fetched and scored as a dev platform in `harassment_dev.py` with no recorded decision. `gate4-data-harm-tone.md` §2 says "not fetched and not used", which is now false |
| P24 | Wikipedia / Wikidata | kind/food/danger (shipped), gate 4 | CC BY-SA 4.0 / CC0 | OK. Per-row revision URLs recorded |

### Encoders and connectome
- **E1 (S1, disclosure).** nomic-embed-text-v1.5 at `e9b6763`: Apache 2.0. Its training data is as in C4:
  scraped, and includes MS MARCO (non-commercial). Nothing on the cards or the pitch discloses it (decision 14).
  **nomic-embed-vision-v1.5 was CC BY-NC 4.0 until 2025-01-16.** The pinned `e3a725b` (2025-03-31) carries
  `apache-2.0`, so keep that pin and never an older one. Its training set, DFN-2B, carries Apple's `apple-amlr`
  licence tag (text not fetched; believed research-only) over CommonPool crawl. That is REVIEW before the eyes
  ship.
- **E2 (S2).** No NOTICE or attribution files exist for self-hosting (decision 13).
  - Nothing in the repo holds the MaleCNS attribution, the licence link, or the "list of changes" that CC BY 4.0
    §3(a)(1)(B) requires (neuron subset, 5-synapse cut, sign from NT prediction, rate model, trained per-type
    parameters and KC→MBON weights).
  - There is no Shiu MIT notice. `kernel/lif.h` and `kernel.py` say "Shiu et al. 2024 defaults" but carry no
    copyright line.
  - The only attribution is a string in `family.json`.
  - The MaleCNS download page says "licensed under CC-BY" and names Janelia FlyEM, Cambridge/MRC LMB and Google.
    Credit them, and do not imply endorsement.
- **E3 (S2).** FlyWire annotations. `ops/fetch_data.sh` still fetches the GitHub copy.
  - That repo has **no licence** (GitHub API `None`), and FlyWire's ToS puts community annotations under
    **CC BY-NC 4.0**.
  - They are used only to verify the hard-coded GRN modality map (`populations.py`) and are not in the served
    family.
  - Decision 38 (taste on the gustatory neurons) will lean on that map. Before then, switch to the CC BY 4.0
    Nature supplement or get FlyWire's answer, and cite the four papers their README lists.

---

## Part 2 (cont.). sha256, the cutoff, PII and LLM rows

### Checksums
- **Recorded (full sha256):**
  - the 10 datasets in `docs/clean-data-manifest.json`;
  - every source in the three gate-4 manifests;
  - the `FETCH.json` files.
- **S3 gaps:**
  - **SMS Spam** (shipped) has only truncated hashes in `clean-data.md` and no machine-readable record or
    FETCH.json. On disk, `SMSSpamCollection` is `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`,
    which matches the truncated hash.
  - **SE Politeness zip:** `d5be7586c3f4224cffdb45cf87af17b5ad17c0ed4d2d5f6560bcb201703681a3` (matches).
  - **DynaSent** has only truncated hashes.

### The 2022-11-01 cutoff (grep of the builders)
- **Enforced in code:**
  - Wikipedia (`rvstart=2022-11-01T00:00:00Z`; gate 3 l.389, gate 4 l.542);
  - Wikinews (l.989);
  - multilingual Wikipedia (l.1062);
  - SE and Money SE (`CreationDate < CUTOFF`, intent-finance l.227);
  - CFPB (`dt <= CUTOFF`, l.269);
  - Tatoeba (added and modified dates, l.1542);
  - FDA, CPSC and NHTSA by record date.
- **Holes:**
  - **NWS** is exempt (P6).
  - **Tatoeba keeps undated sentences** (`\N`, empty or zero dates count as OK). These are probably the oldest
    sentences, but that is unproven. S3: bound them by sentence id.
  - **Wikidata class labels** were queried live on 2026-09-27. They are labels, not text, so this is acceptable;
    note it.

### PII (text fields scanned by regex; examples in the audit transcript)
- **P-II1 (S1). Junk (shipped).**
  - The only scrub is phone numbers (`v1_gate2_data.py` l.43–47).
  - `clean-data.md` claims "**Phone numbers and name-like strings are scrubbed**". **No name scrub exists.**
  - Emails survive in training text, including a personal one (`...@hotmail.com` in a ham message).
  - First names survive in private messages ("Spoke with uncle john today ...").
  - Fix the doc claim, or add the scrub and retrain.
- **P-II2 (S3). Hate.** DynaHate text has no scrub. It holds `@` handles of public figures and organisations only
  (for example @BorisJohnson). Acceptable; say so.
- **P-II3 (S2). Stack Exchange @usernames in the text.**
  - Decision 31 says "Usernames go only into attribution manifests".
  - In fact `@Name` mentions stay inside the text of the politeness training data (about 300 items), `stackexchange`
    (34) and `money_se` (8).
  - The intent-finance scrub (`v1_gate4_intent_finance.py` l.1094) handles only emails and digits: **no URLs**
    (113 Money SE train items carry URLs, some to personal image hosts) and **no @mentions**.
  - `gate4-data-topic-language-danger.md` says "emails/URLs/phones scrubbed" for SE. That is true in that builder,
    not in intent-finance.
- **P-II4 (S3). DBpedia-14 (topic)** includes biographies of living people from 2014 (Artist, Athlete,
  OfficeHolder), unscreened. The gate-3 living-people rule does not reach it.
- **Clean:**
  - the gate-4 topic-language-danger and harm-tone texts: 0 emails, and Wikipedia user signatures replaced with
    `<user>`;
  - the long digit runs in FDA are UPCs, not phone numbers;
  - the CFPB masks (`XXXX`) are intact.

### LLM-generated rows
- **None found in any training pool.**
- **SGD** is crowd-paraphrased templates (2019).
- **MultiDoGO** is Wizard-of-Oz, human on both sides.
- **NLU++** was written by dialogue experts.
- **HH-RLHF assistant turns** are excluded in code.
- **Aegis** responses are excluded in code (`response.isna()`, user_message only). The residual is the LLM-jury
  labels (P7), and they are test-only.
- **DynaHate/Hatemoji** are "synthetic" in the sense of human adversarial writing (£16/h), not LLM output.

---

## Part 3. Fix list, in order
1. **Replace the pitch lines** (C1, C2, C4, C5) with the honest wordings above, everywhere Ask Bosco copy is drafted.
   Update `ASK-BOSCO-PRIORITIES.md` §2/§5, `HANDOFF-HALTERES.md` and `BRIEF.md`.
2. **Card schema** (`export_family.py`): add
   - the encoder's training-data disclosure;
   - the plain-probe and controls footnote;
   - "trained by gradient descent";
   - "50,140-neuron central brain";
   - single-source status (decision 29).
3. **Junk:** record Nick's acceptance of the SMS Spam upstream risk, or drop it. Scrub emails and names, and correct
   `clean-data.md`.
4. **Write LICENSE and NOTICE files** before any self-host bundle:
   - MaleCNS credit, licence link and list of changes;
   - the Shiu MIT notice;
   - per-source credits;
   - the BY-SA manifests. Intent-finance and harm-tone have **no attribution manifests yet**. Only
     topic-language-danger has them.
5. **ConvAbuse:** take it to Nick, and fix `gate4-data-harm-tone.md` §2.
6. **NWS:** take it to Nick as a decision-33 exception with the formatter fact, or drop it from the danger pool.
7. **Aegis:** filter the held-out set to human-labelled rows, or state that labels are partly LLM-jury.
8. **The intent-finance scrub:** add URLs and @mentions. Strip @mentions from SE text across all builders.
9. **FlyWire:** switch to the Nature supplement before taste (decision 38) uses the GRN map.
10. **Records:** write machine-readable sha256 entries for SMS Spam, SE Politeness and DynaSent, and bound undated
    Tatoeba rows.
