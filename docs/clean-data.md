# Clean training data (track B), fetched 2026-09-25

This covers track B of `docs/PLAN-V1.md` and decision 10 of `docs/DECISIONS-2026-09-25.md`. Each dataset was fetched from its original publisher, not from tasksource. The files sit under `data/raw/clean/<name>/` exactly as released. That directory is gitignored.

The machine-readable record is `data/raw/clean/MANIFEST.json`. It gives, per dataset:
- source URL and version (commit, revision or DOI);
- licence, with the evidence it was checked against;
- how the data was collected, and whether it holds personal data;
- splits, row counts and label names;
- the SHA-256 of every file.

The ledger this checks against is `docs/audit-2026-09-25/provenance.md`. Nothing is legal advice.

## Table

| name | fetched from | version | licence (verified at) | ledger | task | rows |
|---|---|---|---|---|---|---|
| dynahate | authors' GitHub `bvidgen/Dynamically-Generated-Hate-Speech-Dataset` | commit `36f9dc8`, v0.2.3 (and v0.2.2) | CC BY 4.0 (repo README; there is no LICENSE file) | KEEP | text → hate / nothate | train 32,924, dev 4,100, test 4,120 |
| hatemojibuild | author's GitHub `HannahKirk/Hatemoji` (same CSVs as the gated HF repo) | commit `a626f63` | CC BY 4.0 (repo LICENSE; HF card) | KEEP | text → hateful 0/1 | train 4,728, val 591, test 593 |
| civil_comments | HF `google/civil_comments` (HF-staff Parquet conversion, see note 3) | rev `f2970eb` | CC0 1.0 (TFDS catalog; HF card) | REVIEW | text → toxicity score in [0, 1], plus 6 subtypes | train 1,804,874, val 97,320, test 97,320 |
| dbpedia_14 | author's original `dbpedia_csv.tar.gz` (Google Drive, linked from `zhangxiangxiao/Crepe`) | version 2, 2015-09-09 | CC BY-SA and GFDL, dual (tarball readme.txt) | KEEP | title + abstract → 14 classes | train 560,000, test 70,000 |
| snips_built_in_intents | publisher's GitHub `sonos/nlu-benchmark` | commit `b86ac7f` | CC0 1.0 (repo LICENSE) | KEEP | text → 10 intents | 328, no official split |
| massive | Amazon S3 `amazon-massive-dataset-1.1.tar.gz` (en-US extracted) | 1.1 | CC BY 4.0 (tarball LICENSE and NOTICE; SLURP LICENSE.txt, also CC BY 4.0) | KEEP | text → 60 intents / 18 scenarios | en-US: train 11,514, dev 2,033, test 2,974 |
| clinc150 | publisher's GitHub `clinc/oos-eval` (`data_full.json`) | commit `828f809` | CC BY 3.0 (repo LICENSE) | KEEP | text → 150 intents + out-of-scope | train 15,000 + 100 oos; val 3,000 + 100; test 4,500 + 1,000 |
| boolq | HF `google/boolq` (HF-staff Parquet conversion, see note 3) | rev `35b264d` | CC BY-SA 3.0 (GitHub `google-research-datasets/boolean-questions` README) | KEEP | (passage, question) → yes / no | train 9,427, val 3,270 |
| multinli | authors' release `cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip` | 1.0 | per genre (paper.pdf in the zip; OANC page); see below | KEEP | (premise, hypothesis) → entailment / neutral / contradiction | train 392,702, dev matched 10,000, dev mismatched 10,000 |
| flywire_annotations | Nature Supplementary Data 1 (`41586_2024_7686_MOESM5_ESM.tsv`) | as published with the paper | CC BY 4.0 (article licence, PMC11446831) | REVIEW | — | 139,255 |

**Labels.**
- **DynaHate** has `label` (hate, nothate), `type` (Animosity, Derogation, Dehumanization, Threatening, Support, none, notgiven) and `target`.
- **Civil Comments** columns: toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit. Each is the fraction of raters who marked it. 8% of train rows have toxicity ≥ 0.5.
- **DBpedia** classes, 1 to 14: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.
- **SNIPS** intents: ShareCurrentLocation, ComparePlaces, GetPlaceDetails, SearchPlace, BookRestaurant, RequestRide, GetDirections, ShareETA, GetTrafficInformation, GetWeather.

## MultiNLI by genre

Source: paper.pdf §2, shipped in the release zip, and https://anc.org/data/oanc/.

| genre | split | source | licence | use? |
|---|---|---|---|---|
| government | train, dev matched | OANC: public-domain US government sites | OANC (unrestricted, commercial use allowed) | yes |
| slate | train, dev matched | OANC: Slate magazine, 1996–2000 | OANC | yes (rests on OANC's grant from Slate) |
| telephone | train, dev matched | OANC: Switchboard transcripts, 1990–91 | OANC | yes (recruited speakers; real conversations) |
| travel | train, dev matched | OANC: Berlitz guides | OANC | yes (rests on OANC's grant) |
| **fiction** | train, dev matched | 10 works, 1912–2010 | mixed: Seven Swords is CC BY-SA 3.0; Living History and Password Incorrect are CC BY 3.0; the rest are **public domain in the US only** | **no**, outside the US (see flags) |
| letters | dev mismatched only | OANC: fundraising letters | OANC | yes |
| oup | dev mismatched only | OANC: five OUP non-fiction works | OANC | yes (rests on OANC's grant) |
| verbatim | dev mismatched only | OANC: Verbatim posts, 1990–96 | OANC | yes |
| facetoface | dev mismatched only | OANC: Charlotte conversation collection | OANC | yes (recruited speakers) |
| nineeleven | dev mismatched only | OANC: the 9/11 Commission report | OANC (public domain) | yes |

Train without fiction has 315,354 rows. The hypotheses were written by 387 anonymous workers on Hybrid. Base pay is not stated; the ledger records a $1 validation bonus.

## Flags: what should not be used as fetched

1. **MultiNLI fiction genre: exclude.**
   - The paper itself says the non-CC works are public domain "in the United States (but may be licensed differently elsewhere)".
   - Christie (d. 1976), Asimov (d. 1992), Del Rey (d. 1993), Norton (d. 2005) and Piper (d. 1964) are still in copyright in the EU and UK. Decision 7 serves from a Kimsufi (OVH, France), and decision 13 allows self-hosting.
   - Seven Swords also carries share-alike.
   - The ledger's row ("CC BY-SA 3.0 and CC BY 3.0 fiction") misses this.
   - Use the other four training genres; all five mismatched genres are fine for evaluation.
2. **FlyWire annotations: the ledger's Zenodo route does not exist.**
   - Zenodo 10.5281/zenodo.10877326 (CC BY 4.0) holds only NBLAST scores and skeletons. Its description sends readers to the GitHub repo for annotations.
   - The CC BY 4.0 copy is the Nature supplement (Supplementary Data 1), which was fetched instead.
   - **It is not a drop-in replacement** for `data/raw/flywire/Supplemental_file1_neuron_annotations.tsv`, the GitHub copy the code uses:
     - it has 7 more rows;
     - `cell_type` differs on 47,612 shared neurons and `super_class` on 639;
     - it lacks five GitHub columns;
     - the gustatory map in `populations.grn_modality_map()` resolves 4 types instead of 38.
   - Switching `paths.FLYWIRE_ANNOTATIONS` would change that verification table. `ops/fetch_data.sh` was **not** changed.
   - Options: use the published supplement and accept the older typing; or keep the GitHub file and ask FlyWire for a licence (the ledger already says to email flywire@princeton.edu).
3. **Civil Comments and BoolQ are not the publisher's own files.**
   - `google/civil_comments` and `google/boolq` live under the Google org on Hugging Face. Their Parquet files were converted by HF staff in January 2024 from the original releases.
   - The originals could not be fetched: `gs://jigsaw-unintended-bias-in-toxicity-classification/civil_comments_v1.2.zip` and `gs://boolq/*.jsonl` return HTTP 403; Kaggle needs a login and acceptance of the competition rules.
   - The BoolQ conversion drops the `title` field.
   - If byte-level provenance to the publisher matters, fetch the Kaggle zip and the BoolQ jsonl with a Google or Kaggle login and compare.
4. **Civil Comments remains REVIEW.**
   - CC0 is verified at Google's TFDS catalog; the paper says only "a Creative Commons license".
   - The text is real people's comments, including hate and threats.
   - The two open questions from the ledger are still unanswered: what the Civil Comments terms let commenters agree to, and the rater platform and pay.
   - It was downloaded because the ledger says REVIEW. **Do not train on it until those are cleared.**
5. **DynaHate: rounds 3 and 4 are adapted from real posts.**
   - Annotators "searched for real-world hateful online content to inspire their entries", each "subject to at least one substantial adjustment" (paper §5.3–5.4).
   - The ledger names round 3 only; round 4 is the same.
   - Use v0.2.3 (deduplicated) for training.
6. **MASSIVE carries `worker_id`.** Drop the column before use.
7. **DBpedia is dual-licensed, CC BY-SA and GFDL.** The readme does not state the CC BY-SA version; the HF card and DBpedia 2014 say 3.0. Any redistributed derivative must be share-alike, and so must BoolQ's.
8. **SNIPS is 328 queries of unknown authorship.** The JSON also contains the commercial NLU services' predictions. Use only `text` and the intent key.

## Licence status

Every licence was verified at the source, and each matches the ledger. There are four exceptions:
- the ledger's **FlyWire Zenodo** claim is wrong (flag 2);
- the ledger's **MultiNLI fiction** reading is incomplete (flag 1);
- **DynaHate** has no LICENSE file, only a README statement;
- the **Civil Comments** CC0 rests on Google's TFDS catalog and HF card, not a Jigsaw-hosted page that could be read.

Nothing is marked UNVERIFIED.

## Added 2026-09-26 (for specialist gate 2; from `docs/free-datasets.md`)

| dataset | source | licence | fetched files (SHA-256) | notes |
|---|---|---|---|---|
| SMS Spam Collection v.1 | UCI archive.ics.uci.edu/static/public/228 | CC BY 4.0 (UCI) | `SMSSpamCollection` 7d039a24…c239d; `readme` 8753cd2d…a84e | 5,574 messages (747 spam). **Phone numbers and name-like strings are scrubbed on read, before embedding.** Cite Almeida & Gómez Hidalgo |
| DynaSent round 2 | github.com/cgpotts/dynasent (v1.1 zip, sha 33001cf3…edfd; zip deleted after extraction) | CC BY 4.0 (README) | r2 train da7fbad5…cc8, dev f9cb7eaa…ca, test 8924060e…32 | **Round 1 (Yelp text) and the bundled SST file were deleted** (non-commercial / no licence). **16,911 of about 18,500 r2 train rows have `has_prompt=true`**: a worker wrote the sentence after seeing a Yelp sentence, and `prompt_data` *contains that Yelp sentence*. prompt_data is never read. Whether prompted rows count as clean is **Nick's call** (see gate 2) |
| Stack Exchange Politeness corpus | ConvoKit (zissou.infosci.cornell.edu …/stack-exchange-politeness-corpus.zip, sha d5be7586…81a3) | CC BY 4.0 (ConvoKit); SE text CC BY-SA | `utterances.jsonl` 7b21d502…2fa4 | 6,603 requests with a normalised politeness score; usernames in metadata are not read |
