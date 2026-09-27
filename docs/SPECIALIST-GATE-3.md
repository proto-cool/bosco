# Specialist gate 3: the Wikipedia/Wikidata specialists (pre-registered 2026-09-26)

It is the same as `docs/SPECIALIST-GATE-2.md` (the v1 brain, the fixed family v1 antenna, the pinned nomic
encoder, the DN read, the preflight, CPU scoring once, the production bar), on the data in `docs/wiki-data.md`
(built by `scripts/v1_gate3_fetch.py` and `scripts/v1_gate3_data.py`).

## Data and leakage check
- **Text:** English Wikipedia leads (and Simple English for `plain`), the last revision at or before
  2022-11-01. Labels from Wikidata (CC0). No living people: persons need a birth before 1900 and a date of
  death, and there are no humans in food, danger or plain.
- **Splits** by Wikidata item (a subject never spans two splits; both sides of a plain pair are in the same
  split). 27 near-duplicate val/test items were dropped.
- **plain:** each complex lead is cut to its Simple partner's word count, so length gives nothing away (the
  mean after the cut is 57 vs 53 words).

| specialist | design | options | train / val / test | bar |
|---|---|---|---|---|
| **kind** | A | person, place, organisation, creative work, event, species, product or technology | 5,597 / 377 / 1,088 | 0.80 |
| **food** | A | food or drink / not food | 5,080 / 308 / 936 | 0.80 |
| **danger** | A | dangerous / not dangerous | 5,549 / 363 / 1,092 | 0.80 |
| **plain** | A | plain language / complex language | 3,984 / 250 / 862 | 0.80 |

**Ceilings on validation** (logistic on the encoder / through the family antenna): kind 0.986 / 0.978,
food 1.000 / 1.000, danger 0.994 / 0.978, plain 0.702 / 0.613. There are **no disputed-label exceptions:
all four have the 0.80 bar**, stated to Nick before any gate-3 test number existed.

**Disclosure:** the report-only baseline scores (plain logistic, nose alone, prototype alone) were computed
on the gate-3 test sets **before** this document was written, and seen. They are comparison models, not the
fly. The bars had already been fixed and stated (0.80 for all four, no exceptions), so nothing here was
changed by them.

## Honest limits that go on every card
- **Wikipedia leads often state the answer outright** ("is a species of venomous snake", "is a dish"). Test
  scores will be **optimistic** for real-world text.
- **danger's labels come from our own rule over Wikidata classes** (disasters, infectious diseases with an ICD
  code, carcinogen-flagged or toxic substances excluding medicines, venomous animals, weapons). Some members are
  debatable (industrial dyes, mild infections, calibres).
- **plain** is written style, which a meaning-based encoder barely carries (ceiling 0.61 through his nose). It
  is expected to fail, and is run because it is cheap and the result is informative.

## Rules (fixed now)
As gate 2: a specialist ships if its sealed test balanced accuracy ≥ 0.80 **and** ECE ≤ 0.10. The baselines are
report-only.

Runner: `scripts/v1_gate2.py --gate 3 …`. Results: `docs/specialist-gate-3-results.md`.
