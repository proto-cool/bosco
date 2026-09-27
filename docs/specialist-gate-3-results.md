# Specialist gate 3 results

Pre-registration: `docs/SPECIALIST-GATE-3.md`. Sealed test sets, balanced accuracy, scored once on the CPU; ECE after temperature.

| specialist | design | options | chance | **brain** | ECE | bar | ships | nose alone | prototype alone | plain baseline | CPU s/question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kind | A | 7 | 0.143 | **0.940** | 0.020 | 0.800 | **yes** | 0.743 | 0.959 | 0.991 | 0.37 |
| food | A | 2 | 0.500 | **0.991** | 0.006 | 0.800 | **yes** | 0.522 | 0.986 | 0.995 | 0.10 |
| danger | A | 2 | 0.500 | **0.974** | 0.009 | 0.800 | **yes** | 0.553 | 0.986 | 0.993 | 0.10 |
| plain | A | 2 | 0.500 | — | — | 0.800 | — | 0.497 | 0.633 | 0.759 | — |

**Ships: kind, food, danger.**


## Reading (after the numbers; the rules are as pre-registered)

- **Three of four ship: kind (0.940), food (0.991) and danger (0.974).** All are well above the 0.80 bar and
  well calibrated (ECE 0.006–0.020). They run on the same family v1 brain and fixed antenna as gate 2.
- **plain did not run.** It failed its preflight, as the pre-registration expected: written style barely
  survives a meaning-based encoder (a 0.61 ceiling through his nose). It needs a style sense of its own, not
  more training.
- **These scores are optimistic for real-world text.** Wikipedia leads often state the answer outright ("is a
  dish", "is a venomous snake"). The report-only prototype matcher alone reaches 0.959–0.986, so the smell
  carries most of the answer here; the fly's part is steering it to the right descending neurons. Every card
  says this, and that danger's labels come from our own rule over Wikidata classes.
- **CPU time:** 0.10 s per yes/no question, 0.37 s for kind's 7 options.
