# A3 phase 1 results

Rules: `docs/A3-PHASE1.md`. Accuracy of a logistic model; held-out (large) / bench (50 items).

| task | Jev | single raw | single antenna | pair raw | pair antenna |
|---|---|---|---|---|---|
| route | 0.76 | 0.820 / 0.80 | 0.865 / 0.82 | — | — |
| yesno | 0.94 | 0.683 / 0.64 | 0.656 / 0.68 | 0.692 / 0.70 | 0.652 / 0.58 |
| rate | 0.62 | 0.671 / 0.64 | 0.666 / 0.60 | — | — |
| ambig | 0.58 | 0.657 / 0.48 | 0.634 / 0.52 | 0.677 / 0.48 | 0.661 / 0.52 |

## By the rules

- route: best through the antenna 0.865 vs Jev 0.76 → **in A3**
- yesno: best through the antenna 0.656 vs Jev 0.94 → **out: the nose cannot carry it**
  - two pieces vs one vector through the antenna: -0.003 → **no fix of ours recovers it**
- rate: best through the antenna 0.666 vs Jev 0.62 → **in A3**
- ambig: best through the antenna 0.661 vs Jev 0.58 → **in A3**
  - two pieces vs one vector through the antenna: +0.027 → **no fix of ours recovers it**
- rate, stars off on average (bench, antenna): 0.42 (Jev 0.42)
- ambig, distance from the human label split (single antenna): 0.073 (Jev 0.149; uniform guess 0.127)
- ambig, distance from the human label split (pair antenna): 0.072 (Jev 0.149; uniform guess 0.127)

## Reading (after the numbers; the rules are unchanged)

- **Route, rate and ambig go into A3.** Yes/no does not: the nose tops out
  at 0.66 against Jev's 0.94. Reading a passage to answer a question does
  not survive one vector, and keeping the two pieces apart does not help
  (−0.003).
- **Two sniffs does not earn a gate.** Pair beats single by −0.003 (yes/no)
  and +0.027 (ambig), both under the 0.05 rule. The relation between two
  texts is mostly gone once each is a vector. This limits A4's
  "question first, then the thing" design, which has to be settled in A4's
  pre-registration.
- **The comparison with Jev is not like for like.** These ceilings come from
  models *trained* on each task's own data; Jev answered cold. The rule
  compared the large held-out sets with Jev's 50 bench items. On those 50,
  the antenna ceilings are route 0.82, rate 0.60 (0.42 stars off, Jev 0.42)
  and ambig 0.52 (Jev 0.58). Ambig's large held-out set is ordinary MultiNLI,
  easier than ChaosNLI's items, which were chosen because humans disagree.
- On ChaosNLI, a trained model's probabilities sit closer to the human split
  (0.073) than Jev's (0.149). Also trained, not cold.
