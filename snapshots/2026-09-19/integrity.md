# Nightly 2026-09-19

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000410929532.npz to 439701532 ms: PASS (logged 2258e77fcc63, got 2258e77fcc63)
- KC q0.99 over the day 0.1233 at or under the recorded top 0.1364: PASS; busiest window 0.1905
- KC median over the last 24 h (2304 episodes): 0.0245 in [0.0082, 0.0511]: PASS
- rate caps (config/caps_v1.yaml) over 773 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924', '2->3@1789758710']
- weight digest chain (9240 rows): PASS

**ALL: PASS**
