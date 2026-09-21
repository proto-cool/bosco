# Nightly 2026-09-21

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000410929532.npz to 612522532 ms: PASS (logged 2258e77fcc63, got 2258e77fcc63)
- KC q0.99 over the day 0.1306 at or under the recorded top 0.1364: PASS; busiest window 0.1661
- KC median over the last 24 h (1885 episodes): 0.0283 in [0.0082, 0.0511]: PASS
- rate caps (config/caps_v1.yaml) over 1343 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924', '2->3@1789758710']
- weight digest chain (12991 rows): PASS

**ALL: PASS**
