# Nightly 2026-09-23

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000675054532.npz to 742658532 ms: PASS (logged 70dca52b170d, got 70dca52b170d)
- KC q0.99 over the day 0.1218 at or under the recorded top 0.1414: PASS; busiest window 0.1609
- KC median over the last 24 h (1957 episodes): 0.0295 in [0.0107, 0.0671]: PASS
- rate caps (config/caps_v1.yaml) over 1678 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924', '2->3@1789758710']
- weight digest chain (15700 rows): PASS

**ALL: PASS**
