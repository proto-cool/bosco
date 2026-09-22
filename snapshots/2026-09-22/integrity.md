# Nightly 2026-09-22

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000675054532.npz to 698909532 ms: PASS (logged 70dca52b170d, got 70dca52b170d)
- KC q0.99 over the day 0.1228 at or under the recorded top 0.1414: PASS; busiest window 0.2352
- KC median over the last 24 h (1862 episodes): 0.0306 in [0.0107, 0.0671]: PASS
- rate caps (config/caps_v1.yaml) over 1627 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924', '2->3@1789758710']
- weight digest chain (14922 rows): PASS

**ALL: PASS**
