# Nightly 2026-09-20

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000410929532.npz to 526097532 ms: PASS (logged 2258e77fcc63, got 2258e77fcc63)
- KC q0.99 over the day 0.1366 at or under the recorded top 0.1364: FAIL; busiest window 0.1671
- KC median over the last 24 h (1676 episodes): 0.0340 in [0.0082, 0.0511]: PASS
- rate caps (config/caps_v1.yaml) over 1045 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924', '2->3@1789758710']
- weight digest chain (11002 rows): PASS

**ALL: FAIL**
