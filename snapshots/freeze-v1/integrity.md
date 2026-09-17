# Integrity at freeze-v1

Run in his own image against snapshots/freeze-v1/ledger.sqlite and the live state dir,
2026-09-17. 5993 episodes, 2026-09-13 18:42:54 to 2026-09-17 19:29:53 UTC.

# Integrity checks

- replay from snapshot 000313481532.npz to 324508532 ms: PASS (logged 7b2b58a291c1, got 7b2b58a291c1)
- KC q0.99 over the day 0.0928 at or under the recorded top 0.1364: PASS; busiest window 0.1319
- KC median over the last 24 h (1751 episodes): 0.0192 in [0.0082, 0.0511]: PASS
- rate caps (config/caps_v1.yaml) over 318 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924']
- weight digest chain (5993 rows): PASS

**ALL: PASS**

