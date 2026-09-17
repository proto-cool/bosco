# Nightly 2026-09-17

text-like values in ledger: none

# Integrity checks

- replay from snapshot 000313481532.npz to 322468532 ms: PASS (logged 7b2b58a291c1, got 7b2b58a291c1)
- KC q0.99 over the day 0.0924 at or under the recorded top 0.1355: PASS; busiest window 0.1319
- KC median in band: no kc_band in thresholds.json yet, run the calibration (SKIP)
- rate caps (config/caps_v1.yaml) over 292 real actions: PASS
- no post text in database: PASS
- learning-rule changes logged: ['1->2@1789537924']
- weight digest chain (5891 rows): PASS

**ALL: PASS**
