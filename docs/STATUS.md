# Build status (2026-09-13)

| Phase | Gate | Status | Report |
|---|---|---|---|
| 0 Data + annotations | coverage report | PASS | `phase0-coverage.md` |
| 1 Kernel | Shiu sugar → MN9 reproduces | PASS | `phase1-shiu-gate.md` |
| 2 Prune + port | ~5% KCs, stable | PASS (2.5–5%, eLN fix) | `phase2-stability.md`, `phase2-wsyn-calibration.md` |
| 3 Plasticity | learn / forget | PASS | `phase3-plasticity-gate.md` |
| 4 Encoder + readout | unit tests | done (17 tests) | `encoder.md`, `config/readout_populations.yaml` |
| 5 Bluesky loop | dry-run, snapshots, nightly | code done, **not yet run against a live account** | `src/bosco/bsky.py`, `ops/` |
| 6 Calibration + tag | thresholds from dev activity | tool ready; needs ≥200 dev episodes | `scripts/calibrate_thresholds.py` |
| 7 dunce + monthly note | | tools ready | `scripts/make_dunce.py`, `scripts/replay_controls.py`, `scripts/monthly_report.py` |

## What only Nick can do

1. **Phrasebook lines.** `phrasebook.yaml` has structure and zero lines.
   Until it has lines, `reply` and `spontaneous_post` decisions are logged
   with `note = no_line` and nothing is posted.  `like` and `leave` work.
2. **Account.** Create `bosco.proto.cool`, an app password, the bot
   self-label, "in development" in the bio, and a list named `bosco-ignore`
   on `@proto.cool`.  Fill `ops/env.example` → `.env`.
3. **Box.** Vultr VPS, `podman build`, quadlet, timers (`ops/README.md`).
   Feathers go in `data/raw/` (1.1 GB; `scripts/` has the URLs in `paths.py`).

## Try it now

```
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate poke --did did:plc:you --text "hi bosco" --mention
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate outcome --episode 1 --valence reward
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate replay --episode 1
uv run python scripts/integrity_checks.py --ledger /tmp/dev.sqlite --state-dir /tmp/devstate
```

## Known limits to state at tag

- Thresholds are null → every decision is `nothing` until calibration.
- The valence key (reward-vs-punishment MBON balance) is crude and untested
  against behaviour; it only selects phrasebook lines.
- Some odors leave a small persistent population after input off; episodes
  start from rest, so it never carries over.
- Being blocked is detected via `getRelationships` for accounts Bosco acted
  toward in the last 7 days, not for the whole network.
