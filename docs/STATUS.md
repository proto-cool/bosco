# Build status (2026-09-13)

| Phase | Gate | Status | Report |
|---|---|---|---|
| 0 Data + annotations | coverage report | PASS | `phase0-coverage.md` |
| 1 Kernel | Shiu sugar → MN9 reproduces | PASS | `phase1-shiu-gate.md` |
| 2 Prune + port | ~5% KCs, stable | PASS (2.5–5%, eLN fix) | `phase2-stability.md`, `phase2-wsyn-calibration.md` |
| 3 Plasticity | learn / forget | PASS; two timescales (STM hours, LTM spaced, a month) | `phase3-plasticity-gate.md` |
| 3b Learning → behaviour | spaced rewards change action | PASS (follow after 3 rewards; ignore after 3 punishments; persists 10 days) | `scripts/phase3_mbon_out_gain.py`, README §12 |
| 4 Encoder + readout | unit tests | done (22 tests) | `encoder.md`, `config/readout_populations.yaml` |
| 4b Generator | tests | done; minimal fly-world corpus (~1600 words, tagged) | `src/bosco/textgen.py`, `corpus/`, `docs/corpus-audit.md` |
| 4d Timing + caps | tests | dust drive (no timer), per-kind caps with thread/account guards, episode budget | `config/caps_v1.yaml`, `config/encoder_v1.yaml` |
| 4e Remote management | tests | sleep/wake/delete/ignore/unfollow/reload/restart/status/people/memory/forget by mention | `src/bosco/control.py` |
| 4f Topics as smells | tests | keyword map → glomeruli mixtures; logged as names | `config/topics_v1.yaml` |
| 4g What-to-say learning | tests | per-document preferences from outcomes, decaying | `Agent.voice_update`, `state/voice.json` |
| 4h Habituation | tests | STD on sensory afferents; same smell fades, recovers in minutes | `config/model_v1.yaml` |
| 4i Outcomes | | likes/reposts/follows from anyone, kind replies, known-account returns, blocks | `src/bosco/bsky.py` |
| 4c Moderation | tests | labels → bitter, no approach, no reward; ignore → unfollow | `config/moderation_v1.yaml`, `src/bosco/moderation.py` |
| 5 Bluesky loop | dry-run, snapshots, nightly | code done, **not yet run against a live account** | `src/bosco/bsky.py`, `ops/` |
| 6 Calibration + tag | thresholds from dev activity | provisional synthetic thresholds written; re-run on real dev activity before tag | `scripts/calibrate_thresholds.py` |
| 7 dunce + monthly note | | tools ready | `scripts/make_dunce.py`, `scripts/replay_controls.py`, `scripts/monthly_report.py` |

## Parked

- A Bosco panel page at `bosco.proto.cool` with live stats from the ledger
  (episodes, actions, people, integrity), in ARC UI v4.2.2. After he flies.

## Deploy

`docs/DEPLOY.md`. His first post is the introduction in `config/identity_v1.yaml`.

## What only Nick can do

1. **Corpus and phrasebook.** `corpus/` holds a minimal fly-world corpus in
   the curious register; extend it under the rules in `docs/corpus-audit.md`.
   `phrasebook.yaml` has zero lines.  Decided 2026-09-13: closed vocabulary
   for the dev period; revisit word-learning from liked accounts before the
   tag, with `bosco people` data in hand.
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

- Thresholds are provisional (synthetic battery); ~46% of synthetic episodes
  cross some threshold, so the rate caps, not the thresholds, bound activity.
- The song-DN population (`reply`) is almost silent in the model (99th pct
  1.1 Hz); replies mostly come from `engage` on already-followed accounts.
- Bitter taste does not reach the escape DNs, so an insult makes Bosco
  approach (follow) rather than leave; aversion arrives only via learning.
- Accounts generalise: two accounts share ~4 of 12 glomeruli, so learning
  about one leaks a little onto the other. This is the fly's olfactory
  code, not a bug.
- The valence key (reward-vs-punishment MBON balance) is crude and untested
  against behaviour; it only selects phrasebook lines.
- Some odors leave a small persistent population after input off; episodes
  start from rest, so it never carries over.
- Being blocked is detected via `getRelationships` for accounts Bosco acted
  toward in the last 7 days, not for the whole network.
