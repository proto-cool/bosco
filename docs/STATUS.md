# Build status (2026-09-17)

| Phase | Gate | Status | Report |
|---|---|---|---|
| 0 Data + annotations | coverage report | PASS | `phase0-coverage.md` |
| 1 Kernel | Shiu sugar → MN9 reproduces | PASS | `phase1-shiu-gate.md` |
| 2 Prune + port | ~5% KCs, stable | PASS (2.5–5%, eLN fix) | `phase2-stability.md`, `phase2-wsyn-calibration.md` |
| 3 Plasticity | learn / forget | PASS; two timescales (STM hours, LTM spaced, a month) | `phase3-plasticity-gate.md` |
| 3b Learning → behaviour | spaced rewards change action | PASS (follow after 3 rewards; ignore after 3 punishments; persists 10 days) | `scripts/phase3_mbon_out_gain.py`, README §12 |
| 4 Encoder + readout | unit tests | done | `encoder.md`, `config/readout_populations.yaml` |
| 4b Generator | tests | done; retrieval steers and the walk speaks since 2026-09-18 (`scripts/quotation_gate.py`), the phrasebook a last resort | `src/bosco/textgen.py`, `corpus/`, `docs/corpus-audit.md` |
| 4d Timing + caps | tests | dust drive (no timer), per-kind caps with thread/account guards, episode budget | `config/caps_v1.yaml`, `config/encoder_v1.yaml` |
| 4e Remote management | tests | sleep/wake/delete/ignore/unfollow/reload/restart/status/people/memory/forget/primer by mention | `src/bosco/control.py` |
| 4f Topics as smells | tests | keyword map → glomeruli mixtures; logged as names | `config/topics_v1.yaml` |
| 4h Habituation | tests | STD on sensory afferents; same smell fades, recovers in minutes | `config/model_v1.yaml` |
| 4i Outcomes | | likes/reposts/follows from anyone, kind replies, known-account returns, blocks | `src/bosco/bsky.py` |
| 4g What-to-say learning | tests | per-document preferences from outcomes, decaying | `Agent.voice_update`, `state/voice.json` |
| 4j Questions, memory told back, topic corpus | tests | `?` is a stronger touch and a reply; "what do you think of me"; `topic=` corpus files | `docs/CORPUS-PLAN.md` |
| 4c Moderation | tests | labels → bitter, no approach, no reward; ignore → unfollow | `config/moderation_v1.yaml`, `src/bosco/moderation.py` |
| 8 Senses (2026-09-15) | gates | taste while browsing, α'3 familiarity, retina, open nose, associations, walk | `phase8-taste.md`, `phase8-retina.md` |
| 5 Bluesky loop | dry-run, snapshots, nightly | **live since 2026-09-13**, on the dedicated box since 2026-09-16 | `src/bosco/bsky.py`, `ops/`, `docs/MIGRATE.md` |
| 6 Calibration + tag | thresholds from dev activity | engage/like/leave/reply/walk from the 09-14 dev ledger; **groom still synthetic**; not yet re-run on the full dev period | `scripts/calibrate_thresholds.py` |
| 7 dunce + monthly note | | offline tools ready; live-beside-him path added 2026-09-17 (`BOSCO_BRAIN`, `ops/bosco-dunce.container`); no account, no `dunce_v1.npz` built, no identity file | `scripts/make_dunce.py`, `scripts/replay_controls.py`, `scripts/monthly_report.py` |

## Before `freeze-v1`

1. ~~Re-calibrate thresholds on the full dev period.~~ Done 2026-09-17 off
   `snapshots/2026-09-17/`, every population from real activity and `min_hz` binding on none of
   them: engage 4.85, like 3.84 (over `tasted`), leave 4.00, reply 8.75, groom 12.48, walk 2.05,
   plus `kc_band`. `_calibration.kept` is empty; nothing synthetic survives. Never from outcomes.
2. ~~Phrasebook.~~ Done 2026-09-17 (version 3, 72 lines): `reply` covers all 27 keys twice over
   and `groom` all nine, so no groom borrows a line through the fallback chain any more. The
   seven new groom keys were drafted by Claude at Nick's direction, strictly in the fly's
   physical world -- his day picks the key, the line never says what happened to him.
3. ~~Pin the artifacts.~~ Done: `config/frozen_digests.json` holds corpus, phrasebook, topics and
   identity as the code reads them, and `tests/test_corpus.py` fails if any of the four moves.
   EXPERIMENT.md §3 carries sha256s for 22 artifacts, ten more than it used to ask for.
4. ~~Snapshot the freeze.~~ `snapshots/freeze-v1/` holds his ledger (5993 episodes, 2026-09-13
   18:42:54 to 2026-09-17 19:29:53 UTC) and `brain_state.npz`, with a clean integrity report
   beside it: replay, both sparseness tests, rate caps, no text, and the weight digest chain all
   PASS. EXPERIMENT.md is out of DRAFT. Still to do by hand: "in development" out of the bio.
5. ~~Confirm the nightly publishes.~~ It does, and ALL: PASS, since 2026-09-17 19:00 UTC.
   What went wrong, for the record. The repo had no snapshot commit between 2026-09-14 and
   2026-09-17. The timer was enabled and firing; `uv` was never installed on the new box, so the
   job exited 127 after copying the ledger and before the integrity report, and the failed unit
   went unread -- the dead-man watches episodes, and he was living fine. Fixed 2026-09-17: `uv`
   installed, `PATH` set in the unit (a user unit does not get the login shell's), and the script
   now names a missing tool in an alert instead of dying at 127. Behind it sat a second fault:
   the rate-cap check called legal activity a violation, which would have aborted the job on the
   next line; it now reads `config/caps_v1.yaml`, and a failing night commits its report anyway.

## Before dunce goes live (weeks 3–4 after the tag)

- Account, app password, bot self-label, `~/bosco/.env.dunce` (`ops/env.dunce.example`).
- `scripts/make_dunce.py` on the box → `data/cache/dunce_v1.npz`; `podman` quadlet
  `ops/bosco-dunce.container`, its own `state-dunce/`. `BOSCO_BRAIN` refuses to start if the
  matrix is missing rather than falling back to his wiring.
- `config/identity_v1_dunce.yaml`: hand-authored, Nick's. dunce must not answer that he is Bosco.
- Undecided: whether the panel shows one fly or two, and dunce's own nightly snapshot.

## Deploy

`docs/DEPLOY.md`, `docs/MIGRATE.md`. His first post is the introduction in
`config/identity_v1.yaml`; the pinned primer thread is `primer`.

## What only Nick can do

1. **Corpus and phrasebook.** Extend under the rules in `docs/corpus-audit.md`.
2. **Accounts.** `bosco.proto.cool` exists; dunce's does not. The `bosco-ignore` list lives on
   `@proto.cool`.
3. **Box.** Dedicated server since 2026-09-16 (`docs/MIGRATE.md`), rootless under `nick`.

## Try it now

```
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate poke --did did:plc:you --text "hi bosco" --mention
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate outcome --episode 1 --valence reward
uv run bosco --ledger /tmp/dev.sqlite --state-dir /tmp/devstate replay --episode 1
uv run python scripts/integrity_checks.py --ledger /tmp/dev.sqlite --state-dir /tmp/devstate
```

## Known limits to state at tag

- Every threshold now comes from his own activity: groom off 44 real landings (2026-09-17), and
  `like` off the windows in which something tasted sweet. `min_hz` binds on none of them. Over
  every window `like` was 75% exact zeros, so its quantile fell inside that silence and the floor
  had been deciding it; the policy already answered that shape for reply and groom by naming the
  windows in which the behaviour can happen, and `tasted` is that answer for like. The rate caps,
  not the thresholds, still bound how much he does.
- The song-DN population (`reply`) has no sensory input in the model; since 2026-09-14 being
  addressed drives pC1 in proportion to appetite (README §26), and engage maps to reply when he
  was addressed.
- Bitter taste does not reach the escape DNs, so an insult makes Bosco approach (follow) rather
  than leave; aversion arrives only through learning, now from outcomes and from the taste of
  what he reads (2026-09-15).
- Accounts generalise: two accounts share ~4 of 12 glomeruli, so learning about one leaks a
  little onto the other. This is the fly's olfactory code, not a bug.
- The valence key (reward-vs-punishment MBON balance) is crude and untested against behaviour;
  since 2026-09-16 it selects the corpus subset and the fallback phrasebook line, not the
  sentence, which is retrieval by smell.
- Some odors leave a small persistent population after input off; episodes start from rest, so
  it never carries over.
- Being blocked is detected via `getRelationships` for accounts Bosco acted toward in the last
  7 days, not for the whole network.
- Opening a ledger normally runs the schema migrations, which rewrites the file; tools that read
  a published dump open it read-only (2026-09-17), so an audit cannot alter the evidence.
