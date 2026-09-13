# Deploying Bosco

One box, one container, one account. In this order.

## 1. The account (you)

1. Create `bosco.proto.cool` (custom handle on your domain: DNS TXT `_atproto.bosco` → `did=...`).
2. Bio, in his register, saying he is an automated fruit fly brain, in development,
   and that questions go to `@proto.cool`. Avatar of your choosing.
3. Settings → App passwords → create one for the box.
4. Moderation: subscribe the account to the labelers you want him to respect
   (the default Bluesky moderation service is enough to start). Labels from
   these are what `config/moderation_v1.yaml` reads.
5. On `@proto.cool`: create a list named `bosco-ignore` (can be empty).

## 2. The box

```
git clone <repo> ~/bosco && cd ~/bosco
ops/fetch_data.sh                      # ~1.1 GB
cp ops/env.example .env && $EDITOR .env  # handle, app password, operator
podman build -t bosco -f ops/Containerfile .
```

First run builds `data/cache/mcns_cb_v1.npz` (15 s). Then:

```
podman run --rm --env-file .env -v ./data/raw:/app/data/raw:ro,Z -v ./data/cache:/app/data/cache:Z \
  -v ./state:/app/state:Z localhost/bosco run --dry-run --once
```

You should see his introduction printed (not posted), the notifications and
feeds read, and a poll line. Run it a few more times with `--dry-run` while
you mention him from your account; `bosco memory --did <your did>` shows
what he made of it.

## 3. Live

```
mkdir -p ~/.config/containers/systemd && cp ops/bosco.container ~/.config/containers/systemd/
systemctl --user daemon-reload && systemctl --user start bosco && journalctl --user -fu bosco
```

His first post is the introduction (`config/identity_v1.yaml` `intro`).
Pin it from the app. From then on he reads, learns, and acts on his own.

Timers: copy `ops/bosco-nightly.{service,timer}` and a 30-minute timer for
`ops/bosco-deadman.sh` into the same systemd directory and enable them.

## 4. Watching

- `@bosco.proto.cool status` from your account, or `bosco status` on the box.
- `bosco people`, `bosco memory --did`, `bosco replay`.
- `snapshots/<date>/integrity.md` every night.
- `@bosco.proto.cool sleep` if anything looks wrong. He keeps perceiving; he stops acting.

## 5. Before `freeze-v1`

- ≥ 200 real event windows in the ledger, then `scripts/calibrate_thresholds.py --ledger state/ledger.sqlite --write`.
- Corpus and phrasebook final. Decide on word-learning (EXPERIMENT.md §2).
- Fill the artifact table in EXPERIMENT.md, tag, publish.
