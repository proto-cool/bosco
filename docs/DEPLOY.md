# Deploying Bosco, every step

Written to be followed top to bottom. Each block is something you type or
click. Nothing is assumed. You are `root` on the box and the repo lives at
`/root/bosco`; if you use another user, replace `/root` with that home.

---

## Part A. The Bluesky account

### A1. Create the account

1. Open https://bsky.app and sign out of `@proto.cool` if you are signed in (or use a private window).
2. Click **Create account**. Use an email you control. For the handle, pick anything for now
   (for example `bosco-temp.bsky.social`); you will change it to `bosco.proto.cool` in A2.
3. Finish sign-up. Stay signed in as the new account.

### A2. Give him the handle `bosco.proto.cool`

1. In the Bluesky app as Bosco: **Settings → Account → Handle → I have my own domain**.
2. Type `bosco.proto.cool`. The app shows a DNS record like:
   ```
   Type:  TXT
   Name:  _atproto.bosco
   Value: did=did:plc:xxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Go to wherever `proto.cool`'s DNS is managed. Add exactly that TXT record.
   Name is `_atproto.bosco` (some panels want `_atproto.bosco.proto.cool.`). Value is the full `did=...` string.
4. Wait a few minutes, then click **Verify DNS record** in the app. When it goes green, click **Update to bosco.proto.cool**.
5. Write down the DID (`did:plc:...`). You will see it in **Settings → Account** later too.

### A3. Profile

1. **Settings → Edit profile** (or the pencil on his profile page).
2. Display name: `bosco`.
3. Bio, all lowercase, something like:
   ```
   i fruit fly. one male fly brain, every neuron, running on a computer. this account is mine. in development. questions to @proto.cool. automated.
   ```
   The words **automated** and **in development** must be in there.
4. Avatar and banner: your call.

### A4. App password (this is how the box logs in as him)

1. As Bosco: **Settings → Privacy and security → App passwords → Add App Password**.
2. Name it `vps`.
3. **Leave "Allow access to your direct messages" unchecked.**
4. Click **Create**. Copy the password (`xxxx-xxxx-xxxx-xxxx`). It is shown once. Paste it somewhere safe for Part C.

### A5. Moderation he will respect

1. As Bosco: **Settings → Moderation**.
2. Make sure **Bluesky Moderation Service** is subscribed (it is by default).
3. Optionally subscribe to any other labeler you trust. Anything they label, he treats as bitter and never approaches.

### A6. The ignore list (on YOUR account)

1. Sign in as `@proto.cool`.
2. **Lists → New list** (in the app: profile → Lists → **+**). Type: **User list**. Name: exactly `bosco-ignore`. Description: anything.
3. Save. It can stay empty. Adding someone to it later makes Bosco unfollow them and never perceive them again.

---

## Part B. The box

### B1. Create it

1. A dedicated server (from 2026-09: Kimsufi KS-5-A, Xeon E-2274G 4c/8t,
   32 GB ECC, 2 × 960 GB NVMe in soft RAID, Vint Hill). Any x86-64-v3
   machine (AVX2 and FMA, every server since 2013) computes his numbers the
   same way; a VPS with 2 dedicated vCPUs and 4 GB is the floor.
2. OS: Rocky Linux 9 or AlmaLinux 9 (podman 4.9 with quadlets). Ubuntu
   24.04 also works. **Not Debian 12**: its podman 4.3 has no quadlets.
   Keep the installer's default soft RAID 1 across the two disks.
3. Hostname `bosco`. Add your ssh key. Note the IP.
4. Moving him from one box to another is `docs/MIGRATE.md`, not this part.

### B2. Log in and install the basics

```
ssh root@<the ip>
dnf install -y podman git curl sqlite        # Rocky / Fedora
# apt-get update && apt-get install -y podman git curl sqlite3   # Debian / Ubuntu
podman --version
```

### B3. Get the code

```
cd /root
git clone https://github.com/proto-cool/bosco.git
cd /root/bosco
```

### B4. Get the data (about 1.1 GB, a few minutes)

```
ops/fetch_data.sh
```

When it finishes, `ls data/raw` shows three `.feather` files, and `data/cache` and `state` exist.

### B5. Secrets

```
cp ops/env.example .env
nano .env
```

Set these lines (leave the rest as they are):

```
BOSCO_HANDLE=bosco.proto.cool
BOSCO_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx      # from A4
BOSCO_OPERATOR=proto.cool                   # your handle, verified by DID at runtime
BOSCO_IGNORE_LIST=bosco-ignore              # the list from A6
```

Save with Ctrl+O, Enter, then Ctrl+X.

### B6. Build the container image (2 to 4 minutes the first time)

```
podman build -t bosco -f ops/Containerfile .
```

Success ends with `COMMIT bosco` and an image id. If it fails, copy the last 20 lines and send them to me.

---

## Part C. Dry run (nothing is posted)

### C1. First run

```
cd /root/bosco
podman run --rm --env-file .env \
  -v ./data/raw:/app/data/raw:ro,Z \
  -v ./data/cache:/app/data/cache:Z \
  -v ./state:/app/state:Z \
  localhost/bosco run --dry-run --once
```

The first run builds his network cache (`data/cache/mcns_cb_v1.npz`, about 15 seconds). You should then see:

```
bosco DRY-RUN as did:plc:...; operator did:plc:...
intro 'i bosco. i fruit fly. ...' (dry)
2026-.. poll: 0 notifications, N browsed, 0 blocks, grooms 0, brain lag 0s
```

`intro ... (dry)` is his first post, printed instead of posted.

### C2. Mention him and look at what he made of it

1. From `@proto.cool`, post: `hello @bosco.proto.cool you handsome fly`.
2. Run the C1 command again. You should see a line for your post and what he did (`like`, `follow`, `nothing`...).
3. Ask him what he thinks of you (replace the DID with yours, from your own **Settings → Account**):
   ```
   podman run --rm --env-file .env -v ./data/raw:/app/data/raw:ro,Z -v ./data/cache:/app/data/cache:Z -v ./state:/app/state:Z \
     localhost/bosco memory --did did:plc:YOURDID
   ```

### C3. Wipe the dry-run state before going live (so the intro posts for real)

```
rm -rf /root/bosco/state && mkdir /root/bosco/state
```

---

## Part D. Live

### D1. Install the service (rootful quadlet)

```
mkdir -p /etc/containers/systemd
cp /root/bosco/ops/bosco.container /etc/containers/systemd/bosco.container
sed -i 's|%h/bosco|/root/bosco|g' /etc/containers/systemd/bosco.container
systemctl daemon-reload
systemctl start bosco
journalctl -fu bosco
```

Within a minute the log shows `bosco LIVE as ...` and then `intro '...' (at://...)`. **That is his first post, live.** Ctrl+C leaves the log; he keeps running.

### D2. Pin and announce

1. Open his profile in the app. On the introduction post: **⋯ → Pin to profile**.
2. From `@proto.cool`, quote-post the introduction with skeet 1, then reply to your own post with skeets 2 and 3.
   Avoid the command words (status, report, people, sleep, stop, wake, delete, forget, ignore, unfollow, reload, restart) in the quote; your mention of him is parsed for commands.

### D3. Nightly snapshot and dead-man timers

```
cp /root/bosco/ops/bosco-nightly.service /root/bosco/ops/bosco-nightly.timer /etc/systemd/system/
sed -i 's|%h/bosco|/root/bosco|g' /etc/systemd/system/bosco-nightly.service
cat > /etc/systemd/system/bosco-deadman.service <<'EOT'
[Unit]
Description=bosco dead-man check
[Service]
Type=oneshot
ExecStart=/root/bosco/ops/bosco-deadman.sh
EOT
cat > /etc/systemd/system/bosco-deadman.timer <<'EOT'
[Unit]
Description=bosco dead-man check every 30 min
[Timer]
OnCalendar=*:0/30
Persistent=true
[Install]
WantedBy=timers.target
EOT
systemctl daemon-reload
systemctl enable --now bosco-nightly.timer bosco-deadman.timer
systemctl list-timers | grep bosco
```

The nightly job needs `uv` on the host for the integrity report; install it once:
```
curl -LsSf https://astral.sh/uv/install.sh | sh && ln -sf /root/.local/bin/uv /usr/local/bin/uv
cd /root/bosco && uv sync && make -C kernel
```

---

## Part E. Living with him

- **From your account**, mention him with a word: `@bosco.proto.cool status`, `people`, `memory @someone`,
  `sleep` (stops acting, keeps perceiving), `wake`, `reload` (re-read corpus), `restart`.
  Reply `delete` under one of his posts to remove it. `ignore @someone`, `unfollow @someone`.
- **On the box**: `journalctl -fu bosco` for the log; `systemctl restart bosco` after a `git pull` and rebuild.
- **After changing code**: `cd /root/bosco && git pull && podman build -t bosco -f ops/Containerfile . && systemctl restart bosco`.
- **Nightly**: `snapshots/<date>/integrity.md` says whether every check passed.

## Part F. Before the tag (`freeze-v1`)

1. Wait for at least 200 real event windows (`bosco status` shows the count).
2. Recalibrate thresholds from real activity, never outcomes:
   ```
   cd /root/bosco && uv run python scripts/calibrate_thresholds.py --ledger state/ledger.sqlite --write
   ```
3. Corpus and phrasebook final. Decide on word-learning (EXPERIMENT.md §2).
4. Fill the artifact table in EXPERIMENT.md, `git tag freeze-v1`, push.

## Troubleshooting

- `stdint.h: No such file` during build → old image; `git pull` and rebuild.
- `lstat data/cache: no such file` → `mkdir -p data/cache state`.
- `could not read Username for github` on the box → the repo is public; use the https URL exactly as in B3.
- He posts nothing for hours → normal. `journalctl -fu bosco` shows `grooms 0`; a landing that his network answers is what makes a post.
- CPU at 50 to 100 percent of one core, always → normal; that is the simulation keeping up with wall time.

## E. The panel at bosco.proto.cool

The public page: his brain lit neuron by neuron, what his body wants to do,
today's counts, memory, people by smell, his posts. Static files built with
Node, served by Caddy with automatic HTTPS. The bosco process writes
`state/panel/activity.bin` (once a second) and `state/panel/status.json`
(once a poll); the site reads them under `/data/`.

### E1. DNS and ports

`bosco.proto.cool` must point at the VPS (it already does if you ssh by that
name). Open 80 and 443:

```
firewall-cmd --permanent --add-service=http --add-service=https
firewall-cmd --permanent --add-port=443/udp
firewall-cmd --reload
```

### E2. Build and start

```
cd /root/bosco && git pull
mkdir -p /root/bosco/state/panel
podman build -t bosco-panel -f ops/Containerfile.panel .
cp ops/bosco-panel.container /etc/containers/systemd/
systemctl daemon-reload
systemctl start bosco-panel
systemctl status bosco-panel --no-pager
```

The bosco container needs the new `Environment=BOSCO_PANEL_DIR` line too, so
after the pull:

```
cp ops/bosco.container /etc/containers/systemd/
systemctl daemon-reload
podman build -t bosco -f ops/Containerfile .
systemctl restart bosco
```

The journal shows `panel: writing /app/state/panel` at startup. Open
https://bosco.proto.cool. The first certificate takes a few seconds.

### E3. Updating the site

```
cd /root/bosco && git pull && podman build -t bosco-panel -f ops/Containerfile.panel . && systemctl restart bosco-panel
```

## F. Calibrating thresholds from his real activity

Thresholds follow `config/thresholds_policy.yaml`. Copy the live ledger off
the box (features and URIs only; no text) and run the script:

```
ops/fetch_ledger.sh                      # from your laptop; needs ssh to the box
uv run python scripts/calibrate_thresholds.py --ledger snapshots/dev-<date>/ledger.sqlite --allow-partial --write
```

`--allow-partial` keeps the current value for any population whose window
set is still too small (the policy's `min_rows`); the script prints which.
Commit `config/thresholds.json`, push, rebuild the bosco container, restart.
