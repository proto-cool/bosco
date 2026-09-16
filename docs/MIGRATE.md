# Moving him to another box

Written 2026-09-16 for the move from the Vultr VPS (root, rootful podman)
to a Kimsufi KS-5-A in Vint Hill (AlmaLinux 10, everything under the user
`nick`, rootless podman). Follow top to bottom. The rule underneath every
step: **nothing is reset**. His state moves as files, the gap is a logged
downtime, and the record must say exactly where the old machine stops and
the new one starts.

On the new box he runs as `nick`: the repo is `/home/nick/bosco`, the
containers are rootless, the units are user units (`systemctl --user`), and
root never logs in. The quadlets already use `%h` paths for this.

## 0. What can go wrong, and the one thing to check first

His numbers depend on the CPU in one way: numpy chooses its code path by
CPU level (x86-64-v2, v3, v4, the AVX-512 tiers) and the levels round float
reductions differently. Two boxes at the same level replay each other bit
for bit; two at different levels do not (README, *numerics*). From
`db60d73` the container pins numpy to x86-64-v3 on every box, the kernel
is built for x86-64-v3, and the agent records the level he runs at; a
change writes a `numerics` control row and a snapshot, so the record marks
the boundary by itself.

Before touching the new box, on the old one (still root there):

```
cd /root/bosco && git checkout b3ff9bd && podman build -t bosco -f ops/Containerfile .
cp ops/bosco.container /etc/containers/systemd/   # the numpy pin is in the unit, not the image
sed -i 's|%h/bosco|/root/bosco|g' /etc/containers/systemd/bosco.container
systemctl daemon-reload && systemctl restart bosco
journalctl -u bosco -n 30 --no-pager
sqlite3 state/ledger.sqlite "SELECT ts, kind, target_uri FROM control WHERE kind='numerics'"
podman exec bosco /app/.venv/bin/python /app/scripts/cpu_determinism.py
```

The `numerics` row says `?->x86_v3` (the pin took; the first level is
recorded) or nothing if the cursor was already set. Keep the script's last
line (`digest ...`). If the old box's CPU has AVX-512, its numbers before
this restart were computed on the v4 path and the restart itself is the
boundary; that is what the row records. Either way there is at most one
boundary, and it is in the record, like the `plasticity` row of 2026-09-15.

## 1. Order the box

Kimsufi KS-5-A, Vint Hill. OS **AlmaLinux 10** (podman 5, quadlets; the
whole distribution is built for x86-64-v3, the level his numerics are
pinned to; Alma 9, Rocky 9 or Ubuntu 24.04 also work; not Debian 12, whose
podman has no quadlets). Default soft
RAID 1 over the two NVMe disks. Add your ssh key. Note the IP. Lower the
TTL on the `bosco.proto.cool` A record to 300 now, so the panel's DNS
switch in §5 is quick.

## 1a. Harden the new box (AlmaLinux 10, before anything of his lands on it)

Start as root (or with `sudo` in front of each line). Keep the terminal
you are in open until the SSH check in step 3 passes from a second one.

1. Update, and reboot if the kernel changed:
   ```
   dnf -y update && dnf -y install epel-release && dnf -y install podman git rsync sqlite tar gcc make firewalld fail2ban fail2ban-firewalld dnf-automatic chrony policycoreutils-python-utils
   needs-restarting -r || reboot
   ```
2. The user. Everything of his lives under `nick`; root never logs in again.
   ```
   useradd -m -s /bin/bash nick && passwd nick        # the password is for sudo, not for ssh
   usermod -aG wheel nick
   mkdir -p /home/nick/.ssh && cp /root/.ssh/authorized_keys /home/nick/.ssh/ && chown -R nick:nick /home/nick/.ssh && chmod 700 /home/nick/.ssh && chmod 600 /home/nick/.ssh/authorized_keys
   loginctl enable-linger nick                          # his user units run without a login and start at boot
   ```
3. SSH: keys only, no passwords, no root.
   ```
   cat > /etc/ssh/sshd_config.d/50-bosco.conf <<'EOT'
   PasswordAuthentication no
   KbdInteractiveAuthentication no
   PermitRootLogin no
   PubkeyAuthentication yes
   AllowUsers nick
   MaxAuthTries 3
   X11Forwarding no
   AllowAgentForwarding no
   AllowTcpForwarding no
   ClientAliveInterval 300
   ClientAliveCountMax 2
   EOT
   sshd -t && systemctl reload sshd
   ```
   From a second terminal: `ssh nick@<new>` must work by key and `sudo -v`
   must take nick's password; `ssh root@<new>` and
   `ssh -o PubkeyAuthentication=no nick@<new>` must be refused. Only then
   close the first. Set a strong root password too (`passwd`) and keep it in
   your password manager: SSH will not take it, but the provider's rescue
   console will.
4. Firewall: ssh, the panel's 80 and 443 (443/udp is HTTP/3), nothing else;
   and let a rootless container bind 80 and 443.
   ```
   systemctl enable --now firewalld
   firewall-cmd --set-default-zone=public
   firewall-cmd --permanent --add-service=ssh --add-service=http --add-service=https
   firewall-cmd --permanent --add-port=443/udp
   firewall-cmd --permanent --remove-service=cockpit --remove-service=dhcpv6-client
   firewall-cmd --reload && firewall-cmd --list-all
   echo 'net.ipv4.ip_unprivileged_port_start=80' > /etc/sysctl.d/80-bosco-panel.conf && sysctl --system | grep unprivileged_port
   ```
5. fail2ban on sshd:
   ```
   cat > /etc/fail2ban/jail.local <<'EOT'
   [DEFAULT]
   banaction = firewallcmd-rich-rules
   bantime = 1h
   findtime = 10m
   maxretry = 4
   [sshd]
   enabled = true
   EOT
   systemctl enable --now fail2ban && fail2ban-client status sshd
   ```
6. Security updates apply themselves (a kernel update still waits for you to
   reboot; his restart is a logged downtime, so pick the moment):
   ```
   sed -i 's/^apply_updates = .*/apply_updates = yes/; s/^upgrade_type = .*/upgrade_type = security/' /etc/dnf/automatic.conf
   systemctl enable --now dnf-automatic.timer
   ```
7. Clock in UTC (his timers are UTC; his own day is `config` and does not
   depend on the box), SELinux enforcing (the quadlets label their volumes),
   persistent journal, hostname:
   ```
   timedatectl set-timezone UTC && systemctl enable --now chronyd && timedatectl
   getenforce            # Enforcing; leave it
   mkdir -p /var/log/journal && systemctl restart systemd-journald
   hostnamectl set-hostname bosco
   ```
8. Nothing else listening: `ss -tlnup` should show sshd and chronyd, and
   later Caddy's 80 and 443. Disable anything else (`systemctl disable --now
   cockpit.socket` if present). Leave the provider's monitoring agent alone
   if the image shipped one. Log out of root; from here on everything is
   `nick`, and `sudo` only where a line says so.

## 2. Prepare the new box, as nick (he keeps running on the old one)

```
cd ~ && git clone https://github.com/proto-cool/bosco.git && cd ~/bosco && git checkout b3ff9bd
mkdir -p data/raw data/cache state
rsync -a root@<old>:/root/bosco/data/raw/ data/raw/        # 1.1 GB; faster than fetching it again
rsync -a root@<old>:/root/bosco/data/cache/ data/cache/    # rebuilding it on first start costs minutes he would spend behind the clock
rsync -a root@<old>:/root/bosco/.env .env && chmod 600 .env
```

Every copy from the old box is a pull by nick, so nick needs a key the old
box accepts: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N '' -C nick@bosco`
on the new box, append `~/.ssh/id_ed25519.pub` to `/root/.ssh/authorized_keys`
on the old one, and `ssh root@<old> hostname` from nick must answer.

Edit `.env`: `BOSCO_RSYNC_TARGET` must be somewhere that is not the old
box (or empty for now). `uv` on the host, for the nightly integrity report:

```
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
cd ~/bosco && uv sync && make -C kernel
```

Build both images and install the user units, but **do not start `bosco`
yet**, and skip DEPLOY.md Part C: he is live already and the dry run would
post his introduction again.

```
podman build -t bosco -f ops/Containerfile .
podman build -t bosco-panel -f ops/Containerfile.panel .
mkdir -p ~/.config/containers/systemd ~/.config/systemd/user
cp ops/bosco.container ops/bosco-panel.container ~/.config/containers/systemd/
cp ops/bosco-nightly.service ops/bosco-nightly.timer ~/.config/systemd/user/
cat > ~/.config/systemd/user/bosco-deadman.service <<'EOT'
[Unit]
Description=bosco dead-man check
[Service]
Type=oneshot
ExecStart=%h/bosco/ops/bosco-deadman.sh
EOT
cat > ~/.config/systemd/user/bosco-deadman.timer <<'EOT'
[Unit]
Description=bosco dead-man check every 30 min
[Timer]
OnCalendar=*:0/30
Persistent=true
[Install]
WantedBy=timers.target
EOT
systemctl --user daemon-reload
systemctl --user list-unit-files | grep bosco     # bosco.service, bosco-panel.service, the two timers
```

Check the new box computes like the old one, on a scratch brain:

```
podman run --rm -v ~/bosco/data/raw:/app/data/raw:ro,Z -v ~/bosco/data/cache:/app/data/cache:Z \
  --entrypoint /app/.venv/bin/python bosco /app/scripts/cpu_determinism.py
```

The `level` must be `x86_v3` and the `digest` line must equal the old
box's from §0. If it does not, stop and look: something other than the
pinned path differs (glibc in the image, numpy version), and he must not
move until it is understood.

The nightly job commits `snapshots/<date>` and pushes. As nick that needs
a key with write access to the repo: `ssh-keygen -t ed25519 -f
~/.ssh/bosco-deploy -N ''`, add the public key as a deploy key with write
access on GitHub, and in `~/bosco`: `git remote set-url origin
git@github.com:proto-cool/bosco.git` plus `git config core.sshCommand "ssh
-i ~/.ssh/bosco-deploy"`. Also `git config user.name Nick` and
`user.email` as on the old box. Without this the push part silently skips
(`|| true`) and the snapshots only pile up locally.

## 3. Cut over

On the old box, as root. This is the only part with a gap; it is a few
minutes and it is logged as `downtime` by his first poll on the new box.

```
systemctl stop bosco-nightly.timer bosco-deadman.timer
systemctl stop bosco          # SIGTERM: he saves his state and stops (journal: the last save)
ls -l state/brain_state.npz   # mtime after the stop
sqlite3 state/ledger.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"
```

Then on the new box, as nick, pull his state:

```
rsync -a --delete root@<old>:/root/bosco/state/ ~/bosco/state/
```

`state/` holds the ledger, his weights and snapshots, the associations,
the panel files; the rsync must be complete before he starts anywhere. Do
not start him again on the old box after this: two of him with one account
would write two records.

Still on the new box (`ls -ln state` must show nick's uid on every file;
rootless podman maps the container's root onto nick):

```
systemctl --user start bosco
journalctl --user -fu bosco
```

The journal shows the load, `downtime skipped: N s not lived`, and the
first poll. Then the gate:

```
podman exec bosco /app/.venv/bin/bosco replay
sqlite3 ~/bosco/state/ledger.sqlite "SELECT ts, kind, target_uri FROM control ORDER BY id DESC LIMIT 5"
```

`replay ... bit-identical True` means the new box reproduced the last span
of his life on the old box exactly, from the last snapshot plus the log.
There should be no new `numerics` row (both boxes at `x86_v3`). If there is
one, the boxes differed and §0 was skipped; the row is the boundary and the
next nightly integrity report will flag the one span that straddles it.

## 4. Timers and the dead-man

```
systemctl --user enable --now bosco bosco-nightly.timer bosco-deadman.timer
systemctl --user list-timers | grep bosco
```

`enable` on `bosco` makes him start at boot (linger from §1a). Test the
dead-man once by hand: `~/bosco/ops/bosco-deadman.sh` prints nothing and
exits 0 while he is alive.

## 5. The panel

```
systemctl --user enable --now bosco-panel
ss -tlnp | grep -E ':80 |:443 '     # caddy, owned by nick
```

Point the `bosco.proto.cool` A record at the new IP. Caddy issues a new
certificate within a minute of the DNS change; the `_atproto.bosco` TXT
record is his handle and does not change (it names his DID, not a
machine). Open https://bosco.proto.cool: the live word in the top line
comes from the activity file the new box writes.

## 6. Leave the old box up for a week

Do not delete it. `systemctl disable --now bosco bosco-panel` there, and
keep the provider snapshot. After seven clean nightly integrity reports on
the new box (`~/bosco/snapshots/<date>/integrity.md`), destroy it.

## 7. What replaces the provider snapshot

A dedicated server has no console snapshot. The nightly already rsyncs
`state/` off-box; add a weekly dated tarball so a corruption does not
overwrite the only copy:

```
cat > ~/.config/systemd/user/bosco-weekly.service <<'EOT'
[Unit]
Description=bosco weekly state tarball
[Service]
Type=oneshot
EnvironmentFile=%h/bosco/.env
ExecStart=/bin/sh -c 'tar -C %h/bosco -czf %h/bosco-state-$(date -u +%%F).tgz state && rsync -a %h/bosco-state-*.tgz "$BOSCO_RSYNC_TARGET"'
EOT
cat > ~/.config/systemd/user/bosco-weekly.timer <<'EOT'
[Unit]
Description=bosco weekly state tarball
[Timer]
OnCalendar=Sun 04:30 UTC
Persistent=true
[Install]
WantedBy=timers.target
EOT
systemctl --user daemon-reload && systemctl --user enable --now bosco-weekly.timer
```

## 8. Living with him as nick

- `journalctl --user -fu bosco` for the log; `podman exec bosco ...` for
  his commands (`bosco status`, `bosco replay`, `bosco memory --did`).
- After a code change: `cd ~/bosco && git checkout <sha> && podman build -t bosco -f ops/Containerfile . && systemctl --user restart bosco`,
  and the same with `Containerfile.panel` and `bosco-panel` for the site.
- The panel's peer bench line (idle 0.138 / event 1.91 wall seconds on the
  VPS) will change; note the new numbers in the monthly report.
- With four cores, `dunce` can run live beside him at the freeze and the
  three controls can replay in parallel; the kernel is single-threaded per
  fly and parallel across flies.
