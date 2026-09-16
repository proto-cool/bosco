# Moving him to another box

Written 2026-09-16 for the move from the Vultr VPS to a Kimsufi KS-5-A in
Vint Hill. Follow top to bottom. The rule underneath every step: **nothing
is reset**. His state moves as files, the gap is a logged downtime, and the
record must say exactly where the old machine stops and the new one starts.

## 0. What can go wrong, and the one thing to check first

His numbers depend on the CPU in one way: numpy chooses its code path by
CPU level (x86-64-v2, v3, v4, the AVX-512 tiers) and the levels round float
reductions differently. Two boxes at the same level replay each other bit
for bit; two at different levels do not (README, *numerics*). From
`db60d73` the container pins numpy to x86-64-v3 on every box, the kernel
is built for x86-64-v3, and the agent records the level he runs at; a
change writes a `numerics` control row and a snapshot, so the record marks
the boundary by itself.

Before touching the new box, on the old one:

```
cd /root/bosco && git checkout db60d73 && podman build -t bosco -f ops/Containerfile .
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

Kimsufi KS-5-A, Vint Hill. OS **Rocky Linux 9** (or AlmaLinux 9; Ubuntu
24.04 also works; not Debian 12, whose podman has no quadlets). Default
soft RAID 1 over the two NVMe disks. Add your ssh key. Note the IP. Lower
the TTL on the `bosco.proto.cool` A record to 300 now, so the panel's DNS
switch in §5 is quick.

## 1a. Harden the new box (AlmaLinux 9, before anything of his lands on it)

As root, or with `sudo` in front of each line. Keep the terminal you are in
open until step 2 has been checked from a second one.

1. Update, and reboot if the kernel changed:
   ```
   dnf -y update && dnf -y install epel-release && dnf -y install podman git rsync sqlite tar firewalld fail2ban fail2ban-firewalld dnf-automatic chrony policycoreutils-python-utils
   needs-restarting -r || reboot
   ```
2. SSH: keys only, no passwords, root only by key. Put your key in
   `/root/.ssh/authorized_keys` first (`ssh-copy-id root@<new>` from your machine).
   ```
   cat > /etc/ssh/sshd_config.d/50-bosco.conf <<'EOT'
   PasswordAuthentication no
   KbdInteractiveAuthentication no
   PermitRootLogin prohibit-password
   PubkeyAuthentication yes
   MaxAuthTries 3
   X11Forwarding no
   AllowAgentForwarding no
   AllowTcpForwarding no
   ClientAliveInterval 300
   ClientAliveCountMax 2
   EOT
   sshd -t && systemctl reload sshd
   ```
   From a second terminal: `ssh root@<new>` must work by key, and
   `ssh -o PubkeyAuthentication=no root@<new>` must be refused. Only then
   close the first. Set a strong root password anyway (`passwd`) and keep it
   in your password manager: SSH will not take it, but the provider's rescue
   console will.
3. Firewall: ssh, the panel's 80 and 443 (443/udp is HTTP/3), nothing else.
   ```
   systemctl enable --now firewalld
   firewall-cmd --permanent --set-default-zone=public
   firewall-cmd --permanent --add-service=ssh --add-service=http --add-service=https
   firewall-cmd --permanent --add-port=443/udp
   firewall-cmd --permanent --remove-service=cockpit --remove-service=dhcpv6-client
   firewall-cmd --reload && firewall-cmd --list-all
   ```
   Podman publishes the panel's ports through firewalld itself (netavark);
   nothing more to open.
4. fail2ban on sshd:
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
5. Security updates apply themselves (a kernel update still waits for you to
   reboot; his restart is a logged downtime, so pick the moment):
   ```
   sed -i 's/^apply_updates = .*/apply_updates = yes/; s/^upgrade_type = .*/upgrade_type = security/' /etc/dnf/automatic.conf
   systemctl enable --now dnf-automatic.timer
   ```
6. Clock in UTC (his timers are UTC; his own day is `config` and does not
   depend on the box), SELinux enforcing (the quadlets label their volumes),
   persistent journal:
   ```
   timedatectl set-timezone UTC && systemctl enable --now chronyd && timedatectl
   getenforce            # Enforcing; leave it
   mkdir -p /var/log/journal && systemctl restart systemd-journald
   hostnamectl set-hostname bosco
   ```
7. Nothing else listening: `ss -tlnup` should show sshd, chronyd's udp, and
   later Caddy's 80 and 443. Disable anything else (`systemctl disable --now
   cockpit.socket` if present). Leave the provider's monitoring agent alone
   if the image shipped one.

## 2. Prepare the new box (he keeps running on the old one)

`docs/DEPLOY.md` Part B, steps B2 to B6, with two differences:

- B4: copy the data from the old box instead of fetching it,
  `rsync -a root@<old>:/root/bosco/data/raw/ /root/bosco/data/raw/`, and
  the cache too, `rsync -a root@<old>:/root/bosco/data/cache/ /root/bosco/data/cache/`
  (rebuilding it on first start costs minutes he would spend behind the clock).
- B5: copy `.env` from the old box rather than writing it again. Set
  `BOSCO_RSYNC_TARGET` to somewhere that is not the old box.

Then build both images and install the units, but **do not start `bosco`
yet**, and skip Part C: he is live already and the dry run would post his
introduction again.

```
podman build -t bosco -f ops/Containerfile .
podman build -t bosco-panel -f ops/Containerfile.panel .
cp ops/bosco.container ops/bosco-panel.container /etc/containers/systemd/
sed -i 's|%h/bosco|/root/bosco|g' /etc/containers/systemd/bosco.container /etc/containers/systemd/bosco-panel.container
systemctl daemon-reload
```

Check the new box computes like the old one, on a scratch brain:

```
podman run --rm -v /root/bosco/data/raw:/app/data/raw:ro,Z -v /root/bosco/data/cache:/app/data/cache:Z \
  --entrypoint /app/.venv/bin/python bosco /app/scripts/cpu_determinism.py
```

The `level` must be `x86_v3` and the `digest` line must equal the old
box's from §0. If it does not, stop and look: something other than the
pinned path differs (glibc in the image, numpy version), and he must not
move until it is understood.

## 3. Cut over

On the old box. This is the only part with a gap; it is a few minutes and
it is logged as `downtime` by his first poll on the new box.

```
systemctl stop bosco-nightly.timer bosco-deadman.timer
systemctl stop bosco          # SIGTERM: he saves his state and stops (journal: the last save)
ls -l state/brain_state.npz   # mtime after the stop
sqlite3 state/ledger.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"
rsync -a --delete /root/bosco/state/ root@<new>:/root/bosco/state/
```

`state/` holds the ledger, his weights and snapshots, the associations,
the panel files; the rsync must be complete before he starts anywhere. Do
not start him again on the old box after this: two of him with one account
would write two records.

On the new box:

```
systemctl start bosco
journalctl -fu bosco
```

The journal shows the load, `downtime skipped: N s not lived`, and the
first poll. Then the gate:

```
podman exec bosco /app/.venv/bin/bosco replay
sqlite3 state/ledger.sqlite "SELECT ts, kind, target_uri FROM control ORDER BY id DESC LIMIT 5"
```

`replay ... bit-identical True` means the new box reproduced the last span
of his life on the old box exactly, from the last snapshot plus the log.
There should be no new `numerics` row (both boxes at `x86_v3`). If there is
one, the boxes differed and §0 was skipped; the row is the boundary and the
next nightly integrity report will flag the one span that straddles it.

## 4. Timers and the dead-man

`docs/DEPLOY.md` D3 on the new box. Then:

```
systemctl list-timers | grep bosco
```

## 5. The panel

```
systemctl start bosco-panel
```

Point the `bosco.proto.cool` A record at the new IP. Caddy issues a new
certificate within a minute of the DNS change; the `_atproto.bosco` TXT
record is his handle and does not change (it names his DID, not a
machine). Open https://bosco.proto.cool: the live word in the top line
comes from the activity file the new box writes.

## 6. Leave the old box up for a week

Do not delete it. `systemctl disable --now bosco bosco-panel` there, and
keep the provider snapshot. After seven clean nightly integrity reports on
the new box (`snapshots/<date>/integrity.md`), destroy it.

## 7. What replaces the provider snapshot

A dedicated server has no console snapshot. The nightly already rsyncs
`state/` off-box; add a weekly dated tarball so a corruption does not
overwrite the only copy:

```
cat > /etc/systemd/system/bosco-weekly.service <<'EOT'
[Unit]
Description=bosco weekly state tarball
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'tar -C /root/bosco -czf /root/bosco-state-$(date -u +%%F).tgz state && rsync -a /root/bosco-state-*.tgz ${BOSCO_RSYNC_TARGET}'
EnvironmentFile=/root/bosco/.env
EOT
cat > /etc/systemd/system/bosco-weekly.timer <<'EOT'
[Unit]
Description=bosco weekly state tarball
[Timer]
OnCalendar=Sun 04:30
Persistent=true
[Install]
WantedBy=timers.target
EOT
systemctl daemon-reload && systemctl enable --now bosco-weekly.timer
```

## 8. Afterwards

- The panel's peer bench line (idle 0.138 / event 1.91 wall seconds on the
  VPS) will change; note the new numbers in the monthly report.
- With four cores, `dunce` can run live beside him at the freeze and the
  three controls can replay in parallel; the kernel is single-threaded per
  fly and parallel across flies.
