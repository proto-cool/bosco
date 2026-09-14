# ops

- `Containerfile` — one image, `uv` + C kernel.  Build: `podman build -t bosco -f ops/Containerfile .`
- `bosco.container` — quadlet unit (restart-on-failure).  Copy to `~/.config/containers/systemd/`.
- `bosco-nightly.{sh,service,timer}` — ledger + weights into `snapshots/<date>/`, integrity report, rsync, commit.
- `bosco-deadman.sh` — exits 1 and POSTs to `BOSCO_ALERT_URL` if no episode in 4 h.
- `env.example` — copy to `~/bosco/.env`.

Weekly Vultr snapshot is configured in the Vultr console, not here.
- `Containerfile.panel`, `Caddyfile`, `bosco-panel.container` — the public panel (`panel/`), built with Node, served by Caddy; reads `state/panel/`.
