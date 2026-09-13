#!/usr/bin/env bash
# Dead-man: alert if no episode in 4 h.  Run from a timer every 30 min.
set -euo pipefail
cd "$(dirname "$0")/.."
last=$(sqlite3 state/ledger.sqlite "SELECT COALESCE(MAX(ts),0) FROM episodes")
now=$(date +%s)
if (( now - ${last%.*} > 14400 )); then
  msg="bosco: no episode in $(( (now - ${last%.*}) / 3600 )) h"
  echo "$msg" >&2
  if [ -n "${BOSCO_ALERT_URL:-}" ]; then curl -fsS -m 10 -d "$msg" "$BOSCO_ALERT_URL" || true; fi
  exit 1
fi
