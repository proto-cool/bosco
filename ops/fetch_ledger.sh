#!/usr/bin/env bash
# Copy the live ledger off the VPS for calibration (features and URIs only; no text).
#   ops/fetch_ledger.sh [root@bosco.proto.cool]
set -euo pipefail
host="${1:-root@bosco.proto.cool}"
day=$(date -u +%F)
mkdir -p "snapshots/dev-$day"
ssh "$host" "sqlite3 /root/bosco/state/ledger.sqlite \".backup '/root/bosco/state/ledger.dev.sqlite'\""
scp "$host:/root/bosco/state/ledger.dev.sqlite" "snapshots/dev-$day/ledger.sqlite"
echo "snapshots/dev-$day/ledger.sqlite"
echo "calibrate: uv run python scripts/calibrate_thresholds.py --ledger snapshots/dev-$day/ledger.sqlite --allow-partial --write"
