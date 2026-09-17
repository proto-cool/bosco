#!/usr/bin/env bash
# Copy the live ledger off the box for calibration (features and URIs only; no text).
#   ops/fetch_ledger.sh [nick@bosco.proto.cool]
#
# Since the move to the dedicated box (docs/MIGRATE.md, 2026-09-16) he lives under nick in
# ~/bosco, rootless; root does not log in.
set -euo pipefail
host="${1:-nick@bosco.proto.cool}"
day=$(date -u +%F)
mkdir -p "snapshots/dev-$day"
ssh "$host" "sqlite3 ~/bosco/state/ledger.sqlite \".backup '/tmp/ledger.dev.sqlite'\""
scp "$host:/tmp/ledger.dev.sqlite" "snapshots/dev-$day/ledger.sqlite"
ssh "$host" "rm -f /tmp/ledger.dev.sqlite"
echo "snapshots/dev-$day/ledger.sqlite"
echo "calibrate: uv run python scripts/calibrate_thresholds.py --ledger snapshots/dev-$day/ledger.sqlite --allow-partial --write"
