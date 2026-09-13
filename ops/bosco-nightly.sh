#!/usr/bin/env bash
# Nightly: copy ledger + weights into snapshots/YYYY-MM-DD/, run integrity checks,
# rsync state off-box, commit the snapshot to the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
day=$(date -u +%F)
dst="snapshots/$day"
mkdir -p "$dst"
sqlite3 state/ledger.sqlite ".backup '$dst/ledger.sqlite'"
cp state/mb_state.npz "$dst/mb_state.npz"
uv run bosco --ledger "$dst/ledger.sqlite" integrity
uv run python scripts/integrity_checks.py --ledger "$dst/ledger.sqlite" --state-dir state > "$dst/integrity.md"
if [ -n "${BOSCO_RSYNC_TARGET:-}" ]; then rsync -a state/ "$BOSCO_RSYNC_TARGET"; fi
git add "$dst" && git commit -q -m "snapshot $day" && git push -q || true
