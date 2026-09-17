#!/usr/bin/env bash
# Nightly: copy ledger + weights into snapshots/YYYY-MM-DD/, run the integrity checks,
# rsync state off-box, commit the snapshot to the repo.
#
# The integrity report is the evidence (EXPERIMENT.md 5), so a failing night must still be
# published and must be heard: the checks record their exit status and fire the alert, but they
# never abort the snapshot.  Until 2026-09-17 they did, under `set -e`, and a stale rate-cap
# check meant no snapshot reached the repo at all.
set -euo pipefail
cd "$(dirname "$0")/.."
day=$(date -u +%F)
dst="snapshots/$day"
mkdir -p "$dst"

alert() {
  echo "$1" >&2
  if [ -n "${BOSCO_ALERT_URL:-}" ]; then curl -fsS -m 10 -d "$1" "$BOSCO_ALERT_URL" || true; fi
}

sqlite3 state/ledger.sqlite ".backup '$dst/ledger.sqlite'"
cp state/brain_state.npz "$dst/brain_state.npz"

rc=0
{
  echo "# Nightly $day"
  echo
} > "$dst/integrity.md"
uv run bosco --ledger "$dst/ledger.sqlite" integrity >> "$dst/integrity.md" 2>&1 || rc=$?
echo >> "$dst/integrity.md"
uv run python scripts/integrity_checks.py --ledger "$dst/ledger.sqlite" --state-dir state >> "$dst/integrity.md" 2>&1 || rc=$?

if [ -n "${BOSCO_RSYNC_TARGET:-}" ]; then
  rsync -a state/ "$BOSCO_RSYNC_TARGET" || { rc=1; alert "bosco: nightly rsync off-box failed"; }
fi

git add "$dst"
if ! git diff --cached --quiet; then
  git commit -q -m "snapshot $day"
  git push -q || { rc=1; alert "bosco: snapshot $day committed but did not push"; }
fi

if [ "$rc" -ne 0 ]; then alert "bosco: integrity FAIL, see $dst/integrity.md"; fi
exit "$rc"
