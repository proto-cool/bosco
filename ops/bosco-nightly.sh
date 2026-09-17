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

# A missing tool must be heard, not exit 127 into a failed unit nobody reads: between
# 2026-09-14 and 2026-09-17 `uv` was not installed on the box and the snapshot stopped
# being published, while the fly went on living and the dead-man stayed quiet.
for cmd in sqlite3 podman git; do
  command -v "$cmd" >/dev/null || { alert "bosco: nightly cannot run, $cmd is not on PATH"; exit 127; }
done

sqlite3 state/ledger.sqlite ".backup '$dst/ledger.sqlite'"
cp state/brain_state.npz "$dst/brain_state.npz"

rc=0
{
  echo "# Nightly $day"
  echo
} > "$dst/integrity.md"

# The report must come from the same numerics as the fly.  The image's kernel is built for
# x86-64-v3 and numpy's CPU path is pinned there (docs/MIGRATE.md, README: numerics); a host
# build with other flags could vectorise a sum differently and fail the replay check for no
# reason but the compiler.  So the checks run in his own image, against the night's copy.
npy="NPY_DISABLE_CPU_FEATURES=AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
in_image() {
  podman run --rm -e "$npy" \
    -v "$PWD/state:/app/state:z" -v "$PWD/$dst:/app/snapshot:z" \
    -v "$PWD/data/raw:/app/data/raw:ro,z" -v "$PWD/data/cache:/app/data/cache:ro,z" \
    "$@"
}
in_image localhost/bosco:latest --ledger /app/snapshot/ledger.sqlite integrity >> "$dst/integrity.md" 2>&1 || rc=$?
echo >> "$dst/integrity.md"
in_image --entrypoint /app/.venv/bin/python localhost/bosco:latest \
  /app/scripts/integrity_checks.py --ledger /app/snapshot/ledger.sqlite --state-dir /app/state \
  >> "$dst/integrity.md" 2>&1 || rc=$?

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
