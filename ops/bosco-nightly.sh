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
# freeze-v2: the ownership shares a verdict is read through are derived from the signature cache,
# so a span replayed without it computes different shares and will not reproduce.
cp state/account_kcs.json "$dst/account_kcs.json" 2>/dev/null || true

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

if [ "$rc" -ne 0 ]; then alert "bosco: integrity FAIL, see $dst/integrity.md"; fi

# Publishing is its own failure, with its own alert.  From 2026-09-18 to 09-21 deploys left the
# checkout detached (`git checkout <sha>`); each night committed onto the detached HEAD, the push
# had no branch to send, the next deploy walked away from the commits, and the alert said
# "integrity FAIL" over a report that was all PASS.  So: a detached HEAD at origin/main, or behind
# it by nothing but snapshots, is put back on main (the working tree is only the host's copy; the
# running image does not move).  Any other detached HEAD is left alone and the snapshot stays on
# disk, untracked, which survives checkouts, and is picked up by the first night that can publish.
pub=0
publish() {
  git fetch -q origin main || { alert "bosco: nightly cannot fetch origin; snapshot $day kept on disk"; return 1; }
  if ! git symbolic-ref -q HEAD >/dev/null; then
    local head; head=$(git rev-parse HEAD)
    if git merge-base --is-ancestor "$head" origin/main \
       && [ -z "$(git diff --name-only "$head" origin/main -- . ':(exclude)snapshots/')" ]; then
      git checkout -q -B main origin/main && git branch -q -u origin/main \
        || { alert "bosco: could not put the box back on main; snapshot $day kept on disk"; return 1; }
      alert "bosco: the box was detached at ${head:0:7}; put back on main"
    else
      alert "bosco: the box is detached at ${head:0:7}, not on main; snapshot $day kept on disk, not published"
      return 1
    fi
  elif [ "$(git symbolic-ref --short HEAD)" != main ]; then
    alert "bosco: the box is on $(git symbolic-ref --short HEAD), not main; snapshot $day kept on disk, not published"
    return 1
  else
    git merge -q --ff-only origin/main \
      || { alert "bosco: main on the box has diverged from origin; snapshot $day kept on disk"; return 1; }
  fi
  # every dated snapshot, so a night that could not publish is carried by the next one that can
  git add snapshots/20[0-9][0-9]-*
  if ! git diff --cached --quiet; then
    git commit -q -m "snapshot $day"
    git push -q origin main || { alert "bosco: snapshot $day committed but did not push"; return 1; }
  fi
}
publish || pub=1

[ "$rc" -eq 0 ] && [ "$pub" -eq 0 ]
