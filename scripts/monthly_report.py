"""Monthly descriptive note (EXPERIMENT.md §6): numbers only."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from bosco.ledger import Ledger


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--month", help="YYYY-MM (default: previous month)")
    ap.add_argument("--dunce-ledger")
    a = ap.parse_args(argv)
    if a.month:
        y, m = map(int, a.month.split("-"))
    else:
        first = dt.date.today().replace(day=1) - dt.timedelta(days=1)
        y, m = first.year, first.month
    t0 = dt.datetime(y, m, 1, tzinfo=dt.UTC).timestamp()
    t1 = dt.datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=dt.UTC).timestamp()
    L = Ledger(a.ledger, read_only=True)
    ev = [r for r in L.episodes(since_ts=t0, kind="event") if r["ts"] < t1]
    sp = [r for r in L.episodes(since_ts=t0, kind="spontaneous") if r["ts"] < t1]
    pr = [r for r in L.episodes(since_ts=t0, kind="pairing") if r["ts"] < t1]
    acts = [r for r in L.actions_since(t0) if r["ts"] < t1]
    mix = {}
    for r in ev + sp:
        mix[r["action"]] = mix.get(r["action"], 0) + 1
    silence = mix.get("nothing", 0) / max(1, len(ev) + sp.__len__())
    blocks = L.db.execute("SELECT COUNT(*) FROM outcomes WHERE source='block' AND ts>=? AND ts<?", (t0, t1)).fetchone()[
        0
    ]
    rewards = L.db.execute(
        "SELECT COUNT(*) FROM outcomes WHERE valence='reward' AND ts>=? AND ts<?", (t0, t1)
    ).fetchone()[0]
    punish = L.db.execute(
        "SELECT COUNT(*) FROM outcomes WHERE valence='punishment' AND ts>=? AND ts<?", (t0, t1)
    ).fetchone()[0]
    ctrl = dict(
        L.db.execute("SELECT kind, COUNT(*) FROM control WHERE ts>=? AND ts<? GROUP BY kind", (t0, t1)).fetchall()
    )
    out = [
        f"# bosco monthly note {y}-{m:02d}\n",
        f"- event episodes: {len(ev)}; spontaneous episodes: {len(sp)}; pairings: {len(pr)}",
        f"- silence rate: {silence:.3f}",
        f"- action mix: {json.dumps(mix, sort_keys=True)}",
        f"- real actions posted: {len(acts)}",
        f"- outcomes: reward {rewards}, punishment {punish} (blocks {blocks})",
        f"- operator control events: {json.dumps(ctrl, sort_keys=True)}",
    ]
    # what he did on his own feet (2026-09-15): walks, quotes, taste pairings, familiarity
    walks = sum(1 for r in acts if r["kind"] == "walk")
    quotes = sum(1 for r in acts if r["kind"] == "quote")
    taste = {"reward": 0, "punishment": 0}
    fam_first, fam_later = [], []
    seen_dids: dict[str, int] = {}
    for r in ev:
        note = r["note"] or ""
        for k in taste:
            if f"taste:{k}" in note:
                taste[k] += 1
        try:
            f = json.loads(r["mbon"]).get("_familiar")
        except (TypeError, ValueError):
            f = None
        if f is not None and r["did"]:
            n = seen_dids.get(r["did"], 0)
            (fam_first if n == 0 else fam_later if n >= 5 else []).append(float(f))
            seen_dids[r["did"]] = n + 1
    out += [
        f"- walks (posts read on his own feet): {walks}; quotes in answers: {quotes}",
        f"- taste pairings while reading: {json.dumps(taste)}",
        f"- familiarity of an account on first meeting vs from the sixth: "
        f"{sum(fam_first) / max(1, len(fam_first)):.3f} (n={len(fam_first)}) vs "
        f"{sum(fam_later) / max(1, len(fam_later)):.3f} (n={len(fam_later)})",
    ]
    if a.dunce_ledger:
        D = Ledger(a.dunce_ledger)
        by_src = {r["source_uri"]: r["action"] for r in D.episodes(since_ts=t0, kind="event") if r["ts"] < t1}
        paired = [(r["action"], by_src[r["source_uri"]]) for r in ev if r["source_uri"] in by_src]
        agree = sum(1 for x, y_ in paired if x == y_) / max(1, len(paired))
        out.append(f"- dunce action agreement over {len(paired)} shared stimuli: {agree:.3f}")
    # which integrity checks passed, from the nightly reports of the month
    from pathlib import Path

    from bosco import paths

    passed, failed, nights = 0, 0, 0
    for d in sorted(Path(paths.SNAPSHOTS).glob(f"{y}-{m:02d}-*")):
        f = d / "integrity.md"
        if f.exists():
            nights += 1
            txt = f.read_text()
            passed += txt.count("PASS")
            failed += txt.count("FAIL")
    if nights:
        out.append(f"- integrity checks over {nights} nightly reports: {passed} PASS, {failed} FAIL")
    else:
        out.append("- integrity checks: see snapshots/<date>/integrity.md for each nightly run")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
