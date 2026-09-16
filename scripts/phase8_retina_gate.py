"""Phase 8 gate: the retina (decided 2026-09-15; config/retina_v1.yaml).

A picture is a few channel names driving the visual Kenyon cells.  Before it goes live:

  1. a picture alone lights a sparse KC code (fraction active in the recorded range), and a
     picture with an account odor stays in range too;
  2. MBON responses to a picture are non-zero (the visual KCs reach the mushroom body), and
     the visual KCs are the cells that carry it (most active KCs for the picture alone are
     visual KCs);
  3. the same picture five times, thirty seconds apart, habituates: the visual KCs' output
     resources are used (STD on their outputs, model_v1.yaml habituation_extra_types) and his
     familiarity with it rises; a different picture is fresh;
  4. nothing runs away: activity settles within a few seconds after the picture.

Writes docs/phase8-retina.md.
"""

from __future__ import annotations

import json
import sys
import tempfile

import numpy as np

from bosco import paths
from bosco.agent import Agent
from bosco.encoder import Features
from bosco.ledger import Ledger

T0 = 1_800_000_000.0


def main() -> int:
    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/gate.sqlite"), state_dir=tmp)
    fly, enc = ag.fly, ag.enc
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.bio_ms(T0)
    kc_range = json.load(open(paths.CONFIG / "thresholds.json")).get("kc_range", [0.0, 0.15])
    lines = ["# Phase 8 gate: the retina", ""]
    ok = True

    cat = ("img", "hue:1", "hue:0", "lum:2", "edge:1", "sat:1")  # a warm, busy-ish picture
    sky = ("img", "hue:5", "lum:3", "edge:0", "sat:1")

    # 1. sparseness: picture alone, picture with an account
    ag.live.set_base(ag._base_drives(ag.live.t_ms))
    w = ag.live.present(enc.embed_drives(cat), 1000.0)
    kc = w.counts[fly.kc]
    frac_alone = float((kc > 0).mean())
    vis = np.isin(fly.kc, enc.visual_kcs)
    share_visual = float((kc[vis] > 0).sum() / max(1, (kc > 0).sum()))
    mbon = fly.mbon_rates_from_counts(w.counts, w.ms)
    mbon_hz = float(np.mean(list(mbon.values())))
    ag.live.idle(3000.0)
    o = ag.run(
        Features("did:plc:someone", 0.0, False, 0, False, (), False, (), (), None, (), cat), T0 + 10, "p://1", fast=True
    )
    frac_with = ag.ledger.episode(o.episode_id)["kc_active"] / len(fly.kc)
    lines += [
        "## Sparseness and reach",
        "",
        f"- KC fraction active, picture alone: {frac_alone:.4f} (range at tag {kc_range[0]:.4f} .. {kc_range[1]:.4f})",
        f"- of the KCs active for the picture alone, visual KCs: {share_visual:.2f}",
        f"- KC fraction active, picture with an account odor: {frac_with:.4f}",
        f"- mean MBON rate to the picture alone: {mbon_hz:.2f} Hz; non-zero types: {sum(1 for v in mbon.values() if v > 0)} of {len(mbon)}",
        "",
    ]
    c1 = kc_range[0] <= frac_alone <= kc_range[1] and kc_range[0] <= frac_with <= kc_range[1]
    c2 = mbon_hz > 0 and share_visual > 0.5
    ok &= c1 and c2
    lines.append(
        f"**Check 1 (sparse): {'PASS' if c1 else 'FAIL'}**  **Check 2 (reaches the MB, carried by visual KCs): {'PASS' if c2 else 'FAIL'}**\n"
    )

    # 3. habituation and familiarity
    fams, rates = [], []
    for i in range(5):
        o = ag.run(
            Features("did:plc:someone", 0.0, False, 0, False, (), False, (), (), None, (), cat),
            T0 + 100 + 30 * i,
            f"p://{i + 2}",
        )
        fams.append(o.decision.familiar)
        rates.append(json.loads(ag.ledger.episode(o.episode_id)["scores"]).get("engage", 0.0))
    x = ag.live.net.x()
    used = float((1.0 - x[enc.visual_channel_idx("hue:1")]).mean())
    o = ag.run(Features("did:plc:someone", 0.0, False, 0, False, (), False, (), (), None, (), sky), T0 + 300, "p://sky")
    fam_sky = o.decision.familiar
    lines += [
        "## The same picture five times, thirty seconds apart",
        "",
        f"- familiarity before each: {[round(f, 3) for f in fams]} (want rising)",
        f"- output resources used on a channel's visual KCs after: {used:.3f} (want > 0)",
        f"- a different picture right after: familiarity {fam_sky:.3f} (want below the fifth of the same)",
        "",
    ]
    c3 = fams[-1] > fams[0] and used > 0 and fam_sky < fams[-1]
    ok &= c3
    lines.append(f"**Check 3 (habituates, familiar): {'PASS' if c3 else 'FAIL'}**\n")

    # 4. settles
    w1 = ag.live.idle(1000.0)
    w2 = ag.live.idle(1000.0)
    ag.live.idle(3000.0)
    w5 = ag.live.idle(1000.0)
    a1, a2, a5 = (float((w.counts[fly.kc] > 0).mean()) for w in (w1, w2, w5))
    lines += [
        "## After the picture",
        "",
        f"- KC fraction active in the seconds after: 1 s {a1:.4f}, 2 s {a2:.4f}, 6 s {a5:.4f} (want falling to ~0)",
        "",
    ]
    c4 = a5 <= max(a1, 0.002)
    ok &= c4
    lines.append(f"**Check 4 (settles): {'PASS' if c4 else 'FAIL'}**\n")
    lines.append(f"**GATE: {'PASS' if ok else 'FAIL'}**")
    (paths.DOCS / "phase8-retina.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
