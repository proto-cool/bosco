"""The 64 x 48 dot-matrix brain map ("the wave"), generated from the real brain and a trained specialist.

uv run python scripts/brain_map.py --spec topic-real

- Rows: left hemisphere (top 23), the unpaired midline neurons (2 rows), right hemisphere mirrored (bottom 23). Fixed bands per
  hemisphere: smell 3, taste 1, vision 3, learning 4, innate 2, navigation 2, output 2, other 6.
- Columns: 60 of response time (when a neuron starts reacting, measured on real sniffs), then 4 for the named
  answer cells (the approach / avoid descending neurons).
- A dot = the real neurons in that band row and time bin; brightness = their mean distance from rest now.
Writes runs/brain-map/map.json (dot -> neurons) and PNG frames + a GIF of a real decision.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from bosco import data, paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4b_dev as D  # noqa: E402
import v1_pilot as V  # noqa: E402

OUT = paths.ROOT / "runs" / "brain-map"
W, H, HEMI, GAP, TIME_COLS = 64, 48, 23, 2, 60
BANDS = [  # (name, rows, colour)
    ("smell", 3, (86, 140, 255)),
    ("taste", 1, (255, 170, 60)),
    ("vision", 3, (70, 200, 220)),
    ("learning", 4, (180, 120, 255)),
    ("innate", 2, (255, 110, 150)),
    ("navigation", 2, (120, 220, 120)),
    ("output", 2, (255, 120, 40)),
    ("other", 6, (150, 150, 160)),
]
MIDLINE_COLOUR = (220, 220, 235)


def band_rows(m):
    """Each neuron -> (band, sub-row within band)."""
    R = {k: v.cpu().numpy() for k, v in m.regions.items()}
    a = data.annotations().reindex(__import__("bosco.model2", fromlist=["x"]).load_or_build().brain.ids)
    cls = a["class"].fillna("").to_numpy().astype(str)
    band = np.full(m.n, "other", object)
    sub = np.zeros(m.n, int)
    for k, (b, s) in {"orn": ("smell", 0), "alpn": ("smell", 1), "alln": ("smell", 2), "vpn": ("vision", 0),
                      "kc": ("learning", 0), "mbon": ("learning", 2), "dan": ("learning", 3), "lh": ("innate", 0),
                      "cx": ("navigation", 0), "dn": ("output", 0)}.items():
        band[R[k]], sub[R[k]] = b, s
    band[cls == "gustatory"] = "taste"
    # spread big bands over their rows by a stable hash of the neuron index
    rows = dict((b, r) for b, r, _ in BANDS)
    for b, r in rows.items():
        idx = np.nonzero(band == b)[0]
        if b == "learning":  # KCs over rows 0-1, MBON row 2, DAN row 3
            kc = idx[sub[idx] == 0]
            sub[kc] = kc % 2
        elif b == "smell":
            pass
        else:
            sub[idx] = idx % r
    side = a["somaSide"].fillna(a.get("rootSide", "")).astype(str).str.upper().str[:1].to_numpy()
    side = np.where(np.isin(side, ["L", "R"]), side, "M").astype(object)
    return band, sub, side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="topic-real")
    ap.add_argument("--item", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    task, arm = a.spec.rsplit("-", 1)
    meta, X, L, zl, sets = V.load()
    m, info = V.make_brain(arm, device="cpu")
    m.load_state_dict(torch.load(V.RUNS / f"{a.spec}.pt", weights_only=True)["state"])
    va = sets[task if task != "all" else "topic"]["val"]
    # rest and response times from 16 validation sniffs (the first option of each)
    b = [it | {"opts": it["opts"][:1], "gold": 0} for it in va[:16]]
    s, _, _ = D.fly_sniffs(b, zl, "bi46")
    with torch.no_grad():
        _, _, tr = m.run(torch.tensor(s), record=True)
        _, _, rest = m.run(torch.full((1, s.shape[1]), 0.5), record=True)
    tr, rest = tr.float().numpy(), rest.float().numpy()[:, :, 0]  # (steps, n, B), (steps, n)
    dev = np.abs(tr - rest[:, :, None]).mean(2)  # (steps, n)
    peak = dev.max(0)
    t_resp = np.argmax(dev >= 0.5 * peak[None, :], axis=0).astype(float)
    # structural depth: synapse hops from the sensory neurons (breaks ties, places non-responders)
    Wb = (m.W.coalesce().abs() + torch.sparse_coo_tensor(torch.stack([m.kp_post, m.kp_pre]), m.kp_w.abs(), (m.n, m.n))).coalesce()
    sens = torch.zeros(m.n, 1)
    for k in ("orn", "other_sensory", "vpn"):
        sens[m.regions[k]] = 1.0
    hops = np.full(m.n, 9.0)
    reach, frontier = sens[:, 0] > 0, sens
    hops[reach.numpy()] = 0
    for h in range(1, 9):
        frontier = (torch.sparse.mm(Wb, frontier) > 0).float()
        new_ = (frontier[:, 0] > 0) & ~reach
        hops[new_.numpy()] = h
        reach |= new_
    t_resp[peak < 1e-5] = m.steps + hops[peak < 1e-5]  # non-responders go after responders, by depth
    key = t_resp + hops / 100.0
    band, sub, side = band_rows(m)
    # answer cells: columns 60-63 of the output rows (avoid 60-61, approach 62-63)
    ap_i, av_i = (x.cpu().numpy() for x in m.read_groups["dn"])
    named = np.zeros(m.n, int) - 1
    named[av_i], named[ap_i] = 0, 1
    row_start, r0 = {}, 0
    for b_, r, _ in BANDS:
        row_start[b_] = r0
        r0 += r
    cells: dict[tuple[int, int], list[int]] = {}

    def fill(dots, neurons):
        """Every dot gets neurons: neurons in response-time order, split evenly across dots in order (a
        neuron is repeated when a block has fewer neurons than dots)."""
        neurons = sorted(neurons, key=lambda i: key[i])
        if not neurons:
            return
        if len(neurons) >= len(dots):
            for d, part in zip(dots, np.array_split(np.array(neurons), len(dots)), strict=True):
                cells[d] = [int(i) for i in part]
        else:
            for j, d in enumerate(dots):
                cells[d] = [int(neurons[j * len(neurons) // len(dots)])]

    for hemi_left in (True, False):
        for b_, rows, _ in BANDS:
            for rr in range(rows):
                y = row_start[b_] + rr
                y = y if hemi_left else H - 1 - y
                sel = (band == b_) & (sub == rr) & (side == ("L" if hemi_left else "R")) & (named < 0)
                ncols = TIME_COLS if b_ == "output" else W
                fill([(x, y) for x in range(ncols)], np.nonzero(sel)[0].tolist())
            if b_ == "output":
                for g, cols in ((0, (60, 61)), (1, (62, 63))):
                    ys = [row_start["output"] + rr for rr in range(2)]
                    ys = ys if hemi_left else [H - 1 - y for y in ys]
                    sel = (named == g) & (side == ("L" if hemi_left else "R"))
                    if not sel.any():
                        sel = named == g
                    fill([(x, y) for y in ys for x in cols], np.nonzero(sel)[0].tolist())
    mid = np.nonzero(side == "M")[0]  # unpaired, midline neurons: the two middle rows
    half = sorted(mid.tolist(), key=lambda i: key[i])
    fill([(x, HEMI) for x in range(W)], half[::2])
    fill([(x, HEMI + 1) for x in range(W)], half[1::2])
    colour = {b_: c for b_, _, c in BANDS} | {"midline": MIDLINE_COLOUR}
    dot_band = {k: ("midline" if k[1] in (HEMI, HEMI + 1) else band[v[0]]) for k, v in cells.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"width": W, "height": H, "bands": [[b_, r, c] for b_, r, c in BANDS],
               "dots": [{"x": int(x), "y": int(y), "band": dot_band[(x, y)], "neurons": [int(i) for i in v]}
                        for (x, y), v in cells.items()],
               "named": {"avoid": info["avoid_types"], "approach": info["approach_types"]}},
              open(OUT / "map.json", "w"))
    # a real decision: one validation item, its right option
    it = va[a.item]
    s1, _, _ = D.fly_sniffs([it | {"opts": [it["opts"][it["gold"]]], "gold": 0}], zl, "bi46")
    with torch.no_grad():
        _, _, t1 = m.run(torch.tensor(s1), record=True)
    d1 = np.abs(t1.float().numpy()[:, :, 0] - rest)  # (steps, n)
    from PIL import Image, ImageDraw

    frames = []
    keys = list(cells)
    dmat = np.stack([d1[:, cells[k]].mean(1) for k in keys], 1)  # (steps, dots)
    # brightness scale per band: the 90th percentile of its dots' peaks, so weak reactors stay dim
    bscale = {}
    for b_ in [b for b, _, _ in BANDS] + ["midline"]:
        js = [j for j, k in enumerate(keys) if dot_band[k] == b_]
        if js:
            bscale[b_] = max(float(np.percentile(dmat[:, js].max(0), 90)), 1e-6)
    dpeak = np.array([bscale[dot_band[k]] for k in keys])
    kidx = {k: j for j, k in enumerate(keys)}
    for st in list(range(0, m.steps, 4)) + [m.steps - 1]:
        img = Image.new("RGB", (W * 10, H * 10), (10, 10, 14))
        dr = ImageDraw.Draw(img)
        for gx in range(W):
            for gy in range(H):
                v = cells.get((gx, gy))
                if v is None:
                    c, lvl = (255, 0, 0), 1.0  # an empty dot would be a bug: show it loudly
                else:
                    c = colour[dot_band[(gx, gy)]]
                    lvl = 0.15 + 0.85 * float(np.clip(dmat[st, kidx[(gx, gy)]] / dpeak[kidx[(gx, gy)]], 0, 1)) ** 1.5
                col_ = tuple(int(ch * lvl) for ch in c)
                dr.ellipse([gx * 10 + 1, gy * 10 + 1, gx * 10 + 8, gy * 10 + 8], fill=col_)
        dr.text((4, H * 10 - 12), f"{st * 5} ms", fill=(200, 200, 200))
        frames.append(img)
    strip = Image.new("RGB", (W * 10 * 2 + 10, H * 10 * 2 + 10), (0, 0, 0))
    for j, fi in enumerate([2, 5, 10, len(frames) - 1]):
        strip.paste(frames[fi], ((j % 2) * (W * 10 + 10), (j // 2) * (H * 10 + 10)))
    strip.save(OUT / "frames-4.png")
    frames[-1].save(OUT / "frame-last.png")
    frames[len(frames) // 3].save(OUT / "frame-early.png")
    frames[0].save(OUT / "wave.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    used = len(cells)
    print(f"dots used {used} of {W * H}; frames {len(frames)}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
