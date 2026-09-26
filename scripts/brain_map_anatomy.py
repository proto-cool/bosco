"""The 64 x 48 anatomical brain map: a real front view, every dot used, coloured like calcium imaging.

uv run python scripts/brain_map_anatomy.py --spec topic-real [--mix 0.6]

- Each neuron sits at its real soma position (MaleCNS somaLocation; missing ones take their cell type's
  median, else their class's). Front view: x = left-right, y = dorsal-ventral.
- Every dot is used: columns take equal numbers of neurons in x order, and within a column, rows take equal
  numbers in y order. `--mix` blends that rank layout with the raw positions (1 = pure rank, every dot
  equally full; lower keeps more of the true shape, and emptier dots take their nearest neurons).
- Colour like GCaMP imaging: firing above rest glows green to white; pushed below rest dips magenta; the
  resting brain is a dim blue-grey with a faint tint per region.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from bosco import data, paths
from bosco import model2 as M2

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4b_dev as D  # noqa: E402
import v1_pilot as V  # noqa: E402

OUT = paths.ROOT / "runs" / "brain-map"
W, H = 64, 48


def positions(m) -> np.ndarray:
    a = data.annotations().reindex(M2.load_or_build().brain.ids)
    loc = a["somaLocation"].to_numpy()
    P = np.full((m.n, 3), np.nan)
    ok = np.array([isinstance(v, (list, np.ndarray)) and len(v) == 3 for v in loc])
    P[ok] = np.stack(loc[ok]).astype(float)
    for col in ("type", "class"):
        g = a[col].fillna("?").to_numpy()
        miss = np.isnan(P[:, 0])
        if not miss.any():
            break
        df = __import__("pandas").DataFrame(P[~miss], columns=list("xyz"))
        df["g"] = g[~miss]
        med = df.groupby("g")[list("xyz")].median()
        for i in np.nonzero(miss)[0]:
            if g[i] in med.index:
                P[i] = med.loc[g[i]].to_numpy()
    miss = np.isnan(P[:, 0])
    P[miss] = np.nanmedian(P, 0)
    return P


def layout(P, mix):
    x, y = P[:, 0], P[:, 1]
    n = len(x)
    rx = np.empty(n)
    rx[np.argsort(x, kind="stable")] = np.arange(n) / n
    cx = np.clip(((mix * rx + (1 - mix) * (x - x.min()) / np.ptp(x)) * W).astype(int), 0, W - 1)
    cy = np.zeros(n, int)
    for c in range(W):
        idx = np.nonzero(cx == c)[0]
        if len(idx) == 0:
            continue
        ry = np.empty(len(idx))
        ry[np.argsort(y[idx], kind="stable")] = np.arange(len(idx)) / len(idx)
        yy = (y[idx] - y.min()) / np.ptp(y)
        cy[idx] = np.clip(((mix * ry + (1 - mix) * yy) * H).astype(int), 0, H - 1)
    cells = {}
    for i in range(n):
        cells.setdefault((int(cx[i]), int(cy[i])), []).append(i)
    # any empty dot takes its nearest neurons (by grid distance), so every dot is used
    have = np.array(list(cells))
    for gx in range(W):
        for gy in range(H):
            if (gx, gy) not in cells:
                j = np.argmin(((have - [gx, gy]) ** 2).sum(1))
                cells[(gx, gy)] = list(cells[tuple(have[j])])
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="topic-real")
    ap.add_argument("--item", type=int, default=0)
    ap.add_argument("--mix", type=float, default=0.6)
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    task, arm = a.spec.rsplit("-", 1)
    meta, X, L, zl, sets = V.load()
    m, info = V.make_brain(arm, device="cpu")
    m.load_state_dict(torch.load(V.RUNS / f"{a.spec}.pt", weights_only=True)["state"])
    P = positions(m)
    cells = layout(P, a.mix)
    va = sets[task if task != "all" else "topic"]["val"]
    it = va[a.item]
    s1, _, _ = D.fly_sniffs([it | {"opts": [it["opts"][it["gold"]]], "gold": 0}], zl, "bi46")
    with torch.no_grad():
        _, _, t1 = m.run(torch.tensor(s1), record=True)
        _, _, rest = m.run(torch.full((1, s1.shape[1]), 0.5), record=True)
    sd = t1.float().numpy()[:, :, 0] - rest.float().numpy()[:, :, 0]  # signed distance from rest
    keys = sorted(cells)
    smat = np.stack([sd[:, cells[k]].mean(1) for k in keys], 1)
    scale = float(np.percentile(np.abs(smat).max(0), 97))
    # a faint resting tint per region
    reg = {k: v.cpu().numpy() for k, v in m.regions.items()}
    tint_of = {"kc": (70, 55, 95), "mbon": (70, 55, 95), "dan": (70, 55, 95), "alpn": (45, 60, 90), "orn": (45, 60, 90),
               "alln": (45, 60, 90), "lh": (85, 50, 65), "cx": (50, 80, 60), "vpn": (40, 70, 80), "dn": (90, 65, 40)}
    tint = np.tile(np.array([48, 52, 64], float), (m.n, 1))
    for k, c in tint_of.items():
        tint[reg[k]] = c
    base = {k: tint[cells[k]].mean(0) for k in keys}
    from PIL import Image, ImageDraw

    frames, glow = [], np.zeros(len(keys))
    sign = np.zeros(len(keys))
    cnt = np.array([len(cells[k]) for k in keys], float)
    dens = (np.log1p(cnt) - np.log1p(cnt).min()) / max(np.ptp(np.log1p(cnt)), 1e-9)  # 0..1 per dot
    S = 14  # pixels per grid cell (finer rendering, sizes vary within it)
    for st in range(m.steps):
        v = np.clip(np.abs(smat[st]) / scale, 0, 1)
        up = v >= glow
        sign = np.where(up, np.sign(smat[st]), sign)
        glow = np.where(up, v, glow * 0.9)
        if st % 4 and st != m.steps - 1:
            continue
        img = Image.new("RGB", (W * S, H * S), (5, 6, 9))
        halo = Image.new("RGB", (W * S, H * S), (0, 0, 0))
        dr, dh = ImageDraw.Draw(img), ImageDraw.Draw(halo)
        for j, (gx, gy) in enumerate(keys):
            g = float(glow[j]) ** 1.2
            if sign[j] < 0:
                g *= 0.55  # inhibition: a subtler dip than excitation
            rest_lvl = 0.35 + 0.45 * dens[j]  # denser regions rest brighter
            b = base[(gx, gy)] * rest_lvl
            hot = np.array([130, 255, 150]) if sign[j] >= 0 else np.array([200, 80, 170])
            if sign[j] >= 0 and g > 0.7:
                hot = hot + (np.array([255, 255, 255]) - hot) * (g - 0.7) / 0.3
            c = b * (1 - g) + hot * g
            r = 1.8 + 1.4 * dens[j] + 2.2 * g  # size: density at rest, swelling with activity
            cx, cy = gx * S + S / 2, gy * S + S / 2
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(x) for x in c))
            if g > 0.35 and sign[j] >= 0:
                R = r + 4
                dh.ellipse([cx - R, cy - R, cx + R, cy + R], fill=tuple(int(x * 0.35) for x in hot))
        from PIL import ImageChops, ImageFilter

        img = ImageChops.add(img, halo.filter(ImageFilter.GaussianBlur(4)))
        ImageDraw.Draw(img).text((6, H * S - 14), f"{st * 5} ms", fill=(150, 150, 160))
        frames.append(img)
    strip = Image.new("RGB", (W * S * 2 + 10, H * S * 2 + 10), (0, 0, 0))
    for j, fi in enumerate([2, 5, 10, len(frames) - 1]):
        strip.paste(frames[fi], ((j % 2) * (W * S + 10), (j // 2) * (H * S + 10)))
    OUT.mkdir(parents=True, exist_ok=True)
    strip.save(OUT / f"anatomy-frames-mix{a.mix:g}.png")
    frames[0].save(OUT / f"anatomy-mix{a.mix:g}.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    json.dump({"width": W, "height": H, "mix": a.mix,
               "dots": [{"x": x, "y": y, "neurons": [int(i) for i in cells[(x, y)]]} for x, y in keys]},
              open(OUT / f"anatomy-map-mix{a.mix:g}.json", "w"))
    print(f"dots {len(cells)} of {W * H}; wrote anatomy-frames-mix{a.mix:g}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
