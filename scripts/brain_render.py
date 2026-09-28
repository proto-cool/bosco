"""Shared renderer for the 64 x 48 brain views (anatomical flat and wave): one look for both.

Calcium-imaging palette: firing above rest glows green to white; pushed below rest is a subtler magenta dip.
Resting dots carry a faint tint (region or band) and their size and brightness follow how many real neurons
sit under them; active dots swell and get a soft halo. `emphasis` dots (e.g. the named answer cells) render
larger.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 14  # pixels per grid cell
EXCITE = np.array([130, 255, 150])
INHIBIT = np.array([200, 80, 170])
WHITE = np.array([255, 255, 255])


def render(keys, counts, tints, smat, steps_to_draw, W=64, H=48, emphasis=frozenset(), afterglow=0.9, groups=None):
    """keys: [(x, y)] dots; counts: neurons per dot; tints: resting RGB per dot; smat: (steps, dots) signed
    activity from rest. `groups` (one label per dot) scales brightness per group (the wave's bands), so every
    system's activity shows; default one scale for the whole brain. Returns PIL frames for the given steps."""
    cnt = np.asarray(counts, float)
    lc = np.log1p(cnt)
    dens = (lc - lc.min()) / max(np.ptp(lc), 1e-9)
    tints = np.asarray(tints, float)
    peaks = np.abs(smat).max(0)
    if groups is None:
        scale = np.full(len(keys), float(np.percentile(peaks, 97)) or 1.0)
    else:
        groups = np.asarray(groups)
        scale = np.ones(len(keys))
        for gname in set(groups.tolist()):
            m_ = groups == gname
            scale[m_] = max(float(np.percentile(peaks[m_], 90)), 1e-9)
    emph = np.array([k in emphasis for k in keys])
    glow, sign, frames = np.zeros(len(keys)), np.zeros(len(keys)), []
    for st in range(smat.shape[0]):
        v = np.clip(np.abs(smat[st]) / scale, 0, 1)
        up = v >= glow
        sign = np.where(up, np.sign(smat[st]), sign)
        glow = np.where(up, v, glow * afterglow)
        if st not in steps_to_draw:
            continue
        img = Image.new("RGB", (W * S, H * S), (6, 7, 10))
        halo = Image.new("RGB", (W * S, H * S), (0, 0, 0))
        dr, dh = ImageDraw.Draw(img), ImageDraw.Draw(halo)
        for j, (gx, gy) in enumerate(keys):
            g = float(glow[j]) ** 1.2
            if sign[j] < 0:
                g *= 0.55
            b = tints[j] * (0.55 + 0.5 * dens[j])  # resting: visible structure, denser = brighter
            hot = EXCITE if sign[j] >= 0 else INHIBIT
            if sign[j] >= 0 and g > 0.7:
                hot = hot + (WHITE - hot) * (g - 0.7) / 0.3
            c = np.clip(b * (1 - g) + hot * g, 0, 255)
            r = 1.8 + 1.4 * dens[j] + 2.2 * g + (1.5 if emph[j] else 0)
            cx, cy = gx * S + S / 2, gy * S + S / 2
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(x) for x in c))
            if g > 0.35 and sign[j] >= 0:
                R = r + 4
                dh.ellipse([cx - R, cy - R, cx + R, cy + R], fill=tuple(int(x * 0.35) for x in hot))
        img = ImageChops.add(img, halo.filter(ImageFilter.GaussianBlur(4)))
        ImageDraw.Draw(img).text((6, H * S - 14), f"{st * 5} ms", fill=(150, 150, 160))
        frames.append(img)
    return frames


def strip(frames, idx, W=64, H=48):
    out = Image.new("RGB", (W * S * 2 + 10, H * S * 2 + 10), (0, 0, 0))
    for j, fi in enumerate(idx):
        out.paste(frames[fi], ((j % 2) * (W * S + 10), (j // 2) * (H * S + 10)))
    return out
