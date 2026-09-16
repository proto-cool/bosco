"""The retina: a thumbnail -> a few channel names (config/retina_v1.yaml).

A fixed, public, parameter-free transform at a fly's resolution: downsample to a grid of
facets, take each facet's hue, saturation and brightness, and name the dominant hues, the
brightness, the edge strength and the saturation as bins.  No learned weights anywhere;
what a picture means is the mushroom body's to work out.  The image is decoded, reduced and
discarded; only the channel names leave this module.
"""

from __future__ import annotations

import io

import numpy as np
import yaml

from bosco import paths


def load_retina_cfg(path=paths.CONFIG / "retina_v1.yaml") -> dict:
    return yaml.safe_load(open(path))


def facets(data: bytes, grid: tuple[int, int]) -> np.ndarray:
    """Decode an image and reduce it to a grid of RGB facets in [0, 1] (box filter)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB").resize((int(grid[0]), int(grid[1])), Image.BOX)
        return np.asarray(im, dtype=np.float64) / 255.0


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in [0, 1), saturation, value; vectorised over facets."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0.0)
    h = np.zeros_like(mx)
    nz = d > 0
    rc = np.where(nz, (mx - r) / np.maximum(d, 1e-12), 0.0)
    gc = np.where(nz, (mx - g) / np.maximum(d, 1e-12), 0.0)
    bc = np.where(nz, (mx - b) / np.maximum(d, 1e-12), 0.0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(nz, (h / 6.0) % 1.0, 0.0)
    return h, s, mx


def code_facets(rgb: np.ndarray, cfg: dict) -> tuple[str, ...]:
    """Channel names for a grid of facets."""
    h, s, v = hsv(rgb)
    out: list[str] = []
    nb = int(cfg["hue_bins"])
    coloured = s >= float(cfg["sat_floor"])
    w = np.where(coloured, s, 0.0)
    total = float(w.sum())
    if total > 0:
        bins = np.minimum(nb - 1, (h * nb).astype(int))
        hist = np.bincount(bins.ravel(), weights=w.ravel(), minlength=nb) / total
        order = np.argsort(-hist, kind="stable")
        for i in order[: int(cfg["top_hues"])]:
            if hist[i] >= float(cfg["hue_min_share"]):
                out.append(f"hue:{int(i)}")
    lum = float(v.mean())
    out.append(f"lum:{int(np.searchsorted(np.asarray(cfg['lum_cuts'], dtype=float), lum, side='right'))}")
    gy = np.abs(np.diff(v, axis=0)).mean() if v.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(v, axis=1)).mean() if v.shape[1] > 1 else 0.0
    edge = float((gy + gx) / 2.0)
    out.append(f"edge:{int(np.searchsorted(np.asarray(cfg['edge_cuts'], dtype=float), edge, side='right'))}")
    sat = float(s.mean())
    out.append(f"sat:{int(np.searchsorted(np.asarray(cfg['sat_cuts'], dtype=float), sat, side='right'))}")
    return tuple(out)


def code(data: bytes, cfg: dict | None = None) -> tuple[str, ...]:
    """Channel names for an encoded image (JPEG/PNG/...).  Empty if it cannot be decoded."""
    cfg = cfg or load_retina_cfg()
    try:
        rgb = facets(data, tuple(cfg["grid"]))
    except Exception:  # noqa: BLE001  (not an image, truncated, unsupported)
        return ()
    return code_facets(rgb, cfg)


def is_channel(token: str) -> bool:
    return token in ("img", "motion") or token.split(":", 1)[0] in ("hue", "lum", "edge", "sat")


def fetch(url: str, timeout_s: float) -> bytes | None:
    """One thumbnail, once, with a timeout; None on any failure."""
    import httpx

    try:
        r = httpx.get(url, timeout=timeout_s, follow_redirects=True)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:  # noqa: BLE001
        return None
    return None


def look(urls: tuple[str, ...], cfg: dict | None = None, fetcher=None) -> tuple[str, ...]:
    """The channels of the first max_images thumbnails, in order, each once."""
    cfg = cfg or load_retina_cfg()
    fetcher = fetcher or fetch
    out: list[str] = []
    for u in urls[: int(cfg.get("max_images", 2))]:
        data = fetcher(u, float(cfg.get("fetch_timeout_s", 3.0)))
        if not data:
            continue
        for t in code(data, cfg):
            if t not in out:
                out.append(t)
    return tuple(out)
