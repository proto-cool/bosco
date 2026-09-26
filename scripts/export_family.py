"""Export a brain family and its specialists for the service (docs/SERVICE-R10.md, docs/BRAIN-VIEWS.md).

uv run python scripts/export_family.py                  # family v1 from the specialist pilot's data
uv run python scripts/export_family.py --specialist topic-real --name topic

A family is shared by every specialist: the connectome build, the nose's antenna (fit once, label-free),
the answer's DN groups, steps, the brain maps and the resting trace. A specialist is only its learned
weights plus a card. Written to `service/families/<id>/` and `service/specialists/<name>/<version>/`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys

import numpy as np
import torch

from bosco import encoders, paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import v1_pilot as V  # noqa: E402

SERVICE = paths.ROOT / "service"


def antenna(meta, X, L):
    """Exactly v1_pilot.load's antenna (seeded), returned as parameters."""
    tr = np.where([it["split"] == "train" for it in meta["items"]])[0]
    Xf = X[tr[np.random.default_rng(V.SEED).permutation(len(tr))[:4000]]]
    F = np.concatenate([Xf, np.repeat(L, max(1, len(Xf) // len(L)), 0)])
    mu = F.mean(0)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / s[:46, None]
    norm = float(np.percentile(np.abs((F - mu) @ W.T), 99))
    return mu.astype(np.float32), W.astype(np.float32), norm


def export_family(fid: str) -> int:
    meta = json.load(open(V.OUT / "items.json"))
    e = np.load(V.OUT / "emb.npz")
    mu, W, norm = antenna(meta, e["X"], e["L"])
    m, info = V.make_brain("real", device="cpu")
    order = hashlib.sha256(np.ascontiguousarray(M2ids()).tobytes()).hexdigest()[:16]
    d = SERVICE / "families" / fid
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "antenna.npz", mu=mu, W=W, norm=np.float32(norm))
    for src, dst in (("runs/brain-map/anatomy-map-mix0.3.json", "anatomy.json"), ("runs/brain-map/map.json", "wave.json")):
        shutil.copy(paths.ROOT / src, d / dst)
    json.dump({
        "id": fid, "neuron_count": m.n, "neuron_order_hash": order, "steps": m.steps, "dt_ms": 5.0,
        "read_steps": m.read_steps, "dn_groups": info, "gain": V.GAIN,
        "encoder": encoders.TEXT, "antenna": "bi46: 46 whitened components, resting rate 0.5 (docs/SPECIALIST-PILOT.md)",
        "connectome": "MaleCNS v1.0 (Janelia FlyEM et al., CC BY 4.0), cut at 5 synapses, all KC->MBON kept",
        "created": datetime.date.today().isoformat(),
    }, open(d / "family.json", "w"), indent=1)
    print(f"family {fid}: {m.n} neurons, order {order}")
    return 0


def M2ids():
    from bosco import model2 as M2

    return M2.load_or_build().brain.ids


def export_specialist(spec: str, name: str, fid: str) -> int:
    task, arm = spec.rsplit("-", 1)
    ck = torch.load(V.RUNS / f"{spec}.pt", weights_only=True)
    meta = json.load(open(V.OUT / "items.json"))
    tj = json.load(open(V.RUNS / f"{spec}.json"))
    version = f"{name}-{datetime.date.today().isoformat()}"
    d = SERVICE / "specialists" / name / version
    d.mkdir(parents=True, exist_ok=True)
    learned = {k: v for k, v in ck["state"].items() if k in ("log_g", "b", "log_tau", "kp_logm", "log_k", "c")}
    torch.save(learned, d / "weights.pt")
    card = {
        "specialist": name, "version": version, "family": fid, "task": task, "arm": arm,
        "options": meta["tasks"][task]["options"], "best_val": tj["best_val"],
        "status": "development: not yet passed the production bar (docs/SPECIALIST-PILOT.md)",
    }
    json.dump(card, open(d / "card.json", "w"), indent=1)
    print(f"specialist {version} ({len(card['options'])} options)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="v1")
    ap.add_argument("--specialist")
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    if a.specialist:
        return export_specialist(a.specialist, a.name or a.specialist.rsplit("-", 1)[0], a.family)
    return export_family(a.family)


if __name__ == "__main__":
    sys.exit(main())
