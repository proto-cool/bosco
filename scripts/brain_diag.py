"""Where the item's signal goes, label-free (a diagnostic for BRAIN-SPEC L5; no labels, no training).

From the saved label-free start (scripts/brain_check.py), on the same 200 unlabelled items:
1. the spread across items of each stage's mean rate (ORN fed, ALPN, KC, MBON approach/avoid, DN approach/avoid)
   and the mean per-cell spread within each stage;
2. how far the read can move through the mushroom body: random KC->MBON memories (log-multipliers ~ N(0, s),
   seeded, not learned) and the read's spread across items under each; and the read's response to a fixed
   extra drive on the approach or the avoid MBONs.

    uv run python scripts/brain_diag.py
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import brain_check as B  # noqa: E402

from bosco import model2 as M2  # noqa: E402


def stages(m, rr):
    apm, avm = m.read_groups["mbon"]
    apd, avd = m.read_groups["dn"]
    groups = {"orn_fed": m.orn_idx, "alpn": m.regions["alpn"], "lh": m.regions["lh"], "kc": m.kc,
              "mbon_approach": apm, "mbon_avoid": avm, "dan": m.regions["dan"], "dn_approach": apd,
              "dn_avoid": avd}
    out = {}
    for k, v in groups.items():
        x = rr[v]
        out[k] = {"mean": float(x.mean()), "sd_of_group_mean": float(x.mean(0).std()),
                  "mean_cell_sd": float(x.std(1).mean())}
    return out


def read_sd(m, smells):
    with torch.no_grad():
        m.freeze()
        m.r_rest = None
        m.settle(REST)
        _, _, rr = m.run(smells, record="read")
    ap, av = m.read_groups["dn"]
    d = rr[ap].mean(0) - rr[av].mean(0)
    return float(d.std()), float(d.mean()), rr


def main() -> int:
    global REST
    torch.set_num_threads(10)
    b2 = M2.load_or_build()
    m = B.brain("real", b2)
    B.load_start(m, "real")
    ant = B.antenna()
    REST = torch.tensor(ant.resting())
    smells = torch.tensor(ant(B.items()[0]))
    res = {}
    sd0, mu0, rr = read_sd(m, smells)
    res["untrained"] = {"read_sd": sd0, "read_mean": mu0, "stages": stages(m, rr)}
    g = torch.Generator().manual_seed(7)
    res["random_memory"] = {}
    for s in (0.5, 1.0, 2.0, 3.0):
        with torch.no_grad():
            m.kp_logm.copy_(torch.randn(m.kp_logm.shape, generator=g) * s)
        sd, mu, rr = read_sd(m, smells)
        mb = stages(m, rr)
        res["random_memory"][s] = {"read_sd": sd, "read_mean": mu,
                                   "mbon_approach_sd": mb["mbon_approach"]["sd_of_group_mean"],
                                   "mbon_avoid_sd": mb["mbon_avoid"]["sd_of_group_mean"]}
    with torch.no_grad():
        m.kp_logm.zero_()
    res["mbon_push"] = {}
    for name in ("approach", "avoid"):
        idx = m.read_groups["mbon"][0 if name == "approach" else 1]
        for dx in (0.05, 0.2):
            with torch.no_grad():
                m.b_cell[idx] += dx
            _, mu, rr = read_sd(m, smells)
            ap, av = m.read_groups["mbon"]
            res["mbon_push"][f"{name}+{dx}"] = {"read_mean_change": mu - mu0,
                                                "mbon_group_rate_change": float(rr[idx].mean()
                                                                                - res["untrained"]["stages"][f"mbon_{name}"]["mean"])}
            with torch.no_grad():
                m.b_cell[idx] -= dx
    json.dump(res, open(B.OUT / "diag.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
