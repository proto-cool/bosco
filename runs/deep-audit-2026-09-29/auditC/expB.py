"""E5 (odour-specific depression), E6 (MBON activation -> read), E8 (DN pathways)."""
import json
import sys

sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *  # noqa

ver, arm = sys.argv[1], sys.argv[2]
DRIVE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
t = T()
m, rest, ctx = load(ver, arm)
typ = types(ctx)
X, S = smells_items(ctx["B"], 192)
r_rest = m.r_rest.clone()
rr, d = sniff(m, S)
rest_d = rest_read(m)
kc = m.kc
act = rr[kc] > 0.01
J = jaccard_pairs(act)
t("intact")
# literature transmitter (Aso et al. 2014a) -> Aso 2014b valence: Glu avoid, GABA/ACh approach
LIT = {**{f"MBON{i:02d}": "glu" for i in range(1, 8)}, **{f"MBON{i:02d}": "gaba" for i in range(8, 12)},
       **{f"MBON{i:02d}": "ach" for i in range(12, 20)}}
mbon = m.regions["mbon"].numpy()
mtypes = sorted({typ[i] for i in mbon if typ[i][:6] in LIT})
apm, avm = (x.numpy() for x in m.read_groups["mbon"])
out = {"ver": ver, "arm": arm, "drive": DRIVE, "E5": {}, "E6": {}, "E8": {}}
for T_ in mtypes:
    cells = torch.tensor(np.nonzero((typ == T_) & np.isin(np.arange(len(typ)), mbon))[0])
    lit = LIT[T_[:6]]
    exp_sign = -1 if lit == "glu" else 1  # activation: glu -> avoid (read down)
    model_side = "approach" if np.isin(cells.numpy(), apm).all() else "avoid" if np.isin(cells.numpy(), avm).all() else "none"
    # E6: activate the type at rest
    with Perturb(m, rest, cells, db=DRIVE) as mm:
        dread = rest_read(mm) - rest_d
        dm = float((mm.r_rest[cells] - r_rest[cells]).mean())
    out["E6"][T_] = {"lit_nt": lit, "model_group": model_side, "d_rate_mbon": dm, "d_read": dread,
                     "expected_sign": exp_sign, "sign_ok": bool(np.sign(dread) == exp_sign)}
    # E5: pair odour A with dopamine in this compartment
    ev = (rr[cells] - r_rest[cells][:, None]).mean(0)
    A = int(ev.argmax())
    res = {"evoked_A_before": float(ev[A]), "evoked_sd": float(ev.std())}
    if ev[A] > 1e-4:
        cand = torch.nonzero(ev >= 0.5 * ev[A])[:, 0]
        cand = cand[cand != A]
        pool = cand if len(cand) else torch.tensor([i for i in range(len(ev)) if i != A])
        Bi = int(pool[J[A, pool].argmin()])
        kcA = kc[act[:, A]]
        mask = torch.isin(m.kp_post, cells) & torch.isin(m.kp_pre, kcA)
        kp = m.kp_logm.detach().clone()
        kp[mask] += float(np.log(0.2))
        with Perturb(m, rest, kp=kp) as mm:
            rr2, d2 = sniff(mm, S[[A, Bi]])
            ev2 = (rr2[cells] - mm.r_rest[cells][:, None]).mean(0)
        dropA = 100 * (1 - float(ev2[0]) / float(ev[A]))
        dropB = 100 * (1 - float(ev2[1]) / float(ev[Bi])) if ev[Bi] > 1e-6 else float("nan")
        dd = float(d2[0] - d[A])
        # depressing an avoid (glu) MBON -> approach (read up); depressing approach -> read down
        res.update({"B": Bi, "jaccard_AB": float(J[A, Bi]), "evoked_B_before": float(ev[Bi]),
                    "synapses_depressed": int(mask.sum()), "kcs_A": int(len(kcA)),
                    "drop_A_pct": dropA, "drop_B_pct": dropB, "d_read_A": dd, "d_read_B": float(d2[1] - d[Bi]),
                    "read_sd_items": float(d.std()),
                    "pass_dropA": dropA >= 50, "pass_specific": (dropA - dropB) >= 30 if dropB == dropB else False,
                    "pass_read_dir": bool(np.sign(dd) == -exp_sign)})
    out["E5"][T_] = res
    t(T_)

# ---------------- E8 ----------------
def act_type(names, watch, db=DRIVE):
    idx = idx_of(typ, names)
    with Perturb(m, rest, idx, db=db) as mm:
        r = {"cells": len(idx), "d_read": rest_read(mm) - rest_d,
             "d_self": float((mm.r_rest[idx] - r_rest[idx]).mean())}
        for w in watch:
            wi = idx_of(typ, [w])
            r[f"d_{w}"] = float((mm.r_rest[wi] - r_rest[wi]).mean())
            r[f"rest_{w}"] = float(r_rest[wi].mean())
    return r


E8 = out["E8"]
E8["read_sd_items"] = float(d.std())
E8["MDN_in_avoid_read"] = bool(np.isin(idx_of(typ, ["MDN"]).numpy(), m.read_groups["dn"][1].numpy()).all())
E8["act_MDN"] = act_type(["MDN"], ["MDN"])
E8["act_LC16"] = act_type(["LC16"], ["MDN", "DNp42", "DNa13"])
E8["act_LC4"] = act_type(["LC4"], ["DNp01", "DNp02", "DNp04"])
# MDN's strongest in-model input types (signed normalised weight summed over MDN cells)
W = m.W.coalesce()
mdn = idx_of(typ, ["MDN"])
ii = W.indices()
sel = torch.isin(ii[0], mdn)
pre = ii[1, sel].numpy()
val = W.values()[sel].numpy()
import pandas as pd
df = pd.DataFrame({"type": typ[pre], "w": val}).groupby("type").w.sum().sort_values()
E8["MDN_top_inputs"] = {k: float(v) for k, v in pd.concat([df.head(4), df.tail(6)]).items()}
for k in df.tail(4).index:
    if k and k != "MDN":
        E8[f"act_input_{k}"] = act_type([k], ["MDN"])
t("E8")
json.dump(out, open(f"/Users/nickd/.claude/jobs/4059153c/tmp/auditC/B_{ver}_{arm}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
