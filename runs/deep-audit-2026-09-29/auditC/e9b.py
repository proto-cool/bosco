"""E9: does a KC->MBON memory move the DN read (fixed scale: raw read d, no k, no c)? C0 only."""
import json
import sys

sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *  # noqa

arm = sys.argv[1]
TR = "/Users/nickd/projects/bosco/runs/brain-train"
t = T()
m, rest, ctx = load("C0", arm)
X, S = smells_items(ctx["B"], 192)
mbon = m.regions["mbon"]
rr0, d0 = sniff(m, S)
t("untrained")
out = {"arm": arm, "n_kp": len(m.kp_logm), "read_sd_untrained": float(d0.std()), "read_mean_untrained": float(d0.mean())}


def measure(kp, name):
    with Perturb(m, rest, kp=kp) as mm:
        rr, d = sniff(mm, S)
        dd = d - d0
        dm = (rr[mbon] - rr0[mbon])
        res = {"logm_sd": float(kp.std()), "logm_mean": float(kp.mean()),
               "read_sd": float(d.std()), "delta_read_sd": float(dd.std()), "delta_read_mean": float(dd.mean()),
               "delta_mbon_abs_mean": float(dm.abs().mean()), "delta_mbon_item_sd": float(dm.std(1).mean()),
               "transfer_sd_ratio": float(dd.std() / dm.std(1).mean().clamp(min=1e-12)),
               "kc_active": float((rr[m.kc] > 0.01).float().mean())}
    out[name] = res
    t(name)
    return dd


own = f"{TR}/real-s1.pt"
st = torch.load(own)
assert len(st["kp_logm"]) == len(m.kp_logm)
dT = measure(st["kp_logm"], "real_memory_by_index")
out["trained_own_k"] = float(torch.exp(st["log_k"]))
g = torch.Generator().manual_seed(3)
for s in range(2):
    perm = torch.randperm(len(st["kp_logm"]), generator=g)
    dR = measure(st["kp_logm"][perm], f"permuted_{s}")
    out[f"permuted_{s}"]["corr_with_trained"] = float(np.corrcoef(dR, dT)[0, 1])
if False:
    fl = torch.load(f"{TR}/real-flip-s1.pt")
    dF = measure(fl["kp_logm"], "flip_trained")
    out["flip_trained"]["corr_with_trained"] = float(np.corrcoef(dF, dT)[0, 1])
json.dump(out, open(f"/Users/nickd/.claude/jobs/4059153c/tmp/auditC/E9b_{arm}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
