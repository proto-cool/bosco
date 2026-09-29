"""Capped v3.1 label-free start (rest_start31) on 64 unlabelled explore smells, to see whether it converges. Read-only."""
import sys, time, json, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "/Users/nickd/projects/bosco/scripts")
import brain_check as B
from bosco import model2 as M2, v1, ratebrain3 as R3
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
v1.START_ITERS = ITERS
X = np.load("/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz")["X"]
ant = B.antenna()
cal = torch.tensor(ant(X[:64])); ev = torch.tensor(ant(X[64:128])); rest = torch.tensor(ant.resting())
b2 = M2.load_or_build()
m = v1.build("real", device="cpu", b2=b2)
t0 = time.time()
op = v1.rest_start31(m, cal, rest)
wall = time.time() - t0
h = op["history"]
print(json.dumps({k: v for k, v in op.items() if k != "history"}, default=float))
for r in h[:: max(1, len(h) // 10)] + [h[-1]]:
    print(r)
# where are the scales, and which regions are silent/saturated on held-out smells
with torch.no_grad():
    _, _, rr = m.run(ev, record="read")
reg = {k: v for k, v in m.regions.items()}
out = {}
for k, v in reg.items():
    x = rr[v]
    out[k] = {"mean": float(x.mean()), "frac_cells_silent(<0.002)": float((x.mean(1) < 0.002).float().mean()),
              "s_in_median": float(m.s_in[v].median()), "s_in_at_max": float((m.s_in[v] >= 99.9).float().mean()),
              "s_in_at_min": float((m.s_in[v] <= 0.1001).float().mean()), "item_sd_median": float(x.std(1).median())}
print(json.dumps(out, indent=0))
kc = rr[m.kc]; act = kc > R3.ACTIVE
print("KC active/sniff", float(act.float().mean()), "median active rate", float(kc[act].median()) if act.any() else None)
print("KC b_cell median/frac>0", float(m.b_cell[m.kc].median()), float((m.b_cell[m.kc] > 0).float().mean()))
print("wall s", wall)
