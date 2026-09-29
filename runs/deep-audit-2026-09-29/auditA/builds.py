"""Run in a given tree: model build stats + the saved start's static stats."""
import sys, json, time, numpy as np, torch, pandas as pd
torch.set_num_threads(2)
from bosco import model2 as M2, v1, data, ratebrain3 as R3
tree = sys.argv[1]
t0 = time.time()
b2 = M2.load_or_build()
m = v1.build("real", device="cpu", b2=b2)
out = {"tree": tree, "BETA": R3.BETA, "W_nnz": int(m.W._nnz()), "plastic": int(m.kp_w.numel()), "n_units": m.n_units,
       "W_sum_pos": float(m.W.values().clamp(min=0).sum()), "W_sum_neg": float(m.W.values().clamp(max=0).sum()),
       "kp_w_sum": float(m.kp_w.sum()), "approach_mbon": int(len(m.read_groups["mbon"][0])), "avoid_mbon": int(len(m.read_groups["mbon"][1]))}
# per-post-class input weight sums (row sums of |W| and net) for the cut matrix
W = m.W.coalesce(); idx = W.indices().numpy(); val = W.values().numpy()
kp_post = m.kp_post.numpy(); kp = m.kp_w.numpy()
n = m.n
net = np.bincount(idx[0], weights=val, minlength=n) + np.bincount(kp_post, weights=kp, minlength=n)
pos = np.bincount(idx[0], weights=np.clip(val, 0, None), minlength=n) + np.bincount(kp_post, weights=kp, minlength=n)
a = data.annotations().reindex(b2.brain.ids)
cls = a["class"].fillna(a["superclass"]).fillna("?").to_numpy()
df = pd.DataFrame({"cls": cls, "net": net, "pos": pos})
out["row_sums_median_by_class"] = df.groupby("cls").median().round(3).loc[["Kenyon_Cell", "MBON", "ALPN", "ALLN", "olfactory", "DAN", "descending_neuron", "cb_intrinsic", "visual_projection", "CX"]].to_dict("index")
# the saved start
st = torch.load(f"{tree}/runs/brain-check/real-start.pt", weights_only=False)
out["start_keys"] = list(st.keys())
try:
    with torch.no_grad():
        m.b.copy_(st["b"])
    out["start_loads"] = True
except Exception as e:
    out["start_loads"] = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
b = st["b"]; unit = m.unit
if b.numel() == m.n_units:
    thr = (-b[unit]).numpy()
    reg = {k: v.numpy() for k, v in m.regions.items()}
    out["threshold_by_region"] = {k: [float(np.median(thr[v])), float(thr[v].min()), float(thr[v].max())] for k, v in reg.items() if len(v)}
    bc = st["b_cell"].numpy()[m.kc.numpy()]
    out["kc_b_cell"] = {"median": float(np.median(bc)), "frac_pos": float((bc > 0).mean()), "min": float(bc.min()), "max": float(bc.max())}
    out["kc_effective_bias_median"] = float(np.median(bc + b[unit].numpy()[m.kc.numpy()]))
    rr = st["r_rest"].numpy()
    out["rest_rate_by_region"] = {k: float(rr[v].mean()) for k, v in reg.items() if len(v)}
    out["log_g"] = float(st["log_g"].mean())
out["wall"] = time.time() - t0
print(json.dumps(out, indent=1, default=str))
