import numpy as np, pandas as pd, torch, json
torch.set_num_threads(2)
from bosco import data, model2 as M2, wiring3 as W3, populations as pop, v1, controls as C, senses as S
from bosco import ratebrain3 as R3, ratebrain2 as R2
from bosco.model import AL_LN_PREFIXES
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
OUT = "/Users/nickd/.claude/jobs/4059153c/tmp/auditA/"
a = data.annotations()
b2 = M2.load_or_build(); b = b2.brain; n = b.n; ids = b.ids
am = a.reindex(ids)
cls = am["class"].fillna(am["superclass"]).fillna("?").to_numpy().astype(str)
typ = am["type"].fillna("").to_numpy().astype(str)
sup = am["superclass"].fillna("").to_numpy().astype(str)
ft = W3.fast_tables(ids)

# MBON08 and MBON class coverage
print("MBON08 rows:", a[a.type.fillna("").str.startswith("MBON08")][["type", "class", "superclass", "status"]].to_dict("records"))
print("MBON-typed bodies not class MBON (in model):", sorted(set(typ[(np.char.startswith(typ.astype(str), "MBON")) & (cls != "MBON")])))
print("class MBON types:", sorted(set(typ[cls == "MBON"])))

# ---- pathways: uncut vs cut, and fast (v3.1)
pre = b.pre_of_edges(); post = b.indices; cnt = b.count.astype(float)
kc = cls == "Kenyon_Cell"; mb = cls == "MBON"
keep = (b.count >= 5) | (kc[pre] & mb[post])
ed = W3.edge_drives(b, ft)
lh = np.array([t.startswith(R3.LH_PREFIXES) for t in typ])
grp = {
    "ORN": np.char.startswith(typ.astype(str), "ORN_"), "uPN": (cls == "ALPN") & ~np.char.startswith(typ.astype(str), "M_") & ~np.char.startswith(typ.astype(str), "CB"),
    "mPN": (cls == "ALPN") & (np.char.startswith(typ.astype(str), "M_") | np.char.startswith(typ.astype(str), "CB")),
    "ALLN": cls == "ALLN", "KC": kc, "APL": typ == "APL", "MBON": mb, "DAN": cls == "DAN", "LHN": lh,
    "DN": sup == "descending_neuron", "CX": cls == "CX",
}
rows = []
for s_, t_ in [("ORN", "uPN"), ("ORN", "mPN"), ("ORN", "ALLN"), ("ORN", "ORN"), ("ALLN", "ORN"), ("ALLN", "uPN"), ("uPN", "KC"), ("mPN", "KC"),
               ("uPN", "LHN"), ("mPN", "LHN"), ("KC", "KC"), ("KC", "APL"), ("APL", "KC"), ("KC", "MBON"), ("KC", "DAN"), ("DAN", "MBON"),
               ("MBON", "MBON"), ("MBON", "DAN"), ("MBON", "LHN"), ("MBON", "DN"), ("LHN", "DN"), ("LHN", "LHN"), ("MBON", "CX"), ("uPN", "DN")]:
    m = grp[s_][pre] & grp[t_][post]
    tot = cnt[m].sum()
    rows.append({"path": f"{s_}->{t_}", "edges": int(m.sum()), "syn": int(tot), "kept_cut": round(cnt[m & keep].sum() / max(tot, 1), 3),
                 "fast_kept_cut": round(cnt[m & keep & ed].sum() / max(tot, 1), 3)})
P = pd.DataFrame(rows); print(P.to_string())
# direct MBON->DN and 2-hop
print("DNs with any MBON input (uncut/cut):", int(np.unique(post[grp["MBON"][pre] & grp["DN"][post]]).size), int(np.unique(post[grp["MBON"][pre] & grp["DN"][post] & keep]).size))
# read DN cells' input sources (v3.0 read)
rd = json.load(open("/Users/nickd/projects/bosco/runs/brain-check/dn_read.json"))
for side in ("approach", "avoid"):
    cells = np.array(rd[side]); m = np.isin(post, cells) & keep
    s = pd.Series(cnt[m]).groupby(cls[pre[m]]).sum().sort_values(ascending=False)
    print(side, "read DN input by pre class (cut):", (s / s.sum()).round(3).head(6).to_dict(), " from MBON direct:", int(cnt[m & mb[pre]].sum()))
lab = ft["label"]
print("read DN types and labels:", {side: sorted({(typ[i], lab[i]) for i in rd[side]}) for side in ("approach", "avoid")})
print("STEERING present:", {t: int((typ == t).sum()) for t in v1.STEERING_DNS})

# ---- controls under the v3.1 rule
for arm in ("layered", "hash"):
    w = v1.wiring(arm, b, 1)
    edw = W3.edge_drives(w, ft)
    inf = W3.own_in_fast(b, w, ft)
    wc = v1.cut(w)
    pre_w = w.pre_of_edges()
    kc_mb_w = kc[pre_w] & mb[w.indices]
    kk = kc[pre_w] & kc[w.indices]
    print(f"\n[{arm}] edges {w.nnz} vs {b.nnz}; fast syn {int(w.count[edw].sum())} vs {int(b.count[ed].sum())};"
          f" KC->KC syn {int(w.count[kk].sum())} vs {int(b.count[kc[pre] & kc[post]].sum())}; KC->MBON edges {int(kc_mb_w.sum())} vs {int((kc[pre] & mb[post]).sum())}")
    print(f"  in_fast equal to real? max|diff| {np.abs(inf - ft['in_fast']).max():.0f}; neurons with different in_fast: {int((np.abs(inf - ft['in_fast']) > 0.5).sum())}")
    print(f"  cut edges {wc.nnz} vs real cut {int(keep.sum())}; cut fast syn {int(wc.count[W3.edge_drives(wc, ft)].sum())} vs {int(cnt[keep & ed].sum())}")
    # shortcuts
    print("  shortcuts:", C.shortcut_report(w))
    # ORN->uPN block specifics: does layered keep glomerular identity? share of ORN->uPN syn onto matching glomerulus
    if arm == "layered":
        orn_g = np.array([t[4:] if t.startswith("ORN_") else "" for t in typ])
        pn_g = np.array([t.split("_")[0] for t in typ])
        def match(bb):
            p_ = bb.pre_of_edges(); m = grp["ORN"][p_] & grp["uPN"][bb.indices]
            return float((bb.count[m] * (orn_g[p_[m]] == pn_g[bb.indices[m]])).sum() / bb.count[m].sum())
        print("  ORN->uPN syn onto the matching glomerulus: real", round(match(b), 3), "layered", round(match(w), 3))
    # shuffled edges that land on self?
    print("  self-loops:", int((w.pre_of_edges() == w.indices).sum()), "real:", int((pre == post).sum()))
    # sign of edges into KCs in hash: APL share of KC input variance
    if arm == "hash":
        pw = w.pre_of_edges(); m = kc[w.indices] & (typ[pw] == "APL")
        per = np.bincount(w.indices[m], weights=w.count[m], minlength=n)[kc]
        per0 = np.bincount(post[kc[post] & (typ[pre] == "APL")], weights=cnt[kc[post] & (typ[pre] == "APL")], minlength=n)[kc]
        print("  APL->KC syn per KC: real mean/sd/zero", per0.mean().round(1), per0.std().round(1), int((per0 == 0).sum()), " hash", per.mean().round(1), per.std().round(1), int((per == 0).sum()))
        m2 = kc[w.indices] & (cls[pw] == "ALPN")
        pn_per = np.bincount(w.indices[m2], weights=w.count[m2], minlength=n)[kc]
        pn0 = np.bincount(post[kc[post] & (cls[pre] == "ALPN")], weights=cnt[kc[post] & (cls[pre] == "ALPN")], minlength=n)[kc]
        print("  PN->KC syn per KC: real zero-KCs", int((pn0 == 0).sum()), " hash zero-KCs", int((pn_per == 0).sum()))

# ---- per-type sharing
t30 = R2.type_of_neurons(b); t31 = R3.type_of_neurons(b)
print("\nunits v3.0", len(np.unique(t30)), "v3.1", len(np.unique(t31)))
u30 = pd.Series(t30).value_counts(); print("largest v3.0 units", u30.head(6).to_dict())
u31 = pd.Series(t31).value_counts(); print("largest v3.1 units", u31.head(8).to_dict())
# mixed sensory/central in v3.1 pseudo-types?
sens = np.char.find(sup.astype(str), "sensory") >= 0
mix = pd.DataFrame({"t": t31, "s": sens}).groupby("t").s.agg(["mean", "size"]); mix = mix[(mix["mean"] > 0) & (mix["mean"] < 1)]
print("v3.1 units mixing sensory and non-sensory cells:", mix.to_dict("index"))
# types spanning classes (a named type used across different classes)
tc = pd.DataFrame({"t": t31, "c": cls}).groupby("t").c.nunique(); print("v3.1 units spanning >1 class:", tc[tc > 1].head(15).to_dict(), "count", int((tc > 1).sum()))
# units mixing KCs etc with regions used by rest_start31 masks: tuned vs sensory mask overlap
reg = R3.regions(b)
sens_u = set(t31[np.concatenate([reg["orn"], reg["other_sensory"], reg["vpn"]])])
kc_u = set(t31[reg["kc"]]); mb_u = set(t31[reg["mbon"]])
print("units both 'sensory' (kept s_in=1) and containing non-sensory cells:",
      {u: int(((t31 == u) & ~np.isin(np.arange(n), np.concatenate([reg['orn'], reg['other_sensory'], reg['vpn']]))).sum()) for u in sens_u
       if ((t31 == u) & ~np.isin(np.arange(n), np.concatenate([reg['orn'], reg['other_sensory'], reg['vpn']]))).any()})
print("KC units containing non-KCs:", {u: int(((t31 == u) & ~kc).sum()) for u in kc_u if ((t31 == u) & ~kc).any()})
print("MBON units containing non-MBONs:", {u: int(((t31 == u) & ~mb).sum()) for u in mb_u if ((t31 == u) & ~mb).any()})

# ---- nose
orn = pop.orns(); gl = sorted(orn.glomerulus.unique())
nose = S.nose(b)
usable = [g for g in gl if g not in S.INNATE_GLOMERULI]
print("\nORN glomeruli:", len(gl), "usable", len(usable), "fed", nose.n, "dropped (usable, unfed):", sorted(set(usable) - set(nose.glomeruli)))
print("innate present:", [g for g in S.INNATE_GLOMERULI if g in gl], "innate absent from data:", [g for g in S.INNATE_GLOMERULI if g not in gl])
print("glomerulus labels odd:", [g for g in gl if not g.replace("_", "").isalnum() or "+" in g or g.startswith("VP")])
fed = np.concatenate(nose.orn_idx); orn_all = np.nonzero(cls == "olfactory")[0]
print("olfactory-class cells:", len(orn_all), "fed:", len(fed), "unfed olfactory:", len(set(orn_all) - set(fed)),
      "unfed types:", pd.Series(typ[list(set(orn_all) - set(fed))]).value_counts().head(12).to_dict())
side = am["somaSide"].to_numpy()
print("fed ORN sides:", pd.Series(side[fed]).value_counts(dropna=False).to_dict())
print("ORNs per fed glomerulus min/median/max:", min(map(len, nose.orn_idx)), int(np.median(list(map(len, nose.orn_idx)))), max(map(len, nose.orn_idx)))
# other sensory neurons: counts by class
os_ = reg["other_sensory"]; print("other sensory (no input):", len(os_), pd.Series(cls[os_]).value_counts().head(8).to_dict())
print("hygro/thermo classes in 'orn' region?", pd.Series(cls[reg["orn"]]).value_counts().to_dict())

# ---- unit function
for beta in (50.0, 500.0):
    R3.BETA = beta
    x = torch.tensor([-0.05, -0.01, 0.0, 0.01, 0.05, 0.5])
    xg = x.clone().requires_grad_(True); y = R3.RateBrain3.unit_fn(xg); y.sum().backward()
    print(f"BETA {beta}: f(x) {dict(zip(x.tolist(), y.detach().numpy().round(6).tolist()))}  f'(x) {xg.grad.numpy().round(8).tolist()}")
R3.BETA = 500.0
x_act = float(np.log(np.expm1(500 * np.arctanh(0.01))) / 500); print("x at ACTIVE (BETA 500):", x_act)
print("f(+inf) upper bound tanh(x) -> rate 0.3 needs x =", float(np.arctanh(0.3)))
