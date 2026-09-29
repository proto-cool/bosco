"""Audit A: wiring and data rules, v3.0 vs v3.1. Read-only."""
import json
import numpy as np, pandas as pd
from bosco import data, model2 as M2, wiring3 as W3, populations as pop, v1
from bosco.model import AL_LN_PREFIXES, CB_SUPERCLASSES

OUT = "/Users/nickd/.claude/jobs/4059153c/tmp/auditA/"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
a = data.annotations(); nt = data.neurotransmitters()
b2 = M2.load_or_build(); b = b2.brain; ids = b.ids; n = b.n
print("model n", n, "ids equal fresh", np.array_equal(ids, M2.model_body_ids()))
am = a.reindex(ids)
cls = am["class"].fillna(am["superclass"]).fillna("?").to_numpy().astype(str)
typ = am["type"].fillna("").to_numpy().astype(str)

# ---- selection: what's excluded that matters
print("\n== selection")
nonT = a[a.superclass.isin(M2.V2_SUPERCLASSES) & (a.status != "Traced")]
print("model-superclass bodies not Traced:", nonT.status.value_counts(dropna=False).to_dict())
print(" of which typed:", nonT.type.notna().sum(), nonT[nonT.type.notna()].type.value_counts().head(10).to_dict())
tr_other = a[(a.status == "Traced") & ~a.superclass.isin(M2.V2_SUPERCLASSES)]
print("Traced excluded by superclass:", tr_other.superclass.value_counts(dropna=False).to_dict())
print(" NaN-superclass traced, class:", tr_other[tr_other.superclass.isna()]["class"].value_counts(dropna=False).head(8).to_dict(),
      "types e.g.", tr_other[tr_other.superclass.isna()].type.dropna().unique()[:10])

# ---- transmitter paths
print("\n== transmitter paths (model neurons)")
ntm = nt.reindex(pd.Index(ids))
print("model ids missing from NT table:", int(ntm.body.isna().sum()), "classes:", pd.Series(cls[ntm.body.isna().to_numpy()]).value_counts().to_dict())
cons = ntm.consensus_nt.str.lower(); pred = ntm.predicted_nt.str.lower(); ctp = ntm.celltype_predicted_nt.str.lower()
uns = lambda s: s.isna() | s.isin(W3.UNSURE)
path31 = np.where(~uns(cons), "consensus", np.where(~uns(pred), "predicted", np.where(~uns(ctp), "celltype", "unknown")))
lab31 = W3.transmitter(ids).to_numpy()
path31[typ == "DPM"] = "override"
sign30, src30 = M2.signs(ids)
assert np.array_equal(sign30, b.nt_sign), "cache nt_sign != fresh signs"
pre = b.pre_of_edges(); post = b.indices; cnt = b.count.astype(float)
out_syn = np.bincount(pre, weights=cnt, minlength=n)
keep_cut = (b.count >= 5) | (np.isin(np.arange(n), b.index_of_present(pop.kenyon_cells()))[pre] & np.isin(np.arange(n), b.index_of_present(pop.mbons()["bodyId"]))[post])
out_cut = np.bincount(pre[keep_cut], weights=cnt[keep_cut], minlength=n)
df = pd.DataFrame({"id": ids, "type": typ, "cls": cls, "src30": src30, "sign30": sign30, "path31": path31, "lab31": lab31,
                   "sign31": W3._fast_sign(pd.Series(lab31)), "pconf": ntm.predicted_nt_confidence.to_numpy(),
                   "cconf": ntm.celltype_predicted_nt_confidence.to_numpy(), "out": out_syn, "out_cut": out_cut})
print(df.groupby("src30").agg(neurons=("id", "size"), out_cut=("out_cut", "sum")).to_string())
g = df.groupby(["path31", "lab31"]).agg(neurons=("id", "size"), out_cut=("out_cut", "sum"), pconf_med=("pconf", "median"), cconf_med=("cconf", "median"))
print(g.to_string())
# celltype fallback: which types
cf = df[df.path31 == "celltype"]
print("\ncelltype-fallback neurons:", len(cf), "labels", cf.lab31.value_counts().to_dict())
print(cf.groupby(["type", "lab31"]).agg(n=("id", "size"), out_cut=("out_cut", "sum"), cconf=("cconf", "median")).sort_values("out_cut", ascending=False).head(40).to_string())
# how confident are the 'predicted' path calls
pp = df[df.path31 == "predicted"]
print("\npredicted-path conf quantiles", np.nanquantile(pp.pconf, [0.1, 0.25, 0.5, 0.75]).round(3), "n<0.5:", int((pp.pconf < 0.5).sum()))
# serotonin totals
for lab in ("serotonin", "octopamine", "dopamine", "histamine"):
    s = df[df.lab31 == lab]
    print(f"v3.1 {lab}: {len(s)} neurons, by path {s.path31.value_counts().to_dict()}")
df.to_pickle(OUT + "neurons.pkl")

# ---- sign changes per type, v3.0 fast sign (with v3.0 wiring fixes) vs v3.1
print("\n== sign changes by type (out synapses after the cut)")
ch = df[df.sign30 != df.sign31]
t = ch.groupby(["type", "cls", "lab31", "sign30", "sign31"]).agg(n=("id", "size"), out_cut=("out_cut", "sum")).reset_index()
print("neurons changing sign:", len(ch), "out_cut syn", int(ch.out_cut.sum()), "of", int(df.out_cut.sum()))
print(t.groupby(["sign30", "sign31", "lab31"]).agg(neurons=("n", "sum"), syn=("out_cut", "sum")).to_string())
t = t.sort_values("out_cut", ascending=False)
t.to_csv(OUT + "sign_changes_by_type.csv", index=False)
print(t.head(60).to_string())
# changes other than 'to 0 because monoamine'
nz = t[(t.sign31 != 0)]
print("\nnonzero-to-other-nonzero or 0->nonzero changes:\n", nz.to_string())

# ---- edge-level fast rule
ft = W3.fast_tables(ids)
kc = ft["kc"]
print("\ncache ids ok", np.array_equal(ft["ids"], ids), "cache label==fresh", np.array_equal(ft["label"], lab31.astype("U16")))
ed31 = W3.edge_drives(b, ft)
# v3.0 zeroing rules
eln30 = np.array([x.startswith(AL_LN_PREFIXES) for x in typ]) & (b.nt_sign > 0)
dan = np.isin(ids, pop.dans()["bodyId"])
ed30 = ~eln30[pre] & ~(dan[pre] & kc[post])
reason = np.full(b.nnz, "", object)
lab_pre = lab31[pre]
reason[~ed31 & np.isin(lab_pre, ["dopamine", "octopamine", "serotonin"])] = "monoamine"
reason[~ed31 & (lab_pre == "unknown")] = "unknown"
reason[~ed31 & kc[pre] & kc[post]] = "KC->KC"
reason[~ed31 & ft["silenced"][pre]] = "AL ACh LN"
lost = ed30 & ~ed31 & keep_cut
gained = ~ed30 & ed31 & keep_cut
print("\n== fast drive removed in v3.1 (cut edges that drove in v3.0):", int(cnt[lost].sum()), "syn of", int(cnt[keep_cut & ed30].sum()))
L = pd.DataFrame({"reason": reason[lost], "pre_cls": cls[pre[lost]], "post_cls": cls[post[lost]], "syn": cnt[lost]})
print(L.groupby("reason").syn.sum().to_string())
print(L.groupby(["reason", "pre_cls"]).syn.sum().sort_values(ascending=False).head(25).to_string())
print("by post class (share of that class's v3.0 fast in-model cut input lost):")
tot_post = pd.Series(cnt[keep_cut & ed30]).groupby(cls[post[keep_cut & ed30]]).sum()
lp = L.groupby("post_cls").syn.sum()
print((lp / tot_post).dropna().sort_values(ascending=False).head(25).round(3).to_string())
print("edges re-enabled in v3.1 (v3.0 zeroed):", int(cnt[gained].sum()), pd.Series(cnt[gained]).groupby(typ[pre[gained]]).sum().sort_values(ascending=False).head(10).to_dict())

# ---- input totals
print("\n== input totals")
it30 = b2.in_total; it31 = ft["in_fast"]
in_model = np.bincount(post, weights=cnt, minlength=n)
in_model_cut = np.bincount(post[keep_cut], weights=cnt[keep_cut], minlength=n)
fast_model_cut = np.bincount(post[keep_cut & ed31], weights=cnt[keep_cut & ed31], minlength=n)
fast_model = np.bincount(post[ed31], weights=cnt[ed31], minlength=n)
ratio = it31 / np.maximum(it30, 1)
print("neurons with in_fast < 0.5 in_total:", int((ratio < 0.5).sum()), " <0.25:", int((ratio < 0.25).sum()), " ==0:", int((it31 == 0).sum()), "in_total==0:", int((it30 == 0).sum()))
T = pd.DataFrame({"cls": cls, "type": typ, "it30": it30, "it31": it31, "ratio": ratio,
                  "out30": 1 - in_model / np.maximum(it30, 1), "out31": 1 - fast_model / np.maximum(it31, 1),
                  "kept30": in_model_cut / np.maximum(it30, 1), "kept31": fast_model_cut / np.maximum(it31, 1)})
print(T[T.ratio < 0.5].cls.value_counts().head(20).to_string())
print(T[T.ratio < 0.5].groupby("type").size().sort_values(ascending=False).head(25).to_dict())
agg = T.groupby("cls").agg(n=("cls", "size"), ratio_med=("ratio", "median"), outside30=("out30", "median"), outside31=("out31", "median"),
                           kept30=("kept30", "median"), kept31=("kept31", "median"))
print(agg[agg.n >= 10].round(3).sort_values("n", ascending=False).to_string())
T.to_pickle(OUT + "totals.pkl")

# decomposition of the lost denominator: synapses from all bodies onto model neurons, by pre status/label
w = data.weights()
idx = pd.Index(ids)
pp_ = idx.get_indexer(w["body_post"].to_numpy()); m_ = pp_ >= 0
wb = w[m_]; ppost = pp_[m_]
bodies = np.unique(wb.body_pre.to_numpy())
labb = W3.transmitter(bodies).to_numpy()
pi = np.searchsorted(bodies, wb.body_pre.to_numpy())
inmodel = idx.get_indexer(wb.body_pre.to_numpy()) >= 0
st = a.reindex(bodies)["status"].fillna("none").to_numpy()
sup = a.reindex(bodies)["superclass"].fillna("none").to_numpy()
D = pd.DataFrame({"src": np.where(inmodel, "model", np.where(st[pi] == "Traced", "traced:" + sup[pi].astype(str), "untraced:" + st[pi].astype(str))),
                  "lab": labb[pi], "w": wb.weight.to_numpy()})
dd = D.groupby(["src"]).w.sum().sort_values(ascending=False)
print("\nall synapses onto model neurons by source:", (dd / dd.sum()).round(4).head(15).to_dict())
dl = D.groupby(["src", "lab"]).w.sum()
print("label mix of untraced sources:", (D[D.src.str.startswith("untraced")].groupby("lab").w.sum() / D[D.src.str.startswith("untraced")].w.sum()).round(3).to_dict())
print("label mix of model sources:", (D[D.src == "model"].groupby("lab").w.sum() / D[D.src == "model"].w.sum()).round(4).to_dict())
print("share of v3.0 in_total from untraced fragments (all):", round(float(D[D.src.str.startswith('untraced')].w.sum() / D.w.sum()), 4))
out_fast_frag = D[D.src.str.startswith("untraced") & D.lab.isin(["acetylcholine", "gaba", "glutamate", "histamine"])].w.sum()
print("share of v3.1 in_fast from untraced fragments:", round(float(out_fast_frag / it31.sum()), 4))
# per-class outside split
Dp = pd.DataFrame({"cls": cls[ppost], "src": D.src.str.split(":").str[0].to_numpy(), "w": D.w.to_numpy(),
                   "fast": D.lab.isin(["acetylcholine", "gaba", "glutamate", "histamine"]).to_numpy()})
pc = Dp.groupby(["cls", "src"]).w.sum().unstack(fill_value=0)
pc = pc.div(pc.sum(1), axis=0)
print("per-class share of all input by source (v3.0 denominators):\n", pc.loc[agg[agg.n >= 20].index].round(3).to_string())
json.dump({"n_ratio_lt_half": int((ratio < 0.5).sum())}, open(OUT + "wiring.json", "w"))
