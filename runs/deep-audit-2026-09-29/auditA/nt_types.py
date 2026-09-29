import numpy as np, pandas as pd, json
from bosco import data, model2 as M2, wiring3 as W3
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 30)
OUT = "/Users/nickd/.claude/jobs/4059153c/tmp/auditA/"
df = pd.read_pickle(OUT + "neurons.pkl")
a = data.annotations().reindex(df.id)
df["fwtype"] = a["flywireType"].to_numpy()
df["sub"] = a["subclass"].to_numpy()
fw = data.flywire_annotations()
fwt = fw.groupby("cell_type").agg(fw_top=("top_nt", lambda s: s.mode().iat[0] if s.notna().any() else None),
                                  fw_conf=("top_nt_conf", "median"),
                                  fw_known=("known_nt", lambda s: s.dropna().iat[0] if s.notna().any() else None))
# hemibrain_type fallback
fwh = fw.groupby("hemibrain_type").agg(fw_top=("top_nt", lambda s: s.mode().iat[0] if s.notna().any() else None))
def fwlook(r):
    for key in (r.fwtype, r.type):
        if isinstance(key, str) and key in fwt.index:
            return fwt.loc[key, "fw_top"], fwt.loc[key, "fw_known"]
    return None, None
look = [fwlook(r) for r in df.itertuples()]
df["fw_top"] = [x[0] for x in look]; df["fw_known"] = [x[1] for x in look]

def show(title, m):
    s = df[m].groupby(["type", "lab31", "path31", "sign30", "sign31", "fw_top", "fw_known"], dropna=False).agg(
        n=("id", "size"), out_cut=("out_cut", "sum"), pconf=("pconf", "median")).reset_index()
    print(f"\n== {title}\n", s.to_string())

# 1. neurons whose v3.1 label came from 'predicted'/'celltype' and is serotonin: FlyWire agreement
m5 = (df.lab31 == "serotonin") & df.path31.isin(["predicted", "celltype"])
s = df[m5]
print("predicted/celltype serotonin:", len(s), "FlyWire top_nt for their types:", s.fw_top.value_counts(dropna=False).to_dict(),
      "known_nt:", s.fw_known.value_counts(dropna=False).to_dict())
print(s.groupby("type").agg(n=("id", "size"), out=("out_cut", "sum"), pconf=("pconf", "median"), fw=("fw_top", "first"), fwk=("fw_known", "first")).sort_values("out", ascending=False).head(30).to_string())
# consensus serotonin
s = df[(df.lab31 == "serotonin") & (df.path31 == "consensus")]
print("consensus serotonin types:", s.groupby("type").size().to_dict(), "fw:", s.fw_top.value_counts(dropna=False).to_dict())
# unclear
u = df[df.lab31.isin(["unclear", "unknown"])]
print("\nno-transmitter neurons:", len(u), "FlyWire top_nt for types:", u.fw_top.value_counts(dropna=False).to_dict(),
      "their out_cut by fw_top:", u.groupby(u.fw_top.fillna("none")).out_cut.sum().to_dict())
print(u.groupby(["cls"]).agg(n=("id", "size"), out=("out_cut", "sum")).sort_values("out", ascending=False).head(12).to_string())
# predicted path other labels vs FlyWire
p = df[df.path31 == "predicted"]
print("\npredicted-path label vs FlyWire top (neurons):\n", pd.crosstab(p.lab31, p.fw_top.fillna("none")).to_string())
# whole-model agreement with FlyWire by path
for path in ("consensus", "predicted", "celltype"):
    q = df[(df.path31 == path) & df.fw_top.notna()]
    print(path, "agree with FlyWire top_nt:", round(float((q.lab31 == q.fw_top).mean()), 3), "n", len(q),
          " fast-sign agree:", round(float((W3._fast_sign(q.lab31.reset_index(drop=True)) == W3._fast_sign(q.fw_top.reset_index(drop=True))).mean()), 3))
q = df[df.fw_top.notna() & ~df.lab31.isin(["unclear", "unknown"])]
dis = q[q.lab31 != q.fw_top]
print("disagreeing (label vs FlyWire) neurons:", len(dis), "out syn", int(dis.out_cut.sum()))
print(dis.groupby(["lab31", "fw_top"]).agg(n=("id", "size"), out=("out_cut", "sum")).sort_values("out", ascending=False).head(15).to_string())
print(dis.groupby(["type", "lab31", "fw_top", "path31"]).agg(n=("id", "size"), out=("out_cut", "sum")).sort_values("out", ascending=False).head(30).to_string())

# 2. famous types
T = df.type
fam = {
    "APL": T == "APL", "DPM": T == "DPM",
    "OA-VUM": T.str.startswith("OA-VUM"), "MDN": T == "MDN", "DNp09": T == "DNp09", "GF (DNp01)": T.isin(["DNp01", "GF"]),
    "PFR": T.str.startswith("PFR"), "PFN": T.str.startswith("PFN"), "DANs(class)": df.cls == "DAN",
}
for k, m in fam.items():
    show(k, m.to_numpy())
# PNs: uni vs multi-glomerular by type name
pn = df.cls == "ALPN"
print("\n== ALPN by label x path\n", df[pn].groupby(["lab31", "path31"]).agg(n=("id", "size"), out=("out_cut", "sum")).to_string())
ptype = df[pn].type
multi = ptype.str.startswith(("M_", "mPN", "M_l", "M_v", "M_s")) | ptype.str.contains("PN_", regex=False)
print("ALPN type prefixes:", ptype.str.extract(r"^([A-Za-z]+)")[0].value_counts().head(15).to_dict())
print("ALPN GABA types:", df[pn & (df.lab31 == "gaba")].type.value_counts().head(30).to_dict())
print("ALPN ACh 'uni' prefixed?:", df[pn & (df.lab31 == "acetylcholine")].type.str.extract(r"^([A-Za-z]+_?[a-z]*)")[0].value_counts().head(10).to_dict())
# LNs
ln = df.cls == "ALLN"
print("\n== ALLN label x path\n", df[ln].groupby(["lab31", "path31"]).agg(n=("id", "size"), out=("out_cut", "sum")).to_string())
print("ALLN no-fast (unclear) types with FlyWire:", df[ln & df.lab31.isin(["unclear", "unknown"])].groupby(["type", "fw_top"], dropna=False).agg(n=("id", "size"), out=("out_cut", "sum")).sort_values("out", ascending=False).to_string())
print("ALLN silenced (ACh) v3.1:", df[ln & (df.lab31 == "acetylcholine")].groupby(["type", "fw_top"], dropna=False).agg(n=("id", "size"), out=("out_cut", "sum")).to_string())
df.to_pickle(OUT + "neurons_fw.pkl")

# 3. MBON valence: v3.0 by DAN input, v3.1 by transmitter
b2 = M2.load_or_build()
tab0, ap0, av0 = M2.mbon_groups(b2.brain)
ap1, av1, tab1 = W3.mbon_valence(b2.brain)
mm = tab0.merge(tab1[["type", "nt", "side"]], on="type", suffixes=("_v30", "_v31"))
mm["hb"] = [a.reindex(df.id[df.type == t])["hemibrainType"].dropna().mode().iat[0] if (df.type == t).any() and a.reindex(df.id[df.type == t])["hemibrainType"].notna().any() else "" for t in mm.type]
print("\n== MBON valence\n", mm.to_string())
print("approach cells v3.0/v3.1:", len(ap0), len(ap1), " avoid:", len(av0), len(av1))
mm.to_csv(OUT + "mbon_valence.csv", index=False)
