import sys
sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *  # noqa
import collections

ver, arm = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "real"
t = T()
m, rest, ctx = load(ver, arm)
t("loaded")
typ = types(ctx)
mb = m.regions["mbon"].numpy()
print("MBON types", collections.Counter(typ[mb]))
for nm in ["APL", "DPM", "MDN", "LC16", "LC4", "DNp01", "DNp02", "DNp11", "DNp42", "DNa13", "DNa02", "LC6", "LPLC2", "LPLC1"]:
    print(nm, int((typ == nm).sum()))
print("regions", {k: len(v) for k, v in m.regions.items()})
print("glomeruli", m.nose.glomeruli)
pn = m.regions["alpn"].numpy()
print("alpn types (n types)", len(set(typ[pn])), list(collections.Counter(typ[pn]).most_common(15)))
print("kp synapses", len(m.kp_logm), "read", [len(x) for x in m.read_groups[m.read]])
X, S = smells_items(ctx["B"], 64)
rr, d = sniff(m, S)
t("sniff 64")
print("read sd", float(d.std()), "rest read", rest_read(m))
