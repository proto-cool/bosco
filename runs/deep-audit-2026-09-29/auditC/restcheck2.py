import sys, collections; sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *
m, rest, ctx = load(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "real")
typ = types(ctx)
X, S = smells_items(ctx["B"], 96)
r0 = m.r_rest.clone(); _, d0 = sniff(m, S)
m.r_rest = None; st = m.settle(rest); r2 = m.r_rest.clone(); _, d2 = sniff(m, S)
diff = (r0 - r2).abs() > 0.01
print("settle", st, "cells differ >0.01:", int(diff.sum()))
print({k: int(diff[v].sum()) for k, v in m.regions.items()})
print("top types", collections.Counter(typ[diff.numpy()]).most_common(12))
print("read mean saved %.4e fresh %.4e ; item read corr %.4f ; sd %.2e %.2e" % (d0.mean(), d2.mean(), np.corrcoef(d0, d2)[0,1], d0.std(), d2.std()))
# zero-start vs 5 random starts
fins = []
for s in range(3):
    g = torch.Generator().manual_seed(s); m.r_rest = torch.rand(m.n, generator=g) * 0.5; m.settle(rest); fins.append(m.r_rest.clone())
print("random starts: cells differing from saved >0.01:", [int(((f - r0).abs() > 0.01).sum()) for f in fins], "from fresh:", [int(((f - r2).abs() > 0.01).sum()) for f in fins])
