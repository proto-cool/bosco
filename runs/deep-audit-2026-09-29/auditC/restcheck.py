import sys; sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *
m, rest, ctx = load(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "real")
r0 = m.r_rest.clone(); a = rest_read(m)
st = m.settle(rest); b = rest_read(m); r1 = m.r_rest.clone()
m.r_rest = None; st2 = m.settle(rest); c = rest_read(m); r2 = m.r_rest.clone()
print("saved", a, "continued", b, st, "fresh", c, st2)
print("max|saved-cont|", float((r0-r1).abs().max()), "max|saved-fresh|", float((r0-r2).abs().max()), "n cells differ>0.01", int(((r0-r2).abs()>0.01).sum()))
d = (r0-r2).abs(); top = torch.topk(d, 10).indices; typ = types(ctx); print([(typ[i], float(r0[i]), float(r2[i])) for i in top.tolist()])
