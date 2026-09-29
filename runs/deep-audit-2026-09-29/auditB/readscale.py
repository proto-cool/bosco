import sys, json, numpy as np, torch
from collections import Counter
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
m, rest, SM, ann, b2 = load.load('C1', 4)
typ = ann['type'].fillna('?').to_numpy().astype(str)
ap, av = m.read_groups['dn']; rd = torch.cat([ap, av])
print('read s_in by type', {t: round(float(m.s_in[rd][typ[rd.numpy()] == t].mean()), 2) for t in sorted(set(typ[rd.numpy()]))})
print('KC s_in', float(m.s_in[m.kc].mean()), 'MBON s_in', float(m.s_in[m.regions['mbon']].mean()))
a1 = torch.full((m.n,), .25)
def test(tag):
    m.freeze()
    for nm, s_ in (('rest', rest), ('smell', SM[12])):
        b = dyn.bias_vec(m, s_)
        r, d, _ = dyn.run_long(m, b, m.r_rest, 3000, a1)
        lo = r.clone(); hi = r.clone(); rr = r.clone()
        with torch.no_grad():
            for t in range(400):
                rr = dyn.F(m, rr, b, a1); lo = torch.minimum(lo, rr); hi = torch.maximum(hi, rr)
        amp = (hi - lo).numpy()
        print(tag, nm, 'last200 max step', float(d[-200:].max()), 'osc cells', int((amp > .01).sum()), Counter(typ[amp > .01]).most_common(5),
              'read', float(r[ap].mean() - r[av].mean()), 'MDN rate', r[av].numpy().round(3))
test('saved C1')
s0 = m.s_in.clone()
with torch.no_grad(): m.s_in[rd] = 4.0
test('read cells s_in=G')
with torch.no_grad(): m.s_in.copy_(s0); m.s_in[av] = 4.0
test('only MDN s_in=G')
with torch.no_grad(): m.s_in.copy_(s0); m.s_in[ap] = 4.0
test('only approach s_in=G')
