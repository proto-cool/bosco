import sys, json, numpy as np, torch
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
D = sys.argv[1]; G = float(sys.argv[2]) if len(sys.argv) > 2 else 4
m, rest, SM, ann, b2 = load.load(D, G)
a1 = dyn.alpha_vec(m); b = dyn.bias_vec(m, rest)
ap, av = m.read_groups[m.read]
rR = m.r_rest if m.r_rest is not None else torch.zeros(m.n)
g = torch.Generator().manual_seed(7)
inits = {'rest': rR, 'zero': torch.zeros(m.n), 'rand': torch.rand(m.n, generator=g), 'sat': torch.full((m.n,), .99)}
R = torch.stack(list(inits.values()), 1)
with torch.no_grad():
    for t in range(3000): R = dyn.F(m, R, b[:, None], a1[:, None])
cls = ann['class'].fillna('?').to_numpy().astype(str); typ = ann['type'].fillna('?').to_numpy().astype(str)
sup = ann['superclass'].fillna('?').to_numpy().astype(str)
for j, nm in enumerate(list(inits)[1:], 1):
    dif = (R[:, j] - R[:, 0]).numpy(); big = np.abs(dif) > 0.01
    print(nm, 'n diff', big.sum(), 'up', (dif > 0.01).sum(), 'down', (dif < -0.01).sum())
    from collections import Counter
    print('  class', Counter(cls[big]).most_common(8))
    print('  types', Counter(typ[big]).most_common(12))
    print('  read cells differing', int(big[ap.numpy()].sum()), int(big[av.numpy()].sum()))
# 'up' cells in zero-init: are they the ones at rest ~0?
dif = (R[:, 1] - R[:, 0]).numpy(); big = np.abs(dif) > 0.01
print('rest-state rates of differing cells (quantiles)', np.quantile(R[big, 0].numpy(), [0, .25, .5, .75, 1]).round(3))
print('zero-state rates of differing cells (quantiles)', np.quantile(R[big, 1].numpy(), [0, .25, .5, .75, 1]).round(3))
# pairwise distinct attractors among the 4
for i in range(4):
    print([round(float((R[:, i]-R[:, j]).abs().max()), 3) for j in range(4)])
torch.save(R, f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/multi_{D}.pt')
