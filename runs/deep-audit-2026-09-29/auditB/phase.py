"""Does the served answer depend on the phase of an oscillating rest? usage: phase.py DESIGN G"""
import sys, json, numpy as np, torch
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
D = sys.argv[1]; G = float(sys.argv[2])
m, rest, SM, ann, b2 = load.load(D, G)
ap, av = m.read_groups[m.read]
a1 = torch.full((m.n,), .25); b = dyn.bias_vec(m, rest)
r = m.r_rest.clone() if m.r_rest is not None else torch.zeros(m.n)
r, _, _ = dyn.run_long(m, b, r, 1000, a1)
starts = []
with torch.no_grad():
    for k in range(40):
        if k % 4 == 0: starts.append(r.clone())
        r = dyn.F(m, r, b, a1)
NI = 64; S = SM[400:400+NI]
Bm = torch.stack([dyn.bias_vec(m, s_) for s_ in S], 1)
reads = []
for r0 in starts:
    R = r0[:, None].repeat(1, NI); acc = 0
    with torch.no_grad():
        for t in range(80):
            R = dyn.F(m, R, Bm, a1[:, None])
            if t >= 72: acc = acc + (R[ap].mean(0) - (R[av].mean(0) if len(av) else 0)) / 8
    reads.append(acc.numpy())
reads = np.array(reads)  # (phases, items)
item_sd = reads.mean(0).std()
res = {'n_phases': len(starts), 'read_item_sd': float(item_sd), 'phase_sd_per_item_median': float(np.median(reads.std(0))),
       'phase_sd_over_item_sd': float(np.median(reads.std(0)) / item_sd),
       'min_rank_corr_between_phases': float(min(np.corrcoef(reads)[np.triu_indices(len(starts), 1)])),
       'max_abs_read_change': float((reads.max(0) - reads.min(0)).max()),
       'sign_flips_centered_share': float((np.sign(reads - np.median(reads, 1, keepdims=True)) != np.sign(reads[0] - np.median(reads[0]))).any(0).mean())}
print(json.dumps(res))
json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/phase_{D}_G{G:g}.json', 'w'))
