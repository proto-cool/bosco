"""Characterise oscillation at rest (and one sniff) vs the Euler step alpha = dt/tau. usage: osc.py DESIGN G alphas [beta]"""
import sys, json, numpy as np, torch
from collections import Counter
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
D = sys.argv[1]; G = float(sys.argv[2]); alphas = [float(x) for x in sys.argv[3].split(',')]
m, rest, SM, ann, b2 = load.load(D, G)
if len(sys.argv) > 4:
    sys.modules[type(m).__module__].BETA = float(sys.argv[4])
typ = ann['type'].fillna('?').to_numpy().astype(str)
ap, av = m.read_groups[m.read]
out = {}
for inp_name, s_ in (('rest', rest), ('smell', SM[12])):
    b = dyn.bias_vec(m, s_)
    for a in alphas:
        al = torch.full((m.n,), a)
        T = int(round(15000 / (5 * a / 0.25)))  # 15 s of model time? steps of dt=5ms*a/0.25 -> keep 3000 steps at a=0.25
        T = int(3000 * 0.25 / a)
        r = (m.r_rest.clone() if m.r_rest is not None else torch.zeros(m.n))
        with torch.no_grad():
            for t in range(T): r = dyn.F(m, r, b, al)
            L = int(800 * 0.25 / a); tail = torch.zeros(L, m.n); dd = []
            for t in range(L):
                rn = dyn.F(m, r, b, al); dd.append(rn - r); r = rn; tail[t] = r
        amp = (tail.max(0).values - tail.min(0).values).numpy()
        osc = amp > 0.01
        rd = (tail[:, ap].mean(1) - tail[:, av].mean(1)).numpy()
        # lag-1 correlation of successive differences (≈ -1 means period-2 flip)
        d1 = torch.stack(dd[-50:])
        c = float(torch.nn.functional.cosine_similarity(d1[:-1], d1[1:], dim=1).mean())
        # dominant period from the most oscillating cell and the read
        def period(x):
            x = x - x.mean()
            if np.abs(x).max() < 1e-7: return None
            f = np.abs(np.fft.rfft(x)); f[0] = 0; k = int(np.argmax(f))
            return None if k == 0 else float(len(x) / k * 5 * a / 0.25)  # ms
        j = int(np.argmax(amp))
        out[f'{inp_name}_a{a:g}'] = {'steps': T, 'max_step_change_tail': float(torch.stack(dd).abs().max()),
            'n_cells_amp_gt_0.01': int(osc.sum()), 'max_amp': float(amp.max()), 'read_amp': float(rd.max() - rd.min()),
            'read_mean': float(rd.mean()),
            'succ_diff_cos': c, 'period_ms_top_cell': period(tail[:, j].numpy()), 'period_ms_read': period(rd),
            'top_types': Counter(typ[osc]).most_common(12)}
        print(inp_name, a, json.dumps({k: v for k, v in out[f'{inp_name}_a{a:g}'].items()}, default=str), flush=True)
json.dump(out, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/osc_{D}_G{G:g}{"_b"+sys.argv[4] if len(sys.argv)>4 else ""}.json', 'w'), indent=1, default=str)
