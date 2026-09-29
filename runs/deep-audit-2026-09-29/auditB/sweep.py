"""Stability map at a fixed calibration: scale non-sensory input rows (global gain) by k, BETA, alpha.
usage: sweep.py DESIGN G ks betas alphas [tag]"""
import sys, json, numpy as np, torch
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
D = sys.argv[1]; G = float(sys.argv[2])
ks = [float(x) for x in sys.argv[3].split(',')]; betas = [float(x) for x in sys.argv[4].split(',')]
alphas = [float(x) for x in sys.argv[5].split(',')]; tag = sys.argv[6] if len(sys.argv) > 6 else ''
m, rest, SM, ann, b2 = load.load(D, G)
mod = sys.modules[type(m).__module__]; beta0 = mod.BETA
n = m.n; W0 = m._W_all
sens = torch.cat([m.regions['orn'], m.regions['other_sensory'], m.regions['vpn']])
crow = W0.crow_indices(); rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])
ap, av = m.read_groups[m.read]
r_start = m.r_rest.clone() if m.r_rest is not None else torch.zeros(n)
typ = ann['type'].fillna('?').to_numpy().astype(str)
out = []
for beta in betas:
    mod.BETA = beta
    for k in ks:
        rs = torch.full((n,), k); rs[sens] = 1.0
        m._W_all = torch.sparse_csr_tensor(crow, W0.col_indices(), W0.values() * rs[rows], (n, n))
        for a in alphas:
            al = torch.full((n,), a)
            rec = {'beta': beta, 'k': k, 'alpha': a}
            for nm, s_ in (('rest', rest), ('smell', SM[12])):
                b = dyn.bias_vec(m, s_)
                T = int(3000 * 0.25 / a)
                r, d, _ = dyn.run_long(m, b, r_start, T, al)
                # amplitude over a 400-step tail
                with torch.no_grad():
                    lo = r.clone(); hi = r.clone(); rr = r.clone()
                    for t in range(int(400 * 0.25 / a)):
                        rr = dyn.F(m, rr, b, al); lo = torch.minimum(lo, rr); hi = torch.maximum(hi, rr)
                amp = hi - lo
                rec[nm] = {'max_step_change_last200': float(d[-200:].max()), 'n_osc_cells': int((amp > 0.01).sum()),
                           'mean_rate': float(r.mean()), 'share_sat': float((r > 0.9).float().mean()),
                           'share_dead': float((r < 1e-3).float().mean()),
                           'read': float(r[ap].mean() - r[av].mean())}
                J, fp, x = dyn.jacobian(m, r, b, al)
                try:
                    v, vec = dyn.top_eigs(J, k=4)
                    lr, lvec = dyn.rightmost_M(J, a, k=3)
                    reM = float((lr[0].real - 1 + a) / a)
                    w = np.abs(lvec[:, 0]) ** 2; top = np.argsort(-w)[:40]
                    from collections import Counter
                    tt = Counter()
                    for i in top: tt[typ[i]] += w[i]
                    rec[nm].update({'rho': float(abs(v[0])), 'mu_top': [float(v[0].real), float(v[0].imag)],
                                    'maxRe_M': reM, 'imM': float(lr[0].imag / a),
                                    'lead_types': [(t, round(float(x), 3)) for t, x in tt.most_common(6)],
                                    'participation': float(1 / ((w / w.sum()) ** 2).sum())})
                except Exception as e:
                    rec[nm]['eig_err'] = str(e)[:80]
            print(json.dumps(rec), flush=True)
            out.append(rec)
mod.BETA = beta0; m._W_all = W0
json.dump(out, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/sweep_{D}_G{G:g}{tag}.json', 'w'), indent=1)
