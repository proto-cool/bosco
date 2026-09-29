"""C1/C3 procedure (stage 1 only, no read tuning) at other G, small scale: 64 explore smells, 20 iterations.
Then stability metrics. usage: calib_sweep.py DESIGN G1,G2,..."""
import sys, json, time, numpy as np, torch
from collections import Counter
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB'); sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp')
sys.path.insert(0, '/Users/nickd/projects/bosco/scripts')
import candidates as C, dyn
import brain_check as B
from bosco import model2 as M2, data
D = sys.argv[1]; Gs = [float(x) for x in sys.argv[2].split(',')]
b2 = M2.load_or_build()
ant = B.antenna(); rest = torch.tensor(ant.resting()).float()
X = np.load('/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz')['X']
SM = torch.tensor(ant(X)).float(); cal = SM[:64]
typ = data.annotations().reindex(b2.brain.ids)['type'].fillna('?').to_numpy().astype(str)
out = {}
for G in Gs:
    t0 = time.time()
    m = C.build_model(D, b2)
    info = C.calibrate(m, cal, rest, G, iters=20)
    torch.save({'s_in': m.s_in.clone(), 'b_cell': m.b_cell.clone(), 'r_rest': m.r_rest}, f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/cal_{D}_G{G:g}.pt')
    a1 = torch.full((m.n,), 0.25)
    rec = {'calib': info, 'calib_s': time.time() - t0}
    for nm, s_, r0 in (('rest', rest, m.r_rest), ('smell', SM[300], m.r_rest), ('rest_from0', rest, torch.zeros(m.n)),
                       ('rest_from_sat', rest, torch.full((m.n,), .99))):
        b = dyn.bias_vec(m, s_)
        r, d, _ = dyn.run_long(m, b, r0, 3000, a1)
        with torch.no_grad():
            lo = r.clone(); hi = r.clone(); rr = r.clone()
            for t in range(400):
                rr = dyn.F(m, rr, b, a1); lo = torch.minimum(lo, rr); hi = torch.maximum(hi, rr)
        amp = hi - lo
        e = {'max_step_change_last200': float(d[-200:].max()), 'n_osc_cells': int((amp > .01).sum()),
             'osc_types': Counter(typ[(amp > .01).numpy()]).most_common(6), 'mean_rate': float(r.mean()),
             'share_sat(>0.9)': float((r > .9).float().mean()), 'share_dead(<1e-3)': float((r < 1e-3).float().mean())}
        if nm == 'rest': r_rest_final = r
        if nm in ('rest_from0', 'rest_from_sat'):
            e['maxabs_vs_rest'] = float((r - r_rest_final).abs().max()); e['share_diff'] = float(((r - r_rest_final).abs() > .01).float().mean())
        if nm in ('rest', 'smell'):
            J, fp, x = dyn.jacobian(m, r, b, a1)
            v, _ = dyn.top_eigs(J, k=4); lr, lv = dyn.rightmost_M(J, .25, k=3)
            w = np.abs(lv[:, 0]) ** 2; top = np.argsort(-w)[:40]; tt = Counter()
            for i in top: tt[typ[i]] += w[i] / w.sum()
            e.update({'rho': float(abs(v[0])), 'mu': [float(v[0].real), float(v[0].imag)], 'maxRe_M': float((lr[0].real - .75) / .25),
                      'imM': float(lr[0].imag / .25), 'lead_types': [(t, round(float(q), 3)) for t, q in tt.most_common(5)]})
        rec[nm] = e
    out[G] = rec
    print(G, json.dumps(rec, default=str), flush=True)
    json.dump(out, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/calsweep_{D}_{sys.argv[2]}.json', 'w'), indent=1, default=str)
