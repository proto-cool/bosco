"""Full dynamics analysis of one design. usage: python analyze.py DESIGN [G] -> auditB/res_DESIGN.json"""
import json, sys, time
import numpy as np, torch
torch.set_num_threads(3)
sys.path.insert(0, '/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
D = sys.argv[1]; G = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
phases = sys.argv[3].split(',') if len(sys.argv) > 3 else ['rest','sniff','multi','resp','sens']
T0 = time.time()
m, rest, SM, ann, b2 = load.load(D, G)
n = m.n
a1 = dyn.alpha_vec(m)
res = {'design': D, 'G': G, 'beta': dyn.beta(m), 'nnz': int(m._W_all.values().numel())}
ap, av = m.read_groups[m.read]
res['read'] = {'kind': m.read, 'n_approach': len(ap), 'n_avoid': len(av)}
REG = {k: v for k, v in m.regions.items() if len(v)}
sens_rows = torch.cat([m.regions['orn'], m.regions['other_sensory'], m.regions['vpn']])

def readv(R):
    a_ = R[ap].mean(0) if len(ap) else torch.zeros(R.shape[1])
    v_ = R[av].mean(0) if len(av) else torch.zeros(R.shape[1])
    return a_ - v_

def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

def eig_info(r, bias, alpha, tag, k=8):
    J, fp, x = dyn.jacobian(m, r, bias, alpha)
    err = dyn.check_jvp(m, r, bias, alpha, J)
    vals, vecs = dyn.top_eigs(J, k=k)
    rho = float(abs(vals[0]))
    out = {'rho': rho, 'relax_steps': float(-1/np.log(rho)) if rho < 1 else None,
           'top': [[round(float(v.real), 5), round(float(v.imag), 5)] for v in vals],
           'jvp_relerr': err, 'share_cells_fprime_gt_0.5': float((fp > 0.5).float().mean()),
           'vec0': dyn.describe_vec(vecs[:, 0], ann, top=10)}
    try:
        lr, lv = dyn.rightmost_M(J, 0.25, k=4)
        out['rightmost'] = [[round(float(v.real), 5), round(float(v.imag), 5)] for v in lr]
    except Exception as e:
        out['rightmost_err'] = str(e)[:100]
    log(tag, 'rho', rho, out['top'][:4])
    return out, vecs[:, 0]

# ---------------- rest
bias_r = dyn.bias_vec(m, rest)
r0 = m.r_rest.clone() if m.r_rest is not None else torch.zeros(n)
if 'rest' in phases:
    rR, d, _ = dyn.run_long(m, bias_r, r0, 3000, a1)
    # track the read and a projection over the last 400 steps for period detection
    rr = rR.clone(); tail = []
    with torch.no_grad():
        for t in range(400):
            rr = dyn.F(m, rr, bias_r, a1); tail.append(float(readv(rr[:, None])[0]))
    tail = np.array(tail)
    res['rest'] = {'step_change_at': {str(t): float(d[t-1]) for t in (1, 10, 100, 500, 1000, 2000, 3000)},
                   'max_step_change_last200': float(d[-200:].max()),
                   'drift_from_saved_rest': float((rR - r0).abs().max()),
                   'read_tail_range': float(tail.max() - tail.min())}
    if res['rest']['max_step_change_last200'] > 1e-6:
        x = tail - tail.mean(); ac = np.correlate(x, x, 'full')[len(x)-1:]; ac /= ac[0] + 1e-30
        res['rest']['period_guess'] = int(np.argmax(ac[2:200]) + 2)
    info, v0 = eig_info(rR, bias_r, a1, 'rest')
    res['rest'].update(info)
    # nonlinear perturbation
    g = torch.Generator().manual_seed(3)
    for eps in (1e-3, 3e-2):
        dl = eps * torch.randn(n, generator=g)
        rp = rR + dl; norms = []
        with torch.no_grad():
            for t in range(601):
                if t in (0, 20, 50, 100, 200, 400, 600): norms.append(float((rp - rR).norm()))
                rp = dyn.F(m, rp, bias_r, a1)
        res['rest'][f'perturb_{eps:g}'] = norms
        # also along leading eigenvector
    vv = torch.tensor(np.real(v0), dtype=torch.float32); vv /= vv.norm()
    rp = rR + 1e-3 * vv; norms = []
    with torch.no_grad():
        for t in range(401):
            if t in (0, 20, 50, 100, 200, 400): norms.append(float((rp - rR).norm()))
            rp = dyn.F(m, rp, bias_r, a1)
    res['rest']['perturb_eigvec_1e-3'] = norms
    m.r_rest = rR
    json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/res_{D}{"" if D in ("C0","C2") else "_G%g"%G}{"_"+sys.argv[3].replace(",","-") if len(sys.argv)>3 else ""}.json', 'w'), indent=1, default=float)
else:
    rR = r0

# ---------------- sniff: hold smells 2000 steps
if 'sniff' in phases:
    idx = list(range(8))
    Bm = torch.stack([dyn.bias_vec(m, SM[i]) for i in idx], 1)
    R = rR[:, None].repeat(1, len(idx)); dmax = np.zeros((2000, len(idx))); rd = np.zeros((2000, len(idx)))
    R80 = None
    with torch.no_grad():
        for t in range(2000):
            Rn = dyn.F(m, R, Bm, a1[:, None]); dmax[t] = (Rn - R).abs().max(0).values.numpy(); R = Rn
            rd[t] = readv(R).numpy()
            if t == 79: R80 = R.clone()
    s = {'step_change_at': {str(t): float(dmax[t-1].max()) for t in (1, 80, 200, 500, 1000, 2000)},
         'max_step_change_last200': float(dmax[-200:].max()),
         'items_not_converged_1e-6': int((dmax[-200:].max(0) > 1e-6).sum()),
         'read_range_last200_max': float((rd[-200:].max(0) - rd[-200:].min(0)).max()),
         'read80_vs_read2000_absdiff_median': float(np.median(np.abs(rd[79] - rd[-1]))),
         'read_item_sd_80': float(rd[79].std()), 'read_item_sd_2000': float(rd[-1].std())}
    if s['max_step_change_last200'] > 1e-6:
        j = int(np.argmax(dmax[-200:].max(0)))
        x = rd[-400:, j] - rd[-400:, j].mean(); ac = np.correlate(x, x, 'full')[len(x)-1:]; ac /= ac[0] + 1e-30
        s['period_guess'] = int(np.argmax(ac[2:200]) + 2)
    s['eig_fixed'] = []; s['eig_step80'] = []
    for j in range(3):
        e, _ = eig_info(R[:, j].contiguous(), Bm[:, j].contiguous(), a1, f'sniff{j} end', k=6)
        s['eig_fixed'].append(e)
        e, _ = eig_info(R80[:, j].contiguous(), Bm[:, j].contiguous(), a1, f'sniff{j} step80', k=6)
        s['eig_step80'].append({k: e[k] for k in ('rho', 'relax_steps', 'top')})
    res['sniff'] = s
    json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/res_{D}{"" if D in ("C0","C2") else "_G%g"%G}{"_"+sys.argv[3].replace(",","-") if len(sys.argv)>3 else ""}.json', 'w'), indent=1, default=float)

# ---------------- multistability
if 'multi' in phases:
    g = torch.Generator().manual_seed(7)
    smells = [rest, SM[10], SM[11], SM[12]]
    inits = {'rest': rR, 'zero': torch.zeros(n), 'rand': torch.rand(n, generator=g), 'sat': torch.full((n,), 0.99)}
    cols, Bc = [], []
    for si, s_ in enumerate(smells):
        b_ = dyn.bias_vec(m, s_)
        for k, r_ in inits.items():
            cols.append(r_); Bc.append(b_)
    R = torch.stack(cols, 1); Bm = torch.stack(Bc, 1)
    with torch.no_grad():
        for t in range(3000):
            Rn = dyn.F(m, R, Bm, a1[:, None]); last = (Rn - R).abs().max(0).values; R = Rn
    out = {}
    names = list(inits)
    for si in range(len(smells)):
        blk = R[:, si*4:(si+1)*4]
        ref = blk[:, 0]
        out[f'smell{si}'] = {nm: {'maxabs': float((blk[:, j] - ref).abs().max()),
                                  'share_cells_diff_gt_0.01': float(((blk[:, j] - ref).abs() > 0.01).float().mean()),
                                  'read_diff': float(readv(blk[:, j:j+1])[0] - readv(ref[:, None])[0]),
                                  'last_step_change': float(last[si*4+j])} for j, nm in enumerate(names)}
    res['multi'] = out
    log('multi', {k: {nm: round(v['maxabs'], 4) for nm, v in d.items()} for k, d in out.items()})
    json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/res_{D}{"" if D in ("C0","C2") else "_G%g"%G}{"_"+sys.argv[3].replace(",","-") if len(sys.argv)>3 else ""}.json', 'w'), indent=1, default=float)

# ---------------- response dynamics, saturation
if 'resp' in phases:
    NI = 64
    Bm = torch.stack([dyn.bias_vec(m, SM[100 + i]) for i in range(NI)], 1)
    R = rR[:, None].repeat(1, NI); rd = np.zeros((1000, NI)); win = torch.zeros(n, NI)
    stage = {k: np.zeros((1000, NI)) for k in REG}
    with torch.no_grad():
        for t in range(1000):
            R = dyn.F(m, R, Bm, a1[:, None]); rd[t] = readv(R).numpy()
            if t < 200 or t % 10 == 9:
                for k, v in REG.items(): stage[k][t] = R[v].mean(0).numpy()
            if 72 <= t < 80: win += R / 8
    read80 = rd[72:80].mean(0); readinf = rd[-8:].mean(0)
    def settle_t(x, fin):
        dev = np.abs(x - fin[None]); tol = 0.05 * np.abs(fin - x[0])[None] + 1e-12
        ok = dev <= tol
        # first t after which all later ok
        bad = ~ok; last_bad = np.where(bad.any(1))[0]
        per = []
        for j in range(x.shape[1]):
            lb = np.where(bad[:, j])[0]; per.append(int(lb[-1] + 2) if len(lb) else 1)
        return np.array(per)
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    rc = rd - rd.mean(1, keepdims=True)
    r = {'settle95_to_step1000_median': float(np.median(settle_t(rd, rd[-1]))),
         'settle95_to_step1000_p90': float(np.percentile(settle_t(rd, rd[-1]), 90)),
         'settle95_to_step80_median': float(np.median(settle_t(rd[:80], rd[79]))),
         'centered_settle95_to_step1000_median': float(np.median(settle_t(rc, rc[-1]))),
         'corr_with_read80': {str(t): c(rd[t-1], read80) for t in (10, 20, 40, 60, 120, 200)},
         'corr_read80_readinf': c(read80, readinf), 'read80_sd': float(read80.std()), 'readinf_sd': float(readinf.std()),
         'read80_mean': float(read80.mean()),
         'sign_agree_80_inf_centered': float(((read80 - read80.mean()) * (readinf - readinf.mean()) > 0).mean()),
         'stage_settle95_median_to_step200': {k: float(np.median(settle_t(v[:200], v[199]))) for k, v in stage.items()}}
    # saturation/dead at the read window
    X = win
    sd = {}
    for k, v in list(REG.items()) + [('all', torch.arange(n))]:
        x = X[v]
        sd[k] = {'dead(max<0.001)': float((x.max(1).values < 1e-3).float().mean()),
                 'sat_always(min>0.9)': float((x.min(1).values > 0.9).float().mean()),
                 'sat_any(max>0.9)': float((x.max(1).values > 0.9).float().mean()),
                 'mean': float(x.mean()), 'item_cv_median': float((x.std(1) / (x.mean(1) + 1e-9)).median())}
    r['sat_dead'] = sd
    res['resp'] = r
    log('resp', json.dumps({k: r[k] for k in r if k != 'sat_dead'}, default=float))
    json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/res_{D}{"" if D in ("C0","C2") else "_G%g"%G}{"_"+sys.argv[3].replace(",","-") if len(sys.argv)>3 else ""}.json', 'w'), indent=1, default=float)

# ---------------- sensitivity (80-step sniffs)
if 'sens' in phases:
    NI = 32
    S0 = SM[200:200+NI]
    def sniff(S, Wscale=None):
        Bm = torch.stack([dyn.bias_vec(m, s_) for s_ in S], 1)
        R = rR[:, None].repeat(1, len(S)); acc = 0
        with torch.no_grad():
            for t in range(80):
                R = dyn.F(m, R, Bm, a1[:, None])
                if t >= 72: acc = acc + R / 8
        return acc
    base = sniff(S0)
    g = torch.Generator().manual_seed(11)
    dS = torch.randn(S0.shape, generator=g); dS = dS / dS.norm(dim=1, keepdim=True) * 0.01 * (S0 - rest).norm(dim=1, keepdim=True)
    pert = sniff(S0 + dS)
    rb, rp = readv(base), readv(pert)
    out = {'input_1pct': {'read_change_over_item_sd': float((rp - rb).abs().median() / rb.std()),
                          'stages_rel_change_per_rel_input': {k: float(((pert[v]-base[v]).norm(dim=0)/(base[v].norm(dim=0)+1e-9)).median()/0.01) for k, v in REG.items()}}}
    # global input scale of non-sensory rows +1%
    W0 = m._W_all
    rowscale = torch.full((n,), 1.01); rowscale[sens_rows] = 1.0
    crow = W0.crow_indices(); rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])
    m._W_all = torch.sparse_csr_tensor(crow, W0.col_indices(), W0.values() * rowscale[rows], (n, n))
    # rest moves too: re-settle
    rr2, d2, _ = dyn.run_long(m, bias_r, rR, 1500, a1)
    rR_save = rR; rR = rr2
    pg = sniff(S0)
    rR = rR_save; m._W_all = W0
    rg = readv(pg)
    out['gain_1pct'] = {'read_change_over_item_sd': float((rg - rb).abs().median() / rb.std()),
                        'read_rank_corr': float(np.corrcoef(rb.numpy(), rg.numpy())[0, 1]),
                        'stages_rel_change_per_rel_gain': {k: float(((pg[v]-base[v]).norm(dim=0)/(base[v].norm(dim=0)+1e-9)).median()/0.01) for k, v in REG.items()},
                        'rest_settled': float(d2[-100:].max())}
    res['sens'] = out
    log('sens', json.dumps(out, default=float))
    json.dump(res, open(f'/Users/nickd/.claude/jobs/4059153c/tmp/auditB/res_{D}{"" if D in ("C0","C2") else "_G%g"%G}{"_"+sys.argv[3].replace(",","-") if len(sys.argv)>3 else ""}.json', 'w'), indent=1, default=float)
log('done')
