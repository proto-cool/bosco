"""E1 (ORN->PN), E2 (APL), E3 (KC sparseness/decorrelation), E4 (MBONs KC-driven), E7 (innate path)."""
import json
import sys

sys.path.insert(0, "/Users/nickd/.claude/jobs/4059153c/tmp/auditC")
from common import *  # noqa
from scipy import stats

ver, arm = sys.argv[1], sys.argv[2]
N_ITEMS = 384
t = T()
m, rest, ctx = load(ver, arm)
typ = types(ctx)
B = ctx["B"]
X, S_items = smells_items(B, N_ITEMS)
nch = S_items.shape[1]
REST = float(rest[0])
rng = np.random.default_rng(7)
single = torch.full((nch, nch), REST)
single[range(nch), range(nch)] = 1.0
few = torch.full((100, nch), REST)
for i in range(100):
    ch = rng.choice(nch, 3, replace=False)
    few[i, ch] = torch.tensor(rng.uniform(0.5, 1.0, 3), dtype=torch.float32)
out = {"ver": ver, "arm": arm, "n_items": N_ITEMS, "rest_input": REST}

r_rest = m.r_rest.clone()
rr_items, d_items = sniff(m, S_items)
rr_syn, _ = sniff(m, torch.cat([single, few]))
t("intact sniffs")

# ---------------- E1: ORN -> PN ----------------
orn = m.orn_idx
glom = [str(g) for g in m.nose.glomeruli]
pn_all = m.regions["alpn"]
pn_type_glom = np.array([s.split("_")[0] for s in typ])
uni_pn = torch.tensor(np.nonzero(np.isin(pn_type_glom, glom) & np.isin(np.arange(len(typ)), pn_all.numpy()))[0])
uni_pn_ch = torch.tensor([glom.index(pn_type_glom[i]) for i in uni_pn.numpy()])


def lifetime_sparseness(R):  # R (cells, odours) >= 0 ; Vinje & Gallant / Willmore
    N = R.shape[1]
    m1 = R.mean(1)
    m2 = (R ** 2).mean(1)
    ok = m2 > 1e-8
    s = (1 - m1[ok] ** 2 / m2[ok]) / (1 - 1 / N)
    return float(s.mean()), int(ok.sum())


def e1(rr, name):
    res = {}
    for lab, idx in (("orn", orn), ("uni_pn", uni_pn), ("all_pn", pn_all)):
        ev = rr[idx] - r_rest[idx][:, None]
        ls, n_resp = lifetime_sparseness(ev.clamp(min=0))
        res[lab] = {"lifetime_sparseness": ls, "responsive_cells": n_resp, "n": len(idx),
                    "skew_pooled": float(stats.skew(ev.flatten().numpy())),
                    "mean_evoked": float(ev.mean()), "sd_evoked": float(ev.std()),
                    "frac_resp_gt_0.01": float((ev > 0.01).float().mean())}
    res["pass_sparseness"] = res["uni_pn"]["lifetime_sparseness"] < res["orn"]["lifetime_sparseness"]
    res["pass_skew"] = res["uni_pn"]["skew_pooled"] < res["orn"]["skew_pooled"]
    return res


out["E1"] = {"items": e1(rr_items, "items"), "synthetic": e1(rr_syn, "syn")}
# glomerular specificity: single-glomerulus pulse -> share of uniPN evoked (positive) response on that glomerulus's PNs
ev = (rr_syn[:, :nch][uni_pn] - r_rest[uni_pn][:, None]).clamp(min=0)  # (uniPN, 46)
share, rank1 = [], 0
for g in range(nch):
    own = uni_pn_ch == g
    if own.sum() == 0:
        continue
    tot = ev[:, g].sum()
    share.append(float(ev[own, g].sum() / tot) if tot > 0 else 0.0)
    per_ch = torch.zeros(nch).index_add(0, uni_pn_ch, ev[:, g]) / torch.bincount(uni_pn_ch, minlength=nch).clamp(min=1)
    rank1 += int(per_ch.argmax() == g)
out["E1"]["glom_specificity"] = {"mean_share_on_own_glom": float(np.mean(share)), "own_glom_is_top": rank1,
                                 "n_glom_with_pns": len(share), "chance_share": float(1 / len(set(uni_pn_ch.tolist())))}
# ORN->PN input-output: per glomerulus, PN evoked vs ORN evoked across items (Olsen 2010: concave, weak inputs boosted)
orn_ev_ch = torch.stack([(rr_items[m.orn_idx[m.orn_chan == g]] - r_rest[m.orn_idx[m.orn_chan == g]][:, None]).mean(0)
                         for g in range(nch)])
pn_ev_ch = torch.zeros(nch, rr_items.shape[1])
cnt = torch.bincount(uni_pn_ch, minlength=nch)
pn_ev_ch.index_add_(0, uni_pn_ch, rr_items[uni_pn] - r_rest[uni_pn][:, None])
pn_ev_ch = pn_ev_ch / cnt.clamp(min=1)[:, None]
have = cnt > 0
cors = [float(np.corrcoef(orn_ev_ch[g], pn_ev_ch[g])[0, 1]) for g in range(nch) if have[g] and orn_ev_ch[g].std() > 0]
out["E1"]["orn_pn_same_glom_corr_median"] = float(np.median(cors))
t("E1")

# ---------------- E3: KC sparseness / decorrelation ----------------
kc = m.kc
act = rr_items[kc] > 0.01
frac = act.float().mean(0)
kc_ev = rr_items[kc] - r_rest[kc][:, None]
pn_ev = rr_items[uni_pn] - r_rest[uni_pn][:, None]
pn_ev_all = rr_items[pn_all] - r_rest[pn_all][:, None]
orn_ev = rr_items[orn] - r_rest[orn][:, None]


def popcorr(E):
    C = np.corrcoef(E.T.numpy())
    iu = np.triu_indices(len(C), 1)
    return float(np.nanmean(C[iu])), C[iu]


kc_c, kc_pairs = popcorr(kc_ev)
pn_c, pn_pairs = popcorr(pn_ev)
pna_c, _ = popcorr(pn_ev_all)
orn_c, _ = popcorr(orn_ev)
J = jaccard_pairs(act)
iu = np.triu_indices(N_ITEMS, 1)
Jp = J.numpy()[iu]
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
cos = (Xn @ Xn.T)[iu]
Sn = S_items.numpy() - REST
inp_c = np.corrcoef(Sn)[iu]
rho_emb = stats.spearmanr(cos, Jp).correlation
rho_inp = stats.spearmanr(inp_c, Jp).correlation
q_hi, q_lo = np.quantile(cos, 0.95), np.quantile(cos, 0.5)
out["E3"] = {"kc_active_frac_mean": float(frac.mean()), "kc_active_frac_min_max": [float(frac.min()), float(frac.max())],
             "frac_odours_in_2_10pct": float(((frac >= 0.02) & (frac <= 0.10)).float().mean()),
             "popcorr_orn": orn_c, "popcorr_uni_pn": pn_c, "popcorr_all_pn": pna_c, "popcorr_kc": kc_c,
             "kc_jaccard_mean": float(Jp.mean()),
             "spearman_embcos_vs_kcjaccard": float(rho_emb), "spearman_inputcorr_vs_kcjaccard": float(rho_inp),
             "jaccard_top5pct_similar": float(Jp[cos >= q_hi].mean()), "jaccard_bottom50pct": float(Jp[cos <= q_lo].mean())}
out["E3"]["pass_sparse"] = 0.02 <= out["E3"]["kc_active_frac_mean"] <= 0.10
out["E3"]["pass_decorrelate"] = kc_c < pn_c
out["E3"]["pass_similar_share_more"] = out["E3"]["jaccard_top5pct_similar"] > out["E3"]["jaccard_bottom50pct"] and rho_emb > 0
t("E3")

# ---------------- E2: APL ----------------
S2 = S_items[:192]
act0 = act[:, :192]
apl = idx_of(typ, ["APL"])
with Perturb(m, rest, apl, set_to=-10.0) as mm:
    rr2, d2 = sniff(mm, S2)
    a2 = rr2[kc] > 0.01
    out["E2"] = {"apl_cells": len(apl), "apl_rest_rate_intact": float(r_rest[apl].mean()),
                 "apl_evoked_intact": float((rr_items[apl, :192] - r_rest[apl][:, None]).mean()),
                 "kc_frac_intact": float(act0.float().mean()), "kc_frac_apl_off": float(a2.float().mean()),
                 "jaccard_intact": mean_offdiag(jaccard_pairs(act0)), "jaccard_apl_off": mean_offdiag(jaccard_pairs(a2)),
                 "kc_rest_active_apl_off": float((mm.r_rest[kc] > 0.01).float().mean())}
e2 = out["E2"]
e2["ratio"] = e2["kc_frac_apl_off"] / max(e2["kc_frac_intact"], 1e-9)
e2["pass"] = e2["ratio"] >= 1.5 and e2["jaccard_apl_off"] > e2["jaccard_intact"]
t("E2")

# ---------------- E4: MBONs KC-driven ----------------
mbon = m.regions["mbon"]
ev_int = (rr_items[mbon, :192] - r_rest[mbon][:, None])
with Perturb(m, rest, kp=torch.full_like(m.kp_logm, -30.0)) as mm:
    rr4, _ = sniff(mm, S2)
    ev_off = rr4[mbon] - mm.r_rest[mbon][:, None]
    rest_off = mm.r_rest[mbon].clone()
typical = torch.tensor([i for i in mbon.tolist() if typ[i].startswith("MBON") and int(typ[i][4:6]) <= 19])
tmask = torch.isin(mbon, typical)
out["E4"] = {"mbon_abs_evoked_intact": float(ev_int.abs().mean()), "mbon_abs_evoked_kp_off": float(ev_off.abs().mean()),
             "typical_abs_evoked_intact": float(ev_int[tmask].abs().mean()),
             "typical_abs_evoked_kp_off": float(ev_off[tmask].abs().mean()),
             "mbon_sd_across_odours_intact": float(ev_int.std(1).mean()), "mbon_sd_across_odours_kp_off": float(ev_off.std(1).mean()),
             "mbon_rest_intact": float(r_rest[mbon].mean()), "mbon_rest_kp_off": float(rest_off.mean())}
e4 = out["E4"]
e4["cut_pct_all"] = 100 * (1 - e4["mbon_abs_evoked_kp_off"] / e4["mbon_abs_evoked_intact"])
e4["cut_pct_typical"] = 100 * (1 - e4["typical_abs_evoked_kp_off"] / e4["typical_abs_evoked_intact"])
e4["cut_pct_sd"] = 100 * (1 - e4["mbon_sd_across_odours_kp_off"] / e4["mbon_sd_across_odours_intact"])
e4["pass"] = e4["cut_pct_typical"] >= 80
t("E4")

# ---------------- E7: innate path ----------------
sd0 = float(d_items[:192].std())
res7 = {"read_sd_intact": sd0}
lh = m.regions["lh"]
for name, idx in (("kc_off", kc), ("lh_off", lh), ("kc_and_lh_off", torch.cat([kc, lh])), ("pn_off", pn_all)):
    with Perturb(m, rest, idx, set_to=-10.0) as mm:
        _, dd = sniff(mm, S2)
        res7[f"read_sd_{name}"] = float(dd.std())
        res7[f"read_corr_with_intact_{name}"] = float(np.corrcoef(dd.numpy(), d_items[:192].numpy())[0, 1])
    t(f"E7 {name}")
res7["kc_off_ratio"] = res7["read_sd_kc_off"] / sd0
res7["lh_off_ratio"] = res7["read_sd_lh_off"] / sd0
res7["pass"] = res7["kc_off_ratio"] >= 0.5 and res7["lh_off_ratio"] < 1.0
out["E7"] = res7
json.dump(out, open(f"/Users/nickd/.claude/jobs/4059153c/tmp/auditC/A_{ver}_{arm}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
