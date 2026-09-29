import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from common import *
out={}
for tag in (None,"real-s1"):
    m=load(tag)
    k=float(torch.exp(m.log_k)); c=float(m.c)
    key=tag or "start"
    z=ans(m)
    d=(z-c)/(10*k)
    np.save(AUD/f"base_{key}.npy", z)
    res={"k":k,"c":c,"logit_mean":float(z.mean()),"logit_sd":float(z.std()),"logit_min":float(z.min()),"logit_max":float(z.max()),
         "read_sd":float(d.std()),"yes_share":float((z>=0).mean()),"n_unique":int(len(np.unique(z)))}
    probes={}
    probes["rest"]=rest[None]
    probes["all_max"]=torch.ones(1,46)
    probes["all_zero"]=torch.zeros(1,46)
    probes["all_0.5"]=torch.full((1,46),0.5)
    pulses=rest.repeat(46,1); pulses[torch.arange(46),torch.arange(46)]=1.0
    probes["pulses"]=pulses
    rng=np.random.default_rng(0); U=rng.normal(size=(20,768)); U/=np.linalg.norm(U,axis=1,keepdims=True)
    probes["rand_unit"]=torch.tensor(ant(U))
    probes["zero_vec"]=torch.tensor(ant(np.zeros((1,768))))
    probes["neg_item"]=torch.tensor(ant(-X[:20]))
    probes["x10_item"]=torch.tensor(ant(10*X[:20]))
    probes["nan"]=torch.full((1,46),float("nan"))
    probes["big"]=torch.full((1,46),100.0)
    probes["neg"]=torch.full((1,46),-5.0)
    pr={}
    for n,s in probes.items():
        zz=m.answer(s).numpy()
        pr[n]={"logit":[float(x) for x in zz[:3]],"min":float(np.nanmin(zz)) if not np.isnan(zz).all() else "nan","max":float(np.nanmax(zz)) if not np.isnan(zz).all() else "nan","nan":int(np.isnan(zz).sum()),"yes":float((zz>=0).mean())}
    res["probes"]=pr
    pz=m.answer(probes["pulses"]).numpy(); res["pulse_minus_rest_sd"]=float((pz-pr["rest"]["logit"][0]).std())
    res["ant_zero_vec_input"]=[float(x) for x in probes["zero_vec"][0,:6]]
    res["rand_unit_input_mean"]=float(probes["rand_unit"].mean()); res["rand_unit_share_at_max"]=float((probes["rand_unit"]>=1).float().mean())
    # clipping dependence: unclipped antenna
    p=((X-ant.mu)@ant.W.T)/ant.norm; pm=np.concatenate([np.maximum(0,p),np.maximum(0,-p)],1)
    Sun=torch.tensor((ant.rest+(1-ant.rest)*pm).astype(np.float32))
    res["clip_items_with_any_clip"]=float((pm>1).any(1).mean()); res["clip_max_unclipped"]=float(pm.max())
    zu=m.answer(Sun).numpy()
    res["clip_flip_share"]=float(((zu>=0)!=(z>=0)).mean()); res["clip_max_logit_change"]=float(np.abs(zu-z).max())
    res["clip_affected_items_mean_change"]=float(np.abs(zu-z)[(pm>1).any(1)].mean()) if (pm>1).any() else 0
    # longer run: convergence at 80 vs 160 steps
    m.steps=160; m.read_steps=8
    z160=ans(m,n=64); m.steps=80
    res["steps160_vs80_corr"]=float(np.corrcoef(z160,z[:64])[0,1]); res["steps160_flip"]=float(((z160>=0)!=(z[:64]>=0)).mean())
    res["steps160_max_change"]=float(np.abs(z160-z[:64]).max())
    # oscillation within the read window
    with torch.no_grad():
        _,_,tr=m.run(S[:8],record=True)
    ap,av=m.read_groups["dn"]
    rd=(tr[:,ap].float().mean(1)-tr[:,av].float().mean(1)).numpy()  # (steps,B)
    res["read_last8_range_mean"]=float((rd[-8:].max(0)-rd[-8:].min(0)).mean()); res["read_sd_items_at_end"]=float(rd[-1].std())
    res["read_step_diff_last"]=float(np.abs(rd[-1]-rd[-2]).max())
    out[key]=res
    print(key, json.dumps(res,indent=0,default=str)[:3000],flush=True)
json.dump(out,open(AUD/"a1.json","w"),indent=1,default=str)
