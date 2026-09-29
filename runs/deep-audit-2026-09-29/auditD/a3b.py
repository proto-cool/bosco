import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from pert import *
from bosco import data
import pandas as pd
m=fresh(); z0=m.answer(s).numpy()
out={}
ids=b2().brain.ids
nt=data.neurotransmitters().reindex(pd.Index(ids))
cons=nt["consensus_nt"].str.lower(); pred=nt["predicted_nt"].str.lower()
uns=cons.isna()|cons.isin(["unclear","unknown"]); lab=cons.where(~uns,pred).fillna("unknown").to_numpy()
typ=data.annotations().reindex(pd.Index(ids))["type"].to_numpy()
src=b2().sign_source
out["n_default_sign"]=int((src=="default").sum()); out["n_unclear_or_unknown"]=int(np.isin(lab,["unclear","unknown"]).sum())
def per_pre(f):
    f=torch.tensor(f,dtype=torch.float32)
    return lambda m: set_w(m, f[m._W0[0][1]], f[m.kp_pre])
def run(name,fn):
    m=fresh(); fn(m); info=reset(m); z=m.answer(s).numpy()
    out[name]={**cmp(z,z0),"settle_conv":info["converged"],"stages":stages(m)}; print(name,out[name],flush=True)
mono=np.isin(lab,["dopamine","octopamine","serotonin"])&(typ!="DPM")
f=np.ones(len(ids)); f[mono]=0; run("monoamine_removed_keepDPM",per_pre(f))
f=np.ones(len(ids)); f[src=="default"]=0; run("default_sign_silenced",per_pre(f))
print(out)
json.dump(out,open(AUD/"a3b.json","w"),indent=1,default=str)
