import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from pert import *
from bosco import data
import pandas as pd
m=fresh(); z0=m.answer(s).numpy(); st0=stages(m)
out={"base":{"stages":st0}}
ids=b2().brain.ids
nt=data.neurotransmitters().reindex(pd.Index(ids))
cons=nt["consensus_nt"].str.lower(); pred=nt["predicted_nt"].str.lower()
uns=cons.isna()|cons.isin(["unclear","unknown"]); lab=cons.where(~uns,pred).fillna("unknown").to_numpy()
conf=nt["predicted_nt_confidence"].to_numpy()
sign=b2().brain.nt_sign.astype(float)
out["counts"]={"glutamate":int((lab=="glutamate").sum()),"uncertain_conf<0.6":int((conf<0.6).sum()),"uncertain_nonzero_sign":int(((conf<0.6)&(sign!=0)).sum()),
  "monoamine":int(np.isin(lab,["dopamine","octopamine","serotonin"]).sum()),"unknown_default_plus":int((lab=="unknown").sum()) if True else 0}
def per_pre(f):
    f=torch.tensor(f,dtype=torch.float32)
    return lambda m: set_w(m, f[m._W0[0][1]], f[m.kp_pre])
def run(name,fn):
    m=fresh(); fn(m); info=reset(m); z=m.answer(s).numpy()
    out[name]={**cmp(z,z0),"settle_conv":info["converged"],"settle_steps":info["steps"],"stages":stages(m)}; print(name,out[name],flush=True)
f=np.ones(len(ids)); f[lab=="glutamate"]=-1; run("glutamate_excitatory",per_pre(f))
f=np.ones(len(ids)); f[conf<0.6]=-1; run("uncertain_flipped",per_pre(f))
f=np.ones(len(ids)); f[np.isin(lab,["dopamine","octopamine","serotonin"])]=0; run("monoamine_removed",per_pre(f))
f=np.ones(len(ids)); f[lab=="unknown"]=0; run("unknown_silenced",per_pre(f))
json.dump(out,open(AUD/"a3.json","w"),indent=1,default=str)
