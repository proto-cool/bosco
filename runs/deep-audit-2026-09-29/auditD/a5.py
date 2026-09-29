import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from common import *
from bosco import data
N=128; s=S[:N]
typ=data.annotations().reindex(b2().brain.ids)["type"].fillna("?").to_numpy()
def settle_from(m,r0):
    m.r_rest=r0.clone(); m.settle(rest); return m.r_rest.clone()
out={}
for tag in (None,"real-s1"):
    key=tag or "start"; m=load(tag)
    if tag is None: saved=m.r_rest.clone()
    rz=settle_from(m,torch.zeros(m.n))
    g=torch.Generator().manual_seed(3)
    rests={"zero":rz}
    if tag is None: rests["saved"]=saved
    for i in range(4): rests[f"rand{i}"]=settle_from(m,torch.rand(m.n,generator=g))
    rests["ones"]=settle_from(m,torch.ones(m.n))
    res={}
    ref=None
    for n,r in rests.items():
        d=(r-rz).abs(); big=torch.nonzero(d>0.1)[:,0].numpy()
        m.r_rest=r; z=m.answer(s).numpy()
        if ref is None: ref=z
        res[n]={"n_cells_diff>0.1":int(len(big)),"max":float(d.max()),"types":sorted(set(typ[big]))[:15],
                "flip_vs_zero":float(((z>=0)!=(ref>=0)).mean()),"max_dlogit":float(np.abs(z-ref).max()),"mean_dlogit":float(np.abs(z-ref).mean())}
        print(key,n,res[n],flush=True)
    out[key]=res
json.dump(out,open(AUD/"a5.json","w"),indent=1,default=str)
