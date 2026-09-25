import sys; sys.path.insert(0,'scripts')
import numpy as np, torch, a4b_dev as D, a4_pilot as P
from bosco import a5, data
meta,X,L=D.load()
zi,zl=D.antenna(X,L,meta,"bi46")
m=P.build_brain("real")
b2=__import__('bosco.model2',fromlist=['x']).load_or_build(); b=a5.cut(b2.brain)
alpn=np.where(data.annotations().reindex(b.ids)["class"].astype(str)=="ALPN")[0]
tr=[i for i,it in enumerate(meta["items"]) if it["split"]=="train"][:64]
s=torch.tensor(zi[tr],device=m.device); v=torch.zeros(64,52,device=m.device)
P0=zl-0.5; P0=P0/np.linalg.norm(P0,axis=1,keepdims=True); cp=P0@P0.T; iu=np.triu_indices(len(zl),1)
pi=[i for i,it in enumerate(meta["items"]) if it["split"]=="practice"]
def run(z):
    out=[]
    with torch.no_grad():
        for i in range(0,len(z),256):
            _,r,_=m.run(torch.tensor(z[i:i+256],device=m.device),torch.zeros(len(z[i:i+256]),52,device=m.device)); out.append(r.cpu().numpy())
    return np.concatenate(out,1)
def sc(A):
    A=A-A.mean(0); n=np.linalg.norm(A,axis=1,keepdims=True)+1e-9; c=(A/n)@(A/n).T; return np.corrcoef(c[iu],cp[iu])[0,1]
for g in (0.25,0.5,1.0,2.0,4.0,8.0):
  for th in (0.05,0.2,0.5):
    m.set_init(g,th)
    try:
        a5._bisect(m,s,v,m.set_kc_threshold,"kc",0.05); a5._bisect(m,s,v,m.set_mbon_threshold,"mbon",0.2); a5._bisect(m,s,v,m.set_kc_threshold,"kc",0.05)
    except Exception as e: print(g,th,e); continue
    st=a5._probe_init(m,s,v)
    R=run(zl); K=R[m.kc.cpu().numpy()].T
    RI=run(zi[pi]); KI=RI[m.kc.cpu().numpy()].T
    Kn=K/(np.linalg.norm(K,axis=1,keepdims=True)+1e-9); KIn=KI/(np.linalg.norm(KI,axis=1,keepdims=True)+1e-9)
    res={}
    for j,i in enumerate(pi):
        it=meta["items"][i]; res.setdefault(it["kind"],[]).append(int(np.argmax(Kn[it["opts"]]@KIn[j])==it["gold"]))
    print(f"g {g:5.2f} th {th:.2f} | kc {st['kc']:.3f} mbon {st['mbon']:.2f} spread {st['spread']:.3f} | sim ORN {sc(R[m.orn_idx.cpu().numpy()].T):.2f} PN {sc(R[alpn].T):.2f} KC {sc(K):.2f} | KC-match practice {D.macro({k:np.mean(x) for k,x in res.items()}):.3f}",flush=True)
