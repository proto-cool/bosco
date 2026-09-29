import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from common import *
N=128
s=S[:N]
def fresh():
    m=load("real-s1"); W=m.W.coalesce(); m._W0=(W.indices().clone(),W.values().clone()); m._kp0=m.kp_w.clone(); return m
def set_w(m,fW=None,fkp=None):
    idx,val=m._W0
    v=val*fW if fW is not None else val
    m.W=torch.sparse_coo_tensor(idx,v,(m.n,m.n)).coalesce()
    m.kp_w=m._kp0*fkp if fkp is not None else m._kp0.clone()
def reset(m):
    m.freeze(); m.r_rest=None; info=m.settle(rest); return info
def stages(m):
    with torch.no_grad():
        _,_,rr=m.run(s[:32],record="read")
    R=m.regions; ap,av=m.read_groups["dn"]; apm,avm=m.read_groups["mbon"]
    f=lambda x: round(float(x),4)
    return {"orn":f(rr[R["orn"]].mean()),"alpn":f(rr[R["alpn"]].mean()),"kc_active":f((rr[m.kc]>R3.ACTIVE).float().mean()),
            "mbon_ap":f(rr[apm].mean()),"mbon_av":f(rr[avm].mean()),"dn_ap":f(rr[ap].mean()),"dn_av":f(rr[av].mean()),
            "all_mean":f(rr.mean()),"sat_share":f((rr>0.95).float().mean()),"rest_mean":f(m.r_rest.mean())}
def cmp(z,z0):
    return {"flip":float(((z>=0)!=(z0>=0)).mean()),"corr":float(np.corrcoef(z,z0)[0,1]) if np.std(z)>0 else None,
            "mean_abs_dlogit":float(np.abs(z-z0).mean()),"yes":float((z>=0).mean()),"nan":int(np.isnan(z).sum())}
