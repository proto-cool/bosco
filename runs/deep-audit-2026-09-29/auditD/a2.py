import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from pert import *
m=fresh(); z0=m.answer(s).numpy(); st0=stages(m)
out={"base":{"yes":float((z0>=0).mean()),"near_boundary_|logit|<0.25":float((np.abs(z0)<0.25).mean()),"logit_sd":float(z0.std()),"stages":st0}}
print(out,flush=True)
g=torch.Generator().manual_seed(7)
nW=m._W0[1].numel(); nk=m._kp0.numel()
def run(name,fn):
    m=fresh(); fn(m); info=reset(m); z=m.answer(s).numpy()
    out[name]={**cmp(z,z0),"settle_conv":info["converged"],"stages":stages(m)}; print(name,out[name],flush=True)
for eps in (0.01,0.05):
    for rep in (0,1):
        a=1+eps*torch.randn(nW,generator=g); b_=1+eps*torch.randn(nk,generator=g)
        run(f"w_{eps}_r{rep}",lambda m,a=a,b_=b_: set_w(m,a,b_))
    a=1+eps*torch.randn(nW,generator=g); run(f"w_fixed_only_{eps}",lambda m,a=a: set_w(m,a,None))
    b_=1+eps*torch.randn(nk,generator=g); run(f"w_kcmbon_only_{eps}",lambda m,b_=b_: set_w(m,None,b_))
    def tau(m,eps=eps):
        with torch.no_grad(): m.log_tau += torch.log1p(eps*torch.randn(m.n_units,generator=g)).float()
    run(f"tau_{eps}",tau)
    def gain(m,eps=eps):
        with torch.no_grad(): m.log_g += torch.log1p(eps*torch.randn(m.n_units,generator=g)).float()
    run(f"gain_{eps}",gain)
    def kcoff(m,eps=eps):
        sd=float(m.b_cell[m.kc].std())
        with torch.no_grad(): m.b_cell[m.kc] += eps*sd*torch.randn(len(m.kc),generator=g)
    run(f"kc_offset_{eps}sd",kcoff)
for beta in (49.5,50.5,45.0,55.0):
    R3.BETA=beta; run(f"beta_{beta}",lambda m: None)
R3.BETA=50.0
json.dump(out,open(AUD/"a2.json","w"),indent=1,default=str)
