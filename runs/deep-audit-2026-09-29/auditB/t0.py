import sys,time,torch
torch.set_num_threads(3)
sys.path.insert(0,'/Users/nickd/.claude/jobs/4059153c/tmp/auditB')
import load, dyn
t=time.time(); m,rest,sm,ann,b2=load.load(sys.argv[1]); print('load',time.time()-t, m.n, m._W_all.values().numel())
bias=dyn.bias_vec(m,rest); a=dyn.alpha_vec(m); print('alpha',float(a.min()),float(a.max()), 'beta',dyn.beta(m))
r=m.r_rest.clone() if m.r_rest is not None else torch.zeros(m.n)
t=time.time(); r2,d,_=dyn.run_long(m,bias,r,200,a); print('200 steps',time.time()-t, d[:3], d[-3:])
t=time.time(); J,fp,x=dyn.jacobian(m,r2,bias,a); print('J',time.time()-t, J.nnz, 'jvp relerr',dyn.check_jvp(m,r2,bias,a,J))
t=time.time(); v,_=dyn.top_eigs(J,k=6); print('eigs',time.time()-t, v, abs(v))
