import sys; sys.path.insert(0,"/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
from common import *
t=time.time(); m=load(); print("build+load", time.time()-t)
t=time.time(); z=ans(m,n=20); print("20 answers", time.time()-t, z[:5])
print("S stats", S.shape, float(S.min()), float(S.max()), float((S>=1).float().mean()), float((S<=0.05+1e-7).float().mean()))
print("X norms", np.linalg.norm(X,axis=1)[:5])
