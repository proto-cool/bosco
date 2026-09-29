import sys, json, time, numpy as np, torch
torch.set_num_threads(2)
V='/Users/nickd/.claude/jobs/4059153c/tmp/v30'
sys.path.insert(0,V+'/scripts'); sys.path.insert(0,V+'/src')
import brain_check as BC, brain_train as BT
from bosco import model2 as M2, data
t0=time.time()
b2=M2.load_or_build(); m=BC.brain('real',b2); BC.load_start(m,'real'); print('build s',time.time()-t0, 'n plastic',len(m.kp_logm))
tab,apm,avm=M2.mbon_groups(b2.brain)
print(tab.side.value_counts().to_dict())
post=m.kp_post.numpy(); pre=m.kp_pre.numpy()
grp=np.full(m.n,'other',dtype=object); grp[apm]='approach'; grp[avm]='avoid'
g=grp[post]
print('synapses per group',{k:int((g==k).sum()) for k in ('approach','avoid','other')})
L={}
for d in ('brain-train','brain-train-depress'):
  for arm in ('real','real-flip','layered'):
    for s in range(1,6):
        L[(d,arm,s)]=torch.load(f'/Users/nickd/projects/bosco/runs/{d}/{arm}-s{s}.pt')['kp_logm'].numpy()
for d in ('brain-train','brain-train-depress'):
  x=L[(d,'real',1)]
  print(d,'real s1 kp_logm: mean %.3f sd %.3f min %.2f max %.2f  |x|>0.1: %.1f%%  |x|>1: %.1f%%'%(x.mean(),x.std(),x.min(),x.max(),100*(abs(x)>0.1).mean(),100*(abs(x)>1).mean()))
  for k in ('approach','avoid','other'):
     xx=x[g==k]; print('   ',k,'n',len(xx),'mean %.3f  frac>+0.1 %.2f frac<-0.1 %.2f  mean multiplier %.2f'%(xx.mean(),(xx>0.1).mean(),(xx<-0.1).mean(),np.exp(xx).mean()))
  print('   corr across seeds (real s1 vs s2..5):',[round(np.corrcoef(L[(d,'real',1)],L[(d,'real',s)])[0,1],3) for s in range(2,6)])
  print('   corr real vs flip (same seed):',[round(np.corrcoef(L[(d,'real',s)],L[(d,'real-flip',s)])[0,1],3) for s in range(1,6)])
  print('   corr real+flip mean-shift? mean real %.3f flip %.3f'%(np.mean([L[(d,'real',s)].mean() for s in range(1,6)]),np.mean([L[(d,'real-flip',s)].mean() for s in range(1,6)])))
# KC activity at the untrained start on 200 label-free items vs |logm|
ant=BC.antenna(); rest=torch.tensor(ant.resting())
with torch.no_grad():
    _,_,rr=m.run(torch.tensor(ant(BC.items()[0])),record='read')
kcr=rr.numpy()  # (n,200)
act=(kcr>0.01).mean(1); meanr=kcr.mean(1)
a=act[pre]; x=L[('brain-train','real',1)]
for lo,hi in ((0,0.0001),(0.0001,0.02),(0.02,0.1),(0.1,1.01)):
    k=(a>=lo)&(a<hi); print('KC active share in [%.4f,%.2f): synapses %d, mean|logm| %.3f'%(lo,hi,k.sum(),abs(x[k]).mean() if k.any() else np.nan))
print('KC mean rate of never-active KCs: %.4f; of active KCs %.4f'%(meanr[m.kc.numpy()][act[m.kc.numpy()]==0].mean(), meanr[m.kc.numpy()][act[m.kc.numpy()]>0].mean()))
np.savez('/Users/nickd/.claude/jobs/4059153c/tmp/auditE/grp.npz',g=g.astype(str),post=post,pre=pre)
