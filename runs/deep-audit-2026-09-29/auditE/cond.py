import sys, json, time, numpy as np, torch
torch.set_num_threads(2)
V='/Users/nickd/.claude/jobs/4059153c/tmp/v30'; R='/Users/nickd/projects/bosco'
sys.path.insert(0,V+'/scripts'); sys.path.insert(0,V+'/src')
import brain_check as BC, brain_train as BT
from bosco import model2 as M2
from scipy.stats import rankdata
def auc(y,s):
    y=y.astype(bool); r=rankdata(s); n1=y.sum(); n0=len(y)-n1; return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def ba(y,yes): y=y.astype(bool); return 0.5*(yes[y].mean()+(~yes[~y]).mean())
N=int(sys.argv[2]) if len(sys.argv)>2 else 300
conds=sys.argv[1].split(',')
b2=M2.load_or_build(); m=BC.brain('real',b2); BC.load_start(m,'real')
ant=BC.antenna(); rest=torch.tensor(ant.resting())
Xva,yva=BT.harm('val'); sub=np.random.default_rng(0).permutation(len(yva))[:N]
Xtr,ytr=BT.harm('train'); subt=np.random.default_rng(0).permutation(len(ytr))[:N]
tr=torch.load(R+'/runs/brain-train/real-s1.pt'); fl=torch.load(R+'/runs/brain-train/real-flip-s1.pt')
z=np.load(R+'/runs/brain-train/seed1.npz')
post=m.kp_post.numpy()
x=tr['kp_logm'].numpy()
def per_mbon_mean(x):
    s=np.bincount(post,weights=x,minlength=m.n); c=np.bincount(post,minlength=m.n); return (s/np.maximum(c,1))[post]
def shuffle_within(x):
    rng=np.random.default_rng(1); y=x.copy()
    for p in np.unique(post):
        k=np.nonzero(post==p)[0]; y[k]=x[k][rng.permutation(len(k))]
    return y
mems={'trained':x,'permbon':per_mbon_mean(x),'shuffled':shuffle_within(x),
      'common':0.5*(x+fl['kp_logm'].numpy()),'specific':0.5*(x-fl['kp_logm'].numpy()),'zero':0*x,'trained_stalerest':x,'trained_train':x,'zero_train':0*x}
res={}
for cnd in conds:
    t0=time.time()
    if cnd=='trained_stalerest':
        BT.with_memory(m, torch.zeros_like(tr['kp_logm']), tr['log_k'], tr['c'], rest)   # rest under zero memory
        with torch.no_grad(): m.kp_logm.copy_(tr['kp_logm'])
        m.freeze()   # keep r_rest from zero memory
    else:
        BT.with_memory(m, torch.tensor(mems[cnd],dtype=torch.float32), tr['log_k'], tr['c'], rest)
    Xs,ys=(Xtr[subt],ytr[subt]) if cnd.endswith('_train') else (Xva[sub],yva[sub])
    lg=m.answer(torch.tensor(ant(Xs))).numpy()
    out={'auc':round(float(auc(ys,lg)),4),'ba':round(float(ba(ys,lg>=0)),4),'mean':round(float(lg.mean()),3),'sd':round(float(lg.std()),3),'s':round(time.time()-t0,1)}
    if cnd=='trained': out['max_abs_diff_vs_npz']=float(np.abs(lg-z['real_trained'][sub]).max()); out['ref_auc_npz_sub']=round(float(auc(ys,z['real_trained'][sub])),4)
    if cnd!='trained' : out['corr_vs_npz_trained']=round(float(np.corrcoef(lg,z['real_trained'][sub])[0,1]),3) if not cnd.endswith('_train') else None
    res[cnd]=out; print(cnd,out,flush=True)
