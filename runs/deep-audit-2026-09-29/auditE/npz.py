import json, numpy as np, collections
from scipy.stats import rankdata
R='/Users/nickd/projects/bosco/'
m=json.load(open(R+'data/cache/v1-harm-dev/items.json')); its=m['items']
va=[it for it in its if it['task']=='harmful' and it['split']=='val']
y=np.array([it['gold']==0 for it in va]); src=np.array([it['source'] for it in va])
def auc(y,s):
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def ece(y,p,nb=15,w=None):
    w=np.ones_like(p) if w is None else w
    b=np.minimum((p*nb).astype(int),nb-1); e=0
    for i in range(nb):
        k=b==i
        if k.any(): e+=w[k].sum()/w.sum()*abs((w[k]*y[k]).sum()/w[k].sum()-(w[k]*p[k]).sum()/w[k].sum())
    return e
sig=lambda z:1/(1+np.exp(-z))
wbal=np.where(y,0.5/y.mean(),0.5/(1-y.mean()))
print('val harmful share',y.mean())
for d in ('brain-train',):
  for s in range(1,6):
    z=np.load(R+f'runs/{d}/seed{s}.npz'); inf=json.load(open(R+f'runs/{d}/seed{s}.json'))
    k,c=inf['real_k'],inf['real_c']
    t,r,sw=z['real_trained'],z['real_reset'],z['real_swapped']
    p=sig(t)
    # decomposition: logit = c + 10k*d ; innate part = reset-c ; memory part = t - r
    inn=r-c; mem=t-r
    print(f's{s} AUC tr {auc(y,t):.3f} reset {auc(y,r):.3f} swap {auc(y,sw):.3f} layered {auc(y,z["layered_trained"]):.3f} lreset {auc(y,z["layered_reset"]):.3f}',
      f'| ECE {ece(y,p):.3f} ECE(bal-weighted) {ece(y,p,w=wbal):.3f} mean p {p.mean():.3f} vs rate {y.mean():.3f}',
      f'| var: logit {t.var():.3f} innate {inn.var():.4f} mem {mem.var():.3f} cov2 {2*np.cov(inn,mem)[0,1]:.3f}; means c {c:.2f} innate {inn.mean():.2f} mem {mem.mean():.2f}; R2 {np.corrcoef(t,r)[0,1]**2:.3f} corr(mem,inn) {np.corrcoef(mem,inn)[0,1]:.3f}',
      f'| sd(logit)/10k {t.std()/(10*k):.4f} lay {z["layered_trained"].std()/(10*inf["layered_k"]):.4f} ratio {(t.std()/k)/(z["layered_trained"].std()/inf["layered_k"]):.2f} k-ratio {inf["layered_k"]/k:.2f}')
    # T2 thresholded vs label convention: flip yes/no => innate AUC 1-x
    # within-source AUC
    print('   within-source AUC trained:', {s_:round(auc(y[src==s_],t[src==s_]),3) for s_ in np.unique(src)}, 'reset:',{s_:round(auc(y[src==s_],r[src==s_]),3) for s_ in np.unique(src)})
    # mean logit by source
    if s==1: print('   mean reset logit by source', {s_:round(float(r[src==s_].mean()),3) for s_ in np.unique(src)}, 'harm rate', {s_:round(float(y[src==s_].mean()),3) for s_ in np.unique(src)})
    # reliability
    if s==1:
        b=np.minimum((p*10).astype(int),9)
        print('   reliability (bin, n, mean p, frac harmful):',[(i,int((b==i).sum()),round(float(p[b==i].mean()),2),round(float(y[b==i].mean()),2)) for i in range(10) if (b==i).any()])
