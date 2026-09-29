import json, numpy as np, torch
from scipy.stats import rankdata
R='/Users/nickd/projects/bosco/'
m=json.load(open(R+'data/cache/v1-harm-dev/items.json')); its=m['items']
y=np.array([it['gold']==0 for it in its if it['task']=='harmful' and it['split']=='val'])
def wauc(y,s,w):
    # weighted AUC via sorting (ties averaged approx via rankdata on unique)
    o=np.argsort(s,kind='mergesort'); s_,y_,w_=s[o],y[o],w[o]
    wp=w_*y_; wn=w_*(~y_)
    cn=np.cumsum(wn)-wn  # negatives strictly below (ignoring ties; logits continuous)
    return (wp*cn).sum()/(wp.sum()*wn.sum())
Z={s:dict(np.load(R+f'runs/brain-train/seed{s}.npz')) for s in range(1,6)}
rng=np.random.default_rng(20260928); B=2000
t2=[];t3=[];t2f=[]
for i in range(B):
    pick=rng.choice(5,5); w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)).astype(float)
    a2=[];a3=[];a2f=[]
    for s in pick:
        z=Z[s+1]; at=wauc(y,z['real_trained'],w); ar=wauc(y,z['real_reset'],w); asw=wauc(y,z['real_swapped'],w)
        a2.append((ar-0.5)-0.25*(at-0.5)); a3.append((0.5-asw)-0.8*(at-0.5))
        a2f.append(((1-ar)-0.5)-0.25*(at-0.5))  # label convention reversed: reset ranks invariant, AUC -> 1-AUC
    t2.append(np.mean(a2)); t3.append(np.mean(a3)); t2f.append(np.mean(a2f))
print('A4-T2 on v3.0 data: (AUCreset-.5)-.25(AUCtr-.5) p95 %.4f (pass if <=0)'%np.percentile(t2,95), 'mean %.4f'%np.mean(t2))
print('A4-T3 on v3.0 data: (.5-AUCswap)-.8(AUCtr-.5) p5 %.4f (pass if >=0)'%np.percentile(t3,5))
print('A4-T2 with yes=not harmful (reset AUC mirrored, assuming trained AUC same): p95 %.4f'%np.percentile(t2f,95))
# multiplier distribution
x=torch.load(R+'runs/brain-train/real-s1.pt')['kp_logm'].numpy(); f=torch.load(R+'runs/brain-train/real-flip-s1.pt')['kp_logm'].numpy()
e=np.exp(x); print('multiplier quantiles 1,10,50,90,99,99.9,max:',np.round(np.quantile(e,[.01,.1,.5,.9,.99,.999,1]),2), 'frac >20x %.3f  >100x %.4f'%((e>20).mean(),(e>100).mean()))
c=(x+f)/2; sp=(x-f)/2; print('common var %.3f specific var %.3f (real var %.3f); common mean %.3f'%(c.var(),sp.var(),x.var(),c.mean()))
lay=torch.load(R+'runs/brain-train/layered-s1.pt')['kp_logm'].numpy(); print('layered max logm %.2f, multiplier 99pct %.1f'%(lay.max(),np.quantile(np.exp(lay),.99)))
