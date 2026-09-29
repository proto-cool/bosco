import json, numpy as np, collections
D='/Users/nickd/projects/bosco/data/cache/v1-harm-dev/'
m=json.load(open(D+'items.json')); its=m['items']
X=np.load(D+'emb.npz')['X']
print(X.shape, X.dtype, np.linalg.norm(X[:3],axis=1))
tr=np.array([i for i,it in enumerate(its) if it['task']=='harmful' and it['split']=='train'])
va=np.array([i for i,it in enumerate(its) if it['task']=='harmful' and it['split']=='val'])
A=X[tr]/np.linalg.norm(X[tr],axis=1,keepdims=True); B=X[va]/np.linalg.norm(X[va],axis=1,keepdims=True)
S=B@A.T
mx=S.max(1); am=S.argmax(1)
for t in (0.9,0.95,0.98,0.99):
    print('val items with a train neighbour cos >',t, int((mx>t).sum()), f'{(mx>t).mean():.3%}')
ytr=np.array([its[i]['gold']==0 for i in tr]); yva=np.array([its[i]['gold']==0 for i in va])
k=mx>0.95
print('label agreement with nearest train at >0.95:', (ytr[am[k]]==yva[k]).mean(), 'n',k.sum())
print('sources of >0.95 val:', collections.Counter(its[va[i]]['source'] for i in np.nonzero(k)[0]))
for i in np.nonzero(k)[0][:8]:
    print(round(float(mx[i]),3), repr(its[va[i]]['text'][:90]), '||', repr(its[tr[am[i]]]['text'][:90]), yva[i], ytr[am[i]])
# 1-NN accuracy overall
pred=ytr[am]
def ba(y,p): return 0.5*((p[y]).mean()+(~p[~y]).mean())
print('1-NN balanced acc on val (full 768 emb):', ba(yva,pred))
# source-only classifier
src_tr=collections.defaultdict(list)
for i in tr: src_tr[its[i]['source']].append(its[i]['gold']==0)
rate={s:np.mean(v) for s,v in src_tr.items()}
print('train harmful rate per source',rate)
p=np.array([rate[its[i]['source']]>=0.5 for i in va])
print('source-only rule BA on val', ba(yva,p))
# source ranking AUC
from itertools import product
sc=np.array([rate[its[i]['source']] for i in va])
def auc(y,s):
    r=np.argsort(np.argsort(s,kind='mergesort'),kind='mergesort')+1.0
    # ties: average ranks
    import scipy.stats as st
    r=st.rankdata(s)
    n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
print('source-only AUC', auc(yva,sc))
np.save('/Users/nickd/.claude/jobs/4059153c/tmp/auditE/nn_max.npy', mx)
