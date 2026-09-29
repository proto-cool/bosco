import json, numpy as np, sys, collections
sys.path.insert(0,'/Users/nickd/projects/bosco/scripts')
import brain_check as BC
R='/Users/nickd/projects/bosco/data/cache/'
g=json.load(open(R+'v1-gate2/items.json'))['items']; Xg=np.load(R+'v1-gate2/emb.npz')['X']
split=np.array([it['split'] for it in g]); rng=np.random.default_rng(BC.SEED)
val=rng.permutation(np.nonzero(split=='val')[0])[:200]; tr=rng.permutation(np.nonzero(split=='train')[0])
used={'measure200':val,'cal256':tr[:256],'antfit4000':tr[256:4256]}
h=json.load(open(R+'v1-harm-dev/items.json'))['items']; Xh=np.load(R+'v1-harm-dev/emb.npz')['X']
hv=np.array([i for i,it in enumerate(h) if it['task']=='harmful' and it['split']=='val'])
ht=np.array([i for i,it in enumerate(h) if it['task']=='harmful' and it['split']=='train'])
hvt=set(h[i]['text'] for i in hv); htt=set(h[i]['text'] for i in ht)
for k,ix in used.items():
    tx=[g[i]['text'] for i in ix]
    print(k, 'tasks',collections.Counter(g[i]['task'] for i in ix).most_common(3),'exact∩harm val',len(set(tx)&hvt),'exact∩harm train',len(set(tx)&htt))
    S=Xh[hv]@Xg[ix].T
    print('   harm val items with cos>0.95 to',k, int((S.max(1)>0.95).sum()), ' >0.9', int((S.max(1)>0.9).sum()))
# whole gate2 hate train vs harm val
gh=np.array([i for i,it in enumerate(g) if it['task']=='hate'])
print('gate2 hate (all splits) exact ∩ harm val', len(set(g[i]['text'] for i in gh)&hvt), '∩ harm train', len(set(g[i]['text'] for i in gh)&htt))
