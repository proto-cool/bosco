import pandas as pd, numpy as np
from bosco import data
a = data.annotations()
t = a[a.status=='Traced']
ids = np.sort(t.index.to_numpy().astype(np.int64))
w = pd.read_feather('data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
idx = pd.Index(ids)
pre = idx.get_indexer(w.body_pre.to_numpy()); post = idx.get_indexer(w.body_post.to_numpy())
k = (pre>=0)&(post>=0)
print('all edges', len(w), 'syn', int(w.weight.sum()), 'traced-traced edges', int(k.sum()), 'syn', int(w.weight.to_numpy()[k].sum()))
# in_total from any body for traced
cnt = w.weight.to_numpy()
intot = np.bincount(post[post>=0], weights=cnt[post>=0], minlength=len(ids))
outtot = np.bincount(pre[pre>=0], weights=cnt[pre>=0], minlength=len(ids))
np.savez('/tmp/aud/edges.npz', ids=ids, pre=pre[k].astype(np.int32), post=post[k].astype(np.int32), w=cnt[k].astype(np.int32), intot=intot, outtot=outtot)
