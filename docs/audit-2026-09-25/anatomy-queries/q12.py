import numpy as np, pandas as pd, yaml
from bosco import data, model2 as M2, a5
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, dijkstra
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
b2 = M2.load_or_build(); b = b2.brain
a = data.annotations().reindex(b.ids)
sc=a.superclass.fillna('').to_numpy().astype(str); ty=a.type.fillna('').to_numpy().astype(str); cl=a['class'].fillna('').to_numpy().astype(str)
tab, ap, av = M2.mbon_groups(b)
mball = np.nonzero(cl=='MBON')[0]
lh = np.nonzero(np.char.startswith(ty,'LHAV')|np.char.startswith(ty,'LHAD')|np.char.startswith(ty,'LHPV')|np.char.startswith(ty,'LHPD')|np.char.startswith(ty,'LHCENT'))[0]
# LH output neurons by type prefix; exclude those with superclass not cb_intrinsic
print('LH-named neurons', len(lh))
dn = np.nonzero(sc=='descending_neuron')[0]
cand = {'approach/forward':['DNp09','DNa01','DNa02','DNa03','DNa04','DNb01','DNb02','DNb05','DNb06','DNp42','DNa13'],
        'avoid/escape':['DNp01','DNp02','DNp03','DNp04','DNp06','DNp10','DNp11','MDN','DNp03']}
res={}
for name,br in [('full',b),('cut5',a5.cut(b,5))]:
    pre = br.pre_of_edges(); post=br.indices
    n=br.n
    G = sp.csr_matrix((np.ones(len(pre)),(pre,post)),shape=(n,n))
    P = sp.csr_matrix((br.count/np.maximum(b2.in_total[post],1),(pre,post)),shape=(n,n))
    S = sp.csr_matrix((br.sign*br.count/np.maximum(b2.in_total[post],1),(pre,post)),shape=(n,n))
    res[name]=(G,P,S)
    for src_name, src in [('MBON all',mball),('MBON approach',ap),('MBON avoid',av),('LH',lh)]:
        d = dijkstra(G, indices=src, unweighted=True, min_only=True)
        dd = d[dn]
        print(name, src_name, 'hops to DNs: min', dd.min(), 'median', np.median(dd[np.isfinite(dd)]), 'unreachable', np.isinf(dd).sum(), 'hist', np.unique(dd, return_counts=True))
        # candidate
        row = {t: d[ty==t].min() if (ty==t).any() else None for g in cand.values() for t in g}
        print('   cand hops', row)
    # k-hop influence (unsigned and signed) from each source into candidate DN types
    PT = P.T.tocsr(); ST = S.T.tocsr()
    for src_name, src in [('MBON approach',ap),('MBON avoid',av),('MBON all',mball),('LH',lh)]:
        f = np.zeros(n); f[src]=1; g=f.copy()
        out=[]
        cum = np.zeros(n); cums=np.zeros(n)
        for k in range(1,9):
            f = PT@f; g = ST@g
            f[src]=f[src]  # propagation includes recurrent
            cum+=f; cums+=g
            out.append({'k':k,'DN_all_mean_%':100*f[dn].mean(), 'DNp09':100*f[ty=='DNp09'].mean(),'DNa02':100*f[ty=='DNa02'].mean(),'MDN':100*f[ty=='MDN'].mean(),'DNp01':100*f[ty=='DNp01'].mean(),'DNp09_signed':100*g[ty=='DNp09'].mean(),'MDN_signed':100*g[ty=='MDN'].mean()})
        print(name, src_name); print(pd.DataFrame(out).round(4).to_string(index=False))
