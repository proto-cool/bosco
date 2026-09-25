import numpy as np, pandas as pd
from bosco import data, model2 as M2, a5
import scipy.sparse as sp
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
b2 = M2.load_or_build(); b = b2.brain
a = data.annotations().reindex(b.ids)
sc=a.superclass.fillna('').to_numpy().astype(str); ty=a.type.fillna('').to_numpy().astype(str); cl=a['class'].fillna('').to_numpy().astype(str)
nt = data.neurotransmitters().reindex(pd.Index(b.ids))
mb = cl=='MBON'; dn = sc=='descending_neuron'
pre=b.pre_of_edges(); post=b.indices
m = mb[pre]&dn[post]
e = pd.DataFrame({'mbon':ty[pre[m]],'dn':ty[post[m]],'syn':b.count[m],'sign':b.sign[m],'frac_of_dn_input':b.count[m]/b2.in_total[post[m]]})
g = e.groupby(['mbon','dn']).agg(syn=('syn','sum'),sign=('sign','first'),frac=('frac_of_dn_input','mean')).sort_values('syn',ascending=False)
print('direct MBON->DN total syn', e.syn.sum(), 'edges', len(e), 'edges>=5', (e.syn>=5).sum()); print(g.head(30).to_string())
# rank DN types by 1-3 hop influence from MBONs (cut5)
c = a5.cut(b,5); pre=c.pre_of_edges(); post=c.indices
P = sp.csr_matrix((c.count/np.maximum(b2.in_total[post],1),(pre,post)),shape=(b.n,b.n)).T.tocsr()
S = sp.csr_matrix((c.sign*c.count/np.maximum(b2.in_total[post],1),(pre,post)),shape=(b.n,b.n)).T.tocsr()
tab, ap, av = M2.mbon_groups(b)
def infl(src, M, K=3):
    f=np.zeros(b.n); f[src]=1; cum=np.zeros(b.n)
    for k in range(K): f=M@f; cum+=f
    return cum
ua=infl(np.nonzero(mb)[0],P); sa=infl(ap,S); sv=infl(av,S)
d = pd.DataFrame({'type':ty,'u':ua*100,'s_ap':sa*100,'s_av':sv*100})[dn].groupby('type').mean()
d['ap_minus_av'] = d.s_ap - d.s_av
print('top DN types by unsigned MBON influence (1-3 hops, % of input)'); print(d.sort_values('u',ascending=False).head(25).round(3).to_string())
print('DN types median u', d.u.median().round(4))
# NT of key cells
for t in ['MBON01','MBON02','MBON03','MBON04','MBON05','MBON06','MBON07','MBON09','MBON10','MBON11','MBON12','MBON13','MBON14','MBON15','MBON16','MBON17','MBON18','MBON19','MBON20','MBON21','MBON22','MBON23','MBON24','MBON25','MBON26','MBON27','MBON28','MBON29','MBON30','MBON31','MBON32','MBON33','MBON34','MBON35','APL','DPM','MB-C1','DNp09','DNa02','MDN','DNp01','DNa01']:
    ii = np.nonzero(ty==t)[0]
    print(t, len(ii), nt.iloc[ii][['consensus_nt','predicted_nt']].astype(str).value_counts().to_dict(), 'model sign', set(b.nt_sign[ii].tolist()), set(b2.sign_source[ii].tolist()))
