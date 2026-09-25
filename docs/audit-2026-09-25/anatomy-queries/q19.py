import sys; sys.path.insert(0,'/tmp/aud')
from common import *
vp = sc=='visual_projection'; dn = sc=='descending_neuron'
for t in ['DNp09','DNa02','MDN','DNp01']:
    m=ty==t
    inp=np.asarray(A[vp][:, m].sum(1)).ravel()
    s=pd.Series(inp,index=ty[vp]).groupby(level=0).sum().sort_values(ascending=False)
    print(t, 'in_total', int(intot[m].sum()), 'from VPN', int(inp.sum()), s.head(5).to_dict())
tot = A[vp][:, dn].sum()
print('VPN->DN total syn', int(tot), 'DN in_total', int(intot[dn].sum()))
