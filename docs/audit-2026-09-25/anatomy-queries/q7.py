import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
kc = cl=='Kenyon_Cell'
for t in ['KCg-d','KCab-p','KCg-m',"KCa'b'-ap2"]:
    m = kc & (ty==t)
    inp = np.asarray(A[:,m].sum(1)).ravel()
    s = pd.DataFrame({'type':ty,'cat':cat,'sc':sc,'syn':inp})
    s = s[(s.syn>0)&(s.cat!='KC')].groupby(['type','cat','sc']).syn.agg(['sum','size']).sort_values('sum',ascending=False)
    print('==',t, 'total nonKC', s['sum'].sum()); print(s.head(25).to_string())
# all VPN -> KC by VPN type
m = kc
inp = np.asarray(A[:,m].sum(1)).ravel()
s = pd.DataFrame({'type':ty,'cat':cat,'syn':inp})
print('== VPN/VCN/optic -> all KC'); print(s[s.cat.isin(['VPN','VCN','optic_lobe','photoreceptor'])&(s.syn>0)].groupby(['cat','type']).syn.agg(['sum','size']).sort_values('sum',ascending=False).head(30).to_string())
