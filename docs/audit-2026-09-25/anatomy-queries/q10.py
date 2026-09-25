import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
kc = cl=='Kenyon_Cell'
AT = A.T.tocsr()
alpn = cat=='ALPN'; vpn = cat=='VPN'
fromPN = np.asarray(A[alpn][:,kc].sum(0)).ravel()
fromVPN = np.asarray(A[vpn][:,kc].sum(0)).ravel()
# ORN-dominated PN? use all ALPN (includes thermo/hygro PNs)
k_ty = ty[kc]
df = pd.DataFrame({'type':k_ty,'pn':fromPN,'vpn':fromVPN,'intot':intot[kc]})
for th in [5,10,20]:
    df[f'both{th}'] = (df.pn>=th)&(df.vpn>=th)
print(df.groupby('type')[['pn','vpn','both5','both10','both20']].agg({'pn':'median','vpn':'median','both5':'sum','both10':'sum','both20':'sum'}).to_string())
print('KCs with vpn>=5:', (df.vpn>=5).sum(), ' with pn>=5:', (df.pn>=5).sum(), ' both>=5', df.both5.sum(), ' both>=10', df.both10.sum())
# number of KCs with any VPN input >= 5 by type
print(df[df.vpn>=5].type.value_counts().to_string())
