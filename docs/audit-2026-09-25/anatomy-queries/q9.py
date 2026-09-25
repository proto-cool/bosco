import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
kc = cl=='Kenyon_Cell'
pr = cat=='photoreceptor'
# photoreceptor -> X -> KC
toK = np.asarray(A[:,kc&np.isin(ty,['KCg-d','KCab-p'])].sum(1)).ravel()
fromPR = np.asarray(A[pr,:].sum(0)).ravel()
s = pd.DataFrame({'type':ty,'cat':cat,'fromPR':fromPR,'toVisKC':toK})
print(s[(s.fromPR>0)&(s.toVisKC>0)].groupby(['cat','type'])[['fromPR','toVisKC']].sum().sort_values('toVisKC',ascending=False).head(15).to_string())
print('PR total output', A[pr,:].sum(), 'PR out to optic_lobe', A[pr][:, cat=='optic_lobe'].sum(), 'to VPN', A[pr][:, cat=='VPN'].sum(),'to VCN',A[pr][:, cat=='VCN'].sum())
# KC types -> MBON types
mb = cl=='MBON'
types=['KCg-m','KCg-d','KCab-s','KCab-m','KCab-c','KCab-p',"KCa'b'-ap1","KCa'b'-ap2","KCa'b'-m"]
M = {t: pd.Series(np.asarray(A[kc&(ty==t)][:,mb].sum(0)).ravel(), index=ty[mb]).groupby(level=0).sum() for t in types}
df = pd.DataFrame(M).fillna(0).astype(int)
df = df[df.sum(1)>0]
print(df.to_string())
