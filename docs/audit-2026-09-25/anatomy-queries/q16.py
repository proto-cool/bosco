import sys; sys.path.insert(0,'/tmp/aud')
from common import *
side = a.somaSide.fillna('').to_numpy().astype(str)
for t in ['L1','L2','L3','Mi1','Tm5a','Dm8a','Dm8b','Tm20']:
    m = ty==t
    if not m.any(): continue
    fromPR = np.asarray(A[cat=='photoreceptor'][:,m].sum(0)).ravel()
    for s in ['L','R']:
        mm = side[m]==s
        print(t,s,'n',mm.sum(),'median in_total',np.median(intot[m][mm]),'median from traced PR',np.median(fromPR[mm]), 'frac cells with PR input>0', (fromPR[mm]>0).mean().round(3))
# how much of all input to photoreceptors' targets... also R7/R8 synapse count
for t in ['R1-R6','R7p','R7y','R8p','R8y']:
    m=ty==t; out=np.asarray(A[m].sum(1)).ravel(); print(t, m.sum(), 'median output syn', np.median(out))
