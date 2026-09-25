import pandas as pd, numpy as np
from bosco import data
pd.set_option('display.width',250)
a = data.annotations()
pr = a[a.superclass=='ol_sensory']
print(pr.groupby(['status']).size())
print(pr.groupby(['type','rootSide'],dropna=False).size().unstack(fill_value=0).to_string())
print(a[a.superclass.isin(['ol_intrinsic','visual_projection','visual_centrifugal','ol_sensory'])].groupby(['superclass','status']).size().unstack(fill_value=0))
t=a[a.status=='Traced']
for sc in ['ol_intrinsic','visual_projection','visual_centrifugal','ol_sensory']:
    s=t[t.superclass==sc]
    print(sc, 'somaSide', s.somaSide.value_counts(dropna=False).to_dict(), 'rootSide', s.rootSide.value_counts(dropna=False).to_dict())
oc = t[t.type.fillna('').str.match(r'OC[GC]')]
print(oc.groupby(['superclass','class']).size())
print('ocellar photoreceptor-like?', t[t.type.fillna('').str.contains('ocellar|OcP',case=False)].type.unique()[:20])
print('col-hex', t[t.superclass=='ol_intrinsic'].assignedOlHex1.notna().sum())
