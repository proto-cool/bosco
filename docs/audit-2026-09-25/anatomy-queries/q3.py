import pandas as pd, numpy as np, re
from bosco import data
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
a = data.annotations(); t = a[a.status=='Traced']
def show(m, col='type', n=80):
    print(t[m].groupby([col,'somaSide'],dropna=False).size().unstack(fill_value=0).head(n).to_string())
print('--visual ol_sensory'); show(t.superclass=='ol_sensory')
print('--thermo'); show(t['class']=='thermosensory')
print('--hygro'); show(t['class']=='hygrosensory')
print('--gust cb'); print(t[(t['class']=='gustatory')].groupby(['superclass','subclass']).type.agg(lambda s: sorted(set(s.dropna()))[:40]).to_string())
m = t['class']=='mechanosensory'
print('--mech types', t[m].type.fillna('NA').str.replace(r'_.*','',regex=True).value_counts().head(40).to_string())
print(t[m].groupby('subclass',dropna=False).type.agg(lambda s: sorted(set(s.dropna().str[:6]))[:30]).to_string())
print('--unknown cb', t[(t['class']=='unknown_sensory')&(t.superclass.str.startswith('cb')|t.superclass.str.startswith('sensory'))].type.value_counts().head(30).to_string())
orn = t[(t['class']=='olfactory')]
print('--olf types prefix', orn.type.fillna('NA').str[:4].value_counts().to_string())
g = orn[orn.type.fillna('').str.startswith('ORN_')].type.str[4:]
print('n glomeruli', g.nunique(), 'ORNs', len(g)); print(g.value_counts().sort_index().to_string())
print('non-ORN olfactory types', orn[~orn.type.fillna('').str.startswith('ORN_')].type.value_counts(dropna=False).head(20).to_string())
print('--vis centrifugal', t[t.superclass=='visual_centrifugal'].type.nunique())
print('--ocelli?', t[t.type.fillna('').str.contains('ocell|OCG|OcB|OCC', case=False)].type.value_counts().to_string())
