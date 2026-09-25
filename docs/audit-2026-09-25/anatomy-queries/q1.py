import pandas as pd, numpy as np
from bosco import data
a = data.annotations()
print(a.columns.tolist()); print(len(a))
print(a['status'].value_counts().head(10))
t = a[a.status=='Traced']
print(t['superclass'].value_counts(dropna=False))
w = pd.read_feather('data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
print(w.columns.tolist(), len(w), w['weight'].sum())
