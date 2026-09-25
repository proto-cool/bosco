import pandas as pd, numpy as np
from bosco import data
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
a = data.annotations(); t=a[a.status=='Traced']
print(t[t.superclass=='cb_intrinsic']['class'].value_counts(dropna=False).to_string())
print(t[t['class']=='Kenyon_Cell'].type.value_counts(dropna=False).to_string())
print(t[t.type.astype(str).str.startswith('OC')].groupby(['superclass','class'],dropna=False).size())
print(t[t.superclass=='visual_projection']['class'].value_counts(dropna=False).head(20))
print(t[t.superclass=='descending_neuron']['class'].value_counts(dropna=False))
print(t[t.superclass=='descending_neuron'].type.nunique())
