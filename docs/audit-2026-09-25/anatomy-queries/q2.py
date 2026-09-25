import pandas as pd, numpy as np
from bosco import data, model2 as M2
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
a = data.annotations()
t = a[a.status=='Traced']
model = set(M2.model_body_ids().tolist())
t = t.assign(inmodel=t.index.isin(list(model)))
sens = t[t.superclass.fillna('').str.contains('sensory')]
print(sens.groupby(['superclass','class'],dropna=False).agg(n=('inmodel','size'),inmodel=('inmodel','sum')))
print(sens.groupby(['class','subclass'],dropna=False).agg(n=('inmodel','size'),inmodel=('inmodel','sum')).to_string())
print(sens.receptorType.value_counts().head(40))
print(sens.entryNerve.value_counts().head(30))
print('model n', len(model))
