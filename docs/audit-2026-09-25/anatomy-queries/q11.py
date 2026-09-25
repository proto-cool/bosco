import numpy as np, pandas as pd, yaml
from bosco import data, model2 as M2, a5
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, breadth_first_order
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
b2 = M2.load_or_build(); b = b2.brain
c = a5.cut(b, 5)
print('model n', b.n, 'edges full', len(b.indices), 'syn', int(b.count.sum()), '| cut edges', len(c.indices), 'syn', int(c.count.sum()))
a = data.annotations().reindex(b.ids)
sc=a.superclass.fillna('').to_numpy(); ty=a.type.fillna('').to_numpy().astype(str); cl=a['class'].fillna('').to_numpy()
print('DN superclasses in model', pd.Series(sc[np.char.find(sc.astype(str),'descending')>=0]).value_counts().to_dict())
dn = np.nonzero(sc=='descending_neuron')[0]
print('DN types', len(set(ty[dn])), 'untyped', (ty[dn]=='').sum())
cfg = yaml.safe_load(open('config/readout_populations.yaml'))['populations']
for k,v in cfg.items():
    for t in v:
        m = np.nonzero(ty==t)[0]
        print(k,t,len(m), set(sc[m]))
tab, ap, av = M2.mbon_groups(b)
print(tab.to_string())
print('ap',len(ap),'av',len(av))
