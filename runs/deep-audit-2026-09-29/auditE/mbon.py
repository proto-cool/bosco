import numpy as np, pandas as pd, json
from bosco import model2 as M2, wiring3 as W3, data
b2=M2.load_or_build(); b=b2.brain
tab,ap,av=M2.mbon_groups(b)
ap2,av2,tab2=W3.mbon_valence(b)
t=tab[['type','cells','from_PAM','from_PPL1','side']].merge(tab2[['type','nt','side']],on='type',suffixes=('_v30','_v31'))
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print(t.to_string())
print(pd.crosstab(t.side_v30,t.side_v31))
print('cells approach v30',len(ap),'v31',len(ap2),'; avoid v30',len(av),'v31',len(av2))
# DN read cells' transmitters and types
rd=json.load(open('runs/brain-check/dn_read.json'))
lab=W3.transmitter(b.ids).to_numpy()
a=data.annotations().reindex(b.ids)
for side in ('approach','avoid'):
    idx=np.array(rd[side]); df=pd.DataFrame({'type':a['type'].to_numpy()[idx],'nt':lab[idx]})
    print(side, df.groupby('type').nt.agg(lambda s: s.value_counts().to_dict()).to_dict())
