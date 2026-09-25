import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
from bosco import model2 as M2
inmodel = np.isin(ids, M2.model_body_ids())
print('model n', inmodel.sum(), 'model cats', pd.Series(cat[inmodel]).value_counts().to_dict())
kc = cl=='Kenyon_Cell'
kct = np.where(kc, ty, '')
# direct input to KC types by presyn category
rows=[]
cats = sorted(set(cat))
AT = A.T.tocsr()  # AT[j,i] = syn i->j
for t in ['KCg-m','KCg-d','KCab-s','KCab-m','KCab-c','KCab-p',"KCa'b'-ap1","KCa'b'-ap2","KCa'b'-m",'other']:
    m = (kct==t) if t!='other' else (kc & ~np.isin(kct,['KCg-m','KCg-d','KCab-s','KCab-m','KCab-c','KCab-p',"KCa'b'-ap1","KCa'b'-ap2","KCa'b'-m"]))
    inp = np.asarray(A[:,m].sum(1)).ravel()
    r={'KCtype':t,'n':int(m.sum()),'in_total_any':int(intot[m].sum())}
    for c in cats: r[c]=int(inp[cat==c].sum())
    rows.append(r)
df=pd.DataFrame(rows).set_index('KCtype')
df=df.loc[:,(df!=0).any()]
print(df.to_string())
# fraction
print((df.drop(columns=['n','in_total_any']).div(df.in_total_any,axis=0)*100).round(2).to_string())
df.to_csv('/tmp/aud/kc_direct.csv')
