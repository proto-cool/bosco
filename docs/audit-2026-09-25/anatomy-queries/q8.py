import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250)
senses = ['ORN','thermo','hygro','gustatory','JO','mech_other','other_sensory','photoreceptor','VPN']
F0 = np.stack([(cat==s).astype(float) for s in senses],1)
Pn = sp.csr_matrix(A.multiply(1.0/np.maximum(intot,1)[None,:]))  # col-normalised
PT = Pn.T.tocsr()
F1 = PT@F0; F2 = PT@F1; F3 = PT@F2; F4=PT@F3
kc = cl=='Kenyon_Cell'
types=['KCg-m','KCg-d','KCab-s','KCab-m','KCab-c','KCab-p',"KCa'b'-ap1","KCa'b'-ap2","KCa'b'-m"]
for k,F in enumerate([F1,F2,F3,F4],1):
    rows={t:F[kc&(ty==t)].mean(0)*100 for t in types}
    print(f'-- hop {k}: % of KC input attributable to sense (path-weight product, input-normalised)')
    print(pd.DataFrame(rows,index=senses).T.round(3).to_string())
# absolute 2-hop equivalent synapses: sum_i syn(i->KC)*F1_i
Akc = A[:,kc]
inp = np.asarray(Akc.sum(1)).ravel()
print('-- 2-hop equivalent synapses onto all KCs (sum syn(i->KC)*frac_i(sense))')
print(dict(zip(senses,(inp@F1).round(0))))
print('-- 3-hop equivalent', dict(zip(senses,(inp@F2).round(0))))
np.save('/tmp/aud/F1.npy',F1); np.save('/tmp/aud/F2.npy',F2)
