import sys; sys.path.insert(0,'/tmp/aud')
from common import *
from bosco import model2 as M2, populations as pop
inm = np.isin(ids, M2.model_body_ids())
ol = np.isin(sc,['ol_intrinsic','visual_centrifugal','ol_sensory','visual_projection_tbc'])
kcm = np.isin(ids, pop.kenyon_cells()); mbm = np.isin(ids, pop.mbons()['bodyId'])
def stats(nodes, label):
    k = nodes[pre]&nodes[post]
    kk = k & ((w>=5) | (kcm[pre]&mbm[post]))
    print(f'{label}: neurons {nodes.sum():,} edges {k.sum():,} syn {int(w[k].sum()):,} edges>=5(+KC->MBON) {kk.sum():,} syn in those {int(w[kk].sum()):,}')
    return k.sum(), kk.sum()
e0,c0 = stats(inm,'current model')
e1,c1 = stats(inm|ol,'model + optic lobes (ol_intrinsic+VCN+photoreceptors)')
print('multiplier edges', e1/e0, 'cut edges', c1/c0, 'neurons', (inm|ol).sum()/inm.sum())
# only photoreceptor + ol_intrinsic sides
side = a.somaSide.fillna('').to_numpy().astype(str)
rside = a.rootSide.fillna('').to_numpy().astype(str)
for s in ['L','R']:
    n_ol = ((sc=='ol_intrinsic')&(side==s)).sum(); n_vcn=((sc=='visual_centrifugal')&(side==s)).sum(); n_vpn=((sc=='visual_projection')&(side==s)).sum()
    n_pr = ((sc=='ol_sensory')&(rside==s)).sum()
    olm = (sc=='ol_intrinsic')&(side==s)
    k = olm[post]
    print(f'side {s}: ol_intrinsic {n_ol:,}, VCN {n_vcn}, VPN {n_vpn:,}, photoreceptors {n_pr:,}; synapses onto ol_intrinsic {int(w[k].sum()):,} edges {k.sum():,}')
# synapses within optic lobe superclasses, all traced
olall = np.isin(sc,['ol_intrinsic','visual_centrifugal','ol_sensory','visual_projection'])
k = olall[pre]|olall[post]
print('edges touching optic-lobe/VPN bodies', k.sum(), 'syn', int(w[k].sum()))
print('total traced-traced syn', int(w.sum()), 'edges', len(w))
# per-source: VPN input from ol_intrinsic
vp = sc=='visual_projection'
kv = vp[post]
print('VPN input total (any)', int(intot[vp].sum()), 'from traced ol_intrinsic', int(w[kv & (sc[pre]=='ol_intrinsic')].sum()), 'from photoreceptors', int(w[kv & (sc[pre]=='ol_sensory')].sum()), 'from model neurons', int(w[kv & inm[pre]].sum()))
