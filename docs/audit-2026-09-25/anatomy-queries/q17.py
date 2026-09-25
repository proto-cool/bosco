import sys; sys.path.insert(0,'/tmp/aud')
from common import *
from bosco import model2 as M2
inm = np.isin(ids, M2.model_body_ids())
frm = np.bincount(post[inm[pre]], weights=w[inm[pre]], minlength=N)
for name,m in [('DN',sc=='descending_neuron'),('VPN',sc=='visual_projection'),('KC',cl=='Kenyon_Cell'),('MBON',cl=='MBON'),('ALPN',cl=='ALPN'),('LH',np.char.startswith(ty,'LH')),('all model',inm)]:
    print(name, 'median share of input from model neurons', np.round(np.median(frm[m]/np.maximum(intot[m],1)),3), 'from traced bodies', np.round(np.median(np.bincount(post,weights=w,minlength=N)[m]/np.maximum(intot[m],1)),3))
for t in ['DNp09','DNa02','DNa03','MDN','DNp42','DNp01']:
    m=ty==t; print(t, 'share from model', np.round(frm[m]/np.maximum(intot[m],1),3), 'from VNC', np.round(np.bincount(post[cat[pre]=='VNC'],weights=w[cat[pre]=='VNC'],minlength=N)[m]/intot[m],3))
vis_named = (sc=='cb_intrinsic') & (np.char.startswith(ty,'LoVP')|np.char.startswith(ty,'MeVP')|np.char.startswith(ty,'LC')|np.char.startswith(ty,'LT')|np.char.startswith(ty,'aMe'))
print('cb_intrinsic cells with visual-PN-style names', vis_named.sum(), sorted(set(ty[vis_named]))[:30])
oc = (sc=='visual_projection') & np.char.startswith(ty,'OC')
print('ocellar types inside visual_projection', oc.sum(), sorted(set(ty[oc])))
# VPNs that reach KCs
kc = cl=='Kenyon_Cell'
toKC = np.asarray(A[:,kc].sum(1)).ravel()
vp = sc=='visual_projection'
print('VPN types with >=10 syn onto KCs', len(set(ty[vp & (toKC>=5)])), 'cells', (vp&(toKC>=5)).sum(), 'total VPN->KC syn', int(toKC[vp].sum()))
