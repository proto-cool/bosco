import pandas as pd, numpy as np
from bosco import data
import scipy.sparse as sp
E = np.load('/tmp/aud/edges.npz')
ids = E['ids']; pre=E['pre']; post=E['post']; w=E['w']; intot=E['intot']
a = data.annotations().reindex(ids)
sc = a.superclass.fillna('').to_numpy().astype(str)
cl = a['class'].fillna('').to_numpy().astype(str)
ty = a['type'].fillna('').to_numpy().astype(str)
sub = a['subclass'].fillna('').to_numpy().astype(str)
N=len(ids)
def cat_of():
    c = np.full(N,'other',dtype=object)
    c[cl=='olfactory']='ORN'
    c[cl=='thermosensory']='thermo'
    c[cl=='hygrosensory']='hygro'
    c[cl=='gustatory']='gustatory'
    c[(cl=='mechanosensory')&np.char.startswith(ty,'JO')]='JO'
    c[(cl=='mechanosensory')&~np.char.startswith(ty,'JO')]='mech_other'
    c[np.isin(cl,['mechanosensory_proprioceptive','mechanosensory_tactile','chemosensory','mechanosensory_tbc','unknown_sensory'])]='other_sensory'
    c[sc=='ol_sensory']='photoreceptor'
    c[sc=='ol_intrinsic']='optic_lobe'
    c[sc=='visual_projection']='VPN'
    c[sc=='visual_centrifugal']='VCN'
    c[cl=='ALPN']='ALPN'
    c[cl=='Kenyon_Cell']='KC'
    c[np.char.startswith(sc,'vnc')]='VNC'
    c[sc=='descending_neuron']='DN'
    c[sc=='ascending_neuron']='AN'
    return c
cat = cat_of()
A = sp.csr_matrix((w.astype(np.float64),(pre,post)),shape=(N,N))  # A[i,j] syn i->j
