import sys; sys.path.insert(0,'/tmp/aud')
from common import *
pd.set_option('display.width',250)
senses = ['ORN','thermo','hygro','gustatory','JO','mech_other','other_sensory','photoreceptor','VPN']
F1=np.load('/tmp/aud/F1.npy'); F2=np.load('/tmp/aud/F2.npy')
Pn = sp.csr_matrix(A.multiply(1.0/np.maximum(intot,1)[None,:])); PT=Pn.T.tocsr()
F3 = PT@F2
groups = {'PAM':np.char.startswith(ty,'PAM'),'PPL1':np.char.startswith(ty,'PPL1'),'MBON':cl=='MBON','LH (LH*)':np.char.startswith(ty,'LH'),'DN':sc=='descending_neuron','DNp09':ty=='DNp09','MDN':ty=='MDN','DNa02':ty=='DNa02'}
for k,F in [(1,F1),(2,F2),(3,F3)]:
    print('hop',k); print(pd.DataFrame({g:F[m].mean(0)*100 for g,m in groups.items()},index=senses).T.round(3).to_string())
print('PAM n', groups['PAM'].sum(), 'PPL1 n', groups['PPL1'].sum(), 'DAN class n', (cl=='DAN').sum())
