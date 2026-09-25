import numpy as np, torch, pandas as pd
from bosco import model2 as M2, a5, ratebrain2 as R, data
torch.manual_seed(0)
b2 = M2.load_or_build()
m = a5.build(b2,'real','type',5,device='cpu')
m.load_state_dict(torch.load('runs/gate-a5/real-type-s1.pt',map_location='cpu'))
m.eval()
tau = torch.exp(m.log_tau).clamp(*R.TAU_RANGE)
a = data.annotations().reindex(b2.brain.ids)
sc=a.superclass.fillna('').to_numpy().astype(str); ty=a.type.fillna('').to_numpy().astype(str); cl=a['class'].fillna('').to_numpy().astype(str)
unit = m.unit.numpy()
def tau_of(mask): return float(np.median(tau.detach().numpy()[unit[mask]]))
dn = sc=='descending_neuron'
print('trained tau median ms: all', float(tau.median()), 'KC', tau_of(cl=='Kenyon_Cell'), 'MBON', tau_of(cl=='MBON'), 'DN', tau_of(dn), 'ALPN', tau_of(cl=='ALPN'))
d = a5.load(m.nose.n)
s,v,y = d['sweet']['test']; s2,v2,y2 = d['dangerous']['test']
S = torch.tensor(np.concatenate([s[:64],s2[:64]])); V=torch.tensor(np.concatenate([v[:64],v2[:64]]))
R.STEPS = 160
with torch.no_grad():
    _, r, tr = m.run(S, V, record=True)
tr = tr.float().numpy()  # (T, n, B)
groups = {'ORN':cl=='olfactory','ALPN':cl=='ALPN','KC':cl=='Kenyon_Cell','MBON':cl=='MBON','DN':dn,'LH':np.char.startswith(ty,'LH'),
 'DNp09':ty=='DNp09','DNa02':ty=='DNa02','DNa03':ty=='DNa03','MDN':ty=='MDN','DNp42':ty=='DNp42','DNp52':ty=='DNp52','DNg104':ty=='DNg104'}
ap=m.ap.numpy(); av=m.av.numpy()
rows=[]
sig = {}
for g,msk in groups.items():
    x = tr[:, msk, :]            # T, cells, B
    spread = x.std(2).mean(1)     # stimulus-dependent spread per step
    sig[g]=spread
readd = tr[:,ap,:].mean(1)-tr[:,av,:].mean(1)
sig['MBON ap-av'] = readd.std(1)
out = pd.DataFrame(sig)
fin = out.iloc[-1]
print('stimulus-dependent spread (std across 128 stimuli), selected steps')
print(out.iloc[[0,1,2,3,4,6,8,12,16,20,24,32,39,48,64,80,120,159]].to_string(float_format=lambda z:f'{z:.2e}'))
def t_reach(col, q):
    s = out[col].to_numpy(); f=s[-1]; 
    i = np.argmax(s>=q*f) if f>0 else -1
    return i+1
print({c:(t_reach(c,0.5),t_reach(c,0.9)) for c in out.columns})
print('ratio spread at step40 / step160', (out.iloc[39]/out.iloc[-1]).round(3).to_dict())
# correlation of DN spread with read
last = tr[39]  # step 40
dnm = np.nonzero(dn)[0]
act = (last[dnm]>1e-3).any(1); var = last[dnm].std(1)>1e-4
print('DN cells', len(dnm), 'active (rate>1e-3 for any stimulus) at step40', act.sum(), 'stimulus-varying', var.sum())
print('active DN types', sorted(set(ty[dnm[act]]))[:80])
for k,v in m.behaviour.items():
    x = last[v.numpy()]
    print('behaviour', k, len(v), 'active', int((x>1e-3).any(1).sum()), 'mean', float(x.mean()))
mb = cl=='MBON'
print('MBON active', int((last[mb]>1e-3).any(1).sum()), 'of', mb.sum())
print('KC active frac mean', float((last[cl=='Kenyon_Cell']>0.01).mean()))
print('all neurons active frac', float((last>1e-3).any(1).mean()))
