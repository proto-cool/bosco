"""A4b dev L3: joint smell of (text, label). Ceiling via logistic 'fits' probe on 46 channels; practice test only."""
import sys, json, time; sys.path.insert(0,'scripts')
import numpy as np, a4b_dev as D
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
meta,X,L=D.load(); items=meta["items"]; labels=meta["labels"]
rng=np.random.default_rng(1)
tr=[i for i,it in enumerate(items) if it["split"]=="train"]; tr=list(rng.permutation(tr)[:6000])
pr=[i for i,it in enumerate(items) if it["split"]=="practice"]
m=SentenceTransformer("intfloat/multilingual-e5-large-instruct",device="mps")
def txt(i,o): return f"Instruct: Given a label, judge whether the text fits it\nQuery: label: {labels[o]}\ntext: {' '.join(items[i]['text'].split()[:100])}"
# train pairs: gold + 3 random own-kind wrong labels
pairs=[];y=[]
for i in tr:
    it=items[i]; g=it["opts"][it["gold"]]; wrong=[o for o in it["opts"] if o!=g]; rng.shuffle(wrong)
    for o in [g]+wrong[:3]: pairs.append((i,o)); y.append(int(o==g))
t0=time.time()
E=m.encode([txt(i,o) for i,o in pairs],batch_size=64,normalize_embeddings=True); print("train emb",len(pairs),round(time.time()-t0),flush=True)
pp=[(i,o) for i in pr for o in items[i]["opts"]]
P=m.encode([txt(i,o) for i,o in pp],batch_size=64,normalize_embeddings=True); print("practice emb",len(pp),round(time.time()-t0),flush=True)
np.savez(f"{sys.argv[1]}/joint.npz",E=E,y=np.array(y),P=P,pairs=np.array(pairs),pp=np.array(pp))
mu=E.mean(0); _,s,vt=np.linalg.svd(E-mu,full_matrices=False)
def score(feat_tr,feat_pr,name):
    lr=LogisticRegression(max_iter=3000,C=1.0).fit(feat_tr,y); sc=lr.decision_function(feat_pr)
    res={}; k=0
    for i in pr:
        n=len(items[i]["opts"]); pick=int(np.argmax(sc[k:k+n])); k+=n
        res.setdefault(items[i]["kind"],[]).append(int(pick==items[i]["gold"]))
    print(f"{name:28s} practice macro {D.macro({a:np.mean(b) for a,b in res.items()}):.3f}",flush=True)
score(E,P,"joint, all 1024")
for k in (23,46,100):
    W=vt[:k]; score((E-mu)@W.T,(P-mu)@W.T,f"joint, {k} components")
