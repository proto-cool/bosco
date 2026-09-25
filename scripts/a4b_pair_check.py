import sys, time; sys.path.insert(0,'scripts')
import numpy as np, torch, a4b_dev as D
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
meta,X,L=D.load(); items=meta["items"]; labels=meta["labels"]
z=np.load(sys.argv[1]+"/joint.npz"); pairs=z["pairs"]; y=z["y"]; pp=z["pp"]
name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForSequenceClassification.from_pretrained(name).to("mps").eval()
ent=[i for i,l in m.config.id2label.items() if l.lower().startswith("entail")][0]
def enc(P):
    H,S=[],[]; t0=time.time()
    for k in range(0,len(P),64):
        b=P[k:k+64]
        t=tok([" ".join(items[i]["text"].split()[:200]) for i,_ in b],[f"This text is about {labels[o]}." for _,o in b],truncation="only_first",max_length=256,padding=True,return_tensors="pt").to("mps")
        with torch.no_grad():
            o=m(**t,output_hidden_states=True)
        H.append(o.hidden_states[-1][:,0].float().cpu().numpy()); S.append(o.logits.float().cpu().numpy())
    print("encoded",len(P),round(time.time()-t0),"s",flush=True)
    return np.concatenate(H),np.concatenate(S)
Ht,St=enc(pairs); Hp,Sp=enc(pp)
np.savez(sys.argv[1]+"/pair.npz",Ht=Ht,St=St,Hp=Hp,Sp=Sp)
pr=[i for i,it in enumerate(items) if it["split"]=="practice"]
def pick(sc,name):
    res={}; k=0
    for i in pr:
        n=len(items[i]["opts"]); res.setdefault(items[i]["kind"],[]).append(int(np.argmax(sc[k:k+n])==items[i]["gold"])); k+=n
    print(f"{name:34s} practice macro {D.macro({a:np.mean(b) for a,b in res.items()}):.3f}",{a[:12]:round(np.mean(b),2) for a,b in res.items()},flush=True)
lsm=lambda S: S[:,ent]-np.log(np.exp(S).sum(1))
pick(lsm(Sp),"encoder alone (its own zero-shot)")
mu=Ht.mean(0); _,s,vt=np.linalg.svd(Ht-mu,full_matrices=False)
for k in (46,768):
    W=vt[:k]/s[:k,None] if k<768 else np.eye(768)
    lr=LogisticRegression(max_iter=5000).fit((Ht-mu)@W.T,y); pick(lr.decision_function((Hp-mu)@W.T),f"probe on pair smell, {k} ch")
print("ref: cosine nose 0.418; joint e5-instruct 46ch 0.386; chance 0.196")
