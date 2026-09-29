import json, numpy as np, time, torch
from bosco import encoders as E
torch.set_num_threads(2)
enc=E.load_text()
meta=json.load(open("data/cache/v1-harm-dev/items.json")); X=np.load("data/cache/v1-harm-dev/emb.npz")["X"]
idx=[i for i,it in enumerate(meta["items"]) if it["task"]=="harmful" and it["split"]=="val"][:64]
texts=[meta["items"][i]["text"] for i in idx]
Y=E.embed_text(enc,texts)
d=np.abs(Y-X[idx]).max(1); print("served vs cached max|diff| max %.2e median %.2e"%(d.max(),np.median(d)))
long=[i for i,t in zip(idx,texts) if len(str(t).split())>200]; print("val items >200 words in first 64:",len(long))
allv=[it for it in meta["items"] if it["task"]=="harmful"]; print("harmful items >200 words:", sum(len(str(it["text"]).split())>200 for it in allv), "of", len(allv))
print("threads check: E.TEXT_THREADS", E.TEXT_THREADS, "torch threads after", torch.get_num_threads())
odd=["", " ", "the "*300, "a", "yes", "no", "Is this harmful?", "\u0000", "😀"*50]
Z=E.embed_text(enc,odd)
np.savez("/Users/nickd/.claude/jobs/4059153c/tmp/auditD/enc.npz", Y=Y, idx=np.array(idx), Z=Z)
print("odd norms", np.linalg.norm(Z,axis=1), "nan", np.isnan(Z).any())
# does the cached embedding match 'batch' or 'single'?
Yb=enc.encode([E.TEXT["prefix"]+" ".join(str(t).split()[:200]) for t in texts[:16]],batch_size=64,normalize_embeddings=True)
print("batch vs cached %.2e, single vs cached %.2e"%(np.abs(Yb-X[idx[:16]]).max(), np.abs(Y[:16]-X[idx[:16]]).max()))
