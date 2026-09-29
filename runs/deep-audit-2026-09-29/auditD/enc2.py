import json, numpy as np, torch
from bosco import encoders as E
torch.set_num_threads(2)
enc=E.load_text()
print("max_seq_length", enc.max_seq_length)
meta=json.load(open("data/cache/v1-harm-dev/items.json")); X=np.load("data/cache/v1-harm-dev/emb.npz")["X"]
idx=[i for i,it in enumerate(meta["items"]) if it["task"]=="harmful" and len(str(it["text"]).split())>200][:6]
texts=[meta["items"][i]["text"] for i in idx]
Y=E.embed_text(enc,texts); print("long items: served vs cached max", np.abs(Y-X[idx]).max(1))
Yf=np.stack([enc.encode([E.TEXT["prefix"]+str(t)],normalize_embeddings=True)[0] for t in texts]); print("untruncated vs cached", np.abs(Yf-X[idx]).max(1))
g=json.load(open("data/cache/v1-gate2/items.json")); Xg=np.load("data/cache/v1-gate2/emb.npz")["X"]
gi=[i for i,it in enumerate(g["items"]) if len(str(it.get("text","")).split())>200][:4]
print("gate2 long", len(gi))
if gi:
    Yg=E.embed_text(enc,[g["items"][i]["text"] for i in gi]); print("gate2 long served vs cached", np.abs(Yg-Xg[gi]).max(1))
gs=list(range(0,len(g["items"]),5000))[:6]
Yg=E.embed_text(enc,[g["items"][i]["text"] for i in gs]); print("gate2 sample served vs cached", np.abs(Yg-Xg[gs]).max(1))
