import sys, json, traceback
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "/Users/nickd/projects/bosco/scripts")
import brain_check as BC
from bosco import model2 as M2, ratebrain3 as R3, v1, wiring3 as W3
res={"BETA":R3.BETA}
b2=M2.load_or_build()
# fast_tables cache: recompute to scratch and compare
cached=W3.fast_tables(b2.brain.ids)
W3.CACHE=Path("/Users/nickd/.claude/jobs/4059153c/tmp/auditD/fast_recomputed.npz")
fresh=W3.fast_tables(b2.brain.ids)
res["fast_cache_matches_fresh"]={k:bool(np.array_equal(cached[k],fresh[k])) for k in fresh}
m=BC.brain("real",b2)
res["n_units_v31"]=m.n_units; res["n_plastic_v31"]=int(m.kp_logm.numel())
st=torch.load("/Users/nickd/projects/bosco/runs/brain-check/real-start.pt")
res["v30_start_b_len"]=int(st["b"].numel())
try:
    BC.load_start(m,"real"); res["load_v30_start_with_v31"]="no error"
    X=np.load("/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz")["X"][:16]
    ant=BC.antenna(); z=m.answer(torch.tensor(ant(X))).numpy()
    res["logits"]=[float(x) for x in z[:6]]; res["s_in_all_one"]=bool((m.s_in==1).all())
except Exception as e:
    res["load_v30_start_with_v31"]="ERROR "+repr(e)[:300]
try:
    tr=torch.load("/Users/nickd/projects/bosco/runs/brain-train/real-s1.pt")
    res["v30_kp_len"]=int(tr["kp_logm"].numel())
    m.kp_logm.data.copy_(tr["kp_logm"]); res["load_v30_memory_with_v31"]="no error"
except Exception as e:
    res["load_v30_memory_with_v31"]="ERROR "+repr(e)[:300]
# dn_read.json from v3.0 with v3.1 wiring: check the read cells vs regions
rd=json.load(open("/Users/nickd/projects/bosco/runs/brain-check/dn_read.json"))
res["dn_read_keys"]=list(rd.keys())
# which v3.0 dn read cells have zero fast output / are DNs with unknown/mod transmitters
ft=cached
ap=np.array(rd["approach"]); av=np.array(rd["avoid"])
res["dn_read_labels_ap"]=dict(zip(*np.unique(ft["label"][ap],return_counts=True)))
res["dn_read_labels_av"]=dict(zip(*np.unique(ft["label"][av],return_counts=True)))
res["dn_read_ap_n"]=len(ap); res["dn_read_av_n"]=len(av)
# mbon valence v31 table
ap2,av2,tab=W3.mbon_valence(b2.brain)
res["mbon_valence_counts"]=tab.side.value_counts().to_dict(); res["mbon_unknown_types"]=tab.loc[tab.side=="unknown",["type","nt"]].values.tolist()
res["mbon_ap_cells"]=len(ap2); res["mbon_av_cells"]=len(av2)
# type names parsing check
bad=[t for t in tab.type if not t.startswith("MBON")]
res["mbon_nonstandard_types"]=bad[:20]
res["label_counts"]={k:int(v) for k,v in zip(*np.unique(ft["label"],return_counts=True))}
print(json.dumps(res,indent=1,default=str))
