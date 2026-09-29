import sys, time, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import brain_check as BC
import brain_train as BT
from bosco import model2 as M2, ratebrain3 as R3, v1
AUD = Path("/Users/nickd/.claude/jobs/4059153c/tmp/auditD")
TR = Path("/Users/nickd/projects/bosco/runs/brain-train")
X = np.load("/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz")["X"]
ant = BC.antenna()
rest = torch.tensor(ant.resting())
S = torch.tensor(ant(X))
_b2 = None
def b2():
    global _b2
    if _b2 is None: _b2 = M2.load_or_build()
    return _b2
def load(trained=None, arm="real"):
    m = BC.brain(arm, b2()); BC.load_start(m, arm)
    if trained:
        tr = torch.load(TR / f"{trained}.pt")
        BT.with_memory(m, tr["kp_logm"], tr["log_k"], tr["c"], rest)
    return m
def ans(m, s=None, n=None):
    s = S if s is None else s
    if n: s = s[:n]
    return m.answer(s).numpy()
