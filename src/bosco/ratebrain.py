"""Phase A (docs/GATE-A1.md): the MaleCNS central brain as a trainable rate network.

The graph, the synapse counts and the signs are the connectome's and are never changed.  What is
trained is per neuron: how strongly each neuron's output synapses act (a gain, positive) and its
threshold (a bias).  That is the part evolution sets and a wiring diagram does not record.

    r <- r + alpha * (-r + tanh(relu(W @ (g * r) + b + input)))     T steps from rest

W[i, j] = sign(j) * count(j -> i) / (total synapses onto i): each neuron sums its inputs in
proportion to how many synapses each contributes.  Input: the antenna's 52 glomerular values on
the ORNs of those glomeruli.  Output: approach MBONs (in punishment-DAN compartments) minus avoid
MBONs (in reward-DAN compartments), the same fixed read as gate C1, times one scale plus one
offset.  Nothing but the brain sits between the antenna and that difference.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from bosco import paths
from bosco.model import Brain

T_STEPS = 12
ALPHA = 0.5
VIS_FAN = 6  # visual channels per visual Kenyon cell (a KC takes about six inputs), drawn once, seeded


def mbon_sides(brain: Brain) -> tuple[np.ndarray, np.ndarray]:
    from bosco import populations as pop

    cfg = yaml.safe_load(open(paths.CONFIG / "mb_compartments.yaml"))
    v = cfg["valence"]
    rew = {m for c in cfg["compartments"].values() if set(c["dans"]) & set(v["reward"]) for m in c["mbons"]}
    pun = {m for c in cfg["compartments"].values() if set(c["dans"]) & set(v["punishment"]) for m in c["mbons"]}
    both = rew & pun
    mb = pop.mbons()
    ap = brain.index_of_present(mb.loc[mb["type"].isin(sorted(pun - both)), "bodyId"])
    av = brain.index_of_present(mb.loc[mb["type"].isin(sorted(rew - both)), "bodyId"])
    return ap, av


def orn_groups(brain: Brain, n_glom: int) -> list[np.ndarray]:
    """ORN model indices per glomerulus, in the antenna's order (sorted glomerulus names)."""
    from bosco import populations as pop

    orn = pop.orns()
    gl = sorted(orn["glomerulus"].unique())[:n_glom]
    return [brain.index_of_present(orn.loc[orn["glomerulus"] == g, "bodyId"]) for g in gl]


def free_brain(brain: Brain, seed: int) -> Brain:
    """Same neurons, same out-degree, counts and signs per edge; each edge's target drawn uniformly."""
    rng = np.random.default_rng(seed)
    pre = brain.pre_of_edges()
    post = rng.integers(0, brain.n - 1, size=brain.nnz)
    post = np.where(post >= pre, post + 1, post).astype(np.int32)  # no self-edges
    b = Brain(brain.ids, brain.indptr, post, brain.count, brain.sign, brain.nt_sign)
    return b


class RateBrain(torch.nn.Module):
    def __init__(
        self,
        brain: Brain,
        n_glom: int = 52,
        g0: float = 1.0,
        device: str = "mps",
        n_vis: int = 0,
        vis_scope: str = "visual",
    ):
        super().__init__()
        self.n = brain.n
        self.device = device
        pre = brain.pre_of_edges()
        post = brain.indices.astype(np.int64)
        c = brain.count.astype(np.float64)
        tot = np.bincount(post, weights=c, minlength=brain.n)
        tot[tot == 0] = 1.0
        w = brain.sign.astype(np.float64) * c / tot[post]
        W = torch.sparse_coo_tensor(
            torch.tensor(np.stack([post, pre.astype(np.int64)])), torch.tensor(w, dtype=torch.float32), (self.n, self.n)
        ).coalesce()
        self.W = W.to(device)
        groups = orn_groups(brain, n_glom)
        self.orn_idx = torch.tensor(np.concatenate(groups).astype(np.int64), device=device)
        self.orn_glom = torch.tensor(np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)]), device=device)
        ap, av = mbon_sides(brain)
        self.ap = torch.tensor(ap.astype(np.int64), device=device)
        self.av = torch.tensor(av.astype(np.int64), device=device)
        self.log_g = torch.nn.Parameter(torch.full((self.n, 1), float(np.log(g0)), device=device))
        self.b = torch.nn.Parameter(torch.zeros(self.n, 1, device=device))
        self.k = torch.nn.Parameter(torch.tensor(1.0, device=device))
        self.c = torch.nn.Parameter(torch.tensor(0.0, device=device))
        from bosco import populations as pop

        self.kc = torch.tensor(brain.index_of_present(pop.kenyon_cells()).astype(np.int64), device=device)
        # eyes (A2): picture channels drive the visual Kenyon cells directly, as v1's retina did, each
        # cell the mean of VIS_FAN channels chosen at random once (label-free, the same for every arm)
        self.n_vis = n_vis
        if n_vis:
            # "visual": the fly's own visual Kenyon cells (A2); "all": every Kenyon cell (A2b, not fly anatomy)
            vk = brain.index_of_present(pop.visual_kcs() if vis_scope == "visual" else pop.kenyon_cells())
            rng = np.random.default_rng(20260924)
            M = np.zeros((len(vk), n_vis), np.float32)
            for i in range(len(vk)):
                M[i, rng.choice(n_vis, VIS_FAN, replace=False)] = 1.0 / VIS_FAN
            self.vis_kc = torch.tensor(vk.astype(np.int64), device=device)
            self.vis_M = torch.tensor(M, device=device)

    def activity(self, z: torch.Tensor, zv: torch.Tensor | None = None) -> torch.Tensor:
        """z: (batch, n_glom) antenna values in 0..1; zv: (batch, n_vis) picture channels or None
        -> rates (n, batch) after T_STEPS from rest."""
        bsz = z.shape[0]
        inp = torch.zeros(self.n, bsz, device=self.device)
        inp[self.orn_idx] = z.T[self.orn_glom]
        if zv is not None and self.n_vis:
            inp[self.vis_kc] += self.vis_M @ zv.T
        g = torch.exp(self.log_g)
        r = torch.zeros(self.n, bsz, device=self.device)
        for _ in range(T_STEPS):
            x = torch.sparse.mm(self.W, g * r) + self.b + inp
            r = r + ALPHA * (-r + torch.tanh(torch.relu(x)))
        return r

    def forward(self, z: torch.Tensor, zv: torch.Tensor | None = None, with_rates: bool = False):
        r = self.activity(z, zv)
        d = r[self.ap].mean(0) - r[self.av].mean(0)
        logit = self.k * d * 10.0 + self.c
        return (logit, r) if with_rates else logit
