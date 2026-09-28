"""The v2 trained brain (docs/DECISIONS-2026-09-24.md): real fly time, constrained training, a trace.

    tau_i dr_i/dt = -r_i + tanh(relu(sum_j W_ij g_j r_j + b_i + input_i)),   dt = DT_MS, STEPS steps

- W_ij = sign(j) * count(j -> i) / (all synapses onto i from any MaleCNS body): inputs from neurons
  outside the model show as a deficit, not rescaled away (Brain2.in_total).
- Wiring fixes carried with their reasons: cholinergic antennal-lobe LN chemical output 0 (lateral
  excitation there is electrical, Yaksi & Wilson 2010; as fast chemical excitation it runs away,
  docs/phase2-stability.md); DAN -> KC 0 (dopamine is the teaching signal, not fast drive onto KCs).
- Trained (CLAUDE.md, amended): gain g, threshold b and time constant tau **per cell type**
  (mode 'type'), or per neuron (mode 'neuron', the comparison arm); and in both, a multiplier on every
  KC -> MBON synapse, where a fly stores what it learns. Graph and signs never.
- Read: approach MBONs minus avoid MBONs (model2.mbon_groups, from measured DAN input), averaged over
  the last READ_STEPS steps, times one scale plus one offset. Behaviour neurons are recorded.
"""

from __future__ import annotations

import numpy as np
import torch

from bosco import data
from bosco import device as DV
from bosco import model2 as M2
from bosco import populations as pop
from bosco import senses as S
from bosco.model import AL_LN_PREFIXES

DT_MS = 5.0
STEPS = 40  # 200 ms of fly time
READ_STEPS = 8  # the answer is the mean over the last 40 ms
TAU0_MS = 20.0  # Shiu et al.'s membrane time constant, the start for every type
TAU_RANGE = (DT_MS, 200.0)


def type_of_neurons(b) -> np.ndarray:
    """Cell type per neuron; untyped cells share one pseudo-type per class, so none is a free agent."""
    a = data.annotations().reindex(b.ids)
    t = a["type"].fillna("").to_numpy().astype(object)
    cls = a["class"].fillna("unclassified").to_numpy()
    t[t == ""] = np.array(["untyped:" + c for c in cls[t == ""]], dtype=object)
    return t.astype(str)


def wiring_values(b, in_total: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(post, pre, value) for every edge, with the two documented fixes applied."""
    pre = b.pre_of_edges().astype(np.int64)
    post = b.indices.astype(np.int64)
    w = b.sign.astype(np.float64) * b.count.astype(np.float64) / np.maximum(in_total[post], 1.0)
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    eln = np.array([t.startswith(AL_LN_PREFIXES) for t in typ]) & (b.nt_sign > 0)
    w[eln[pre]] = 0.0
    kc = np.zeros(b.n, bool)
    kc[b.index_of_present(pop.kenyon_cells())] = True
    dan = np.zeros(b.n, bool)
    dan[b.index_of_present(pop.dans()["bodyId"])] = True
    w[dan[pre] & kc[post]] = 0.0
    return post, pre, w


class RateBrain2(torch.nn.Module):
    def __init__(self, b2: M2.Brain2, mode: str = "type", device: str | None = None, wiring=None):
        """`wiring`: another Brain with the same neurons (a control), else b2's own."""
        super().__init__()
        b = wiring if wiring is not None else b2.brain
        device = device or DV.default()
        self.n, self.device, self.mode = b.n, device, mode
        post, pre, w = wiring_values(b, b2.in_total)
        kc = np.zeros(b.n, bool)
        kc[b.index_of_present(pop.kenyon_cells())] = True
        mb = np.zeros(b.n, bool)
        mb[b.index_of_present(pop.mbons()["bodyId"])] = True
        plastic = kc[pre] & mb[post] & (w != 0)
        fixed = ~plastic & (w != 0)
        self.W = (
            torch.sparse_coo_tensor(
                torch.tensor(np.stack([post[fixed], pre[fixed]])),
                torch.tensor(w[fixed], dtype=torch.float32),
                (b.n, b.n),
            )
            .coalesce()
            .to(device)
        )
        self.kp_post = torch.tensor(post[plastic], device=device)
        self.kp_pre = torch.tensor(pre[plastic], device=device)
        self.kp_w = torch.tensor(w[plastic], dtype=torch.float32, device=device)
        self.kp_logm = torch.nn.Parameter(torch.zeros(int(plastic.sum()), device=device))
        # parameters per type or per neuron
        types = type_of_neurons(b2.brain)
        if mode == "type":
            uniq, inv = np.unique(types, return_inverse=True)
            self.n_units = len(uniq)
        else:
            inv = np.arange(b.n)
            self.n_units = b.n
        self.unit = torch.tensor(inv, device=device)
        self.log_g = torch.nn.Parameter(torch.zeros(self.n_units, device=device))
        self.b = torch.nn.Parameter(torch.zeros(self.n_units, device=device))
        self.log_tau = torch.nn.Parameter(torch.full((self.n_units,), float(np.log(TAU0_MS)), device=device))
        self.k = torch.nn.Parameter(torch.tensor(1.0, device=device))
        self.c = torch.nn.Parameter(torch.tensor(0.0, device=device))
        # senses, read, recordings (positions in the real brain are the same neurons in a control)
        self.nose = S.nose(b2.brain)
        self.eyes = S.eyes(b2.brain)
        self.orn_idx = torch.tensor(np.concatenate(self.nose.orn_idx).astype(np.int64), device=device)
        self.orn_chan = torch.tensor(
            np.concatenate([np.full(len(x), i) for i, x in enumerate(self.nose.orn_idx)]), device=device
        )
        self.vis_idx = torch.tensor(self.eyes.cells.astype(np.int64), device=device)
        self.vis_M = torch.tensor(self.eyes.M, device=device)
        _, ap, av = M2.mbon_groups(b2.brain)
        self.ap = torch.tensor(ap, device=device)
        self.av = torch.tensor(av, device=device)
        self.kc = torch.tensor(np.nonzero(kc)[0], device=device)
        self.behaviour = {k: torch.tensor(v, device=device) for k, v in M2.behaviour_groups(b2.brain).items()}

    def set_init(self, gain: float, threshold: float) -> None:
        with torch.no_grad():
            self.log_g.fill_(float(np.log(gain)))
            self.b.fill_(-threshold)

    def set_kc_threshold(self, threshold: float) -> None:
        """Threshold of every Kenyon-cell unit (a fly's KCs fire sparsely: high threshold plus APL)."""
        with torch.no_grad():
            self.b[torch.unique(self.unit[self.kc])] = -threshold

    def set_mbon_threshold(self, threshold: float) -> None:
        """Threshold of the read MBON units (approach and avoid groups): their resting operating point."""
        with torch.no_grad():
            self.b[torch.unique(self.unit[torch.cat([self.ap, self.av])])] = -threshold

    def run(self, smell: torch.Tensor, sight: torch.Tensor | None = None, record: bool = False):
        """smell (B, nose.n), sight (B, eyes.n) in 0..1 -> (logit (B,), rates at the end (n, B), trace)."""
        bsz = smell.shape[0]
        inp = torch.zeros(self.n, bsz, device=self.device)
        inp[self.orn_idx] = smell.T[self.orn_chan]
        if sight is not None:
            inp[self.vis_idx] += self.vis_M @ sight.T
        g = torch.exp(self.log_g)[self.unit][:, None]
        bias = self.b[self.unit][:, None] + inp
        tau = torch.exp(self.log_tau).clamp(*TAU_RANGE)[self.unit][:, None]
        alpha = DT_MS / tau
        kp_w = (self.kp_w * torch.exp(self.kp_logm))[:, None]
        r = torch.zeros(self.n, bsz, device=self.device)
        read, trace = [], []
        for s in range(STEPS):
            x = g * r
            drive = torch.sparse.mm(self.W, x)
            drive = drive.index_add(0, self.kp_post, x[self.kp_pre] * kp_w)
            r = r + alpha * (-r + torch.tanh(torch.relu(drive + bias)))
            if s >= STEPS - READ_STEPS:
                read.append(r[self.ap].mean(0) - r[self.av].mean(0))
            if record:
                trace.append(r.detach().to(torch.float16).cpu())
        d = torch.stack(read).mean(0)
        logit = self.k * d * 10.0 + self.c
        return logit, r, (torch.stack(trace) if record else None)

    def forward(self, smell, sight=None):
        return self.run(smell, sight)[0]
