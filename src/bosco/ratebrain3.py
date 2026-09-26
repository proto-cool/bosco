"""The v1 brain, "The Model" (docs/PLAN-V1.md, track A). ratebrain2 is kept unchanged so earlier gates replay.

    tau_i dr_i/dt = -r_i + tanh(softplus(BETA * (sum_j W_ij g_j r_j + b_i + input_i)) / BETA)

Changes from ratebrain2, each with its reason (docs/audit-2026-09-25/code.md):
- **Graded unit.** tanh(relu(x)) gives a silent neuron no gradient, so 7,046 of 8,257 cell types never
  trained in A4 broad. Real neurons respond gradedly below spike threshold; softplus with a sharp BETA
  keeps a near-zero rate below threshold but a small, nonzero slope, so a silent type can still learn.
  Above threshold it matches tanh(relu) to within ln(2)/BETA.
- **Controls normalised on their own inputs.** W_ij = sign * count / in_total_i, where in_total_i is
  this wiring's own synapses onto i plus the real brain's synapses onto i from outside the model
  (those are not shuffled). ratebrain2 used the real brain's totals for every control.
- **Sparseness on the right measure.** The KC penalty acts on the (soft) fraction of KCs active, the
  quantity the bars count, above the fly's range (KC_ACTIVE_MAX); ratebrain2's penalised the mean
  rate against 0.01 and never fired.
- **Regions** are labelled for the preflight's liveness check and the trace.
The read is pluggable: 'mbon' (approach minus avoid MBONs, as before) or 'dn' (set by `set_dn_read`,
decision 4). Everything else (time step, steps, the wiring fixes) is as in ratebrain2, with its reasons.
"""

from __future__ import annotations

import numpy as np
import torch

from bosco import data
from bosco import device as DV
from bosco import model2 as M2
from bosco import populations as pop
from bosco import ratebrain2 as R2
from bosco import senses as S

DT_MS = R2.DT_MS
TAU0_MS = R2.TAU0_MS
TAU_RANGE = R2.TAU_RANGE
BETA = 50.0  # softplus sharpness: rate 0.014 at threshold, 0.0016 at 0.05 below it (resting, not "active"),
# slope 0.08 there (Adam rescales small gradients, so a silent type still trains)
ACTIVE = 0.01  # a neuron counts as active above this rate (as in ratebrain2's probes)
KC_ACTIVE_MAX = 0.10  # upper edge of the fly's KC range (2-10% active; Honegger et al. 2011)
CHECKPOINT = 10  # steps per recomputed segment when training (memory for long runs and the optic lobes)
LH_PREFIXES = ("LHAV", "LHAD", "LHPV", "LHPD", "LHCENT", "LHLN", "LHPN")


def own_in_total(b2: M2.Brain2, w) -> np.ndarray:
    """Input totals for wiring `w` (same neurons as b2.brain, uncut): its own synapses from model neurons
    plus the real brain's synapses from outside the model."""
    n = b2.brain.n
    real_in = np.bincount(b2.brain.indices, weights=b2.brain.count, minlength=n)
    own_in = np.bincount(w.indices, weights=w.count, minlength=n)
    return own_in + (b2.in_total - real_in)


def regions(b) -> dict[str, np.ndarray]:
    a = data.annotations().reindex(b.ids)
    cls = a["class"].fillna("").to_numpy().astype(str)
    sup = a["superclass"].fillna("").to_numpy().astype(str)
    typ = a["type"].fillna("").to_numpy().astype(str)
    out = {
        "orn": cls == "olfactory",
        "alpn": cls == "ALPN",
        "alln": cls == "ALLN",
        "kc": cls == "Kenyon_Cell",
        "mbon": cls == "MBON",
        "dan": cls == "DAN",
        "lh": np.array([t.startswith(LH_PREFIXES) for t in typ]),
        "cx": cls == "CX",
        "vpn": sup == "visual_projection",
        "dn": sup == "descending_neuron",
        "other_sensory": np.char.find(sup, "sensory") >= 0,
    }
    out["other_sensory"] &= ~out["orn"]
    taken = np.zeros(b.n, bool)
    for v in out.values():
        taken |= v
    out["rest"] = ~taken
    return {k: np.nonzero(v)[0] for k, v in out.items()}


class RateBrain3(torch.nn.Module):
    def __init__(self, b2: M2.Brain2, mode: str = "type", device: str | None = None, wiring=None, in_total=None):
        """`wiring`: a Brain with the same neurons (a control, possibly cut), else b2's own. `in_total`: that
        wiring's own input totals (`own_in_total`, from the uncut wiring); default b2's (the real brain)."""
        super().__init__()
        b = wiring if wiring is not None else b2.brain
        device = device or DV.default()
        self.n, self.device, self.mode = b.n, device, mode
        post, pre, w = R2.wiring_values(b, in_total if in_total is not None else b2.in_total)
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
        types = R2.type_of_neurons(b2.brain)
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
        self.log_k = torch.nn.Parameter(torch.tensor(0.0, device=device))  # read scale, log so it can train
        self.c = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.nose = S.nose(b2.brain)
        self.eyes = S.eyes(b2.brain)
        self.orn_idx = torch.tensor(np.concatenate(self.nose.orn_idx).astype(np.int64), device=device)
        self.orn_chan = torch.tensor(
            np.concatenate([np.full(len(x), i) for i, x in enumerate(self.nose.orn_idx)]), device=device
        )
        self.vis_idx = torch.tensor(self.eyes.cells.astype(np.int64), device=device)
        self.vis_M = torch.tensor(self.eyes.M, device=device)
        _, ap, av = M2.mbon_groups(b2.brain)
        self.read_groups = {"mbon": (torch.tensor(ap, device=device), torch.tensor(av, device=device))}
        self.read = "mbon"
        self.kc = torch.tensor(np.nonzero(kc)[0], device=device)
        self.regions = {k: torch.tensor(v, device=device) for k, v in regions(b2.brain).items()}
        self.behaviour = {k: torch.tensor(v, device=device) for k, v in M2.behaviour_groups(b2.brain).items()}
        self.steps, self.read_steps = R2.STEPS, R2.READ_STEPS

    # ---- operating point (label-free starts set these) ----
    def set_init(self, gain: float, threshold: float) -> None:
        with torch.no_grad():
            self.log_g.fill_(float(np.log(gain)))
            self.b.fill_(-threshold)

    def set_threshold(self, idx: torch.Tensor, threshold: float) -> None:
        with torch.no_grad():
            self.b[torch.unique(self.unit[idx])] = -threshold

    def set_kc_threshold(self, threshold: float) -> None:
        self.set_threshold(self.kc, threshold)

    def set_read_threshold(self, threshold: float) -> None:
        ap, av = self.read_groups[self.read]
        self.set_threshold(torch.cat([ap, av]), threshold)

    def set_dn_read(self, approach: np.ndarray, avoid: np.ndarray, steps: int, read_steps: int) -> None:
        """Decision 4: the answer from descending neurons (groups fixed before training)."""
        self.read_groups["dn"] = (
            torch.tensor(approach, device=self.device),
            torch.tensor(avoid, device=self.device),
        )
        self.read, self.steps, self.read_steps = "dn", steps, read_steps

    # ---- serving fast path ----
    def freeze(self) -> None:
        """Inference only: fold the learned KC->MBON multipliers and each neuron's gain into one CSR matrix,
        so a step is a single sparse product (W_all @ r == W @ (g*r) + plastic(g*r)). Call again after
        loading other weights; `thaw` returns to the training path."""
        with torch.no_grad():
            n = self.n
            g = torch.exp(self.log_g)[self.unit]
            W = self.W.coalesce()
            idx, val = W.indices(), W.values() * g[W.indices()[1]]
            kidx = torch.stack([self.kp_post, self.kp_pre])
            kval = self.kp_w * torch.exp(self.kp_logm) * g[self.kp_pre]
            A = torch.sparse_coo_tensor(torch.cat([idx, kidx], 1), torch.cat([val, kval]), (n, n)).coalesce()
            self._W_all = A.to_sparse_csr()

    def thaw(self) -> None:
        self._W_all = None

    # ---- dynamics ----
    @staticmethod
    def unit_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.nn.functional.softplus(BETA * x) / BETA)

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
        ap, av = self.read_groups[self.read]
        r = torch.zeros(self.n, bsz, device=self.device)
        read, trace = [], []

        W_all = getattr(self, "_W_all", None)

        def step(r):
            if W_all is not None and not torch.is_grad_enabled():
                drive = W_all @ r
            else:
                x = g * r
                drive = torch.sparse.mm(self.W, x)
                drive = drive.index_add(0, self.kp_post, x[self.kp_pre] * kp_w)
            return r + alpha * (-r + self.unit_fn(drive + bias))

        # training keeps only every CHECKPOINT-th state and recomputes the rest in the backward pass
        ckpt = torch.is_grad_enabled() and not record
        s = 0
        while s < self.steps:
            n_seg = min(CHECKPOINT, self.steps - s)
            reads_in = [k for k in range(n_seg) if s + k >= self.steps - self.read_steps]
            if ckpt and not reads_in:
                r = torch.utils.checkpoint.checkpoint(
                    lambda r, n=n_seg: self._seg(step, r, n), r, use_reentrant=False
                )
            else:
                for k in range(n_seg):
                    r = step(r)
                    if k in reads_in:
                        read.append(r[ap].mean(0) - r[av].mean(0))
                    if record:
                        trace.append(r.detach().to(torch.float16).cpu())
            s += n_seg
        d = torch.stack(read).mean(0)
        logit = torch.exp(self.log_k) * d * 10.0 + self.c
        return logit, r, (torch.stack(trace) if record else None)

    @staticmethod
    def _seg(step, r, n):
        for _ in range(n):
            r = step(r)
        return r

    def forward(self, smell, sight=None):
        return self.run(smell, sight)[0]

    # ---- measures ----
    def kc_active_soft(self, r: torch.Tensor) -> torch.Tensor:
        """Differentiable fraction of KCs active (a sigmoid around ACTIVE)."""
        return torch.sigmoid((r[self.kc] - ACTIVE) / (ACTIVE / 3)).mean()

    def kc_penalty(self, r: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.kc_active_soft(r) - KC_ACTIVE_MAX) ** 2

    def liveness(self, r: torch.Tensor) -> dict[str, float]:
        """Share of each region's neurons active (rate > ACTIVE) in at least one of the batch's sniffs."""
        return {k: float((r[v] > ACTIVE).any(1).float().mean()) for k, v in self.regions.items() if len(v)}
