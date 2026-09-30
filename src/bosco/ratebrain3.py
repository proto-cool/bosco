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
**Brain v3.1** (docs/BRAIN-SPEC.md amendment 4; docs/audit-2026-09-28/brain-deep-dive.md):
- BETA 500: the rate at threshold is 0.0014, so a cell below threshold is silent (P1; it was 0.0139, and 59% of KC
  output came from "inactive" KCs).
- Only fast transmitters drive, normalised by fast input totals (wiring3; P2).
- A fixed per-neuron input scale `s_in` (intrinsic excitability, set label-free per type) on the synaptic drive,
  folded into the frozen matrix: operating points come from input gain, not from a threshold bias (P3).
- MBON valence by transmitter (wiring3.mbon_valence; P6). Untyped cells are pooled per class and superclass.
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
from bosco import wiring3 as W3

DT_MS = R2.DT_MS
TAU0_MS = R2.TAU0_MS
TAU_RANGE = R2.TAU_RANGE
BETA = 500.0  # softplus sharpness: rate 0.0014 at threshold, so below threshold is silent (amendment 4, P1)
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


def type_of_neurons(b) -> np.ndarray:
    """Cell type per neuron; untyped cells share one pseudo-type per class and superclass (amendment 4: the 898
    'untyped:unclassified' cells had been one unit, mixing sensory and central cells)."""
    a = data.annotations().reindex(b.ids)
    t = a["type"].fillna("").to_numpy().astype(object)
    cls = a["class"].fillna("unclassified").to_numpy()
    sup = a["superclass"].fillna("unclassified").to_numpy()
    u = t == ""
    t[u] = np.array([f"untyped:{c}:{s}" for c, s in zip(cls[u], sup[u], strict=True)], dtype=object)
    return t.astype(str)


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
        wiring's own fast input totals (`wiring3.own_in_fast`, from the uncut wiring); default the real brain's."""
        super().__init__()
        b = wiring if wiring is not None else b2.brain
        device = device or DV.default()
        self.n, self.device, self.mode = b.n, device, mode
        ft = W3.fast_tables(b2.brain.ids)
        post, pre, w = W3.values(b, in_total if in_total is not None else ft["in_fast"], ft)
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
        ap, av, _ = W3.mbon_valence(b2.brain)
        self.read_groups = {"mbon": (torch.tensor(ap, device=device), torch.tensor(av, device=device))}
        self.read = "mbon"
        self.kc = torch.tensor(np.nonzero(kc)[0], device=device)
        self.regions = {k: torch.tensor(v, device=device) for k, v in regions(b2.brain).items()}
        self.behaviour = {k: torch.tensor(v, device=device) for k, v in M2.behaviour_groups(b2.brain).items()}
        self.steps, self.read_steps = R2.STEPS, R2.READ_STEPS
        # label-free per-cell offsets (docs/BRAIN-SPEC.md: per-KC thresholds), zero unless set
        self.register_buffer("b_cell", torch.zeros(b.n, device=device))
        # fixed per-neuron input scale on the synaptic drive (amendment 4, P3), one unless the start sets it
        self.register_buffer("s_in", torch.ones(b.n, device=device))
        self.r_rest: torch.Tensor | None = None  # the resting state every sniff starts from, once `settle`d

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

    # ---- one learned preference per option (docs/GAP-DEV.md, way B) ----
    def enable_bank(self, n_options: int) -> None:
        """Give every option its own KC->MBON memory (the synapses a fly learns with); everything else is
        shared. A sniff then runs with the memory of the option it asks about (`run(..., opt=...)`)."""
        self.kp_bank = torch.nn.Parameter(self.kp_logm.detach().repeat(n_options, 1))

    # ---- serving fast path ----
    def freeze(self) -> None:
        """Inference only: fold the learned KC->MBON multipliers and each neuron's gain into one CSR matrix,
        so a step is a single sparse product (W_all @ r == W @ (g*r) + plastic(g*r)). Call again after
        loading other weights; `thaw` returns to the training path."""
        with torch.no_grad():
            n = self.n
            g = torch.exp(self.log_g)[self.unit]
            W = self.W.coalesce()
            idx = W.indices()
            val = W.values() * g[idx[1]] * self.s_in[idx[0]]
            kidx = torch.stack([self.kp_post, self.kp_pre])
            kval = self.kp_w * torch.exp(self.kp_logm) * g[self.kp_pre] * self.s_in[self.kp_post]
            A = torch.sparse_coo_tensor(torch.cat([idx, kidx], 1), torch.cat([val, kval]), (n, n)).coalesce()
            self._W_all = A.to_sparse_csr()

    def thaw(self) -> None:
        self._W_all = None

    # ---- dynamics ----
    @staticmethod
    def unit_fn(x: torch.Tensor) -> torch.Tensor:
        # softplus written out as relu(z) + log1p(exp(-|z|)): torch's fused softplus kernel rounds its scalar tail
        # differently from its vector body, so on the CPU its last bit depended on the thread count (where the
        # array is split). These ops round the same either way, so answers are bit-identical at any thread count.
        z = BETA * x
        return torch.tanh((torch.relu(z) + torch.log1p(torch.exp(-z.abs()))) / BETA)

    def run(
        self,
        smell: torch.Tensor,
        sight: torch.Tensor | None = None,
        record: bool | str = False,
        opt=None,
        r0: torch.Tensor | None = None,
    ):
        """smell (B, nose.n), or (B, W, nose.n) for W timed windows of equal length across the steps; sight
        (B, eyes.n) in 0..1; r0 the start state (n,) or (n, B), default the resting state if `settle` has set
        one, else r = 0 (docs/BRAIN-SPEC.md L1: start from rest) -> (logit (B,), rates at the end (n, B),
        trace). record=True keeps every step's rates as the trace; record="read" returns instead the mean rates
        over the read window (n, B)."""
        bsz = smell.shape[0]
        windows = smell[:, None, :] if smell.dim() == 2 else smell
        n_win = windows.shape[1]
        base = torch.zeros(self.n, bsz, device=self.device)
        if sight is not None:
            base[self.vis_idx] += self.vis_M @ sight.T
        inps = []
        for w in range(n_win):
            inp = base.clone()
            inp[self.orn_idx] = windows[:, w, :].T[self.orn_chan]
            inps.append(inp)
        g = torch.exp(self.log_g)[self.unit][:, None]
        biases = [self.b[self.unit][:, None] + self.b_cell[:, None] + inp for inp in inps]
        tau = torch.exp(self.log_tau).clamp(*TAU_RANGE)[self.unit][:, None]
        alpha = DT_MS / tau
        if opt is not None:  # per-sniff memory: (n_plastic, B)
            kp_w = self.kp_w[:, None] * torch.exp(self.kp_bank[opt].T)
        else:
            kp_w = (self.kp_w * torch.exp(self.kp_logm))[:, None]
        ap, av = self.read_groups[self.read]
        if r0 is None:
            r0 = self.r_rest
        if r0 is None:
            r = torch.zeros(self.n, bsz, device=self.device)
        else:
            r = (r0[:, None] if r0.dim() == 1 else r0).expand(self.n, bsz).clone()
        read, trace = [], []
        win_len = -(-self.steps // n_win)

        W_all = getattr(self, "_W_all", None)
        s_in = self.s_in[:, None]

        def step(r, bias):
            if W_all is not None and not torch.is_grad_enabled() and opt is None:
                # one sniff: CSR matvec, ~10x faster than CSR @ (n, 1); bit-identical on the CPU to the batched
                # product (each row is summed in index order), so a served answer equals a scored one
                drive = torch.mv(W_all, r[:, 0])[:, None] if r.shape[1] == 1 else W_all @ r
            else:
                x = g * r
                drive = torch.sparse.mm(self.W, x)
                drive = drive.index_add(0, self.kp_post, x[self.kp_pre] * kp_w) * s_in
            return r + alpha * (-r + self.unit_fn(drive + bias))

        # training keeps only every CHECKPOINT-th state and recomputes the rest in the backward pass
        ckpt = torch.is_grad_enabled() and not record
        window = record == "read"
        acc = 0.0
        s = 0
        while s < self.steps:
            n_seg = min(CHECKPOINT, self.steps - s)
            reads_in = [k for k in range(n_seg) if s + k >= self.steps - self.read_steps]
            if ckpt and not reads_in and (s // win_len) == ((s + n_seg - 1) // win_len):
                bias = biases[s // win_len]
                r = torch.utils.checkpoint.checkpoint(
                    lambda r, n=n_seg, bias=bias: self._seg(step, r, n, bias), r, use_reentrant=False
                )
            else:
                for k in range(n_seg):
                    r = step(r, biases[(s + k) // win_len])
                    if k in reads_in:
                        read.append(r[ap].mean(0) - r[av].mean(0))
                        if window:
                            acc = acc + r
                    if record is True:
                        trace.append(r.detach().to(torch.float16).cpu())
            s += n_seg
        d = torch.stack(read).mean(0)
        logit = torch.exp(self.log_k) * d * 10.0 + self.c
        if window:
            return logit, r, acc / self.read_steps
        return logit, r, (torch.stack(trace) if record else None)

    # ---- the answer (served and scored: one path) ----
    def answer(self, smell: torch.Tensor) -> torch.Tensor:
        """Yes/no logits (B,) for smells (B, nose.n), one sniff each from rest: p(yes) = sigmoid(logit), the read
        being approach minus avoid (docs/BRAIN-SPEC.md). Each sniff runs alone, so a number is the same bit for bit
        whether it is scored in a batch or served on its own: on the CPU, softplus and the read's mean round
        differently in a (n, B) batch than in one column (by ~1 ulp), and a published number must be the served
        one (decision 7). One sniff is also the fast path (CSR matvec). Call `freeze` first."""
        with torch.no_grad():
            return torch.cat([self.run(smell[i : i + 1])[0] for i in range(smell.shape[0])])

    # ---- the resting state (docs/BRAIN-SPEC.md L1) ----
    def settle(self, rest_smell: torch.Tensor, chunk: int = 40, max_chunks: int = 50, tol: float = 1e-5) -> dict:
        """Hold the resting smell (nose.n,) until the rates stop changing, and keep that state as the start of
        every sniff: the fly is alive before the odour. Continues from the current resting state, if any.
        Converged when, after a `chunk`-step stretch, no rate moves more than `tol` in one step. (Comparing states a
        stretch apart, as before, passed a rhythm whose period divides the stretch: the deep dive's review found it.)"""
        saved, saved_read = self.steps, self.read_steps
        r = self.r_rest if self.r_rest is not None else torch.zeros(self.n, device=self.device)
        delta, steps = float("inf"), 0
        try:
            with torch.no_grad():
                while steps < max_chunks * chunk and delta >= tol:
                    self.steps, self.read_steps = chunk - 1, 1
                    _, r, _ = self.run(rest_smell[None], r0=r)
                    self.steps = 1
                    _, r_new, _ = self.run(rest_smell[None], r0=r[:, 0])
                    delta = float((r_new[:, 0] - r[:, 0]).abs().max())
                    r, steps = r_new[:, 0], steps + chunk
        finally:
            self.steps, self.read_steps = saved, saved_read
        self.r_rest = r
        return {"steps": steps, "max_step_change": delta, "converged": delta < tol}

    def preact(self, r: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Input to the unit function of cells `idx` at rates r (n, B), without sensory input: W g r + b + b_cell."""
        with torch.no_grad():
            g = torch.exp(self.log_g)[self.unit][:, None]
            x = g * r
            drive = torch.sparse.mm(self.W, x)
            drive = drive.index_add(0, self.kp_post, x[self.kp_pre] * (self.kp_w * torch.exp(self.kp_logm))[:, None])
            drive = drive * self.s_in[:, None]
            return drive[idx] + self.b[self.unit][idx, None] + self.b_cell[idx, None]

    @staticmethod
    def _seg(step, r, n, bias):
        for _ in range(n):
            r = step(r, bias)
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
