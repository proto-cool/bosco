"""The Bosco service core: one brain family, many specialists (docs/SERVICE-R10.md, docs/BRAIN-VIEWS.md).

CPU only, deterministic (decision 7). A request's text is translated by the family's pinned encoder into a
smell (bi46 antenna); each option is one sniff (the item's smell mixed with the option word's smell, from
rest); the specialist's weights are swapped into the shared brain; the answer is read from his descending
neurons. Every answer keeps a slimmed trace (per-dot activity for both 64 x 48 views, the named answer
cells, the top neurons per step with their cell types). Nothing a visitor sends is stored beyond the
in-memory trace cache.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from bosco import data, v1
from bosco import model2 as M2
from bosco import ratebrain3 as R3

NARRATE = ("alpn", "kc", "mbon", "dan", "lh", "cx", "dn")  # processing regions named in the trace
PER_REGION = 3
TRACE_CACHE = 256
RECORD_ALL = 48  # sniffs per request above which only the picked sniffs are recorded (a second pass)


class NotTaught(Exception):
    pass


class Family:
    REST_Z = np.full(46, 0.5, np.float32)
    def __init__(self, d: Path, threads: int = 4):
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(threads)
        self.dir = d
        self.meta = json.load(open(d / "family.json"))
        a = np.load(d / "antenna.npz")
        self.mu, self.W, self.norm = a["mu"], a["W"], float(a["norm"])
        self.maps = {k: json.load(open(d / f"{k}.json")) for k in ("anatomy", "wave")}
        self.dots = {
            k: [dd["neurons"] for dd in sorted(v["dots"], key=lambda x: (x["y"], x["x"]))] for k, v in self.maps.items()
        }
        self.m = v1.build("real", device="cpu")
        ap, av, info = v1.dn_groups(self.m)
        self.m.set_dn_read(ap, av, self.meta["steps"], self.meta["read_steps"])
        self.base_state = {k: x.clone() for k, x in self.m.state_dict().items()}
        W = self.m.W.coalesce().to_sparse_csr()  # the fixed connectome, shared by every specialist
        # scipy's CSR product is 20-37% faster than torch's on CPU here, same result (runs/speed-spike.json)
        self.W_csr = sp.csr_matrix(
            (W.values().numpy(), W.col_indices().numpy(), W.crow_indices().numpy()), shape=W.shape
        )
        ids = M2.load_or_build().brain.ids
        self.types = data.annotations().reindex(ids)["type"].fillna("").to_numpy().astype(str)
        self.region_of = np.full(self.m.n, "", object)
        for k, v in self.m.regions.items():
            self.region_of[v.numpy()] = k
        self._enc = None
        self.lock = threading.Lock()  # one brain, one sniff batch at a time

    # ---- senses ----
    def encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer

            t = self.meta["encoder"]
            self._enc = SentenceTransformer(t["id"], revision=t["revision"], trust_remote_code=True, device="cpu")
        return self._enc

    def smell(self, texts: list[str]) -> np.ndarray:
        t = self.meta["encoder"]
        X = self.encoder().encode([t["prefix"] + " ".join(x.split()[:200]) for x in texts], normalize_embeddings=True)
        return np.clip(0.5 + ((X - self.mu) @ self.W.T) / (2 * self.norm), 0, 1).astype(np.float32)

    # ---- decisions: every question of a request in one pass ----
    def params(self, spec) -> dict:
        """A specialist's per-neuron settings, computed once: the connectome is shared by every specialist, so
        a batch can hold columns from several specialists (W @ (g * r) with g per column)."""
        if spec.params is None:
            st = {**self.base_state, **spec.weights}
            u = self.m.unit
            tau = torch.exp(st["log_tau"]).clamp(*R3.TAU_RANGE)[u]
            p = {
                "g": torch.exp(st["log_g"])[u], "b": st["b"][u], "alpha": R3.DT_MS / tau,
                "k": float(torch.exp(st["log_k"])) * 10.0, "c": float(st["c"]),
            }
            if "kp_bank" in spec.weights:  # one memory per option
                p["bank"] = self.m.kp_w[None, :] * torch.exp(spec.weights["kp_bank"])
            else:
                p["kpw"] = self.m.kp_w * torch.exp(st["kp_logm"])
            spec.params = p
        return spec.params

    def run_cols(self, cols: list[tuple], record: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """cols: (spec, smell (46,), option index or None). The same dynamics as RateBrain3.run, with every
        column carrying its own specialist's gains, thresholds, time constants and KC->MBON memory."""
        m, B = self.m, len(cols)
        P = [self.params(s) for s, _, _ in cols]
        S = torch.tensor(np.stack([z for _, z, _ in cols]))
        inp = torch.zeros(m.n, B)
        inp[m.orn_idx] = S.T[m.orn_chan]
        G = torch.stack([p["g"] for p in P], 1)
        bias = torch.stack([p["b"] for p in P], 1) + inp
        alpha = torch.stack([p["alpha"] for p in P], 1)
        kpw = torch.stack([p["kpw"] if o is None else p["bank"][o] for p, (_, _, o) in zip(P, cols, strict=True)], 1)
        k = torch.tensor([p["k"] for p in P])
        c = torch.tensor([p["c"] for p in P])
        ap, av = m.read_groups["dn"]
        r = torch.zeros(m.n, B)
        read, trace = [], []
        with self.lock, torch.no_grad():
            for st in range(m.steps):
                x = G * r
                wx = torch.from_numpy(self.W_csr @ np.ascontiguousarray(x.numpy()))
                drive = wx.index_add(0, m.kp_post, x[m.kp_pre] * kpw)
                r = r + alpha * (-r + m.unit_fn(drive + bias))
                if st >= m.steps - m.read_steps:
                    read.append(r[ap].mean(0) - r[av].mean(0))
                if record:
                    trace.append(r.to(torch.float16))
        logit = k * torch.stack(read).mean(0) + c
        return logit.numpy(), (torch.stack(trace).float().numpy() if record else None)

    def rest_of(self, spec, k: int | None = None) -> np.ndarray:
        """The specialist's own resting state (with option k's memory, for design B): every glomerulus at rest
        (0.5), per step."""
        if k not in spec.rest:
            _, tr = self.run_cols([(spec, np.full(46, 0.5, np.float32), k)], record=True)
            spec.rest[k] = tr[:, :, 0]
        return spec.rest[k]

    def prepare(self, spec, text_z: np.ndarray, qtype: str, options, allow_untaught: bool, opt_z) -> dict:
        own = spec.card["options"]
        if qtype == "approach" and (len(own) != 2 or options is not None):
            raise NotTaught(f"{spec.card['specialist']} is not a yes/no specialist (options: {own})")
        if qtype == "rate" and (options is None or sorted(options) != sorted(own)):
            raise NotTaught(f"rate needs the taught options in low-to-high order: {own}")
        opts = list(options) if options is not None else own
        bank = spec.card.get("design") == "B"
        if bank and not set(opts) <= set(own):  # one memory per taught option: nothing to sniff for the rest
            raise NotTaught(f"{spec.card['specialist']} answers only (a subset of): {own}")
        if not bank and sorted(opts) != sorted(own) and not allow_untaught:  # sniffs are independent: any order
            raise NotTaught(f"{spec.card['specialist']} answers only: {own}")
        if bank:  # the item's smell alone, once per option, each with that option's memory
            cols = [(spec, text_z, own.index(o)) for o in opts]
        else:  # the T-maze: the item's smell mixed with each option word's smell, from rest
            cols = [(spec, np.clip(text_z + opt_z[o] - 0.5, 0, 1), None) for o in opts]
        return {"spec": spec, "type": qtype, "opts": opts, "cols": cols,
                "taught": options is None or sorted(opts) == sorted(own) or bank}

    def ask_many(self, text: str, questions: list[tuple], allow_untaught: bool = False) -> list[dict]:
        """questions: (spec, type, options). One encoder call and one brain pass for all of them."""
        words = sorted({o for s, _, opts in questions if s.card.get("design") != "B"
                        for o in (opts if opts is not None else s.card["options"])})
        z = self.smell([text] + words)
        opt_z = dict(zip(words, z[1:], strict=True))
        preps = [self.prepare(s, z[0], t, o, allow_untaught, opt_z) for s, t, o in questions]
        cols = [c for p in preps for c in p["cols"]]
        record = len(cols) <= RECORD_ALL
        logit, tr = self.run_cols(cols, record)
        out, j, picked = [], 0, []
        for p in preps:
            n = len(p["opts"])
            lo = logit[j : j + n] / p["spec"].card.get("temperature", 1.0)
            pr = np.exp(lo - lo.max())
            pr /= pr.sum()
            kk = int(np.argmax(pr))
            picked.append((j + kk, p["cols"][kk]))
            out.append({"p_arr": pr, "lo": lo, "k": kk})
            j += n
        if not record:  # large batches: record only the picked sniffs, in a second pass
            _, tr2 = self.run_cols([c for _, c in picked], True)
        answers = []
        for q, (p, o) in enumerate(zip(preps, out, strict=True)):
            spec, opts, pr, kk = p["spec"], p["opts"], o["p_arr"], o["k"]
            sd = (tr[:, :, picked[q][0]] if record else tr2[:, :, q]) - self.rest_of(spec, picked[q][1][2])
            n = len(opts)
            base = {
                "sure": round(float(max(0.0, (n * pr.max() - 1) / (n - 1))) if n > 1 else 1.0, 4),
                "taught": p["taught"], "answered_as": spec.card["task"], "trace": self._trace(sd, spec),
            }
            if p["type"] == "approach":
                own = spec.card["options"]
                y = opts.index(own[0])
                answers.append({"yes": own[0], "p": round(float(pr[y]), 4),
                                "lean": round(float(o["lo"][y] - o["lo"][1 - y]), 4), **base})
            elif p["type"] == "rate":
                answers.append({"score": round(float(1 + sum(i * x for i, x in enumerate(pr))), 4),
                                "legend": {str(i + 1): x for i, x in enumerate(opts)},
                                "p": {str(i + 1): round(float(x), 4) for i, x in enumerate(pr)}, **base})
            else:
                answers.append({"pick": opts[kk], "p": {x: round(float(v), 4) for x, v in zip(opts, pr, strict=True)},
                                **base})
        return answers

    def decide(self, spec, text: str, options: list[str] | None, allow_untaught: bool = False) -> dict:
        return self.ask_many(text, [(spec, "choose", options)], allow_untaught)[0]

    def ask(self, spec, text: str, qtype: str, options: list[str] | None, allow_untaught: bool = False) -> dict:
        return self.ask_many(text, [(spec, qtype, options)], allow_untaught)[0]

    def _trace(self, sd: np.ndarray, spec) -> dict:
        """sd: signed distance from the specialist's rest, (steps, n). Views are packed per step as int8
        (value = q / 127 * scale) in base64, 3,072 bytes per step per view."""
        import base64

        views = {}
        for k, dots in self.dots.items():
            a = np.stack([sd[:, nb].mean(1) for nb in dots], 1)
            scale = float(np.abs(a).max()) or 1.0
            q = np.clip(np.round(a / scale * 127), -127, 127).astype(np.int8)
            views[k] = {"scale": scale, "int8_b64": [base64.b64encode(row.tobytes()).decode() for row in q]}
        ap, av = (x.numpy() for x in self.m.read_groups["dn"])
        # narration: the most changed cells per processing region (the senses themselves are the input)
        top = []
        for st in range(sd.shape[0]):
            row = []
            for reg in NARRATE:
                idx = self.m.regions[reg].numpy()
                best = idx[np.argsort(-np.abs(sd[st, idx]))[:PER_REGION]]
                row += [[int(i), self.types[i], reg, round(float(sd[st, i]), 3)] for i in best]
            top.append(row)
        return {
            "family": self.meta["id"], "version": spec.card["version"], "steps": self.meta["steps"],
            "dt_ms": self.meta["dt_ms"], "dots_order": "row-major (y, then x)", "views": views,
            "named": {"avoid": sd[:, av].mean(1).round(4).tolist(), "approach": sd[:, ap].mean(1).round(4).tolist()},
            "top_fields": ["neuron", "type", "region", "delta"], "top": top,
        }


class Specialist:
    def __init__(self, d: Path):
        self.card = json.load(open(d / "card.json"))
        w = d / "weights.pt"
        want = self.card.get("weights_sha256")
        if want and hashlib.sha256(w.read_bytes()).hexdigest() != want:
            raise ValueError(f"{d}: weights do not match the card's sha256")
        self.weights = torch.load(w, weights_only=True)
        self.rest: dict[int | None, np.ndarray] = {}  # computed on first use (Family.rest_of)
        self.params: dict | None = None  # per-neuron settings (Family.params)


class Fleet:
    def __init__(self, root: Path, family_id: str = "v1", dev: bool = False):
        """Loads the shipped specialists (registry cards with status "shipped" and a matching weights hash);
        `dev` also loads development exports, never for a public server."""
        self.family = Family(root / "families" / family_id)
        self.specs: dict[str, Specialist] = {}
        self.latest: dict[str, str] = {}
        for d in sorted((root / "specialists").glob("*/*")):
            c = json.load(open(d / "card.json"))
            if not dev and (c.get("status") != "shipped" or not c.get("weights_sha256")):
                continue  # only gate-passed, hash-pinned specialists are served
            s = Specialist(d)
            if s.card["family"] != family_id:
                continue  # a specialist only runs on the family it was trained on
            self.specs[s.card["version"]] = s
            self.latest[s.card["specialist"] + "-latest"] = s.card["version"]
        self.traces: collections.OrderedDict[str, dict] = collections.OrderedDict()

    def get(self, version: str) -> Specialist:
        return self.specs[self.latest.get(version, version)]

    def keep_trace(self, trace: dict) -> str:
        tid = uuid.uuid4().hex
        self.traces[tid] = trace
        while len(self.traces) > TRACE_CACHE:
            self.traces.popitem(last=False)
        return tid
