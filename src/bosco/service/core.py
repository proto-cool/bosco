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
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import torch

from bosco import data, v1
from bosco import model2 as M2

NARRATE = ("alpn", "kc", "mbon", "dan", "lh", "cx", "dn")  # processing regions named in the trace
PER_REGION = 3
TRACE_CACHE = 256


class NotTaught(Exception):
    pass


class Family:
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

    # ---- one decision ----
    def load(self, spec) -> None:
        """Swap a specialist's learned weights into the shared brain (callers hold the lock)."""
        self.m.load_state_dict(self.base_state)
        self.m.load_state_dict(spec.weights, strict=False)

    def rest_of(self, spec) -> np.ndarray:
        """The specialist's own resting state: its brain with every glomerulus at rest (0.5), per step."""
        if spec.rest is None:
            with self.lock:
                self.load(spec)
                with torch.no_grad():
                    _, _, tr = self.m.run(torch.full((1, 46), 0.5), record=True)
            spec.rest = tr.float().numpy()[:, :, 0]
        return spec.rest

    def decide(self, spec, text: str, options: list[str] | None, allow_untaught: bool = False) -> dict:
        own = spec.card["options"]
        if options is not None and list(options) != own:
            if not allow_untaught:
                raise NotTaught(f"{spec.card['specialist']} answers only: {own}")
        opts = list(options) if options is not None else own
        z = self.smell([text] + opts)
        smells = np.clip(z[0][None] + z[1:] - 0.5, 0, 1)
        rest = self.rest_of(spec)
        with self.lock:
            self.load(spec)
            with torch.no_grad():
                logit, _, tr = self.m.run(torch.tensor(smells), record=True)
        lo = logit.numpy() / spec.card.get("temperature", 1.0)
        p = np.exp(lo - lo.max())
        p /= p.sum()
        k = int(np.argmax(p))
        n = len(opts)
        trace = self._trace(tr.float().numpy()[:, :, k] - rest, spec)
        return {
            "pick": opts[k],
            "p": {o: round(float(x), 4) for o, x in zip(opts, p, strict=True)},
            "sure": round(float(max(0.0, (n * p.max() - 1) / (n - 1))) if n > 1 else 1.0, 4),
            "taught": options is None or list(options) == own,
            "answered_as": spec.card["task"],
            "trace": trace,
        }

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
        self.weights = torch.load(d / "weights.pt", weights_only=True)
        self.rest = None  # computed on first use (Family.rest_of)


class Fleet:
    def __init__(self, root: Path, family_id: str = "v1"):
        self.family = Family(root / "families" / family_id)
        self.specs: dict[str, Specialist] = {}
        self.latest: dict[str, str] = {}
        for d in sorted((root / "specialists").glob("*/*")):
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
