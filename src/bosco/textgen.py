"""Bosco's generator: a word-level n-gram model (order 3, absolute-discount
backoff) over the published corpus, conditioned on the fly's state.

Not an LLM: no embeddings, no neural weights, no training on other people's
posts.  What the fly contributes:
- which documents are preferred (tags matching behaviour/valence/arousal),
- the temperature (arousal: low 0.7, mid 1.0, high 1.3),
- the seed (episode seed), so the text replays bit-identically.

Output: 1–3 sentences, <= 280 characters, no URLs, no @mentions except
@proto.cool, tokens drawn only from the corpus.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from bosco import paths

BOS, EOS = "<s>", "</s>"
TOKEN_RE = re.compile(r"[A-Za-z0-9'’\-]+|[.,!?;:]")
END_PUNCT = {".", "!", "?"}
MAX_CHARS = 280
MAX_SENT_TOKENS = 22
ALLOWED_MENTIONS = {"@proto.cool"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def sentences(text: str) -> list[list[str]]:
    out, cur = [], []
    for tok in tokenize(text):
        cur.append(tok)
        if tok in END_PUNCT:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def detokenize(toks: list[str]) -> str:
    s = ""
    cap = True
    for t in toks:
        if t in {".", ",", "!", "?", ";", ":"}:
            s += t
            cap = cap or t in END_PUNCT
            continue
        if t == "i" or t.startswith("i'") or t.startswith("i’"):
            t = "I" + t[1:]
        if cap:
            t = t[0].upper() + t[1:]
            cap = False
        s += (" " if s else "") + t
    return s


class Document:
    def __init__(self, text: str, tags: dict[str, str], name: str) -> None:
        self.tags = tags
        self.name = name
        self.sentences = sentences(text)


class NGram:
    """Absolute-discount backoff trigram over a list of sentences."""

    def __init__(self, sents: list[list[str]], d: float = 0.5) -> None:
        self.d = d
        self.c3: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.c2: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.c1: dict[str, int] = defaultdict(int)
        self.surface: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in sents:
            for t in s:
                self.surface[t.lower()][t] += 1
            toks = [BOS, BOS] + [t.lower() for t in s] + [EOS]
            for i in range(2, len(toks)):
                self.c3[(toks[i - 2], toks[i - 1])][toks[i]] += 1
                self.c2[toks[i - 1]][toks[i]] += 1
                self.c1[toks[i]] += 1
        self.n1 = sum(self.c1.values())
        self.vocab = sorted(self.c1)

    def surface_form(self, w: str) -> str:
        """Most frequent casing of w in the corpus, preferring capitalised forms only when they dominate
        (so sentence-initial capitals do not turn every word into a proper noun)."""
        forms = self.surface.get(w)
        if not forms:
            return w
        best = max(sorted(forms), key=lambda f: forms[f])
        return best if best != w and forms[best] > sum(forms.values()) * 0.6 else w

    def p_uni(self, w: str) -> float:
        return self.c1.get(w, 0) / self.n1 if self.n1 else 0.0

    def p_bi(self, a: str, w: str) -> float:
        row = self.c2.get(a)
        if not row:
            return self.p_uni(w)
        tot = sum(row.values())
        lam = self.d * len(row) / tot
        return max(row.get(w, 0) - self.d, 0) / tot + lam * self.p_uni(w)

    def p_tri(self, a: str, b: str, w: str) -> float:
        row = self.c3.get((a, b))
        if not row:
            return self.p_bi(b, w)
        tot = sum(row.values())
        lam = self.d * len(row) / tot
        return max(row.get(w, 0) - self.d, 0) / tot + lam * self.p_bi(b, w)

    def candidates(self, a: str, b: str) -> list[str]:
        c = set(self.c3.get((a, b), {})) | set(self.c2.get(b, {}))
        if len(c) < 3:
            c |= set(self.vocab)
        return sorted(c)


class _Rng:
    """Deterministic splitmix64, independent of numpy versions."""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def unit(self) -> float:
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        return (z >> 11) / 9007199254740992.0


class Generator:
    def __init__(self, corpus_dir: Path = paths.ROOT / "corpus", phrasebook_lines: list[str] | None = None) -> None:
        self.docs: list[Document] = []
        for p in sorted(corpus_dir.glob("*.txt")):
            text = p.read_text(encoding="utf-8")
            tags: dict[str, str] = {}
            first, _, rest = text.partition("\n")
            if first.startswith("#tags:"):
                for kv in first[6:].split():
                    k, _, v = kv.partition("=")
                    tags[k] = v
                text = rest
            self.docs.append(Document(text, tags, p.name))
        if phrasebook_lines:
            self.docs.append(Document("\n".join(phrasebook_lines), {}, "phrasebook"))
        self._models: dict[tuple, NGram] = {}

    @property
    def empty(self) -> bool:
        return not any(d.sentences for d in self.docs)

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for d in self.docs:
            h.update(d.name.encode())
            for s in d.sentences:
                h.update((" ".join(s) + "\n").encode())
        return h.hexdigest()

    def model_for(self, behaviour: str, valence: str, arousal: str) -> NGram:
        key = (behaviour, valence, arousal)
        if key in self._models:
            return self._models[key]
        want = {"behaviour": behaviour, "valence": valence, "arousal": arousal}
        pool: list[list[str]] = []
        for d in self.docs:
            if not d.tags or all(d.tags.get(k, v) == v for k, v in want.items()):
                weight = 3 if d.tags else 1  # matching tagged docs count triple
                pool.extend(d.sentences * weight)
        m = NGram(pool)
        self._models[key] = m
        return m

    def generate(self, behaviour: str, valence: str, arousal: str, seed: int, max_sentences: int = 3) -> str | None:
        if self.empty:
            return None
        m = self.model_for(behaviour, valence, arousal)
        temp = {"low": 0.7, "mid": 1.0, "high": 1.3}.get(arousal, 1.0)
        rng = _Rng(
            int.from_bytes(
                hashlib.blake2b(f"{seed}|{behaviour}|{valence}|{arousal}".encode(), digest_size=8).digest(), "little"
            )
        )
        n_sent = 1 + int(rng.unit() * max_sentences)
        out: list[str] = []
        a, b = BOS, BOS
        n_done = 0
        sent_len = 0
        for _ in range(120):
            cands = m.candidates(a, b)
            ws = [m.p_tri(a, b, w) ** (1.0 / temp) for w in cands]
            tot = sum(ws)
            if tot <= 0:
                break
            u = rng.unit() * tot
            acc = 0.0
            w = cands[-1]
            for c, p in zip(cands, ws, strict=True):
                acc += p
                if u <= acc:
                    w = c
                    break
            if w == EOS or sent_len >= MAX_SENT_TOKENS:
                if w != EOS and out and out[-1] not in END_PUNCT:
                    out.append(".")
                n_done += 1
                sent_len = 0
                if n_done >= n_sent or len(detokenize(out)) > MAX_CHARS * 0.7:
                    break
                a, b = BOS, BOS
                continue
            if w.startswith("@") and w not in ALLOWED_MENTIONS:
                continue
            if "http" in w or "://" in w:
                continue
            out.append(m.surface_form(w))
            sent_len += 1
            a, b = b, w
        text = detokenize(out).strip()
        if not text:
            return None
        if text[-1] not in END_PUNCT:
            text += "."
        return text[:MAX_CHARS]
