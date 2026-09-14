"""Bosco's generator: a word-level n-gram model (order 3, absolute-discount
backoff) over the published corpus, conditioned on the fly's state.

Not an LLM: no embeddings, no neural weights, no training on other people's
posts.  What the fly contributes:
- which documents are preferred (tags matching behaviour/valence/arousal),
- the temperature (arousal: low 0.8, mid 1.0, high 1.15),
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
MIN_SENT_TOKENS = 3
# a sentence may not end on one of these (function words, dangling pronouns)
NO_END = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "are",
    "am",
    "has",
    "have",
    "i",
    "my",
    "your",
    "it",
    "what",
    "who",
    "where",
    "can",
    "do",
    "does",
    "not",
    "with",
    "from",
    "for",
    "so",
    "then",
    "but",
}
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
    """Join tokens. Bosco talks all lowercase: no capitals, 'i' stays 'i'."""
    s = ""
    for t in toks:
        if t in {".", ",", "!", "?", ";", ":"}:
            s += t
        else:
            s += (" " if s else "") + t.lower()
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
            for t in s[1:]:  # sentence-initial capitals do not count
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
    """Trigram over the corpus.  `doc_weight` is the learned preference per document
    (what-to-say learning): a rewarded reply raises the weight of the documents that
    matched its register, a punished one lowers it; weights decay toward 1 over weeks.
    Counts only; the vocabulary never changes."""

    def __init__(self, corpus_dir: Path = paths.ROOT / "corpus", phrasebook_lines: list[str] | None = None) -> None:
        self.doc_weight: dict[str, float] = {}
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

    @staticmethod
    def _doc_matches(d: Document, want: dict[str, str], topics: tuple[str, ...]) -> bool:
        """Untagged documents always match.  Tagged ones must agree on every register key they carry;
        a `topic=` tag matches only when that topic was smelled in the post."""
        if not d.tags:
            return True
        for k, v in d.tags.items():
            if k == "topic":
                if v not in topics:
                    return False
            elif k in want and want[k] != v:
                return False
        return True

    @staticmethod
    def _want(
        behaviour: str, valence: str, arousal: str, familiarity: str | None, state: dict[str, str] | None = None
    ) -> dict[str, str]:
        want = {"behaviour": behaviour, "valence": valence, "arousal": arousal}
        if familiarity:
            want["familiarity"] = familiarity  # new | known | familiar (phrasebook bins)
        for k, v in (state or {}).items():  # time=night|morning|day|evening, appetite=hungry|sated|""
            want[k] = v  # an empty value still excludes documents tagged with that key
        return want

    def matching_docs(
        self,
        behaviour: str,
        valence: str,
        arousal: str,
        topics: tuple[str, ...] = (),
        familiarity: str | None = None,
        state: dict[str, str] | None = None,
    ) -> list[str]:
        """Names of tagged documents that matched this register (the ones what-to-say learning touches)."""
        want = self._want(behaviour, valence, arousal, familiarity, state)
        return [d.name for d in self.docs if d.tags and self._doc_matches(d, want, topics)]

    def set_doc_weights(self, w: dict[str, float]) -> None:
        self.doc_weight = dict(w)
        self._models.clear()

    def model_for(
        self,
        behaviour: str,
        valence: str,
        arousal: str,
        topics: tuple[str, ...] = (),
        familiarity: str | None = None,
        state: dict[str, str] | None = None,
    ) -> NGram:
        key = (behaviour, valence, arousal, tuple(sorted(topics)), familiarity, tuple(sorted((state or {}).items())))
        if key in self._models:
            return self._models[key]
        want = self._want(behaviour, valence, arousal, familiarity, state)
        pool: list[list[str]] = []
        for d in self.docs:
            if self._doc_matches(d, want, topics):
                base = 1
                if d.tags:
                    base = 3  # matching tagged docs count triple
                    if "topic" in d.tags:
                        base = 5  # a document about what was just smelled counts most
                weight = max(0, int(round(base * self.doc_weight.get(d.name, 1.0))))
                pool.extend(d.sentences * weight)
        m = NGram(pool)
        self._models[key] = m
        return m

    def generate(
        self,
        behaviour: str,
        valence: str,
        arousal: str,
        seed: int,
        max_sentences: int = 3,
        topics: tuple[str, ...] = (),
        familiarity: str | None = None,
        state: dict[str, str] | None = None,
        word_valence: dict[str, float] | None = None,
        air: dict[str, float] | tuple[str, ...] = (),
        beta: float = 1.0,
        gamma: float = 0.5,
    ) -> str | None:
        """`word_valence` is what his mushroom body has learned about each word (from the weights);
        a sweet word is chosen more, a bitter one less.  `air` is what is on his antennae right
        now, each word by how much (0..1; a bare tuple counts as 1 each); they come up more."""
        if self.empty:
            return None
        m = self.model_for(behaviour, valence, arousal, topics, familiarity, state)
        wv = word_valence or {}
        in_air = dict(air) if isinstance(air, dict) else dict.fromkeys(air, 1.0)
        temp = {"low": 0.8, "mid": 1.0, "high": 1.15}.get(arousal, 1.0)
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
        used: set[tuple[str, str, str]] = set()  # no trigram twice in one utterance (loop guard)
        for _ in range(120):
            cands = m.candidates(a, b)
            content = sum(1 for t in out[len(out) - sent_len :] if t not in END_PUNCT and t not in {",", ";", ":"})
            dangling = bool(out) and out[-1].lower() in NO_END
            if (content < MIN_SENT_TOKENS or dangling) and sent_len < MAX_SENT_TOKENS:
                filtered = [c for c in cands if c != EOS and c not in END_PUNCT]
                if filtered:
                    cands = filtered
            fresh = [c for c in cands if (a, b, c) not in used or c == EOS]
            if fresh:
                cands = fresh
            ws = [
                m.p_tri(a, b, w) ** (1.0 / temp)
                * max(0.1, 1.0 + beta * wv.get(w, 0.0))
                * (1.0 + gamma * in_air.get(w, 0.0))
                for w in cands
            ]
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
            out.append(w)
            used.add((a, b, w))
            sent_len += 1
            a, b = b, w
        # drop a trailing dangling fragment left by the token cap
        while out and out[-1] in END_PUNCT:
            out.pop()
        while out and out[-1].lower() in NO_END:
            out.pop()
        text = detokenize(out).strip()
        if not text:
            return None
        if text[-1] not in END_PUNCT:
            text += "."
        return text[:MAX_CHARS]
