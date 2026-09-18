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
EOS_BIAS = 2.5  # weight on ending a sentence where a corpus sentence ended, against splicing on
MIN_SENT_TOKENS = 1  # a one-word sentence is his register ("banana", "rude", "warm")
OPEN_MIN_TOKENS = 3  # but not when the word was handed to him: an answer has to say something
SEEN_WEIGHT = 20  # asked about a smell, his yes or his no (seen=met|fresh) is at least this, and at least half the pool
SEEN_PICK = 3.0  # and in retrieval a line from that file counts three times over (2026-09-16, with a bigger corpus)
# He may borrow a phrase, not a paragraph (decided 2026-09-18).  Once the tail of what he is
# saying has run this many tokens alongside one line of his, he is pushed off it: first towards
# the end of the sentence, which always reads, and only then towards another line of his, which
# is where a splice can turn clumsy.  Nine tokens is a short sentence of his, so a whole small
# thought still comes out in one piece.
COPY_RUN_MAX = 9
GUARD_EOS_BIAS = 6.0  # at the cap he would rather stop than splice: a full stop is always his
# And inside a sentence he stays on the line he is on.  A splice mid-clause is where he turns
# clumsy ("a sweet part in it is far in the dark"), so the words that keep his own clause going
# are weighted up, and he changes his mind between sentences instead, where it always reads.
STAY_BIAS = 4.0
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
    """Sentences end at a period, question mark or exclamation, and at the end of a line: a corpus
    line is one utterance whether or not it carries a final period (most of his do not), so the
    model never learns to run one line into the next."""
    out: list[list[str]] = []
    for line in text.splitlines():
        cur: list[str] = []
        for tok in tokenize(line):
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


def lines_of(text: str) -> list[list[list[str]]]:
    """The corpus lines, each as its sentences: a line is one thought, the unit of retrieval."""
    return [sentences(line) for line in text.splitlines() if line.strip() and not line.startswith("#")]


class Document:
    def __init__(self, text: str, tags: dict[str, str], name: str) -> None:
        self.tags = tags
        self.name = name
        self.sentences = sentences(text)
        self.lines = lines_of(text)


class NGram:
    """Absolute-discount backoff trigram over a list of sentences."""

    def __init__(self, sents: list[list[str]], d: float = 0.5) -> None:
        self.d = d
        self.c4: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.c3: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.c2: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.c1: dict[str, int] = defaultdict(int)
        self.surface: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in sents:
            for t in s[1:]:  # sentence-initial capitals do not count
                self.surface[t.lower()][t] += 1
            toks = [BOS, BOS, BOS] + [t.lower() for t in s] + [EOS]
            for i in range(3, len(toks)):
                self.c4[(toks[i - 3], toks[i - 2], toks[i - 1])][toks[i]] += 1
                self.c3[(toks[i - 2], toks[i - 1])][toks[i]] += 1
                self.c2[toks[i - 1]][toks[i]] += 1
                self.c1[toks[i]] += 1
        self.n1 = sum(self.c1.values())
        self.vocab = sorted(self.c1)
        # contexts that end in a word, for opening an utterance on it: w -> [(a, w), ...]
        self.pred: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for a, b in self.c3:
            self.pred[b].append((a, b))

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

    def p4(self, z: str, a: str, b: str, w: str) -> float:
        row = self.c4.get((z, a, b))
        if not row:
            return self.p_tri(a, b, w)
        tot = sum(row.values())
        lam = self.d * len(row) / tot
        return max(row.get(w, 0) - self.d, 0) / tot + lam * self.p_tri(a, b, w)

    def candidates(self, z: str, a: str, b: str) -> list[str]:
        """Continuations of a context: the words that followed the last three in the corpus; only
        when the corpus never saw those three does the model back off to the last two, then the
        last one, then anything.  So what he says is his own sentences, spliced only where two
        of them share three words in a row, not words strung by chance."""
        row = self.c4.get((z, a, b)) or self.c3.get((a, b))
        if row:
            return sorted(row)
        c = set(self.c2.get(b, {}))
        if len(c) < 2:
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
        self._lines: list[list[str]] | None = None
        self._starts: dict[str, list[tuple[int, int]]] | None = None

    def _copy_index(self) -> tuple[list[list[str]], dict[str, list[tuple[int, int]]]]:
        """Every line of his as one token list, and where each token occurs in them: enough to
        know, while he is talking, how long he has been running alongside one of his own lines."""
        if self._lines is None:
            lines: list[list[str]] = []
            starts: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for d in self.docs:
                for line in d.lines:
                    toks = [t.lower() for sent in line for t in sent]
                    if not toks:
                        continue
                    i = len(lines)
                    lines.append(toks)
                    for j, t in enumerate(toks):
                        starts[t].append((i, j))
            self._lines, self._starts = lines, dict(starts)
        return self._lines, self._starts

    def _copy_step(self, w: str, matches: set[tuple[int, int]], run: int) -> tuple[set[tuple[int, int]], int]:
        """Where he now stands in his own lines, after saying `w`, and how many tokens he has
        said alongside one of them without a break."""
        lines, starts = self._copy_index()
        w = w.lower()
        grown = {(i, j + 1) for i, j in matches if j + 1 < len(lines[i]) and lines[i][j + 1] == w}
        if grown:
            return grown, run + 1
        return set(starts.get(w, ())), 1

    def _copy_next(self, matches: set[tuple[int, int]]) -> set[str]:
        """The words that would carry one of his lines on from here."""
        lines, _ = self._copy_index()
        return {lines[i][j + 1] for i, j in matches if j + 1 < len(lines[i])}

    def _copy_at_stop(self, matches: set[tuple[int, int]]) -> bool:
        """Whether he is standing where one of his own sentences stops.  A word that dangles in
        general ("it", "for") does not dangle there: "today has a sweet part in it" is a line of
        his, and left to end on its own it reads, while carrying it on into another line's clause
        ("...in it is far in the dark") does not."""
        lines, _ = self._copy_index()
        return any(j + 1 >= len(lines[i]) or lines[i][j + 1] in END_PUNCT for i, j in matches)

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
        chosen: list[tuple[Document, int]] = []
        for d in self.docs:
            if self._doc_matches(d, want, topics):
                base = 1
                if d.tags:
                    base = 3  # matching tagged docs count triple
                    if "topic" in d.tags:
                        base = 5  # a document about what was just smelled counts most
                    if "seen" in d.tags and "seen" in want:
                        base = 0  # set below: he was asked about a smell, his yes or his no is half the pool
                chosen.append((d, base))
        # the seen file's weight keeps it at least half the pool however large the corpus grows
        others = sum(len(d.sentences) * b for d, b in chosen if b)
        seen_n = sum(len(d.sentences) for d, b in chosen if not b)
        seen_w = max(SEEN_WEIGHT, -(-others // seen_n)) if seen_n else 0
        pool: list[list[str]] = []
        for d, base in chosen:
            weight = max(0, int(round((base or seen_w) * self.doc_weight.get(d.name, 1.0))))
            pool.extend(d.sentences * weight)
        m = NGram(pool)
        self._models[key] = m
        return m

    def pick_sentence(
        self,
        behaviour: str,
        valence: str,
        arousal: str,
        seed: int,
        air: dict[str, float] | tuple[str, ...] = (),
        topics: tuple[str, ...] = (),
        familiarity: str | None = None,
        state: dict[str, str] | None = None,
        word_valence: dict[str, float] | None = None,
        beta: float = 1.0,
        avoid: set[str] | None = None,
        top_k: int = 5,
    ) -> str | None:
        """The line of his that smells most like the moment: every corpus line (one thought, one
        to a few sentences) in the matching register is scored by the words on his antennae it
        contains (each by its freshness, and by what he has learned of that word with this
        person), and the seed picks among the top few.  Whole, his, and about what is in front
        of him (a line, not a sentence, since 2026-09-16: "you said thank." alone was a stub).
        None when nothing in the air is in any line, or nothing is in the air; the caller then
        stitches as before."""
        in_air = dict(air) if isinstance(air, dict) else dict.fromkeys(air, 1.0)
        if self.empty or not in_air:
            return None
        want = self._want(behaviour, valence, arousal, familiarity, state)
        wv = word_valence or {}
        avoid = avoid or set()
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for d in self.docs:
            if not self._doc_matches(d, want, topics):
                continue
            bonus = 1.0 if not d.tags else (1.5 if "topic" in d.tags else 1.2)
            if d.tags and "seen" in d.tags and "seen" in want:
                bonus = SEEN_PICK  # asked about a smell: his yes or his no leads the answer
            for line in d.lines:
                sent = [t for s in line for t in s]
                toks = [t.lower() for t in sent if t not in END_PUNCT and t not in {",", ";", ":"}]
                if not toks:
                    continue
                shared = [t for t in set(toks) if in_air.get(t, 0.0) > 0]
                hit = sum(in_air[t] * max(0.1, 1.0 + beta * wv.get(t, 0.0)) for t in shared)
                if hit <= 0:
                    continue
                # a sentence that shares two of their words beats one that shares one, whatever
                # its freshness: coverage of the moment, not one loud word
                hit *= 1.0 + 0.5 * (len(shared) - 1)
                text = detokenize(sent)
                if text in seen or text in avoid or len(text) > MAX_CHARS:
                    continue
                seen.add(text)
                # a short sentence that is mostly about the thing beats a long one that mentions it
                scored.append((bonus * hit / (1.0 + 0.05 * len(toks)), text))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:top_k]
        rng = _Rng(int.from_bytes(hashlib.blake2b(f"{seed}|pick".encode(), digest_size=8).digest(), "little"))
        tot = sum(sc for sc, _ in top)
        u = rng.unit() * tot
        acc = 0.0
        for sc, text in top:
            acc += sc
            if u <= acc:
                return text
        return top[-1][1]

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
        prime: bool = False,
        opening: str | None = None,
    ) -> str | None:
        """`word_valence` is what his mushroom body has learned about each word (from the weights);
        a sweet word is chosen more, a bitter one less.  `air` is what is on his antennae right
        now, each word by how much (0..1; a bare tuple counts as 1 each); they come up more.
        With `prime` (he is answering someone) the utterance opens on one of the words in the
        air, drawn by how fresh it is, and walks his own sentences from there: an answer is about
        the word they used, in his words."""
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
        z, a, b = BOS, BOS, BOS
        n_done = 0
        sent_len = 0
        matches: set[tuple[int, int]] = set()  # where he stands in his own lines
        run = 0  # how long he has been running alongside one of them
        opened = False  # he was started on a word, and a word alone is not an answer
        if opening:
            # The thought retrieval found is where he starts, not what he says (decided
            # 2026-09-18: said whole, it was a quotation, and nine words in ten of an utterance
            # were one lifted run).  He opens on the word of theirs that lit that thought up,
            # with the words before it in it as his context, and walks on from there himself.
            toks = tokenize(opening)
            spots = [
                (k, in_air.get(t.lower(), 0.0))
                for k, t in enumerate(toks)
                if t not in END_PUNCT and in_air.get(t.lower(), 0.0) > 0
            ]
            k0 = 0
            if spots:
                tot = sum(d for _, d in spots)
                u = rng.unit() * tot
                acc = 0.0
                k0 = spots[-1][0]
                for k, d in spots:
                    acc += d
                    if u <= acc:
                        k0 = k
                        break
            sb = 0  # the top of the sentence their word sits in
            for k in range(k0 - 1, -1, -1):
                if toks[k] in END_PUNCT:
                    sb = k + 1
                    break
            # He takes the run-up to their word with him when it is short enough to be a phrase
            # rather than a paragraph: "some kind of wind came" reads, "kind of wind came" is a
            # fragment.  Deeper into a sentence than the run guard would let him carry anyway, he
            # opens on the word itself.  Either way the borrowed part is his own clause, it ends
            # on their word, and the guard counts it, so the walk takes over within a few words.
            head = list(toks[sb : k0 + 1]) if k0 - sb < COPY_RUN_MAX else [toks[k0]]
            if head:
                for t in head:
                    out.append(t)
                    matches, run = self._copy_step(t, matches, run)
                sent_len = len(head)
                z, a, b = ([BOS, BOS, BOS] + [t.lower() for t in head])[-3:]
                prime = False
                opened = True
        if prime and in_air:
            # open on one of their words: a context of his that ends in it, chosen by freshness
            options = [(w, d) for w, d in in_air.items() if m.pred.get(w)]
            tot = sum(d for _, d in options)
            if options and tot > 0:
                u = rng.unit() * tot
                acc = 0.0
                w0 = options[-1][0]
                for w, d in options:
                    acc += d
                    if u <= acc:
                        w0 = w
                        break
                ctxs = sorted(m.pred[w0])
                a, b = ctxs[int(rng.unit() * len(ctxs)) % len(ctxs)]
                z = BOS
                out.append(m.surface_form(w0))
                sent_len = 1
                opened = True
                matches, run = self._copy_step(w0, matches, run)
        used: set[tuple[str, str, str, str]] = set()  # no four-gram twice in one utterance (loop guard)
        for _ in range(120):
            cands = m.candidates(z, a, b)
            content = sum(1 for t in out[len(out) - sent_len :] if t not in END_PUNCT and t not in {",", ";", ":"})
            dangling = bool(out) and out[-1].lower() in NO_END and not self._copy_at_stop(matches)
            # a sentence of one word is his register; a sentence of one word he was handed is a
            # stub ("today."), so a started sentence has to get somewhere before it may end
            floor = OPEN_MIN_TOKENS if (opened and n_done == 0) else MIN_SENT_TOKENS
            if (content < floor or dangling) and sent_len < MAX_SENT_TOKENS:
                filtered = [c for c in cands if c != EOS and c not in END_PUNCT]
                if not filtered and content < floor:
                    # he was started on a word that his own line ended on, and one word is not an
                    # answer: he keeps less of that line's context and finds somewhere else of
                    # his to take it.  Only for a stub -- a finished clause is allowed to finish.
                    for zz, aa in ((BOS, a), (BOS, BOS)):
                        alt = [c for c in m.candidates(zz, aa, b) if c != EOS and c not in END_PUNCT]
                        if alt:
                            filtered, z, a = alt, zz, aa
                            break
                if filtered:
                    cands = filtered
            fresh = [c for c in cands if (z, a, b, c) not in used or c == EOS]
            if fresh:
                cands = fresh
            at_cap = run >= COPY_RUN_MAX and content >= floor
            carry_on = self._copy_next(matches) if matches else set()
            if at_cap:
                # he has said as much of that line as he is allowed to: the rest is his, and the
                # heavy bias on ending (GUARD_EOS_BIAS) means he stops here where he can
                own = [c for c in cands if c == EOS or c.lower() not in carry_on]
                if own:
                    cands = own
                carry_on = set()
            ws = [
                m.p4(z, a, b, w) ** (1.0 / temp)
                * max(0.1, 1.0 + beta * wv.get(w, 0.0))
                * (1.0 + gamma * in_air.get(w, 0.0))
                * (STAY_BIAS if w.lower() in carry_on else 1.0)  # finish the clause he is in
                # he ends where his sentences end more often than he splices, and at the cap he
                # would rather end than be pushed into somebody else's clause
                * ((EOS_BIAS * (GUARD_EOS_BIAS if at_cap else 1.0)) if w == EOS else 1.0)
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
                if out and out[-1] not in END_PUNCT:
                    out.append(".")  # a corpus line ends without a period; his sentences still do
                    matches, run = self._copy_step(".", matches, run)
                n_done += 1
                sent_len = 0
                if n_done >= n_sent or len(detokenize(out)) > MAX_CHARS * 0.7:
                    break
                z, a, b = BOS, BOS, BOS
                continue
            if w.startswith("@") and w not in ALLOWED_MENTIONS:
                continue
            if "http" in w or "://" in w:
                continue
            out.append(w)
            used.add((z, a, b, w))
            matches, run = self._copy_step(w, matches, run)
            sent_len += 1
            z, a, b = a, b, w
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
