"""Encoder v1: (account DID, VADER compound, mentioned?) -> Stimulus.

Nothing here reads post text except the VADER score computed upstream; the
encoder never sees the text itself.  See docs/encoder.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.feeds import Feeds
from bosco.model import Brain
from bosco.sim import Drive, Stimulus


@dataclass(frozen=True)
class Features:
    """Everything the fly ever learns about an event.  This is what the stimulus log stores."""

    did: str
    vader: float  # compound score in [-1, 1]
    mentioned: bool
    # informational only (phrasebook key), not fed to the network:
    familiarity: int = 0
    # moderation label present (config/moderation_v1.yaml): bitter at full rate, no approach
    labeled: bool = False
    # topics found by the published keyword map (config/topics_v1.yaml); each is a small odor mixture
    topics: tuple[str, ...] = ()
    # the post asks something ('?'): a stronger touch (JO drive x question_gain); approach becomes a reply
    question: bool = False
    # his own words that were in the post (config/words_v1.yaml): each is an odor.  Never the sentence.
    words: tuple[str, ...] = ()
    # words still in the air from earlier posts in this thread, smelled again at a lower rate
    context: tuple[str, ...] = ()
    # where he read it (config/feeds_v1.yaml): a place has a smell.  None for a mention.
    feed: str | None = None
    # other people in the post (mention facets, a quoted post's author): each an odor at a lower rate
    others: tuple[str, ...] = ()
    # what else the post carried (config/encoder_v1.yaml `embeds`): img, video, card, quote,
    # site:<domain>, and the retina's channels for an image.  Never the image, never the text.
    embed: tuple[str, ...] = ()


class Encoder:
    def __init__(self, brain: Brain, config_path=paths.CONFIG / "encoder_v1.yaml") -> None:
        self.cfg = yaml.safe_load(open(config_path))
        self.brain = brain
        orn = pop.orns()
        excl = set(self.cfg["odor"]["exclude"])
        self.all_glomeruli = sorted(orn["glomerulus"].unique())
        self.neutral = [g for g in self.all_glomeruli if g not in excl]
        self.orn_by_glom = {
            g: brain.index_of_present(orn.loc[orn["glomerulus"] == g, "bodyId"]) for g in self.all_glomeruli
        }
        self.sugar = brain.index_of_present(pop.grns("sugar/water"))
        self.bitter = brain.index_of_present(pop.grns("bitter"))
        jo = pop.johnston_organ()
        groups = set(self.cfg["mechanosensory"]["jo_groups"])
        self.jo = brain.index_of_present(jo.loc[jo["group"].isin(groups), "bodyId"])
        from bosco.topics import TopicMap

        self.topics = TopicMap()
        sp = self.cfg.get("spontaneous", {})
        self.bristles = (
            brain.index_of_present(pop.bodies_of_types(sp.get("bristle_types", []))) if sp else np.zeros(0, np.int32)
        )
        self.pc1 = brain.index_of_present(pop.pc1()) if self.cfg.get("courtship") else np.zeros(0, np.int32)
        self.words_cfg = yaml.safe_load(open(paths.CONFIG / "words_v1.yaml"))
        self.vocab = self._load_vocab()
        self.feeds = Feeds()
        # the retina (config/retina_v1.yaml): channels drive the visual Kenyon cells; colour words
        # and the words for pictures are those channels
        from bosco.retina import load_retina_cfg

        self.retina_cfg = load_retina_cfg()
        self.visual_kcs = brain.index_of_present(pop.visual_kcs(tuple(self.retina_cfg["kc_types"])))
        self.visual_words: dict[str, str] = {}
        for ch, ws in (self.retina_cfg.get("visual_words") or {}).items():
            for w in ws:
                self.visual_words[str(w)] = str(ch)
        # innate smells (config/innate_v1.yaml): word -> the glomeruli a fly is born to answer
        innate = yaml.safe_load(open(paths.CONFIG / "innate_v1.yaml"))
        self.innate: dict[str, tuple[str, list[str]]] = {}
        for name, sm in innate.get("smells", {}).items():
            for w in sm["words"]:
                self.innate[str(w)] = (str(name), [str(g) for g in sm["glomeruli"]])

    # ---- words as smells --------------------------------------------------
    def _load_vocab(self) -> frozenset[str]:
        """His closed vocabulary: content words of the corpus and phrasebook.  Also counts, per
        word, the corpus lines it is in (`word_lines`): how common a word is in his mouth, which
        words_for uses to keep the rarer words of a post when there are more than max_words."""
        from bosco.textgen import Generator, tokenize

        stop = set(self.words_cfg["stopwords"])
        min_len = int(self.words_cfg["min_len"])
        words: set[str] = set()
        self.word_lines: dict[str, int] = {}
        try:
            from bosco.phrasebook import Phrasebook

            lines = [ln.text for ln in Phrasebook().lines]
        except Exception:  # noqa: BLE001
            lines = []
        g = Generator(phrasebook_lines=lines)
        for d in g.docs:
            for line in d.lines:
                in_line: set[str] = set()
                for sent in line:
                    for t in sent:
                        w = t.lower().replace("'", "")
                        if w.isalpha() and len(w) >= min_len and w not in stop:
                            words.add(w)
                            in_line.add(w)
                for w in in_line:
                    self.word_lines[w] = self.word_lines.get(w, 0) + 1
        for t in tokenize(" ".join(lines)):
            w = t.lower().replace("'", "")
            if w.isalpha() and len(w) >= min_len and w not in stop:
                words.add(w)
        return frozenset(words)

    HASH_PREFIX = "h:"

    @staticmethod
    def word_hash(word: str) -> bytes:
        return hashlib.blake2b(f"word|{word}".encode(), digest_size=8).digest()

    @classmethod
    def hashed(cls, word: str) -> str:
        """A word outside his vocabulary as he keeps it: the hash that seeds its smell, never the word."""
        return cls.HASH_PREFIX + cls.word_hash(word).hex()

    def words_for(self, text: str, hashed: bool | None = None) -> tuple[str, ...]:
        """The words of his vocabulary in a post, at most max_words, in order of first
        appearance; when a post has more, the rarer ones in his corpus are kept (decided
        2026-09-16, when the conversational words stopped being stopwords: "i know i like the
        little bird" keeps bird over know).  Then, if the perception vocabulary is on
        (config/words_v1.yaml `perception`), up to max_hashed other content words as `h:`
        tokens.  The text is read once and discarded; only these are kept."""
        import re

        from bosco.textgen import tokenize

        # handles and links are not words: "@bosco.proto.cool" would otherwise smell of bosco and cool
        text = re.sub(r"@[\w.\-]+|https?://\S+|\b[\w\-]+\.[a-z]{2,}(?:/\S*)?", " ", text)
        pc = self.words_cfg.get("perception") or {}
        max_hashed = int(pc.get("max_hashed", 0)) if (hashed is None or hashed) else 0
        stop = set(self.words_cfg["stopwords"])
        min_len = int(self.words_cfg["min_len"])
        # shorthand he understands but never says: "gn" is heard as his words good and night
        shorthand = {
            str(k): [str(x) for x in (v if isinstance(v, list) else [v])]
            for k, v in (pc.get("shorthand") or {}).items()
        }
        found: list[str] = []
        other: list[str] = []
        seen: set[str] = set()
        for t in tokenize(text):
            raw = t.lower().replace("'", "")
            for w in shorthand.get(raw) or (self.fold(raw),):
                if w in seen:
                    continue
                if w in self.vocab:
                    seen.add(w)
                    found.append(w)
                elif max_hashed and w.isalpha() and len(w) >= min_len and w not in stop:
                    seen.add(w)
                    if len(other) < max_hashed:
                        other.append(self.hashed(w))
        max_words = int(self.words_cfg["max_words"])
        if len(found) > max_words:
            lines = getattr(self, "word_lines", {})
            keep = set(sorted(found, key=lambda w: (lines.get(w, 0), found.index(w)))[:max_words])
            found = [w for w in found if w in keep]
        return tuple(found) + tuple(other)

    def fold(self, w: str) -> str:
        """A plural of his word is his word: "cats" is the smell of cat, "pictures" of picture.
        Only the trailing s or es, only when the singular is his; nothing else is folded.  Not a
        stemmer: a rule you can read (config/words_v1.yaml)."""
        if w in self.vocab or not (self.words_cfg.get("perception") or {}).get("fold_plurals", True):
            return w
        if w.endswith("es") and w[:-2] in self.vocab:
            return w[:-2]
        if w.endswith("s") and w[:-1] in self.vocab:
            return w[:-1]
        return w

    def word_glomeruli(self, word: str) -> tuple[list[np.ndarray], float]:
        """A word's glomeruli (the olfactory neurons of each) and its rate.  A word that names a
        smell a fly is born to answer (config/innate_v1.yaml) is those glomeruli; any other word
        is k neutral glomeruli chosen by its hash.  The rate is by hash either way.  An `h:`
        token seeds from the hash it carries, so a hashed word is the word's own smell."""
        c = self.words_cfg
        if word.startswith(self.HASH_PREFIX):
            seed = int.from_bytes(bytes.fromhex(word[len(self.HASH_PREFIX) :]), "little")
        else:
            seed = int.from_bytes(self.word_hash(word), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=int(c["k"]), replace=False)
        lo, hi = c["rate_hz"]
        rate = float(lo + (hi - lo) * rng.random())
        if word in self.innate:
            return [self.orn_by_glom[g] for g in self.innate[word][1] if g in self.orn_by_glom], rate
        if word in self.visual_words:
            # a colour word, or the word for a picture, is the retina's channel (config/retina_v1.yaml)
            return [self.visual_channel_idx(self.visual_words[word])], float(self.retina_cfg["rate_hz"])
        return [self.orn_by_glom[self.neutral[i]] for i in chosen], rate

    def word_drive(self, word: str, scale: float = 1.0) -> Drive:
        """A word is k neutral glomeruli and a rate, both chosen by its hash."""
        gloms, rate = self.word_glomeruli(word)
        idx = np.concatenate(gloms).astype(np.int32)
        return Drive(np.sort(idx), round(rate * scale, 6), f"word:{word}")

    # ---- the retina --------------------------------------------------------
    def visual_channel_idx(self, channel: str) -> np.ndarray:
        """The visual Kenyon cells of a channel: k_per_channel of them chosen by the channel's name."""
        rc = self.retina_cfg
        if len(self.visual_kcs) == 0:
            return np.zeros(0, np.int32)
        seed = int.from_bytes(hashlib.blake2b(f"retina|{channel}".encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        k = min(int(rc["k_per_channel"]), len(self.visual_kcs))
        return np.sort(self.visual_kcs[rng.choice(len(self.visual_kcs), size=k, replace=False)]).astype(np.int32)

    def visual_drive(self, channel: str) -> Drive | None:
        idx = self.visual_channel_idx(channel)
        if len(idx) == 0:
            return None
        return Drive(idx, float(self.retina_cfg["rate_hz"]), f"see:{channel}")

    # ---- account odor ---------------------------------------------------
    def glomeruli_for(self, did: str) -> list[str]:
        k = int(self.cfg["odor"]["k"])
        seed = int.from_bytes(hashlib.blake2b(did.encode("utf-8"), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=k, replace=False)
        return sorted(self.neutral[i] for i in chosen)

    def odor_drive(self, did: str) -> Drive:
        gl = self.glomeruli_for(did)
        idx = np.concatenate([self.orn_by_glom[g] for g in gl]).astype(np.int32)
        return Drive(np.sort(idx), float(self.cfg["odor"]["rate_hz"]), f"odor:{did}")

    def topic_drive(self, topic: str) -> Drive:
        """A topic is k neutral glomeruli chosen by the topic name, driven on top of the account odor."""
        seed = int.from_bytes(hashlib.blake2b(f"topic|{topic}".encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=self.topics.k, replace=False)
        idx = np.concatenate([self.orn_by_glom[self.neutral[i]] for i in chosen]).astype(np.int32)
        return Drive(np.sort(idx), self.topics.rate_hz, f"topic:{topic}")

    # ---- other people, and where a link goes ---------------------------------
    def others_drive(self, did: str) -> Drive:
        """Someone mentioned in or quoted by the post: their odor, fainter than the author's."""
        em = self.cfg.get("embeds") or {}
        gl = self.glomeruli_for(did)
        idx = np.concatenate([self.orn_by_glom[g] for g in gl]).astype(np.int32)
        rate = float(self.cfg["odor"]["rate_hz"]) * float(em.get("others_rate_scale", 0.5))
        return Drive(np.sort(idx), round(rate, 6), f"other:{did}")

    def site_drive(self, domain: str) -> Drive:
        """A link card's site is a place: k neutral glomeruli chosen by the domain, like a feed."""
        em = self.cfg.get("embeds") or {}
        seed = int.from_bytes(hashlib.blake2b(f"site|{domain}".encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=int(em.get("site_k", 2)), replace=False)
        idx = np.concatenate([self.orn_by_glom[self.neutral[i]] for i in chosen]).astype(np.int32)
        return Drive(np.sort(idx), float(em.get("site_rate_hz", 60.0)), f"site:{domain}")

    def embed_drives(self, tokens: tuple[str, ...]) -> list[Drive]:
        """Drives for the embed tokens on a post: a site is a place; the retina's channels (img,
        motion, hue/lum/edge/sat) drive the visual Kenyon cells; card and quote name what was
        there and drive nothing."""
        from bosco.retina import is_channel

        out: list[Drive] = []
        for t in tokens:
            if t.startswith("site:") and len(t) > 5:
                out.append(self.site_drive(t[5:]))
            elif is_channel(t):
                d = self.visual_drive(t)
                if d is not None:
                    out.append(d)
        return out

    # ---- place ------------------------------------------------------------
    def feed_drive(self, feed: str) -> Drive:
        """A feed is a place: k neutral glomeruli chosen by its name, on top of the account odor."""
        seed = int.from_bytes(hashlib.blake2b(f"feed|{feed}".encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=self.feeds.k, replace=False)
        idx = np.concatenate([self.orn_by_glom[self.neutral[i]] for i in chosen]).astype(np.int32)
        return Drive(np.sort(idx), self.feeds.rate_hz, f"feed:{feed}")

    # ---- taste -----------------------------------------------------------
    def gustatory_drive(self, vader: float) -> Drive | None:
        g = self.cfg["gustatory"]
        c = float(np.clip(vader, -1.0, 1.0))
        if abs(c) <= g["dead_zone"]:
            return None
        rate = g["max_rate_hz"] * min(1.0, abs(c) / g["c_sat"])
        return Drive(self.sugar if c > 0 else self.bitter, rate, "sugar" if c > 0 else "bitter")

    # ---- touch -----------------------------------------------------------
    def mention_drive(self, question: bool = False) -> Drive:
        rate = float(self.cfg["mechanosensory"]["rate_hz"])
        if question:
            rate *= float(self.cfg["mechanosensory"].get("question_gain", 1.5))
        return Drive(self.jo, rate, "mention?" if question else "mention")

    # ---- being addressed: the courtship command ----------------------------
    def courtship_drive(self, appetite: float, question: bool = False) -> Drive | None:
        """pC1 driven at rate_hz x appetite (x question_gain for a question).  No appetite, no drive."""
        c = self.cfg.get("courtship")
        if not c or len(self.pc1) == 0:
            return None
        rate = float(c["rate_hz"]) * float(np.clip(appetite, 0.0, 1.0))
        if question:
            rate *= float(c.get("question_gain", 1.0))
        if rate <= 0.0:
            return None
        return Drive(self.pc1, round(rate, 6), "courtship?" if question else "courtship")

    # ---- internal drive (no event) ---------------------------------------
    def spontaneous_drive(self, landing_id: int, drive: float = 1.0) -> Drive | None:
        """Debris on his bristles.  One seeded permutation per landing; the first k_max * drive
        bristles of it carry debris, so as the debris settles the subset shrinks and never adds
        an onset.  drive in [0, 1]."""
        sp = self.cfg.get("spontaneous")
        if not sp or len(self.bristles) == 0:
            return None
        k = min(int(round(int(sp["k"]) * float(np.clip(drive, 0.0, 1.0)))), len(self.bristles))
        if k <= 0:
            return None
        rng = np.random.default_rng(int(landing_id) & 0xFFFFFFFF)
        perm = rng.permutation(self.bristles)
        idx = np.sort(perm[:k]).astype(np.int32)
        return Drive(idx, float(sp["rate_hz"]), "bristles")

    def encode(self, f: Features, appetite: float = 0.0) -> Stimulus:
        """The stimulus for an event.  `appetite` is his state, not a feature of the event: it
        sets how hard being addressed excites the courtship command."""
        drives = [self.odor_drive(f.did)]
        if f.feed:
            drives.append(self.feed_drive(f.feed))
        drives += [self.others_drive(d) for d in f.others if d and d != f.did]
        drives += self.embed_drives(f.embed)
        drives += [self.topic_drive(t) for t in f.topics]
        drives += [self.word_drive(w) for w in f.words]
        scale = float(self.words_cfg.get("context_rate_scale", 0.5))
        drives += [self.word_drive(w, scale) for w in f.context if w not in f.words]
        g = self.gustatory_drive(-1.0) if f.labeled else self.gustatory_drive(f.vader)
        if g is not None:
            drives.append(g)
        if f.mentioned:
            drives.append(self.mention_drive(f.question))
            c = self.courtship_drive(appetite, f.question)
            if c is not None:
                drives.append(c)
        return Stimulus(drives)


def vader_compound(text: str) -> float:
    """The only text scorer in the system.  Text is scored and discarded."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
