"""How much of what he says is a quotation (decided 2026-09-18).

He has a small corpus and a trigram over it, so he will always sound like his own pages.
The question this gate asks is narrower: does he *hand a page back whole*?  It drives the
generator the way `Agent._log` does, over his own logged windows (their words, their
topics, his register, his seed), and measures three things over the utterances that come
out:

  1. how many are exactly a line of his corpus or of the phrasebook;
  2. the longest run of tokens an utterance shares, unbroken, with one line of his;
  3. how many utterances carry a run of eight or more.

Nothing here is a threshold he is tuned against: the numbers are printed, and the note in
EXPERIMENT.md 2 quotes them at the boundary where the policy changed.  The ledger is read
only for its features -- words, context, topics, register, seed -- never for text, which is
not stored.

  uv run python scripts/quotation_gate.py snapshots/dev-2026-09-18/ledger.sqlite [N]
"""

from __future__ import annotations

import sqlite3
import statistics
import sys

from bosco.phrasebook import Phrasebook, familiarity_bin
from bosco.textgen import Generator, detokenize, tokenize

LONG_RUN = 8  # a run of eight tokens is a sentence of his, not a phrase


def his_lines(g: Generator, ph: Phrasebook) -> tuple[list[list[str]], set[str]]:
    lines = [[t.lower() for s in line for t in s] for d in g.docs for line in d.lines]
    whole = {detokenize(toks) for toks in lines} | {ln.text.strip().lower() for ln in ph.lines}
    return lines, whole


def longest_run(toks: list[str], lines: list[list[str]]) -> int:
    """The longest run of tokens he shares, unbroken, with one line of his."""
    best = 0
    for line in lines:
        prev = [0] * (len(line) + 1)
        for a in toks:
            cur = [0] * (len(line) + 1)
            for j, b in enumerate(line, 1):
                if a == b:
                    cur[j] = prev[j - 1] + 1
                    best = max(best, cur[j])
            prev = cur
    return best


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    limit = int(argv[2]) if len(argv) > 2 else 300
    ph = Phrasebook()
    g = Generator(phrasebook_lines=[ln.text for ln in ph.lines])
    lines, whole = his_lines(g, ph)

    db = sqlite3.connect(argv[1])
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT * FROM episodes WHERE words IS NOT NULL AND words != '' AND behaviour IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    runs: list[int] = []
    fracs: list[float] = []
    verbatim = 0
    samples: list[str] = []
    for r in rows:
        words = tuple(w for w in (r["words"] or "").split(",") if w and not w.startswith("h:"))
        context = tuple(w for w in (r["context"] or "").split(",") if w and not w.startswith("h:"))
        air = {w: 1.0 for w in words} | {w: 0.6 for w in context if w not in words}
        if not air:
            continue
        topics = tuple(t for t in (r["topics"] or "").split(",") if t)
        speak_as = "reply" if r["mentioned"] else "groom"
        fb = familiarity_bin(r["familiarity"] or 0)
        seed = r["seed"] or r["id"]
        opening = g.pick_sentence(speak_as, r["valence"], r["arousal"], seed, air=air, topics=topics, familiarity=fb)
        text = g.generate(
            speak_as,
            r["valence"],
            r["arousal"],
            seed,
            max_sentences=2,
            topics=topics,
            familiarity=fb,
            air=air,
            prime=bool(r["mentioned"]),
            opening=opening,
        )
        if not text:
            continue
        toks = [t.lower() for t in tokenize(text)]
        run = longest_run(toks, lines)
        runs.append(run)
        fracs.append(run / len(toks))
        if text.strip().lower() in whole:
            verbatim += 1
        if len(samples) < 10:
            samples.append(f"  [run {run} of {len(toks)}] {text}")

    n = len(runs)
    if not n:
        print("no windows with words in that ledger")
        return 1
    print(f"windows: {n}")
    print(f"exactly a line of his: {verbatim} ({verbatim / n:.0%})")
    print(
        f"longest quoted run: mean {statistics.mean(runs):.1f} tokens, median {statistics.median(runs):.0f}, max {max(runs)}"
    )
    long_ones = sum(1 for r in runs if r >= LONG_RUN)
    print(f"utterances quoting {LONG_RUN}+ tokens: {long_ones} ({long_ones / n:.0%})")
    print(f"quoted share of an utterance: mean {statistics.mean(fracs):.2f}")
    print("\n".join(["", "a few of them:", *samples]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
