# corpus

Training text for Bosco's generator (`src/bosco/textgen.py`).  Every file
here is published and frozen at `freeze-v1`; the generator's digest is part
of the frozen-artifact table.

Rules:
- Plain UTF-8 text.  One document per file.  Paragraphs separated by blank lines.
  Every line is one utterance and the generator treats a line end as a sentence
  end, so a line needs no final period.  Lowercase throughout, including `i`.
- Optional tags on the first line: `#tags: behaviour=groom valence=positive arousal=low`.
  Tagged documents are preferred when the fly's state matches; untagged
  documents are always in the pool.  `topic=code` (a name from
  `config/topics_v1.yaml`) joins only when the post he answers smelled of
  that topic.  See `docs/CORPUS-PLAN.md`.
- **Never** other people's Bluesky posts, quotes of them, or anything scraped
  from the network.  The corpus is authored.
- Phrasebook lines (`phrasebook.yaml`) are added to the pool automatically.

Provenance: `000-seed.txt` was drafted by Nick and workshopped with Claude
on 2026-09-13.  The minimal "human thoughts" set (`010`–`080`) was written
by Claude at Nick's request on 2026-09-13 as a starting register: short,
first-person, everyday, no names, no brands, no claims about the world.
Files are tagged so the fly's state (valence, arousal, behaviour) chooses
the register.  Replace or extend freely; the digest changes with the files.

v2 (2026-09-14): topic files `100`–`119`, questions, night, the
familiarity registers, `095-learning`, and the sweet/bitter topic variants
`120`–`129` were written by Claude against `docs/CORPUS-PLAN.md` and
`docs/corpus-audit.md` at Nick's request.

v3 (2026-09-14): `130`–`169`, one file per new topic, written by Claude in
the register of `011-banana.txt`.

State tags (2026-09-14): `time=night|morning|day|evening` and
`appetite=hungry|sated` select files by his clock and his appetite for
contact, so an own post is about the hour and the itch he is answering,
not a thought. `095-learning.txt` was removed for reading as one.

v5 (2026-09-14): a second pass broke the grammar toward `011`/`012` ("he go",
"big one press", "i want know") without lengthening any line, after a first
pass that had padded short lines was reverted; then everything was lowercased
by script at Nick's instruction, including the original `010`–`080` register.

v4 (2026-09-14): `mood=warm|stung|alone` (`170`–`172`), written by Claude
in the register of `011-banana.txt` at Nick's request: what the ledger says
lately happened to him chooses the file. A reward within two hours is warm,
a punishment within six is stung, a day without anyone coming to him is
alone. Fifteen lines each; no verdicts, no lessons, things that happened
and where he is now.
