# corpus

Training text for Bosco's generator (`src/bosco/textgen.py`).  Every file
here is published and frozen at `freeze-v1`; the generator's digest is part
of the frozen-artifact table.

Rules:
- Plain UTF-8 text.  One document per file.  Paragraphs separated by blank lines.
- Optional tags on the first line: `#tags: behaviour=groom valence=positive arousal=low`.
  Tagged documents are preferred when the fly's state matches; untagged
  documents are always in the pool.
- **Never** other people's Bluesky posts, quotes of them, or anything scraped
  from the network.  The corpus is authored.
- Phrasebook lines (`phrasebook.yaml`) are added to the pool automatically.

Provenance: `000-seed.txt` was drafted by Nick and workshopped with Claude
on 2026-09-13.  The minimal "human thoughts" set (`010`–`080`) was written
by Claude at Nick's request on 2026-09-13 as a starting register: short,
first-person, everyday, no names, no brands, no claims about the world.
Files are tagged so the fly's state (valence, arousal, behaviour) chooses
the register.  Replace or extend freely; the digest changes with the files.
