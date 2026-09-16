# Encoder v1 (frozen at `freeze-v1`)

The encoder turns an inbound event into sensory drive: sets of model neurons
driven at Poisson rates for the one-second window in which the post is
presented (`src/bosco/encoder.py`, `Encoder.encode`).  It reads the post
once and keeps features: never the text, never the image.  Everything
below is a fixed, public function with no trained weights (EXPERIMENT.md
§2c).  Rewritten 2026-09-15; the earlier version of this page predated
words, topics, places, the courtship drive, bristles, and the retina.

Config: `config/encoder_v1.yaml` and the files named per sense.  Drives
are built in this order and merged (a neuron driven twice takes the last
rate): account, place, other people, embed (site, retina), topics, words,
context words, taste, touch, courtship.

## Who: account -> odor

`blake2b(DID)` seeds a draw of 12 glomeruli from the 35 valence-neutral ones
(53 annotated minus 18 with documented innate valence; exclusions and
citations in the config).  All ORNs of those glomeruli at 120 Hz.  Two
accounts share about 4 glomeruli; the code is 2–5% of Kenyon cells and
odor-specific.  The account's own smell, probed alone from rest, is what
`bosco memory` and `bosco people` read (README 39).

## Other people (2026-09-15)

A mention facet's DID, or a quoted post's author, is another person in the
room: that account's odor at half rate, up to three, never the author or
himself (`embeds.others_rate_scale`, `others` column).

## Where: places

Each feed he reads from is two neutral glomeruli chosen by its name
(`config/feeds_v1.yaml`), driven at 60 Hz with every post read there.  A
link card's site is a place the same way, by domain (`site:<domain>` in
the `embed` column; 2026-09-15).  Posts a walk brought are from the place
`walk`.

## What: topics and words

Topics: a published keyword map (`config/topics_v1.yaml`, sixty topics);
whole-word matches on the text (and, since 2026-09-15, on alt text, card
text and quoted text) name up to three, each three neutral glomeruli at
100 Hz.  Words: every content word of his closed vocabulary in the post,
up to six, each three neutral glomeruli and a rate in 90–130 Hz chosen by
`blake2b("word|" + word)` (`config/words_v1.yaml`); words that name a smell
a fly is born to answer take their real glomeruli (`config/innate_v1.yaml`);
colour words and the words for pictures take the retina's channels
(`config/retina_v1.yaml` `visual_words`).  Since 2026-09-15 a word outside
his vocabulary is a smell too: kept as `h:<16 hex>` of the same hash, up to
four per post, the word's own smell and never sayable.  The words of the
posts above (the thread, oldest first) and of alt text, cards and quoted
posts are context: smelled at half rate.

## Taste: sentiment

VADER compound `c` of the post's own text.  `|c| <= 0.05`: no taste.
Otherwise 150 Hz · min(1, |c| / 0.6) on the 83 sugar/water GRNs (c > 0) or
the 57 bitter GRNs (c < 0).  A labeled post is bitter at full rate.  Since
2026-09-15 the taste of a window also pairs its mixture with reward or
punishment at a small strength (README 43).

## Touch and the courtship command: being addressed

A mention, reply or quote drives Johnston's organ groups A and B at 50 Hz
(a question is not louder here; its weight moved to the courtship drive)
and the pC1 courtship command neurons at 12 Hz × appetite (× 1.5 for a
question), the input we chose for the song descending neurons that are the
`reply` population (README 26, 27).

## Sight: the retina (2026-09-15)

A post's thumbnails are reduced to a few channel names by a fixed
transform (`src/bosco/retina.py`, `config/retina_v1.yaml`): two dominant
hues of eight, brightness, edge strength and saturation as bins, `img` for
any picture, `motion` for video.  Each channel drives fifteen visual Kenyon
cells (KCγd, KCα/βp) at 6 Hz; those cells stand in for the pruned visual
projection neurons and habituate like afferents (README 47).

## Background, not the encoder

The clock neurons get a 24 h drive (`docs/circadian.md`); bristle debris
lands as seeded events (`spontaneous`, README 25).  Both are the base
every window sits on.

## Not encoded

Follower counts, time of day as a feature (the clock has its own drive),
the familiarity count (a phrasebook key, from the ledger), and anything a
classifier would give: what a picture is, what a sentence means.
