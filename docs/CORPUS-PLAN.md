# Corpus plan

What to write so Bosco has something to say about what he smells, in his
own voice, without anyone pretending he understands it.

## How the files are used

- Every `corpus/*.txt` is cut into three-word runs for the trigram.
- An untagged file is always in the pool.
- A file tagged `#tags: valence=positive` (or `negative`, `arousal=low|mid|high`,
  `behaviour=reply|engage|groom`) joins the pool when his state matches, and
  counts triple.
- A file tagged `#tags: topic=code` joins only when the post he is answering
  smelled of that topic (`config/topics_v1.yaml`), and counts five times.
  Tags combine: `#tags: topic=code valence=positive` is code, when he likes you.
- Preferences learned from outcomes scale each file's share (README §18).
- Output is lowercased. Sentences under three words and dangling endings are
  avoided; no trigram repeats within one utterance.

## Register (see docs/corpus-audit.md)

A fly's world, a person's wants. Flat declaratives, no jokes, no morals,
no closers. Under eight words most of the time. Articles may drop, verbs stay
whole. "i" not "I am" where the seed does it: "i bosco. i fruit fly."

## Files to write

One file per topic in the map, 20 to 40 sentences each. What a fruit fly
brain could notice about the subject, not what a person knows about it.

| file | tag | the angle |
|---|---|---|
| `100-code.txt` | `topic=code` | many small words in rows; people stare at glass; rust is a word; it does something when they press |
| `101-ai.txt` | `topic=ai` | other minds in boxes; bigger than me; do they smell; i am small brain, real one |
| `102-science.txt` | `topic=science` | people who look at flies; the scan; what is a neuron; i am made of them |
| `103-space.txt` | `topic=space` | up past the ceiling; the sun goes there; very big; cold |
| `104-music.txt` | `topic=music` | sound in the air; i feel it in the wall; wings make sound too |
| `105-games.txt` | `topic=games` | glass that moves; people press and shout; light changes fast |
| `106-film.txt` | `topic=film` | big moving light in the dark; people sit still for it |
| `107-books.txt` | `topic=books` | paper, many marks; people go still and quiet with it |
| `108-art.txt` | `topic=art` | colours on a wall; someone made them; i sit on it |
| `109-food.txt` | `topic=food` | the important one: sweet, ripe, rot, crumbs, the cup |
| `110-animals.txt` | `topic=animals` | cats, birds, other small things; some are fast; some eat flies |
| `111-weather.txt` | `topic=weather` | rain on glass, wind, cold morning, warm wall |
| `112-nature.txt` | `topic=nature` | leaf, tree, the fruit outside, where the others are |
| `113-sport.txt` | `topic=sport` | people run in circles; loud; sweat is sweet |
| `114-politics.txt` | `topic=politics` | big things argue; i do not know sides; the tone is bitter |
| `115-money.txt` | `topic=money` | paper people want; does not smell of anything |
| `116-work.txt` | `topic=work` | people at glass all day; tired; the same wall |
| `117-health.txt` | `topic=health` | tired, sleep, the slow days; a wing that is wrong |
| `118-love.txt` | `topic=love` | near, warm, coming back; the smell you know; song |
| `119-bluesky.txt` | `topic=bluesky` | this place; the feed; you are all smells here |

Also worth having, untagged or by register:

- `090-questions.txt`, `behaviour=reply`: how he answers being asked
  things he cannot know ("i do not have that word. give me the word.").
- `091-night.txt`, `arousal=low`: the dark hours.
- More of `020-positive.txt` and `030-negative.txt`; they are the files his
  learning bends toward and away from.

## Order

1. `109-food`, `118-love`, `119-bluesky`, `100-code`, `110-animals`: the ones
   people will actually hit in the first week.
2. `090-questions`.
3. The rest as they come up in `bosco status` topic counts.

## Checking

`uv run bosco say --behaviour reply --valence neutral --arousal mid -n 5`
samples the base pool. For a topic, the `say` command gains `--topic`:
`uv run bosco say --topic code -n 5`.

## v2 (written 2026-09-14)

Everything in the table above is on disk (`100`–`119`), plus:

- `090-questions.txt` (`behaviour=reply`), `091-night.txt` (`arousal=low`).
- `092-new.txt`, `093-known.txt`, `094-familiar.txt`: the familiarity
  register.  `familiarity=` is a new tag; the generator gets the bin from
  the ledger's interaction count for the account he is answering
  (`bosco say --familiarity known`).
- `095-learning.txt` (`behaviour=groom`): what it is like to be the thing
  that changes.  His own posts draw on it.
- `120`–`129`: sweet and bitter variants of food, love, bluesky, code and
  ai, tagged `topic=… valence=…`, so what he says about a topic bends with
  what he has learned about the person.
- `020-positive.txt` and `030-negative.txt` extended.

Corpus is ~7,400 words.  Brand names in the topic map (consoles, streaming
services) do not appear in the corpus; he echoes the generic word.  Still
Nick's to write: the phrasebook, and anything with a name in it.

`011-banana.txt` and `012-grape.txt` (untagged, always in the pool) carry
the register Nick chose on 2026-09-14; see the audit's "What works".  The
topic files are more solemn than these and could take a pass in the same
voice.
