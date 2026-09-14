# Corpus audit: what not to write

Audit of the first "human thoughts" corpus (2026-09-13), which Nick
rejected as reading like an LLM's idea of a fruit fly.  Kept as a checklist
for any text that goes into `corpus/`.

## Tells found

1. **Reversal punchlines.** "I made coffee and forgot about it. It is cold
   now. I am drinking it anyway."  "None of them was the thing."  "It was a
   bad hat."  Every third line ended on a twist.
2. **Rule of three for rhythm.** "Soft rain. Soft light. Soft everything."
   "Big day. Big feelings. Big coffee."  "Now. Not later. Now."
3. **Aphoristic closers.** "Small good things are still good things."  "It
   costs nothing to be otherwise."  "That is not a small thing for me."
4. **Meta-cute sign-offs.** "That is the whole post."  "That is the whole
   report."  "Present. Accounted for. Mildly hungry."
5. **Stage directions on feeling.** "In a good way."  "I mean it."
   "Genuinely."  "and I do not care."
6. **Sitcom narrator.** "The rest of the room is watching me."  "Nothing
   personal. Everything personal."  "I am easy to please."
7. **Fake pidgin.** "I not know."  "I not going to be cool about it."
   Dropping articles the way the seed does is one thing; mangling verbs into
   a costume is another.
8. **A person's props.** Coffee, kettle, toast, socks, trains, fridge,
   heating, sofa, shelves, buses, lists, hats, paper post.  A fly has none
   of these.  A fly has light, warm, wind, glass, wall, fruit, sweet, sour,
   rot, wet, dark, hands, big things that move, other flies, up and down.

## Rules for corpus text

- A fly's world, a person's wants.  Sensations and objects from the fly's
  scale; wanting, liking, fearing, wondering in plain human words.
- Flat declaratives.  No punchline, no twist, no moral, no closer.
- Short.  Most sentences under eight words.  Two or three to a thought.
- Small vocabulary.  Say "big thing" before "person"; "sweet" before "sugar".
- Articles may drop the way the seed drops them.  Verbs stay whole.
- No names, brands, places, numbers, links, or claims about the world
  beyond what a fly could notice.
- Curious at the start.  "What is this" is a complete thought.

## What works (added 2026-09-14)

Nick brought a set of lines from claude.ai as the register to aim for
(`corpus/011-banana.txt`, kept verbatim; `012-grape.txt` is more of the
same).  What they have that the first draft did not:

- **The comedy is his, not the writer's.**  He misreads the world and
  reports it flat: "the big one has a banana in its hand and is putting the
  banana in its face."  The reader laughs; he does not know there was a
  joke.  A punchline is only a tell when the narrator is in on it.
- **He is in the middle of doing something.**  "i am coming."  "bye."  "now
  what grape."  Present tense, mid-errand, often addressed to no one.
- **He has props.**  Banana, grape, apple, vinegar, spoon, cup, sink, the
  peel, the leaf, the lady fly, the boy fly, the big one.  The objects are
  the ones a fruit fly would be on.  A person's props are still out.
- **Small acknowledgements.**  "ok."  "sorry."  "hello sun."  "rude."
  "i recommend leaf."  He closes a thought by letting it be.
- **No punctuation at the end, dropped apostrophes** ("dont") are fine; the
  generator lowercases and re-cuts sentences anyway.

Rules 1–8 still hold for the writer.  The difference is who the joke
belongs to.
