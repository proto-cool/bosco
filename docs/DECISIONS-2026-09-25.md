# Decisions, 2026-09-25 (Nick), after A4 broad and the L3 check

Context: A4 broad failed G1. The pair-encoder "breakthrough" turned out to be the encoder
deciding. Nick: "we need a jev-style generalist model"; "the entire novelty is being
able to encode human problems in a way a fly understands"; "it still has to be FAST
and ACCURATE, [...] it needs to be usable". Halteres (the GUI for asking him questions
and watching his neurons) is built and "cannot go to waste". "We need a real model to
power it, and it needs the fly to make honest decisions."

## Locked

1. **Gradient training is fine.** "Why not use modern computer methods on a computer
   fly." It is stated plainly on the model card: he learns by gradient descent, not by
   dopamine. His decisions run on his wiring.
2. **Bosco v1 is The Model.** No feature is deferred to an arbitrary later milestone
   ("later for Doom" is withdrawn). That includes the optic lobes; how they are fed is
   open (below).
3. **Text uses more than the nose.** The question and the thing being judged come in
   apart, through different senses.
4. **The answer is read from his descending neurons,** the ones that drive going toward
   or away. The MBON groups become one stage the answer passes through, shown in the
   trace, not the read.
5. **The rate model stays.** Speed is preferred over accuracy, but accuracy may not be
   sacrificed significantly; measured, not assumed.
6. **The A5 start rule and the mashed question are fixed in the rebuild.**
7. **Serving and evaluation run on the CPU; GPUs are for training only.** Published
   numbers are the served numbers, computed the same way and exactly repeatable. Speed
   on the Kimsufi (4 cores, no GPU) is a gate. "I can't afford a cloud GPU for
   Halteres."
8. **Faithful to the fly, but extended.** Every extension points at something real flies
   have (a sense, a pathway, a known site of plasticity) and is written down before it
   is used.
9. **The dropped pair encoder:** a model that sees the question and the answer together
   and was trained to judge them is a decider, and is not allowed as a sense.
10. **Data is ethically sourced, with provenance all the way through:** a ledger per
    dataset and per encoder before training.

11. **Eyes: option (c), tried first.** Real eyes (the picture at about 800 facets per
    eye, then photoreceptors, optic lobes and visual projection neurons, all his own)
    **plus** the encoder feeding in at the optic-lobe handoff, as a translator. Nick:
    "encoders for translation is the perfect analogy". Cost: the optic lobes are
    roughly 2–3× the neurons now simulated, the largest speed risk, measured in the
    audit. The stand-in alone (b) is acceptable as a fallback ("the eye stand-in is
    fine").
12. **Our own encoders are allowed** ("if we have to make our own encoders, that's cool
    too"). This is the route to provenance all the way down: trained only on clean sources.
    Trade-off: at first a weaker translation than e5. Decided per sense after the
    provenance audit.
13. **Self-hosting is allowed** (Nick). It is redistribution, so each package ships:
    - the MaleCNS attribution, licence link (CC BY 4.0) and a list of changes;
    - the Shiu MIT notice;
    - only encoders and weights whose licences allow commercial redistribution.

    That means jina-clip-v2 (CC BY-NC) is replaced or licensed, and every shipped
    specialist is trained on ledger-clean data only.
14. **Encoders: auditable first, clean second** (Nick).
    - **Now:** nomic-embed-text-v1.5 and nomic-embed-vision-v1.5 (Apache 2.0, with the
      training data published) replace e5-large-v2 and jina-clip-v2 for the rebuild and
      the specialist pilot. Each is pinned by commit hash.
    - **In parallel:** our own text encoder (trained on Common Pile, with no MS MARCO and
      no scraped social media) and our own image translator (self-supervised, trained on
      commercially licensed images, with no generated captions), both on the 3080.
    - **Before launch:** a gate compares ours against nomic; ship ours if it is close
      enough, otherwise Nick chooses knowingly.
15. **The fleet: 15–20 specialists at launch,** growing toward about 100. Sources: the clean data
    already fetched (about 8–10), Wikipedia/Wikidata topic specialists (about 6–10), and
    commissioned labelling for must-haves such as sentiment (budget open).
16. **Teach your own is self-hosting only** (Nick). A user's specialist is taught, stored and run
    in their own self-hosted copy; nothing about it touches Nick's server or Halteres. The proposed
    mechanism is the fly's own local rule on KC→MBON synapses (forward passes only, cheap on a CPU),
    measured in the specialist pilot against gradient training. The public specialists never learn
    from anyone.
17. **Nick's server runs only our specialists,** for Halteres.
18. **Two ways to use Bosco** (Nick: "they self host a Bosco and use the APIs locally. Halteres is a
    SaaS"):
    - **Halteres, the SaaS:** the public app, running our specialists on Nick's server. The hosted
      Bosco API stays private to that server; Halteres is its only client (as decided 2026-09-24).
    - **A self-hosted Bosco:** anyone can run their own copy and call its API locally, with our
      specialists and any they teach themselves (decision 16).
19. **No community specialists.** Nick does not want to moderate a public library.
20. ~~Browser feasibility test~~ withdrawn: teaching is self-hosted only, so nothing has to run in
    a browser.
21. **The personality is the playback** (Nick: "that's cool as shit"). The trace names real cell
    types as they fire: the glomeruli, the sparse KCs, the DANs, the lateral horn, and the
    answer's descending neurons (for example MDN, "the moonwalker", backing away; the gnathal DNs
    leaning in). Accuracy ties with a shuffle are a model-card footnote, not the story.
22. **No more shuffled twins per specialist** after the specialist pilot. The fair real-vs-shuffle
    answer is taken once for the family, and again only when the brain changes (for example the
    optic lobes). New specialists are checked against the production bar only.
23. **The production bar is 0.80** balanced accuracy on the sealed test (CPU) with ECE ≤ 0.10. For
    a task flagged, before its test is read, as having disputed labels, the bar is 0.9 × its
    information ceiling, stated on its card (specialist pilot, amendments 1–2).
24. **The architecture, in one line** (Nick): the encoders bridge the gap (they translate human things
    into smells and sights), the specialists steer the problem (each owns one clear question), and
    they work together on more complex problems. The **caller** combines specialists with explicit
    rules (fan-out, composite scores, routing by confidence). Nothing automatic picks or overrides
    them, so every decision stays a fly's and every rule stays visible.
25. **The positioning: lean into the novelty and the playback** (Nick). The pitch is "the AI you
    can watch think, neuron by neuron", not "explainable AI" (it shows which real cells fired,
    not a reason in words). The 0.80 bar stays, so he is right as well as watchable.
    Priorities: the service and named-cell traces first; education and exhibits as the first
    paying market to explore; specialists passing the bar, clear-cut ones first.

## Open (Nick to decide)

- **A budget for commissioned labelling** (sentiment first): paid, consented annotation of
  openly licensed text.

- **Where the question enters,** given the eyes decision. It is decided from the
  anatomy check in the audit: which senses reach his Kenyon cells and descending
  neurons, and how strongly.
- **Plasticity beyond KC→MBON** (for example PN→KC, which changes with experience in
  flies). It needs an amendment to CLAUDE.md's training rule, proposed with evidence.

## Next

An audit, then the architecture doc for v1 "The Model", then gates. The audit checks
every stage (senses, encoding, wiring, dynamics, learning, read, training data,
evaluation, serving) against the real fly, our docs and our claims.
