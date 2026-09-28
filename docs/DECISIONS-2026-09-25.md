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
26. **Mood uses all of DynaSent round 2** (Nick, 2026-09-26), including the crowd-written sentences
    prompted by Yelp sentences. The Yelp prompts themselves are never read, and round 1 (Yelp text)
    was deleted. The model card says so.
27. **Many-option specialists use one learned memory per option** (way B: approach or avoid on the
    item's smell alone). CLINC reached 0.811 on validation at epoch 2, against 0.635 for
    prototype-mixture sniffs. Few-option specialists keep the T-maze (topic 0.933).
28. **Share-alike data and the weights** (Nick, 2026-09-27). All shipped specialists are served from
    Nick's server; answers and traces are not copies of the training text. For self-hosting:
    - specialists trained only on CC BY / CC0 data (junk, support, hate) ship under our own licence,
      with their NOTICE credits;
    - specialists trained on CC BY-SA text (topic from DBpedia, politeness from Stack Exchange; kind,
      food and danger from Wikipedia) ship **CC BY-SA 4.0**, with an attribution manifest (the
      sources, and for Wikipedia each row's revision URL). Whether weights are an adaptation is
      legally unsettled; this is the cautious reading;
    - training text is never shipped with any weights;
    - a lawyer's view on "weights as adaptation" before distributing weights, if Ask Bosco earns
      real money. Serving through the API does not wait on it.
29. **Specialists are generalists in their fields** (Nick, 2026-09-27: "the specialists need to be
    generalists in their fields"). The next fields are **intent, harm, topic, language, tone, danger and
    finance**.
    - Each field gets a taxonomy written before training. Several clean sources are pooled onto it, and
      each source's mapping is pre-registered.
    - Each gate adds a **breadth test**: a held-out source the specialist never trained on, under the same
      bar. A specialist that passes only on its own sources' test sets does not ship as a generalist.
    - Today's 8 were each trained on one source. Their cards say so, and they are candidates to be
      rebuilt into their fields (topic, harm, tone, danger).
30. **Civil Comments is accepted for training** (Nick, 2026-09-27: "accept it and note it"). The licence is
    CC0 (Google's TFDS catalog and HF card; no Jigsaw-hosted page could be read). Two ethics gaps stay open
    and go on every card trained on it: the commenters' consent rests on the Civil Comments platform terms,
    and the raters' platform and pay are undocumented. It unblocks `sexual` in harm and the hostile end of
    tone (docs/research-gate4/harm-tone.md).
31. **Stack Exchange data is accepted** (Nick, 2026-09-27: "it's explicitly not an LLM so I think it's
    fine"). The text is CC BY-SA 4.0. Use the 2024-04 archive.org dump, and only posts from before 2022-11. The
    newer download terms ("not training a large language model") are noted on every card trained on it;
    Bosco is not a language model and generates no text. Usernames go only into attribution manifests.
32. **Language covers the 30–40 languages the current encoder can see** (Nick: "30-40 is plenty").
    nomic-embed-text-v1.5's uncased BERT tokenizer loses most Chinese, Thai and South Asian, Southeast
    Asian and Ethiopic scripts (docs/research-gate4/topic-language-danger.md). Those languages are out of
    scope and are named on the card. A multilingual encoder is revisited with our own encoders.
33. **Text from live sources must predate 2022-11-01** (Nick, 2026-09-27), before ChatGPT's launch
    (2022-11-30). This keeps training and test text human-written, as the no-LLM-data rule requires, where
    rows cannot be checked one by one: Wikipedia, Wikinews, Stack Exchange and CFPB. Fixed crowd-written
    datasets that predate it need no cutoff. Cards say "text up to 2022". It may be relaxed per source only
    for supervised human writing under a no-AI-tools rule, written down before use.
34. **The consensus is the human's** (Nick, 2026-09-27: "it's the human's judgment out of all of the
    questions they asked Bosco and the answers he gave back. You're asking your group of experts"). Bosco
    does not vote or combine; decision 24 stands. Each specialist answers its own question with an honest
    probability, and the person weighs the answers (Nick: "you're asking it a series of questions for
    your task. It gives you more data to make a decision with"). Consequences:
    - Calibration is a first-class requirement. A wrong answer given with confidence misleads the person's
      judgment; an unsure one does not.
    - Specialists that are only meaningful beside others (tone first) may ship as **components**. They
      must be calibrated on the held-back source (ECE ≤ 0.10) and better than chance where they are
      confident. Their cards say they are a second opinion, never a verdict. The exact component bar is
      pre-registered with Gate 4.
35. **Harm is the priority** (Nick, 2026-09-27: "flagging potentially harmful content is a huge value
    add… we need to crack this"). Harm means **content moderation**: is this text itself harmful. It does
    not mean screening prompts sent to an AI, which would be a separate specialist later. It is built as
    separate yes/no experts (hate, threat, harassment, sexual; tone as a component), tested across
    platforms with one source held back at a time. HateCheck stays unscored until it is used as a gate test.
36. **Sharp yes/no questions over broad menus** (Nick, 2026-09-27; runs/gate4-ceilings/reframe.json).
    Across intent, finance, harm and tone, yes/no framings carry to a held-back source and broad
    multiple-choice ones do not (through his nose, held-out dev: p(social) 0.885, p(problem) 0.839, p(harm)
    0.798, p(credit and debt) 0.782, against 0.40–0.61 for the full menus). Gate 4 candidates:
    - **candidates:** p(harm) (the headline harm question), p(problem), p(social), p(credit and debt);
    - **follow-up experts** (asked when p(harm) is high): threat, sexual, hate, harassment;
    - **not ready:** p(cancel or change) and p(fraud or scam);
    - **tone is out of Gate 4.** No framing transfers from Wikipedia to Stack Exchange. It can return
      only as a calibrated component (decision 34);
    - **broad menus** (28 intents, 15 finance areas) stay single-source specialists and say so on their
      cards.
    Topic, language and danger get the same check when their data is built.
37. **Parity with Jev is the yardstick, not Doom** (Nick, 2026-09-27: "No planning for doom specifically I
    just want to make sure we have decent parity with what jev is capable of"). There is no game work and
    no game-specific specialists. General fields stay candidates for the fleet: **valence** (good or bad
    for me, approach or avoid; the fly's native question), **urgency** (does this need action now?) and
    **direction** (left, right or ahead, from a description). The gaps against Jev's contract:
    - **to build (service):** the score type (2–10 levels) and several questions per request;
    - **speed:** many-option choices (7.9 s at 151 options) need all options in one batched pass, or
      choice specialists kept to about 20 options;
    - **coverage:** a fleet of generalist specialists wide enough that most questions map onto one;
    - **structural, stated plainly:** Jev answers questions it has never seen; Bosco answers only what a
      specialist was taught (422 not_taught).
    Where Bosco can pass Jev: images (the eyes) and the playback.

38. **Use every sense that has something honest to carry** (Nick, 2026-09-28: "we should be using all senses at
    least reasonably to help our experts make the best decisions"). The brain cut has olfactory receptors
    (2,639), gustatory neurons (355), visual projection neurons (9,201), Johnston's organ (672) and other
    mechano-, thermo- and hygrosensory neurons. **Each sense carries a different kind of information; meaning is
    never duplicated into extra senses:**
    - **smell = meaning** (the pinned encoder; in use);
    - **taste = form** (letters, word shapes, length, punctuation, capitals), a fixed counting rule rather than a
      learned model. For language, plain language, and possibly junk and tone;
    - **sight = pictures** (option c eyes; optic lobes later);
    - **hearing** only if audio input ever comes. Touch, temperature and humidity stay idle.
    A sense a specialist was not trained with gets no input, so shipped specialists are unchanged. First: an
    anatomy check (do gustatory neurons reach the KCs, DANs or DN read within 80 steps in our cut?), then a taste
    spike after gate 4, tested on language and plain.

39. **No more neurons: use the ones we have properly** (Nick, 2026-09-28: "We can't make it go any slower. I don't
    think we need to add neurons just use them correctly"). This amends decision 2: the optic lobes and the VNC stay
    out, because speed rules. "His brain" means the 50,140-neuron central brain (with the visual projection
    neurons), and the cards say so. Decision 3 (question and item through different senses) was never built and
    stands. Family changes, in order, each tested on dev against the plain nose ceiling before adoption, at no cost
    to neurons or speed:
    1. **Timed sniffs:** the smell changes over the 80 steps, carrying different parts of the encoder's meaning in
       different time windows, as flies sample odour over time. The same 46 channels carry more.
    2. **Taste for form** (decision 38), on the 355 gustatory neurons already in the model.
    3. **Question and item apart** (decision 3): the option word through its own channel, not blended into the
       item's smell.
    4. **Wake idle sensory neurons** only where they carry real input.
    Each change is a new family version; specialists retrain onto it, and the shipped ones stay until then.

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
