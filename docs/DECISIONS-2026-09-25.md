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

## Open (Nick to decide)

- **Where the question enters,** given the eyes decision. It is decided from the
  anatomy check in the audit: which senses reach his Kenyon cells and descending
  neurons, and how strongly.
- **Plasticity beyond KC→MBON** (for example PN→KC, which changes with experience in
  flies). It needs an amendment to CLAUDE.md's training rule, proposed with evidence.

## Next

An audit, then the architecture doc for v1 "The Model", then gates. The audit checks
every stage (senses, encoding, wiring, dynamics, learning, read, training data,
evaluation, serving) against the real fly, our docs and our claims.
