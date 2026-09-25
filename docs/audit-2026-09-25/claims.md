# Audit of claims against results (2026-09-25)

Scope: the claims in `docs/BRIEF.md`, `docs/API.md`, `docs/WEBAPP.md`,
`docs/HANDOFF-HALTERES.md`, `docs/EFFORT-PILOT.md`, `docs/PLAN-A4.md`, `README.md` and
`CLAUDE.md`, checked against the results docs, `ERRATA-2026-09-24.md`,
`AUDIT-2026-09-24.md` and the two decision records. No existing doc or code was changed.
Line numbers are as of commit `78a3f64`.

Ranks: **critical** (would mislead users, the Halteres team or Nick about what he can
do), **important** (stale or overstated in a way that affects design or honesty copy),
**minor** (wording, cross-references).

## Evidence in one place

| what | number | source |
|---|---|---|
| Unseen kinds, A4 pilot | real 0.503, layered 0.516, plain net 0.510, **nose alone 0.631**, chance 0.340; G1 FAIL | `a4-pilot-results.md:32,38` |
| Taught kinds, A4 broad (44 kinds) | real 0.419, layered 0.418, plain net 0.426, **nose alone 0.469 (1,024) / 0.453 (bi46)**, chance 0.322; G1 FAIL | `a4-broad-results.md:57,93` |
| New data, taught labels (tier 1) | real 0.828 vs nose 0.820: a tie ("not a gain") | `a4-broad-results.md:85,94,122-125` |
| New label words / new concepts (tiers 2, 3) | real 0.396 / 0.267 vs nose 0.571 / 0.417 | `a4-broad-results.md:86-87` |
| "By the rule, the broad catalogue is not the public v1" | | `a4-broad-results.md:129-131` |
| A5 (the "internal v0") | real 0.863, lowest of all arms (layered 0.887, hash 0.876, free 0.880); D1 FAIL, D2 FAIL; **not adopted** | `gate-a5-results.md:7-11,25-29` |
| A5 calibration | ECE sweet 0.025, dangerous 0.028, junk 0.022, **pictures 0.105** | `gate-a5-results.md:18-21,95` |
| A5 behaviour neurons vs MBON answer | agree 40% | `gate-a5-results.md:7,96-98` |
| A5 start rule | gain 8 keeps label similarity at only 0.53 (ORN) / 0.37 (KC); "this also bears on A5" | `a4b-dev-results.md:51-56` |
| A2 | real 0.841, lowest (shuffle 0.882, hash 0.847, free 0.899); D2 FAIL | `gate-a2-results.md:9-12,28` |
| A1 | real 0.687 vs shuffle 0.710, hash 0.710, free 0.707; FAIL | `gate-a1-results.md:24-30` |
| C1 (MBON read, lifetime rule) | 0.555, FAIL | `gate-c1-results.md:22` |
| Pair encoder was the decider | encoder alone 0.504 > its smell through the fly 0.468 | `a4b-dev-results.md:84-94`; `DECISIONS-2026-09-25.md:3-5,33-34` |
| Passage questions | best nose 0.687 vs Jev 0.94 | `passage-check-results.md:9,23-25` |
| GPU replay wobble | same start, drift by batch 150; "differences of about 0.01 between arms are inside what a replay moves" | `a4-broad-results.md:96-101` |
| MPS nondeterminism | plain net, same seed: val 0.462 / 0.444 / 0.443, test 0.445 / 0.426 / 0.427 | `A4-BROAD.md` amendment 3 |
| Seed-to-seed disagreement | 4.7–6.7% of A5 test items on different sides; cut vs full 10.6–20.3%, full vs itself 18.8% | `gate-a5-results.md:31-37`; `DECISIONS-2026-09-24.md:44-50` |
| Speed | cut brain 2.6× faster end to end, ~6× per **training batch** (GPU); pair encoder ~85 pairs/s on the Mac GPU; "Server CPU not measured" | `a5-cutoff-pilot2-results.md:25-26`; `a4b-dev-results.md:99-100` |
| A2b | results table empty (run killed) | `gate-a2b-results.md`; `AUDIT-2026-09-24.md:4-5` |

Decisions of 2026-09-25 that bear on the claims (`DECISIONS-2026-09-25.md`): (1) gradient
training, stated plainly on the model card; (2) no "later for Doom", optic lobes are in
v1; (3) question and thing enter apart through different senses; (4) **the answer is read
from descending neurons**, MBONs are a stage in the trace; (5) accuracy may not be
sacrificed significantly; (6) A5 start rule fixed in the rebuild; (7) **serving and
evaluation on CPU; published numbers are the served numbers, exactly repeatable; Kimsufi
speed is a gate**; (9) pair encoder not allowed as a sense; (10) data ledger; (11) eyes:
real optic lobes plus the encoder at the optic-lobe handoff, 2–3× the neurons.

---

## Critical

### C1. The Halteres handoff describes A4 broad as still running and the catalogue as the likely v1
- **HANDOFF-HALTERES.md:45-53** — "The **A4 broad** gate … is training now, with 44 taught
  kinds. Its result decides what the public v1 can offer: If it passes, the public site
  offers a catalogue of named questions…"
- **HANDOFF-HALTERES.md:80-82** — "After A4, expect the public site to lean on the taught
  catalogue, as above."
- **Evidence:** A4 broad has reported: G1 FAIL (real 0.419 vs bar 0.472, below the
  untrained nose 0.469), and "the broad catalogue is not the public v1"
  (`a4-broad-results.md:93,129`). The 09-25 decisions replace the plan with an audit, an
  architecture doc for "The Model" and new gates (`DECISIONS-2026-09-25.md:56-58`).
- **Honest wording:** "A4 broad failed (`docs/a4-broad-results.md`): taught 44 kinds, he
  scored 0.419 macro balanced accuracy, below his untrained nose (0.469). The catalogue is
  not the public v1. What the public v1 offers is undecided until the rebuild ('The
  Model', `DECISIONS-2026-09-25.md`) is gated. Build the UI for a small, fixed set of named
  questions, and treat the set as data, not layout."

### C2. "Internal v0 … well calibrated" for a brain that was not adopted
- **HANDOFF-HALTERES.md:54-55** — "Internal v0 today: the A5 brain, with four trained
  questions (sweet, dangerous, junk, pictures), well calibrated."
- **Evidence:** A5 was **not adopted** (D1 and D2 FAIL, `gate-a5-results.md:25-29`); its
  real brain is the lowest of five arms (0.863). Pictures ECE 0.105
  (`gate-a5-results.md:20`), which is not well calibrated. Its start rule loses label
  similarity (`a4b-dev-results.md:51-56`) and is to be fixed in the rebuild (decision 6).
  Its answer is the MBON read, which decision 4 replaces. Its numbers were computed on a
  GPU, not as served numbers (decision 7).
- **Honest wording:** "Internal, for mock-up only: the A5 brain (four questions: sweet,
  dangerous, junk, pictures), which failed its own adoption gate. Test balanced accuracy
  0.77–0.91 by question; calibration good on text (ECE 0.02–0.03), poor on pictures (0.105).
  It reads the answer from MBONs, which the next version will not, and its numbers are not
  yet served-on-CPU numbers. Do not publish them as Bosco's."

### C3. `taught: true` is presented as the mark of a trustworthy answer, and the API examples imply free-form questions work
- **HANDOFF-HALTERES.md:40, 58-59** — "He answers the kinds of question he was taught." /
  "Untaught answers must look visibly different; they are close to guesses."
- **API.md:52, 137, 155** — `taught` "whether this version was trained on this kind of
  question"; "questions outside his training are guesses (`taught: false`)".
- **API.md:65, 72, 77** — examples "Is the customer asking for a human agent?", "Which
  intent?", "How upset is the customer?", with `"taught": true` on two of them
  (API.md:108, 115).
- **BRIEF.md:21-22** — "he knows only the kinds of questions he was trained on, and says so".
- **Evidence:** on the kinds he *was* taught, A4 broad gives 0.419 against chance 0.322 and
  nose-alone 0.469 (`a4-broad-results.md:57`); many-option taught kinds collapse (ledgar
  0.04, snips 0.33 vs nose 0.69, dbpedia 0.45 vs 0.93, `a4-broad-results.md:118-119`). Being
  taught does not make him good; it makes him about as good as, or worse than, cosine
  matching. No customer-support, intent-routing or upset-rating question has been trained
  in any gate; the only gated questions with good numbers are A2/A5's four. How `taught` is
  decided is itself undesigned (`API.md:162-163`).
- **Honest wording (HANDOFF/API):** "`taught` means the version was trained on this kind
  of question; it does not mean the answer is reliable. Reliability is per question, from
  the model card's measured accuracy and calibration. Only questions with a published
  number are offered as named questions." Mark the API examples: "Illustrative request
  shape; none of these questions is trained or measured."

### C4. "One frozen embedding model … the only ML outside the brain"; "the encoders understand, the fly decides"
- **HANDOFF-HALTERES.md:25-27** — "One frozen embedding model turns text into a smell for
  his nose, and that is the only ML outside the brain. 'The encoders understand, the fly
  decides.'"
- **BRIEF.md:35-37** — "CLIP (his nose: text/image → vector), the fixed projection onto 52
  glomeruli".
- **CLAUDE.md:34-36** — "One frozen embedding model turns the state into a vector, and that
  is the whole of ML in the loop for B."
- **API.md:5** — "The fly decides, and the design follows from that."
- **Evidence:** there are at least two encoders: e5-large-v2 for text and jina-clip-v2 for
  pictures (`a2-phase1-results.md:27-28`; `GATE-A5.md` anatomy table), with
  multilingual-e5-large-instruct chosen for the full A4 (`passage-check-results.md:20-22`).
  Decision 11 adds an encoder at the optic-lobe handoff. CLIP and 52 glomeruli are stale
  (46 glomeruli are driven, 7 innate ones excluded; `GATE-A5.md`). On what he decides, the
  results say the encoder carries it: on taught kinds no trained brain beats the untrained
  nose (`a4-broad-results.md:109-114`); on unseen kinds the nose beats the brain by 0.13
  (`a4-pilot-results.md:38`); the pair-encoder route was the encoder deciding and is now
  banned (decision 9). "The fly decides" is true of the mechanism (the answer is read out
  of the simulated brain) but not shown as a contribution.
- **Honest wording:** "Frozen encoders (text: e5; pictures: jina-clip; listed with
  provenance on the model card) turn the input into what his senses receive; no LLM, no
  text out. The answer is read out of his simulated brain. On every task measured so far
  the brain's accuracy is at or below what the encoder's own similarity gives without him;
  he adds a watchable decision process, not measured accuracy." Update CLAUDE.md's
  "one frozen embedding model" to the set of encoders actually in the loop.

### C5. The answer read: MBONs everywhere, descending neurons nowhere
- **BRIEF.md:15-18** — "He decides with his output neurons. Approach vs avoid, read from
  the mushroom-body output neurons (MBONs) in the live simulated brain, with the MBON→DAN
  feedback left on."
- **BRIEF.md:38-40** — "The fly: everything between the glomeruli and the MBONs".
- **API.md:132-133** — "`lean`: … his approach output minus his avoid output."
- **HANDOFF-HALTERES.md:31-33** — trace shows "which regions and neurons carried the
  decision", with no statement of where the answer is read.
- **Evidence:** decision 4 (`DECISIONS-2026-09-25.md:20-22`): the answer is read from
  descending neurons; MBONs become a stage shown in the trace. The descending/behaviour
  read is unmeasured as an answer: in A5 the behaviour neurons agree with the MBON answer
  40% of the time (`gate-a5-results.md:96-98`). Also "live simulated brain with MBON→DAN
  feedback" describes the LIF gates (C1, 0.555, FAIL), not the rate model since A1.
- **Honest wording:** BRIEF/API: "`lean` is read from his descending neurons (toward vs
  away). The MBON groups are shown in the trace as the stage before. How well this read
  works is not yet measured." HANDOFF: tell Halteres that the trace's final stage is the
  descending neurons, so the brain panel should end there, not at the mushroom body.

### C6. Latency and server cost stated as measured
- **API.md:18-19** — "Measured CPU cost: about 1 s of brain per answer before
  optimisation."
- **API.md:48** — "~1–2 s on the server now".
- **HANDOFF-HALTERES.md:117-121** — "about 1 s of brain per answer on the CPU … The cut
  brain is about 6× faster per batch … Latency is about 1 to 2 s".
- **BRIEF.md:86-87** — "one forward pass is tens of milliseconds on the Mac's GPU."
- **Evidence:** no results doc records a per-answer CPU measurement; the 1 s figure first
  appears in commit `9fff777` (2026-09-24), before the A5 rate model and the cut brain. The
  "6×" is per training batch on a GPU, 2.6× end to end (`a5-cutoff-pilot2-results.md:25-26`).
  "Server CPU not measured" (`a4b-dev-results.md:100`). Decision 7 makes Kimsufi speed a
  gate; decision 11 adds the optic lobes, "roughly 2–3× the neurons now simulated, the
  largest speed risk". The GPU figure is irrelevant to serving (decision 7).
- **Honest wording:** "Serving speed on the Kimsufi (4 cores, no GPU) is not measured and
  is a gate for the next version. Adding the optic lobes may make it 2–3× slower than
  today's brain. Design for an unknown wait of seconds; caps are set after the
  measurement."

---

## Important

### I1. Determinism promised as a strength; the measured record is weaker
- **API.md:27** — Jev "not claimed"; Bosco "**better**: frozen weights, no sampling. Same
  input + same version = same answer".
- **EFFORT-PILOT.md:22** — "The brain is deterministic: the same sniff twice gives exactly
  the same answer."
- **EFFORT-PILOT.md:61-63** — "This keeps the API's determinism promise … which Jev does not
  make."
- **CLAUDE.md:53-55** — "the kernel is fixed point or strictly ordered float; a run is
  reproducible from its config and seed."
- **Evidence:** training replays drift by about 0.01 on the same GPU
  (`a4-broad-results.md:96-101`); the plain net on MPS gave three different results from one
  seed (`A4-BROAD.md` amendment 3); every published A-gate number came from GPU training
  and evaluation, so none is yet "exactly repeatable" in the sense of decision 7. Seeds
  disagree on 4.7–6.7% of A5 items (`gate-a5-results.md:31-37`), so a retrained version
  can flip answers. CPU inference with frozen weights can be deterministic, but it has not
  been checked on the server.
- **Honest wording:** API: "Aim: same input + same version = same answer, on the server's
  CPU; to be verified per version before it is claimed. Training is not bit-for-bit
  repeatable on GPUs (replays move results by about 0.01); versions retrained from the
  same data can disagree on about 5% of items." CLAUDE.md: note that the rate-model
  training runs on GPUs and is replayable only to about 0.01, and that serving and
  evaluation move to deterministic CPU (decision 7).

### I2. "A real fly brain does the deciding" with no controls beside it
- **BRIEF.md:3-5** — "in which **a real fly brain does the deciding**".
- **HANDOFF-HALTERES.md:145** — "Say what is true: a real fly connectome decides".
- **README.md:5-6** / **CLAUDE.md:5-6** — the question "is a real brain's wiring diagram a
  useful prior" has an answer on these tasks and neither says so.
- **Evidence:** the real wiring is the lowest or tied in every gate: A1 0.687 vs
  0.707–0.710; A2 0.841 vs 0.847–0.899; A5 0.863 vs 0.876–0.887; A4 broad 0.419 vs layered
  0.418 and plain net 0.426, all within replay noise. (HANDOFF:149-151 already says this
  correctly; BRIEF and README do not.)
- **Honest wording:** "Decisions are computed by a network wired as the MaleCNS central
  brain. On every task measured, scrambled wirings and plain networks trained the same way
  do as well or better; the wiring is not shown to help accuracy."

### I3. Learning described as the fly's own rule
- **BRIEF.md:5** — "its own mushroom-body learning rule, one fly".
- **BRIEF.md:38-40** — "The fly: … the learning rule, forgetting, spaced-repetition memory,
  and the decision."
- **BRIEF.md:41-43** — "Dropped … the offline feedforward training. Rehearsal (+1 h, +3 h,
  +24 h) stays".
- **CLAUDE.md:3-5, 61-70** and **README.md:5-6** — "the same LIF kernel and mushroom-body
  plasticity"; "LIF neurons … Time step 0.1 ms"; "three-factor rule".
- **Evidence:** every gate since A1 trains a rate model (40 steps × 5 ms in A5) by gradient
  descent; the dopamine neurons do not teach (`AUDIT-2026-09-24.md` §5-6; `GATE-A5.md`
  anatomy table: "the teaching is gradient descent, not dopamine"). Decision 1 requires
  this stated plainly on the model card. The three-factor-rule decider failed (C1, 0.555).
  HANDOFF says nothing about gradient training, so Halteres copy may repeat the BRIEF.
- **Honest wording:** "He is trained offline by gradient descent on labelled examples:
  per-cell-type parameters and the KC→MBON synapses. The wiring and signs are the
  connectome's and are not trained. He does not learn by dopamine, and nothing he answers
  changes him." Add this line to HANDOFF's honesty rules. CLAUDE.md's Model section
  should say the LIF kernel and three-factor rule are the v1 reproductions, and the
  current decider is the rate model.

### I4. Vision and taste described as fly senses
- **BRIEF.md:6-7** — "Sweet/bitter taste of text and/or images is the first question".
- **BRIEF.md:28** — "smells (state + question)"; **BRIEF.md:79-82** — "The picture path from
  A2 first; the strong version gives him back the fly's own visual system".
- **API.md:29** — "add pictures" (as a feature over Jev); **API.md:155-156** — "pictures are
  good only as far as his eyes, which are being rebuilt" (fine).
- **Evidence:** A2's picture path was a side door into KCs (`ERRATA-2026-09-24.md:25`;
  `AUDIT-2026-09-24.md` §1). A5's is jina-clip-v2 → PCs → the 9,201 visual projection
  neurons, "**A stand-in**" (`GATE-A5.md`), with no photoreceptors or optic lobes. Sweet and
  bitter never touch the taste neurons (`AUDIT-2026-09-24.md` §3). A5 pictures fail D1 (0.865
  vs 0.889) and are the worst calibrated (0.105). Decision 11: real eyes plus the encoder as
  translator, not built yet.
- **Honest wording:** "Pictures reach him through an image encoder mapped onto his visual
  projection neurons, a stand-in for eyes and optic lobes (the real ones are planned).
  'Sweet' and 'bitter' are names for a learned approach/avoid, not his sense of taste."

### I5. Accuracy demoted below the trace, against the 09-25 decisions
- **HANDOFF-HALTERES.md:33-35** — "showing it well matters more than raw accuracy. Nick:
  'raw compute wins' on accuracy".
- **Evidence:** `DECISIONS-2026-09-25.md:5-6` ("it still has to be FAST and ACCURATE, […] it
  needs to be usable") and decision 5 ("accuracy may not be sacrificed significantly;
  measured, not assumed").
- **Honest wording:** "The trace is the novelty, but he must also be fast and accurate
  enough to use (decisions 5 and 7, 2026-09-25). Accuracy and speed are gates for the next
  version."

### I6. How the question enters
- **BRIEF.md:11-13** — "The question is part of what he smells".
- **API.md:30** — "His nose is one vector, so pointing it matters more".
- **PLAN-A4.md:26-29** — "Question first, then the thing (two sniffs)".
- **Evidence:** decision 3: the question and the thing come in apart, through different
  senses; where the question enters is open (`DECISIONS-2026-09-25.md:48-50`). A3 found two
  sniffs gave no gain (−0.003, `a3-phase1-results.md:16,30`).
- **Honest wording:** "How the question reaches him is being redesigned: the question and
  the thing being judged will enter through different senses (decision 3); which sense
  carries the question is open."

### I7. Doom deferred, on a GPU
- **BRIEF.md:68-89** — "Later: Doom", "once A2 has said…", "tens of milliseconds on the
  Mac's GPU".
- **Evidence:** decision 2 withdraws "later for Doom" and puts the optic lobes in v1;
  decision 7 puts serving on the CPU. A2 has reported (not adopted).
- **Honest wording:** mark the section superseded by `DECISIONS-2026-09-25.md` (2, 7, 11);
  drop the GPU timing.

### I8. PLAN-A4's premise, left without a status line
- **PLAN-A4.md:3-5** — "train him on many questions so he can answer ones he has never
  been asked … this is the fly's version of that."
- **Evidence:** A4 pilot: unseen kinds 0.503 vs nose 0.631 (`a4-pilot-results.md:38`); A4
  broad tiers 2-3 below the nose (`a4-broad-results.md:86-87,126-128`).
- **Honest wording:** add a status line: "Outcome: he did not answer unseen kinds better
  than his untrained nose (A4 pilot, A4 broad). The goal (a generalist) stands; this route
  did not reach it."

### I9. EFFORT-PILOT assumes a shipped A4 brain and an MBON lean
- **EFFORT-PILOT.md:3-4, 42** — "once A4 broad has reported"; "on the brain that A4 broad
  ships".
- **EFFORT-PILOT.md:34-37** — accumulation "outside the brain … downstream of the mushroom
  body, in circuits this model does not simulate".
- **EFFORT-PILOT.md:82-83** — E1: high ≥ low + 0.02.
- **Evidence:** A4 broad ships nothing (`a4-broad-results.md:129`). Decision 4 moves the read
  to descending neurons, which are in the model, so "not simulated" needs rechecking;
  decision 8 requires the extension to be written down against real anatomy. Differences
  of ~0.01 are replay noise (`a4-broad-results.md:101`), so +0.02 is only twice the noise;
  decision 7 requires the evaluation on CPU.
- **Honest wording:** "Waits for the rebuilt brain (not A4 broad, which failed). The lean
  is the descending-neuron read (decision 4). Evaluated on CPU, deterministic; E1's margin
  set against the measured replay noise of that brain."

### I10. Calibration claimed in general, measured narrowly
- **HANDOFF-HALTERES.md:21-22** — "makes typed, calibrated decisions".
- **API.md:38** — calibration "**better**: ECE and reliability per question family,
  published with every version."
- **Evidence:** calibration is measured only for A2/A5's four questions (A5 ECE 0.022–0.105).
  A4 broad reports no ECE at all. The `p` for `choose`/`rate` on any catalogue question is
  unmeasured.
- **Honest wording:** "Probabilities are calibrated on the questions measured (so far four:
  ECE 0.02–0.03 on text, 0.105 on pictures); every published question will carry its own
  ECE." API.md's "better" is a plan; say "planned: …".

### I11. README points users at the retired decider
- **README.md:21-34** — "The decider": `scripts/decider.py bootstrap/decide/reward/serve`,
  `valence`, `p_right`, CALIBRATION-b4, GATE-B4-note.
- **Evidence:** `scripts/decider.py` is the old swarm with the weight read and CLIP
  (docstring lines 1-16), the design C1 said not to return to quietly
  (`gate-c1-results.md:51-52`) and the audit flagged (`AUDIT-2026-09-24.md` §E). Its `reward`
  command teaches online, against "never taught through the API" (`BRIEF.md:19-20`,
  `API.md:15-16`). B4's pass is optimistic (`ERRATA-2026-09-24.md:18`), CALIBRATION-b4's
  label-free claim is wrong (`ERRATA-2026-09-24.md:24`).
- **Honest wording:** "`scripts/decider.py` is the retired B4 decider (swarm, weight read,
  online reward), kept as a record. There is no current decider service; see
  `docs/DECISIONS-2026-09-25.md`."

### I12. CLAUDE.md points at an empty result for the published controls
- **CLAUDE.md:28-30** — "The controls still run, and their numbers are published beside
  his (`docs/GATE-A2b.md`)."
- **Evidence:** `gate-a2b-results.md` has an empty table; A2b was killed
  (`AUDIT-2026-09-24.md:4-5`). The control numbers are in `gate-a5-results.md` and
  `a4-broad-results.md`.
- **Honest wording:** point at the gate whose controls are published (A5, A4 broad), or
  at the rule itself, not at A2b.

---

## Minor

- **HANDOFF-HALTERES.md:20-21** — "cut to 50,140 neurons": the cut is on synapses (≥5,
  24% dropped); 50,140 is the neurons in scope (`GATE-A5.md`). Decision 11 adds the optic
  lobes (2–3×). Wording: "50,140 neurons of the central brain (connections of ≥5 synapses
  kept; the optic lobes are being added)".
- **HANDOFF-HALTERES.md:43** — "A 50-glomerulus nose": 46 glomeruli are driven, and the
  channels are bi46 (46) (`a4b-dev-results.md:18`; `GATE-A5.md`). "His ~46-channel nose".
- **HANDOFF-HALTERES.md:132-133** — "trained weights for internal v0 (A5)": add "not
  adopted by its gate".
- **HANDOFF-HALTERES.md** (missing) — nothing on decisions 1 (gradient training on the
  model card), 4 (DN read), 7 (CPU serving, exact repeatability), 9 (no pair encoder),
  10 (data ledger on the about page), 11 (eyes). The about/model-card page Halteres builds
  needs all of them.
- **API.md:94-95** — `familiar` "from his novelty compartment (α'3)": no gate has read α'3
  or measured novelty. "Planned; no measurement yet."
- **API.md:101-129** — example numbers (`p` 0.87, `fly_ms` 120, timing 180/950 ms) look like
  measurements; A5 runs 200 ms of fly time. Label the block "illustrative values".
- **API.md:38 vs gate-a2-results.md:73** — API says Jev has "no published metrics" for
  calibration; A2's results quote "Jev on its public bench: 0.161". Say that 0.161 is our
  measurement of Jev on its bench.
- **PLAN-A4.md:64** — "Jev scores 94% on passage yes/no (BoolQ)": 20 of those 50 passages
  are in BoolQ's training set (`AUDIT-2026-09-24.md` §D); note the overlap beside the 0.94.
- **PLAN-A4.md:32-35** — "82k per-neuron settings": superseded by per-cell-type training
  (`DECISIONS-2026-09-24.md` §1).
- **EFFORT-PILOT.md:76-77** — "on the Kimsufi's CPU (or its measured stand-in)": decision 7
  makes the Kimsufi itself the gate.
- **BRIEF.md:47-53** — step 1 ("the fly decides … MBONs in the live brain") ran as C1 and
  failed (0.555); add the outcome.
- **WEBAPP.md:3-5 vs HANDOFF-HALTERES.md:9, 86** — WEBAPP calls the public front end
  bosco.systems; the handoff puts Halteres at halteres.bosco.systems. Say which serves
  what.
- **WEBAPP.md:43-44** — caps "set once the server speed is measured": correct, and consistent
  with C6; no change needed.

## What holds

- HANDOFF-HALTERES.md:149-152 (controls beside every number; no fabricated results) and
  HANDOFF:139-141 (mock data labelled as mock) are right and should stay.
- API.md:153-156 (passage questions weak; pictures limited by the eyes) matches A3 and the
  passage check.
- WEBAPP.md's privacy and login claims are design decisions, not results; nothing in the
  evidence contradicts them.
