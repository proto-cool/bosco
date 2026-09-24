# Bosco — the brief (2026-09-23)

A Jev-style decision API (System One: typed answers with a calibrated
probability, no text) in which **a real fly brain does the deciding**. The
MaleCNS v1.0 central brain, its own mushroom-body learning rule, one fly.
Sweet/bitter taste of text and/or images is the first question it answers,
not the whole of it.

## Decisions

- **One Bosco.** Every question is answered by the same fly. The question is
  part of what he smells: *taste what you have been handed, in the light of
  what you were asked*. People who want their own get a copy of the fly, and
  a copy learns only from what its owner rewards.
- **He decides with his output neurons.** Approach vs avoid, read from the
  mushroom-body output neurons (MBONs) in the live simulated brain, with the
  MBON→DAN feedback left on. Not arithmetic on his synapse weights, not an
  average of several flies, not a cached feedforward shortcut.
- **Trained offline, per version; never taught through the API**
  (revised 2026-09-24, `API.md`). Like Jev he arrives trained; unlike Jev he
  knows only the kinds of questions he was trained on, and says so
  (`taught: false`).

## Every answer type from approach/avoid

| Jev type | what the fly does |
|---|---|
| yes/no | smells (state + question); approach = yes |
| one of N | smells (state + question + each option) in turn; picks what he approaches most (a T-maze of N arms) |
| score in a range | how strongly he approaches vs avoids, on his own zero (where the two sides balance) |
| probability | how clearly one side won, calibrated on his own track record, labelled as a gauge outside the fly |

## What is the fly and what is not

- **Not the fly, said openly:** CLIP (his nose: text/image → vector), the
  fixed projection onto 52 glomeruli (set once, label-free), the single read
  of approach vs avoid, and the calibration map.
- **The fly:** everything between the glomeruli and the MBONs: the wiring,
  the Kenyon-cell code, the learning rule, forgetting, spaced-repetition
  memory, and the decision.
- **Dropped from the current decider:** the swarm of five, the weight-based
  valence read, the offline feedforward training. Rehearsal (+1 h, +3 h,
  +24 h) stays: it is how flies are trained, and it is scheduling, not rule.

## Steps, each written down before it runs

1. **The fly decides (sweet/bitter).** Same training as B4. Score read from
   MBONs in the live brain. Known risk (`plasticity.py`, `learned_valence`):
   single MBON readings are chaotic under small weight changes; the fly's
   answer is then the mean over a few presentations (a fly samples an odour
   for seconds, not once). Measured on the held-out sentences, the pictures
   and Nick's probes, against the weight read and the hash control. About an
   hour of CPU, no overnight runs. The number is reported whatever it is.
2. **The API shape.** `decide(state, questions)` with yes/no, one-of-N and
   score types; `reward`; `state`; copies (`fork`). Sweet/bitter is the
   first question through it.
3. **Can one fly carry several questions?** Pre-registered: 2–3 taught
   questions on one Bosco, question in the smell, against one fly per
   question. The measure is how much the questions interfere. If one fly
   cannot carry them, that is written up, and the answer is not quietly
   swapped for a lab of flies.

## Won't do

No LLM, no model deciding, no training beyond per-cell-type parameters and
the KC→MBON synapses (`CLAUDE.md`), no swarm, no text out, not on a feed.

## Later: Doom (added 2026-09-24)

Jev plays Doom in real time (TypeSafe's launch demo): the game state goes
in as a **text description, not frames**, and Jev picks the next action from
a fixed set about ten times a second (madewithjev.com/builds/jev-plays-doom).
No source says it finishes a level. The same bar for Bosco, once A2 has said whether
one brain carries several questions. Parity first: the same text state
through his nose. Frames through his eyes come after (Jev cannot take images).

- **Each frame is a one-of-N question:** forward, turn left, turn right,
  shoot. He smells each option and goes to the one he approaches most.
- **Then his eyes see the screen.** The picture path from A2 first; the strong
  version gives him back the fly's own visual system (optic
  lobes, pruned from the model since v1): steering by sight is what that
  circuit evolved for.
- **He learns by playing** (reward for progress and kills, cost for damage),
  not from labels. That is reinforcement learning, days of runs, and it is
  pre-registered like every gate.
- **Real time:** ten decisions a second, as Jev; one forward pass is tens of
  milliseconds on the Mac's GPU.
- **Controls:** the scrambled brains play too. "A fly brain plays Doom"
  means something only if they do worse.
