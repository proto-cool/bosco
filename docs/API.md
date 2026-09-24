# Bosco's API (design, revised 2026-09-24)

Our own System One-style decision model, not a Jev clone. The idea is the
same (fast typed decisions with honest confidence, never text), but the
vocabulary, extras and rules are Bosco's.

## Decisions (Nick, 2026-09-24)

- **Our own thing with a similar idea.** The request has the natural shape
  for typed questions about one state (as Jev's does), but the question types
  and answers are the fly's.
- **Answering never changes him.** A request is pure: the same Bosco version
  gives the same answer every time. Nobody teaches him through the API.
- **Teaching is separate and offline.** Examples are collected, he is trained
  outside the inference flow, and a new **version** is published. Every
  answer names the version that gave it.
- **Public at bosco.proto.cool** (Nick's Kimsufi server): the frozen current
  version, with per-visitor caps. Nobody's questions reach his training.

## Request

```json
{
  "state": { "text": "…", "image": "<upload id or url>" },
  "questions": {
    "<id>": {
      "type": "approach" | "choose" | "familiar",
      "instructions": "…",
      "toward": "…", "away": "…",
      "options": { "<name>": "<description>" }
    }
  }
}
```

- `state`: text, an image, or both. Text goes through his nose, images
  through his eyes.
- **approach**: would he go toward it? `toward` and `away` optionally say what
  each side means.
- **choose**: 2 or more `options`; he smells each and goes to one (a T-maze).
  A rating is a `choose` over its levels.
- **familiar**: has he met this before? Read from the novelty compartment
  (α'3). Needs a memory of what he has met, so it applies to a session's own
  stream, not to training. To be designed.
- Questions in one request are answered independently and in parallel.

## Answer

```json
{
  "version": "bosco-2026-10-01",
  "answers": {
    "<id>": {
      "lean": 0.74,
      "p": 0.87,
      "sure": 0.71,
      "taught": true,
      "pick": "<option>",
      "pull": { "<option>": 0.61 }
    }
  },
  "brain": { "trace_id": "…", "fly_ms": 120 },
  "timing_ms": { "senses": 22, "brain": 48 }
}
```

- `lean`: −1 (avoid) … +1 (approach), his approach minus avoid.
- `p`: calibrated probability that the answer is "toward" (approach), or of
  each option (choose, in `pull`).
- `sure`: confidence from the spread of `p`, (N·p_max − 1)/(N − 1), N = 2 for
  approach.
- `taught`: whether this version was trained on this kind of question. When
  it is false, the answer is a guess and says so.
- `brain`: a trace of the decision for the UI's replay. The public site sends
  a slimmed trace: activity per brain region, plus the most active neurons per
  step. The full trace is about 3 MB.

## Caps for the public site

Questions per minute per visitor, a limit on text length and one image per
state, and a limit on options per question (each option is one more sniff).

## Open

- Whether the server's CPU is fast enough. Measure before building.
- `familiar`'s memory: per session, and forgotten after it.
- Where Jev is ahead: it answers questions it was never trained on. Bosco
  needs his questions taught; A4 is the plan for teaching him hundreds.
