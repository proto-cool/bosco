# Bosco's API — Jev's primitives on a fly brain (design, 2026-09-24)

Built from TypeSafe's published docs (docs.typesafe.ai: `/primitives`, `/api`,
`/confidence`, read 2026-09-24) so a Jev client can talk to Bosco unchanged.
It is implemented once A2 adopts a brain (`GATE-A2.md`, D1 and D3).

## Request and response: Jev's shapes

```
POST /v1/systemone
{ "state": string | object | array,
  "model": "bosco-latest",
  "questions": { "<id>": { "type": "noul" | "choice" | "score",
                           "instructions": "...",
                           "criteria": ... } } }

→ { "model": "...", "answers": { "<id>": Answer }, "usage": {...} }
```

- **noul**: `criteria` optional `{"true": ..., "false": ...}` → `{"type": "noul", "noul": p}`.
- **choice**: `criteria` = {option: description}, up to 255 → `choice`, `probabilities`, `confidence`.
- **score**: `criteria` = 2–10 ordered levels → `score` (probability-weighted, can fall between levels), `legend`, `probabilities`, `confidence`.
- `confidence` = Jev's formula: (N · p_max − 1) / (N − 1). All the probability on one option gives 1; evenly spread gives 0.
- Every question in a request is answered in parallel and independently. Answers can only ever be the options supplied.

## How the fly answers each one

Everything below is one mechanism: **he smells something and approaches or
avoids it.** The read is always approach MBONs minus avoid MBONs.

| primitive | what he smells | what is read |
|---|---|---|
| noul | state + instructions (+ the `true` description) | P(approach) = the answer |
| choice | state + instructions + **each option**, one sniff per option, all in one batch (a T-maze with N arms) | approach strength per arm → softmax → probabilities; choice = the arm he goes to |
| score | the same, one arm per level | probabilities over levels; score = their weighted mean |

- **State:** text, or an object/array flattened to text, goes through his
  **nose** (e5-large-v2 → 52 glomeruli). A picture in the state goes through
  his **eyes** (jina-clip-v2 → the visual Kenyon cells). Both can be given at
  once.
- **Two-piece questions** ("does the passage answer this?", "does the
  premise make this true?"): two sniffs, piece one then piece two, if A3
  phase 1 shows the relation survives the nose (`A3-PHASE1.md`).
- **Speed:** a 255-arm choice is one batch through the brain. A forward pass
  is tens of milliseconds on the Mac's GPU, to be measured and published.

## Where Bosco does something Jev does not

Claims to be earned by measurement, not asserted:

1. **Pictures.** Jev "cannot process images (yet)". Bosco sees them.
2. **He learns from you.** `reward(decision_id, correct_answer)` teaches the
   copy that made the decision. Jev is fixed. Each person's copy becomes
   their own fly. The mechanism is to be gated: the fly's own rule on the
   KC→MBON synapses, or a small update of the trained brain.
3. **Confidence on yes/no too.** Jev's noul answers carry no confidence;
   Bosco's do, by the same formula with N = 2.
4. **Published calibration.** Jev's calibration method is unpublished, and an
   independent bench measured its ECE at 0.161. Bosco's is measured per
   question on held-out data, and the numbers ship with the model.
5. **Runs locally and deterministically:** seeded, private, no per-token price.
6. **Open about what decides:** the nose and eyes only perceive. Every
   answer comes out of the MaleCNS circuit, with scrambled-wiring controls
   published beside it.

## Where Jev is ahead, stated

- **Jev answers questions it has never seen.** Bosco has to be trained
  (or rewarded) on a question before his answers mean anything; an untaught
  question comes back near 50/50, and the API says so (`trained: false`).
- **Reasoning-heavy tasks** (reading a passage to answer a question) are
  likely beyond a one-vector nose. A3 measures how far.
