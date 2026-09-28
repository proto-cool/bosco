# Bosco's API (design, revised 2026-09-24 from a full read of TypeSafe's docs)

Our own System One-style decision model, not a Jev clone. The idea is the
same: fast, typed, calibrated decisions for software, never text. The fly
decides, and the design follows from that.

Source for every "Jev" statement below: TypeSafe's docs (docs.typesafe.ai,
46 pages read 2026-09-24; local copy in `data/raw/typesafe-docs/`, not
committed).

## Decisions (Nick, 2026-09-24)

- Our own thing with a similar idea. The request shape (a state plus named,
  typed questions) is the natural one and stays.
- **Answering never changes him.** Teaching is separate, offline and
  versioned. The public site runs a frozen version.
- Public at bosco.systems on the Kimsufi (Xeon E-2274G, 4 cores, 32 GB,
  no GPU). Measured CPU cost: about 1 s of brain per answer before
  optimisation.

## Core concepts: adopt, change, add, drop

| concept | Jev | Bosco |
|---|---|---|
| what it is | "gut-check" decisions a person could make in seconds; no text, no reasoning | **adopt**. One sniff, one lean. |
| stateless requests; questions independent and parallel | yes | **adopt**. Each option and each question is its own sniff, batched. |
| determinism | not claimed: repeat runs vary (std ≈ 0.01, a Noul crossed 0.5) | **better**: frozen weights, no sampling. Same input + same version = same answer, to be verified on the server's CPU and stated per version. |
| per-customer weights / fine-tuning | none; same weights for everyone | **same for the public site.** New versions are trained offline from examples we choose. |
| input | text: string, JSON object, array | **add pictures**: `state.text`, `state.image`, or both. Objects and arrays are flattened to text for the nose. |
| state references (`` `ticket.messages[0].text` ``) | backticked paths inside instructions | **change**: `smell` names the part of the state he smells (a path). Default: all of it. His nose is one vector, so pointing it matters more than it does for Jev. |
| instructions as object/array | yes; field names free | **adopt**, flattened to text. |
| question ids hidden from the model | yes | **adopt**. |
| types | noul, choice (≤255), score (2–10 levels) | **approach**, **choose**, **rate**, and a new **familiar** (below). |
| "none/other" option; a Choice always sums to 1 | recommended; pair with a Noul for existence | **adopt**, and say it plainly: a `choose` always goes somewhere, so ask `approach` whether any option fits. |
| >255 options | two stages in code | **same pattern**, but a tighter public cap: each option is one sniff (CPU). |
| probabilities | per option; Noul returns P(yes) only | **adopt**, plus `lean`, the fly's approach − avoid. |
| confidence | clamp((n·p_max − 1)/(n − 1)); none on Noul | **adopt the formula**, and give it on `approach` too (n = 2). |
| calibration | trained for it (RLCD); **no published metrics** | **better**: ECE and reliability per question family, published with every version. |
| score = Σ level·p, can fall between levels; levels judged alone, descriptive not numeric | yes | **adopt** for `rate`: descriptive levels, each smelled on its own. |
| known limits per version ("jaggedness") | a published list | **adopt**: a limits page per version, from our own gates (e.g. two-text relations, from A3). |
| versions | aliases (`jev-latest`) plus pinned ids; the response names the version | **adopt**: `bosco-latest` and pinned `bosco-YYYY-MM-DD`; every answer names its version. |
| models endpoint | `GET /v1/models` | **adopt**, plus a model card: training questions, calibration, limits, wiring. |
| errors | 401, 422 (names the field), 429, 529; retry with backoff | **adopt**: 400/422 name the field, 429 with `retry-after`, 503 when busy. |
| request id | `x-typesafe-request-id` | **adopt**: `x-bosco-request-id`. |
| rate limits | 250k tokens/s, 1,200 req/min | **public caps**: questions/min per visitor, state size, options per question. |
| streaming / batch jobs | none | none. |
| pricing | $0.042 per M input tokens | free public site, capped; `usage` counts **sniffs** instead of tokens. |
| latency | ~100–150 ms | ~1–2 s on the server now; the brain replay makes the wait legible. |
| SDK | `choice()`, `noul()`, `score()`; typed results | later: `approach()`, `choose()`, `rate()`, `familiar()`. |
| playground | yes | **the chat UI** (Nick is designing it in Claude Design), with the brain panel. |
| the answer's reason | never; "no explanations of reasoning" | **add the brain trace.** Not words: the neurons that carried the decision, replayable. |
| "taught" | not a concept (it answers anything) | **add** `taught`: whether this version was trained on this kind of question. Untaught answers say so. |
| patterns: fan-out, confidence routing, composite scores, hierarchical beam, reranking, extraction by selection, abstention bands | cookbooks | **all hold.** Our docs carry the same patterns, with one warning: pairwise questions (does this passage answer that?) are weak for a one-vector nose (A3). |

## Request

```json
POST /v1/decide
{
  "version": "bosco-latest",
  "state": { "text": "…", "image": "<upload id>" },
  "questions": {
    "<id>": {
      "type": "approach",
      "instructions": "Is the customer asking for a human agent?",
      "smell": "text",
      "toward": "…",
      "away": "…"
    },
    "<id2>": {
      "type": "choose",
      "instructions": "Which intent?",
      "options": { "refund": "…", "other": "None of the above" }
    },
    "<id3>": {
      "type": "rate",
      "instructions": "How upset is the customer?",
      "levels": ["calm and polite", "irritated", "angry, threatening to leave"]
    },
    "<id4>": {
      "type": "familiar",
      "instructions": "Has he met this before?"
    }
  }
}
```

- **approach**: yes/no as going toward (yes) or away. `toward` and `away` are
  optional, and say what each side means. Phrase it so toward means yes.
- **choose**: he smells each option in turn and goes to one (a T-maze). The
  answer always picks something; include `other` if nothing may fit.
- **rate**: 2–10 descriptive, ordered levels, each smelled on its own; `level`
  is the probability-weighted position (it can fall between levels).
- **familiar**: from his novelty compartment (α'3), against what this session
  has shown him. Needs session memory; to be designed.

## Answer

```json
{
  "version": "bosco-2026-10-01",
  "answers": {
    "<id>": {
      "type": "approach",
      "lean": 0.74,
      "p": 0.87,
      "sure": 0.74,
      "taught": true
    },
    "<id2>": {
      "type": "choose",
      "pick": "refund",
      "p": { "refund": 0.81, "other": 0.19 },
      "sure": 0.62,
      "taught": true
    },
    "<id3>": {
      "type": "rate",
      "level": 1.4,
      "legend": ["calm and polite", "irritated", "angry, threatening to leave"],
      "p": { "0": 0.1, "1": 0.4, "2": 0.5 },
      "sure": 0.25,
      "taught": false
    }
  },
  "brain": { "trace_id": "…", "fly_ms": 120 },
  "usage": { "sniffs": 6 },
  "timing_ms": { "senses": 180, "brain": 950 }
}
```

- `lean`: −1 (avoid) … +1 (approach): his approach output minus his avoid
  output.
- `p`: calibrated probability, of toward (approach) or of each option or
  level.
- `sure`: clamp((n·p_max − 1)/(n − 1), 0, 1).
- `taught`: whether this version was trained on this kind of question.
- `brain`: `GET /v1/traces/{trace_id}` returns the replay. The public site
  sends a slimmed trace: activity per brain region, plus the most active
  neurons per step.

## Our own guidance (what the docs will say)

- One condition per `approach`; one dimension per `rate`; descriptive levels,
  never bare numbers.
- Let code do counting, dates, arithmetic and exact matching; ask Bosco the
  gut-check.
- Point `smell` at the part of the state that matters; big states blur his
  one-vector nose.
- Act on high `sure`, confirm on medium, hand low to a person. Set thresholds
  on your own data, per version.
- Pin a version when you depend on thresholds; `bosco-latest` moves.
- **Known limits (current):** two texts related to each other (does the
  passage answer the question?) are weak; questions outside his training are
  guesses (`taught: false`); pictures are good only as far as his eyes, which
  are being rebuilt.

## Open

- Server speed: brain pruning, the C kernel, lower precision. Measure before
  building.
- How `taught` is decided: the question's similarity to the questions he was
  trained on, to be designed and validated.
- `familiar`: session memory.
