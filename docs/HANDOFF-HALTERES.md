# Handoff: Bosco, for the session building Halteres (2026-09-25)

Written by the Claude Code session that built and tested Bosco's brain, for the
session that will build **Halteres**. Nick will also give you a brief from
Claude Design.

**How to weigh the two sources:**
- **The Design brief owns what Halteres is:** its role, flows, layout and look.
  This session knows only that Halteres lives at `halteres.bosco.systems`. (In a
  real fly, halteres are the small balance organs behind the wings.)
- **This document and the repo own everything about Bosco:** what the model is,
  what it can and cannot do, the API, privacy, hosting and honesty rules.
  Where the Design brief says otherwise on these, **this wins.** Tell Nick about
  the mismatch; do not quietly build either version.
- Things marked **open** below are not decided. Do not treat a Design mock-up as
  deciding them.

## What Bosco is

A **fly decision engine.** A real connectome (the MaleCNS v1.0 male fruit-fly
brain, cut to 50,140 neurons) simulated as a rate model makes typed, calibrated
decisions. Its spirit is TypeSafe's Jev (a "System One" model), but **Bosco is
Nick's own thing, not a Jev clone.**

- **No text is ever generated, and there is no LLM anywhere.** One frozen
  embedding model turns text into a smell for his nose, and that is the only ML
  outside the brain. "The encoders understand, the fly decides." Never describe
  or build it as an LLM, a chatbot, or "AI that writes".
- **An answer is a decision:** a lean, a probability, how sure he is, and whether
  he was taught this kind of question. It is never words.
- **You can watch him think.** Each answer comes with a replayable brain trace:
  which regions and neurons carried the decision, step by step, over about
  200 ms of fly time. This is the product's novelty; showing it well matters
  more than raw accuracy. Nick: "raw compute wins" on accuracy, so the value is
  "a real connectome as a fast decision engine you can watch thinking".

## What he can and cannot do (honest state, 2026-09-25)

Nothing may claim more than this:
- **He answers the kinds of question he was taught.** The A4 pilot and the A4b
  development round (`docs/a4-pilot-results.md`, `docs/a4b-dev-results.md`)
  showed that a fly cannot answer unseen kinds of question better than his
  untrained nose. A 50-glomerulus nose cannot carry the fine semantic matching
  that needs.
- **The plan is breadth:** teach him a large catalogue of question kinds
  (sentiment, topic, intent, emotion, toxicity, hate, stance, dialogue act and
  more). The **A4 broad** gate (`docs/A4-BROAD.md`) is training now, with 44
  taught kinds. Its result decides what the public v1 can offer:
  - If it passes, the public site offers a **catalogue of named questions**, each
    with its own fixed options. Free-form custom options are claimed only where
    G2 shows they carry over.
  - If it fails, the offer is narrower. Do not design the UI as if he answers
    anything anyone types.
- **Internal v0 today:** the A5 brain, with four trained questions (sweet,
  dangerous, junk, pictures), well calibrated.
- Two-text questions (does this passage answer that question?) are weak.
  Pictures are only as good as his eyes, which are a stand-in for now.
- Every answer shows `taught`. Untaught answers must look visibly different;
  they are close to guesses.

## The API (`docs/API.md` is the source)

- `POST /v1/decide` with `{version, state: {text, image}, questions: {id:
  {type, instructions, smell, options | levels | toward/away}}}`.
- There are four types:
  - **approach:** yes/no, as going toward or away.
  - **choose:** a T-maze, one sniff per option; it always picks one, so include
    "other".
  - **rate:** 2 to 10 descriptive levels.
  - **familiar:** novelty; not designed yet.
- The answer carries `lean` (−1 to 1), `p`, `sure` = clamp((n·p_max − 1)/(n − 1)),
  `taught`, `pick`/`level`, `brain.trace_id`, `usage.sniffs` and `timing_ms`.
- `GET /v1/traces/{id}` returns the replay. The public site gets a **slimmed**
  trace: activity per brain region, plus the most active neurons per step.
- Versions are `bosco-latest` plus pinned `bosco-YYYY-MM-DD`, and every answer
  names its version. `GET /v1/models` gives a model card: training questions,
  calibration, limits, wiring.
- Errors: 400/422 name the field; 429 comes with `retry-after`; 503 when busy.
  Every response has an `x-bosco-request-id`.
- **Caveat:** API.md was written before A4. It shows free-form `instructions`
  and options as if any question works. After A4, expect the public site to
  lean on the taught catalogue, as above. The request shape stays.

## Product rules (Nick, decided; not negotiable without him)

- **Domain: `bosco.systems`.** Halteres is at `halteres.bosco.systems`. No
  company brand in the product. Arclight Digital (Nick's LLC) appears in a
  footer only.
- **Login only through atproto OAuth, identity scope only** (base `atproto`
  scope: DID and handle, nothing else). **No anonymous use. No accounts of our
  own.** A confidential client using the official atproto OAuth library. Client
  metadata: `https://bosco.systems/oauth/client-metadata.json`.
- **Nothing is ever written to a user's atproto repo.** It is public by design.
  Saved question sets, if they come, live in the browser or on our server,
  keyed by DID.
- **The public Bosco never learns from visitors.** Answering never changes him.
  Teaching is separate, offline and versioned, and the site runs a frozen
  version.
- **The API is private to the server.** The web app is its only public
  consumer. Direct API use needs a dev token issued by Nick: revocable, stored
  hashed, with its own caps.
- **Stored on the server:** DID, session, usage counters and minimal request
  logs; no conversation content. **In the browser:** conversations, in local
  storage.
- **Caps:** per-DID questions per minute and per day, one queue with a maximum
  length, input size caps, and a cap on options per question. The numbers are
  **open** until server speed is measured.

## Hosting and shape (`docs/WEBAPP.md`)

- A Kimsufi dedicated server: Xeon E-2274G (4 cores, 8 threads), 32 GB, **no
  GPU**, US-East.
- **Caddy** for HTTPS, in front of a **TypeScript web app** (OAuth, sessions,
  rate limits, queue, UI), in front of a **Python Bosco service** that listens
  only on the box. The service runs the frozen version and returns answers plus
  slimmed traces.
- **Speed is open.** Earlier measurements: about 1 s of brain per answer on the
  CPU, before optimisation. The cut brain is about 6× faster per batch; the
  server number is not measured yet. Latency is about 1 to 2 s, so the UI should
  make the wait legible, with the brain replay as the thing to watch, not a
  spinner.

## UI shape Nick has described

A two-panel chat, like Claude or ChatGPT, plus a slim **brain panel** on the
right showing the neurons as he decides. The questions are structured: pick a
type, the options or levels, and what he smells. It is not free chat. Nick
designs the look in Claude Design; follow the brief for the look.

## What exists and what doesn't

- **Exists:** the brain model and training code (Python, `src/bosco/`,
  `scripts/`), trained weights for internal v0 (A5), and the docs listed here.
  The repo is `~/projects/bosco`, branch `v2`, with nothing pushed. `main` is
  the archived v1: never touch it.
- **Does not exist yet:** the Bosco service (the HTTP wrapper around the
  brain), the trace format, the web app, OAuth, deployment, and the
  `taught`/calibration machinery for serving.
- It is reasonable to build Halteres against a **mock service** that follows
  API.md's shapes, with fake traces, until the service exists. Label mock data
  as mock everywhere it shows.

## Honesty rules that apply to the UI copy

- Say what is true: a real fly connectome decides, and frozen encoders are his
  nose and eyes. Do not call it "the fly understands English".
- Mention the encoders on an about or model-card page. Nick accepts that some
  will say they make it less pure; hiding them would be worse.
- Keep published numbers next to their controls: shuffled brains and plain
  networks score close to the real wiring. Never claim the wiring is proven
  better.
- Never show fabricated accuracy numbers, testimonials or demo results as real.

## Who to ask

Nick decides product and scope. The brain session (this one) owns the model,
the gates and the service interface. If Halteres needs something from the
service that API.md does not give, write it down and ask. Do not change the
brain repo's model code.
