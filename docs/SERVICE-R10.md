# Answer to Halteres R10: a fleet of specialists (2026-09-25, provisional)

From the brain session. **Provisional:** the fleet is not validated. It rests on the
specialist pilot, which runs after the brain rebuild (the audit found 91% of neurons
silent and the controls mis-scaled, `docs/ERRATA-2026-09-24.md`). Halteres can build
against these shapes, but no real specialist names or numbers should reach visitors
until the pilot reports.

## 1. What a specialist is, physically

**The same fly with different memories.**
- **Shared by the whole fleet (the "family"):** the connectome build (neurons, connections,
  signs), the neuron order, the encoders (the translators for text and pictures), the time
  step and the number of steps.
- **Per specialist (the learned state):** per-cell-type gain, threshold and time constant;
  the Kenyon-cell-to-output synapses; the read scale; and, after the rebuild, the thresholds
  that set the descending neurons' operating point. That is about 100,000 numbers, about
  400 KB. If more plasticity sites are approved later (for example PN→KC), they are also
  weights inside the same family; the neuron set does not change.
- So **one brain map and one neuron order serve the whole family (R5).** Comparing two
  specialists' traces on the same input is meaningful: the same neurons, different
  activity.
- The service may skip neurons that can never fire, for speed. That is internal: traces
  are always reported in the family's full neuron order.
- **Coming with the rebuild (family v1):**
  - The optic lobes add neurons: about 144,000 instead of 50,140.
  - The run grows to about 64–80 steps of 5 ms instead of 40.
  - **The answer is read from his descending neurons.** The trace gains a descending-neuron
    region, and "what the fly would do" becomes the answer; the MBON lean becomes a stage
    it passes through.

  Halteres should take the step count, dt, neuron count and region list from the family,
  not hard-code them.

## 2. Hosting

- **Yes, many at once.** The family loads once (tens of MB). Each specialist adds about
  0.4 MB, so hundreds fit in memory.
- **Switching is a weight swap,** not a reload: milliseconds.
- **CPU is the limit.** A sniff costs the same whichever specialist runs it. Measured on
  the Mac's CPU (M3 Max, 4 threads, the current 50,140-neuron brain):
  - about 25 ms per sniff when batched, 64 at a time;
  - about 0.5 s for a lone sniff.

  The Kimsufi is not measured. The optic lobes multiply the cost by about 3.5–6× before
  speed-ups, and there are several exact speed-ups (skipping never-firing neurons, a faster
  sparse format, batching).
- **Batching is best within one specialist,** since each has its own weights. The service
  groups queued sniffs by specialist; Halteres does not need a queue per specialist.

## 3. Versions

- **Each specialist has its own dated versions** (`bosco-junk-2026-10-15`) and an alias
  (`junk-latest`).
- **The family has its own id** (connectome build, encoders, neuron-order hash).
  Specialists name their family.
- **A family change** (new connectome build, new encoders, optic lobes) means every
  specialist is retrained and re-versioned, and gets a new brain map.
- **Retire `bosco-latest`.** It means nothing in a fleet.

## 4. Routing

**The caller always names the specialist. There is no `auto`.** Choosing a specialist
for a question is itself a generalist decision, which is exactly what Bosco cannot
honestly make; a router would be an encoder deciding.

## 5. Questions outside a specialist's task: refuse (this corrects today's behaviour)

**Default: `422 not_taught`, naming the specialist's kinds.** In a fleet, `taught: false`
is not an honest answer. A specialist trained on one question largely answers *its own*
question whatever it is asked: the junk specialist asked "is this urgent?" mostly
answers "is this junk?". Showing that as an answer to the question asked, even marked
untaught, would mislead.

- **Owner free-form flag:** the service answers anyway, with `taught: false` and
  `answered_as: "<the specialist's own kind>"`, so the tile can say what he actually judged.
- The existing `taught` flag stays for other gaps, such as new label words for a taught
  kind.

## On the proposed shape

- `GET /v1/models`: yes, one card per specialist, with `specialist`, `family` and `taught`.
  - Add to `family`: `build_id`, `neuron_count`, `steps`, `dt_ms`, `regions`, and the
    descending-neuron groups used for the answer.
  - Add to each card: data provenance (sources and licences, from the ledger) and
    calibration (ECE), beside the numbers and controls.
- `POST /v1/decide`: unchanged; one request, one specialist. Each answer names its
  `version` (specialist and date) and its `family`.
- Traces: unchanged per sniff; each names its `version` and `family`.
- Batch input and side-by-side comparison are honest, because specialists share a
  family. Batch runs well within one specialist.
