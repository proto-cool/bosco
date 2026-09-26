# Plan: Bosco v1, "The Model" (2026-09-25; a plan, not a pre-registration)

Built on `docs/DECISIONS-2026-09-25.md` (decisions 1–14) and the audit in
`docs/audit-2026-09-25/` (anatomy, code, provenance, claims). Every step with a result
gets its own pre-registration before it runs.

**The product:** a fleet of specialist Boscos (`docs/SERVICE-R10.md`). One family, meaning
the same connectome, encoders and neuron order; each specialist is the same fly with
different learned weights. Served on CPU; watched in Halteres.

## Track A: an awake, honest brain (first; everything else waits on it)

The audit found him 91% silent, his controls mis-scaled, and his sparseness rule inert.

1. **Neurons that can learn from below threshold.** `tanh(relu)` gives no gradient to a
   silent neuron. Replace it with a smooth onset (for example softplus into a
   saturation), so silent cell types still learn. v2 reason: real neurons have graded,
   subthreshold responses. Checked against the old unit on A5-style data.
2. **A start rule that wakes the right places,** label-free:
   - KCs 2–10% active;
   - the antennal lobe and KC layer keep input similarity (`fly_init_keep`);
   - **descending neurons in a working range,** reachable but not saturated.

   The preflight gains a check: the share of each region that is alive.
3. **Controls rebuilt fairly.** Each control is normalised on its *own* input totals and
   has the same number of trainable KC→MBON synapses as the real brain. The old
   real-vs-shuffle comparisons stay withdrawn until rerun.
4. **The sparseness rule fixed:** it penalises the *fraction* of KCs active, the quantity
   the bars measure, with a v2 reason for its size.
5. **The answer from descending neurons** (decision 4).
   - The approach and avoid DN groups are defined from the anatomy audit's routes (MBON
     → DNa02, DNa03, DNb05, MDN, …) and the literature, and fixed before training.
   - The run grows until the DNs settle, measured, not assumed.
   - **Risk, stated now:** the MB is under 2% of most DNs' input and the lateral horn
     5–10× more, so the DN read may follow innate pathways more than learning. The
     preflight tests whether training can move the DN answer at all. If it cannot, that
     is reported, and Nick decides.
6. **Eyes, option (c):** the optic lobes go in (about 144k neurons in all). Real eyes
   start at the first optic-lobe cells (L1–L3), since R1–R6 are only 9–17% traced.
   The picture translator feeds in at the VPN handoff. Fixes from the audit: stop
   driving ocellar neurons; drive the 30 misfiled visual types.
   - Control: a picture-to-DN path that bypasses the MB (looming and object VPNs →
     DNs) is measured with the MB silenced.
7. **CPU first.**
   - Serving and evaluation run on the CPU with deterministic algorithms.
   - Speed-ups: skip never-firing neurons (exact), use a CSR sparse format, batch within
     a specialist, record lazily.
   - **A speed gate on the Kimsufi**, which needs SSH access from Nick.
   - Training stays on the 3080, then gets re-scored on the CPU.
8. **No sealed data in smoke runs.** Smoke runs score only training and validation data.

**Specialists need no question channel.** Each specialist answers one fixed question,
so only "the thing" comes in, as text on the nose and pictures through the eyes. Where
the anatomy says question and thing would only *add* at the γ MBONs, the fleet design
sidesteps the problem. The two-sense question channel is parked unless a
multi-question specialist needs it.

## Track B: clean data (parallel)

- The candidate specialists are only those with ledger-clean data, fetched from the
  **original sources** (not tasksource):
  - hate: DynaHate, HatemojiBuild;
  - toxicity: Civil Comments;
  - topic: DBpedia-14;
  - intent: SNIPS, MASSIVE, CLINC150;
  - yes/no over a passage: BoolQ;
  - relation: MultiNLI.

  Each one's licence and collection are re-verified at the source, and its row in the
  ledger is completed.
- **Pictures:** commercially licensed images, such as Wikimedia Commons by licence.
- **Sentiment has no clean set.** Options: commissioned, paid, consented labelling of
  openly licensed text, or no sentiment specialist at launch. Nick to decide later.
- `ops/fetch_data.sh`: the FlyWire annotations come from Zenodo, not GitHub.

## Track C: encoders (decision 14)

1. **Now:** nomic-embed-text-v1.5 and nomic-embed-vision-v1.5, pinned by commit, in an
   environment where they load (they failed to load in A2; fix that first). Re-fit the
   nose (bi46) and the eyes on them.
2. **In parallel, on the 3080:**
   - our own text encoder on Common Pile (small, contrastive, with no MS MARCO and no
     scraped social media);
   - our own image translator, self-supervised on licensed images.
3. **Encoder gate before launch:** each specialist's score with ours against with nomic.

## Track D: the specialist-fleet pilot (after A, on B and C1)

Pre-registered. For each clean task, one specialist, compared with:
- one shared brain for all the tasks;
- the fairly rebuilt shuffle;
- the plain baseline (embedding plus a small classifier);
- the nose alone.

It is scored on the CPU as served, with balanced accuracy and calibration, and includes a
few-shot arm (5/10/20 examples) for "teach your own". **It decides the launch catalogue:**
a specialist ships only if it clears its bars, and its numbers are published beside the
controls.

## Track E: the service (after A's interfaces settle)

`/v1/decide`, `/v1/traces`, `/v1/models` as in `docs/API.md` and `docs/SERVICE-R10.md`:
- one family and many specialists, with weight swaps;
- per-specialist versions;
- `422 not_taught` outside a specialist's task;
- traces in the family's full neuron order, with steps, dt and regions taken from the
  family;
- model cards with provenance and controls.

Halteres switches from its mock when the pilot reports.

## Later, not blocking v1

- The effort levels, fast and accurate (`docs/EFFORT-PILOT.md`), on the v1 brain.
- `familiar` (the novelty compartment).
- More plasticity sites (for example PN→KC): a proposal with evidence, then a CLAUDE.md
  amendment signed off by Nick.

## Order

A1–A4 and A8 → A5 → A7 (CPU) → A6 (optic lobes; the biggest speed risk, so it is
measured before being committed) → D, with B and C1 ready by then. C2 runs throughout.
E starts once A's interfaces settle.
