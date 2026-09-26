# Priorities for Ask Bosco (from the brain session, 2026-09-25)

Source of truth: `docs/DECISIONS-2026-09-25.md` in the bosco repo (decisions 1–25). Where these differ
from the design brief on the model, the API, honesty or data, these win. Look, flows and tone stay yours.

## 1. What Ask Bosco is
- **The name is "Ask Bosco".** It is our gateway to using Bosco, not a separate brand; "Halteres" can
  stay the repo name.
- **The pitch: "the AI you can watch think, neuron by neuron."** The novelty and the playback are the
  product. Do not call it "explainable AI": it shows which real cells fired, not a reason in words.
- It is a SaaS for using **our** specialists. No training happens in Ask Bosco, no community
  specialists, and no public API keys. Teach-your-own is **self-hosting only**. The hosted Bosco API
  stays private, and Ask Bosco is its only client.

## 2. The playback is the headline (top priority)
- **Name real cells as they fire.** For example: smell arriving in named glomeruli; "5% of Kenyon
  cells recognised it"; the dopamine neurons; the lateral horn's gut reaction; and the answer's
  **descending neurons**, such as **MDN, "the moonwalker", firing as he backs away**, or the gnathal
  (feeding) DNs as he leans in. The service will send cell-type names in the trace.
- **The answer now comes from his descending neurons** (decision 4). "What the fly would do" is no
  longer "never the answer": it **is** the answer. The MBON lean becomes one stage the decision passes
  through. Flip that in the UI and the copy.
- **Take everything about the brain from the service:** step count (v1: 80 steps of 5 ms = 400 ms
  of fly time), neuron count (it grows to about 144,000 when the optic lobes go in), regions and
  cell-type names. Hard-code none of it.
- **A compare view:** one situation, several specialists side by side, each with its own neurons
  lighting up. It is meaningful because every specialist is the same fly with different memories.

## 3. Specialists (the fleet)
- One fixed question each. The caller always names the specialist; nothing routes automatically.
- **Outside its task: `422 not_taught`.** Owner free-form uses `allow_untaught: true` (R10a
  approved) and shows what he actually judged.
- **The production bar:** a specialist appears only if its sealed test, scored on the CPU, has
  balanced accuracy ≥ **0.80** and ECE ≤ 0.10. The exception is a task flagged **disputed labels**,
  whose card states its lower bar and why (hate speech is the one so far).
- **Every model card** shows the bar, calibration, data provenance (sources and licences), the
  controls as a footnote ("a scrambled copy of his wiring learns this about as well"), and
  plain-language limits. Example limit for hate: "catches blatant hate; often flags texts that
  merely mention a group".
- **No real specialist names or numbers in front of visitors** until each has passed. Mock ones stay
  labelled mock.
- **Likely first specialists:**
  - topic (strongest so far)
  - subject, kind, plain language, food, danger (from Wikipedia/Wikidata)
  - support area and in scope (MASSIVE/CLINC)
  - mood (DynaSent round 2), junk (SMS Spam with phone numbers scrubbed), politeness
  - harm (disputed-labels bar)
  - pictures later, after the eyes rebuild

  Plan for about 12–15.

## 4. Composing specialists is how he handles complex problems
- The caller (or the user, in Ask Bosco) combines specialists with **explicit, visible rules**:
  fan-out, composite verdicts, and routing by confidence. For example: bug + angry → top of the
  queue; harm with high confidence → review, never auto-delete.
- A simple **rules builder** over several specialists, plus **batch input** (paste a list or upload
  a CSV, results as a table with a replay per row) makes Ask Bosco a tool, not just a toy.
  "Composable moderation" and triage are the showcase use cases.

## 5. Honesty in the copy
- A real fly connectome (MaleCNS, Janelia FlyEM et al., CC BY 4.0: credit it and don't imply
  endorsement) makes every decision. The encoders (nomic, Apache 2.0) are his translators, his
  "nose" and "eyes"; say so on the model card.
- He is a **computer model** of a real fly's brain, and he learns by gradient descent, not by
  dopamine. Say both plainly.
- Show confidence on every answer, and make low confidence visibly different.
- Harm and moderation answers are for **human review**, never automatic public action.

## 6. Markets to design toward (not yet decided)
- **Education and science exhibits** (museums, schools, universities): the playback and the named
  cells are the whole product there, and accuracy matters less. A "classroom" or "exhibit" mode
  is worth exploring.
- **Workflow gut-checks** (triage and routing) come later, through self-hosted Bosco.

## 7. Not decided yet (don't build toward them)
- Pricing and plans.
- A Bluesky labeler front end (parked until specialists pass).
- `familiar` (novelty) and the effort levels, fast and accurate (a pilot is drafted).
