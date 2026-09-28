# Audit 2026-09-28: locked decisions against the code

Read-only, adversarial. Every row was checked in code, not in docs. Line numbers are as of HEAD `cc31052`
(2026-09-28 06:50). That commit deleted `runs/`, every results doc and `registry/specialists/` while this audit was
running. Deleted files are cited as `789f77e:<path>`. No trained specialist weights exist any more (`service/` holds
only `families/v1`). So the "consequence" column answers one question: **would a build that follows the decisions
give the specialists different inputs or weights?** "Retrain" means yes.

Status: **DONE**, **PARTIAL**, **NOT BUILT** (nothing in code), **CONTRADICTED** (the code does the opposite, or
later work reversed the decision without a written amendment).

## 1. Decision table

### CLAUDE.md non-negotiables

| # | decision | what it requires | what the code does | status | consequence for trained specialists |
|---|---|---|---|---|---|
| N1 | A model may perceive, never decide; one frozen embedding model; no LLM | the encoder only translates; nothing learned outside the fly decides | nomic-embed-text only (`service/core.py:69-80`), then a fixed antenna. But the answer's **read scale `log_k` and offset `c` are trained on labels** (`ratebrain3.py:118-119, 239`) and are a linear readout outside the brain. `c` cancels in a softmax; `k` works as a learned temperature. The calibration temperature is also fit on labels (`v1_gate2.py:299-307`); BRIEF allows that ("a gauge outside the fly"). | PARTIAL | small: k would be fixed or type-level |
| N2 | Every pre-registration opens with an anatomy check (which real circuits carry the task, are they in the model, what stands in) and a leakage check | a real anatomy section per gate | SPECIALIST-GATE-3.md has **no anatomy section** ("the same as gate 2", line 3). GATE-4 lines 15-24 say "the fly's real circuits for these judgements are **not claimed**". GATE-2 and 4B describe the harness, not circuits. **No gate after the 09-25 decisions mentions decision 2, 3 or 11**, which is the check that would have caught the single-smell input. | PARTIAL / CONTRADICTED in substance | the anatomy check would have forced the question/item and eyes build before gates 2–4 |
| N3 | Training only: per-cell-type gain, threshold and τ, plus KC→MBON; per-neuron only as a stated comparison arm | nothing else trained on labels | `mode="type"` by default (`ratebrain3.py:108-117`), KC→MBON `kp_logm`/`kp_bank` (`:106, :162-165`). **Also trained: `log_k`, `c`** (N1). The **per-neuron comparison arm (decision 09-24 #1) has not run since A5**: `v1_pilot.py` and `v1_gate2.py` build `mode="type"` only. | PARTIAL | the per-neuron arm is missing, not a change to the weights |
| N4 | Controls run first; "a result without them is not a result"; never drop a control arm | shuffle and random-projection arms in the same script as the real brain | Gates 2, 3, 4, 4b and harm-dev build **only `"real"`** (`v1_gate2.py:148-154`). `v1.ARMS` keeps `hash` (`v1.py:23`), but **hash has never run on the v1 brain**; the pilot ran layered only (`v1_pilot.py:417`). Decision 22 allows this per specialist, but **CLAUDE.md was never amended**, so its "never drop a control arm" rule stands as written. The shipped recipe (design B, 10k–33k items, every option trained) was **never compared** with a shuffle. The pilot's recipe was (3k items, 4 epochs). | CONTRADICTED (unamended rule) | every shipped number stands without a control on its own recipe |
| N5 | Deterministic and replayable: seeded; a run reproducible from its config and seed | bitwise training replay, or a stated exception | Training runs on CUDA/MPS (`device.py:10-17`) with **no deterministic flags**. The code audit (`audit-2026-09-25/code.md` §Determinism) found the backward pass not bitwise even on the CPU. Only scoring sets `use_deterministic_algorithms` (`v1_gate2.py:290`). Nowhere is it written that training is exempt. | PARTIAL | a retrain will not reproduce the weights bit for bit |
| N6 | No v1 setting without a v2 reason written next to it | a reason per carried setting | Still unexplained: the logit ×10 (`ratebrain3.py:239`); `TAU_RANGE` clamp; `KC_PENALTY=10` (`v1_pilot.py:31`); `EYE_PCS`/`VIS_FAN` 6→3 (`senses.py:27-28`); **behaviour groups read from v1's account-action config** (`model2.py:159-165`: engage/like/leave/groom, "frozen at freeze-v1"). The last two were code-audit finding I1 and are **unfixed**. | PARTIAL | none to the weights; the behaviour record is v1's |
| N7 | CLAUDE.md "Model" section describes the model | an accurate description | It still says "LIF neurons … time step 0.1 ms", "three-factor rule", "optic lobes and VNC dropped" (CLAUDE.md:63-72). The model is a gradient-trained **rate** model, 5 ms × 80 steps. README.md says the same and points to `scripts/decider.py`, the B-era decider with `POST /reward`, which contradicts "never taught through the API". Claims-audit C4/C5/I3 flagged these; **none were fixed** (BRIEF.md:15-18, 35-43 still say CLIP, 52 glomeruli, MBON read, "its own learning rule"). | CONTRADICTED (stale) | none |
| N8 | Every gate written down before it runs | pre-registration first | GATE-4's harm experts **reuse checkpoints trained in harm-dev** (commit `f4afbc4`) **before** GATE-4 was drafted (`42bdab4`). They were trained under a development heading, and their validation numbers were seen. No sealed test was touched. | PARTIAL | the harm specialists were trained before their gate existed |

### Decisions 2026-09-24

| # | decision | requires | code | status | consequence |
|---|---|---|---|---|---|
| 24-1 | Per-type main arm; **per-neuron comparison arm** | both arms measured | main: DONE. Comparison: not run on v1 (N3) | PARTIAL | — |
| 24-2 | 9,201 VPNs in the model; pictures through a fixed, label-free eyes mapping | VPNs present and fed | VPNs in (`model2.py:28`), stand-in eyes built (`senses.py:79-89`). **No gate or service since A5 feeds a picture**: `sight` is optional and never passed (`v1_pilot.py:106-107`, `v1_gate2.py:104-105`, `core.py:106-108`) | PARTIAL (built, unused) | the 9,201 VPNs sit at rest in every specialist |
| 24-3a | Answer from MBONs grouped by measured DAN input | — | superseded by 25-4 (DN read); `model2.mbon_groups` still defines the stage | DONE / superseded | — |
| 24-3b | Behaviour neurons recorded in every trace; **agreement with the answer reported** | a behaviour record plus an agreement number per gate | `RateBrain3.behaviour` exists (`ratebrain3.py:133`) but reads v1's account groups (N6). **No gate after A5 reports agreement; the service trace has no behaviour field** (`core.py:230-235`) | NOT BUILT | — |
| 24-4 | Cutoff ≥ 5 synapses, every KC→MBON kept | as stated | `v1.cut` (`v1.py:35-45`) | DONE | — |
| 24-C | BRIEF amended: trained offline, never taught through the API | no reward endpoint | service has none. README still advertises `decider.py serve` with `POST /reward` | PARTIAL (stale entry point) | — |

### Decisions 2026-09-25 (1–39)

| # | decision | requires | code | status | consequence |
|---|---|---|---|---|---|
| 1 | Gradient training is fine; **stated plainly on the model card** | card says "he learns by gradient descent, not by dopamine" | card fields (`export_family.py:236-256`; `789f77e:registry/specialists/*/…json`) carry no such statement | PARTIAL (card line missing) | none |
| 2 | **Optic lobes are part of The Model**; nothing deferred to "later" | `ol_*` in the brain | `model2.V2_SUPERCLASSES` excludes the optic lobes (`model2.py:28`), and its docstring still says "Doom phase" (`:5-6`). **Every specialist from the pilot through gate 4b ran without them, with no amendment**, until decision 39 (09-28) withdrew them for speed | CONTRADICTED 09-25→09-28, then amended by 39 | under 39: no change. The docs still promise them: SERVICE-R10 "144,000 neurons", PLAN-V1 A6, REGISTRY "a new family (eyes …)" |
| 3 | **Question and item apart, through different senses** | two input channels | **Never built.** Every sniff is `clip(item + option − 0.5)` on the same 46 glomeruli: `a4b_dev.py:315` (`fly_sniffs`), used by `v1_pilot.py:92,106,122`, `v1_gate2.py:95-96`, and served in `core.py:153-154`. That is the "mashed question" of code-audit C4/I2, **carried into v1 under a new name**. PLAN-V1 ("Specialists need no question channel … parked") parked it without Nick. The service also **drops the question text**: `Question` has no `instructions` field (`app.py:46-49`) and pydantic ignores unknown fields, so a caller's question is silently discarded | **CONTRADICTED** | **Retrain all.** Design-A inputs change for every specialist (topic, junk, hate, politeness, kind, food, danger and the 13 gate-4 experts). Design B (support, language) takes no option word, but a question channel would still change its input |
| 4 | Answer read from DNs; MBON groups shown as a stage in the trace | DN read; MBON stage in the trace | DN read DONE (`v1.dn_groups` `v1.py:129-155`, `set_dn_read`). But: the groups come from a 2-hop signed-effect threshold (1e-4, no reason given) and are "**not yet checked against the behavioural literature**" (`789f77e:docs/v1-preflight-results.md`). DNa02 and DNa03 (steering) are labelled "avoid". The MBON stage is **not a named trace series**: `named` holds DN groups only (`core.py:233`). The PLAN-V1 A5 risk (LH input 5–10× the MB's; DN read may follow innate paths) has **no MB-silenced control** | PARTIAL | a literature check could change the groups → retrain |
| 5 | Rate model stays; speed over accuracy, but accuracy loss measured | measurement per speed change | cutoff measured (24-4); CSR fold "identical picks" (SERVICE.md) | DONE | — |
| 6 | A5 start rule and **mashed question** fixed in the rebuild | new start; mash removed | start: `homeostatic_start` (`v1.py:93-122`) replaces gain 8. Mash: **not removed**, only renamed (see 3) | PARTIAL / CONTRADICTED | retrain (as 3) |
| 7 | Serving and evaluation on the CPU; published = served numbers, exactly repeatable; **Kimsufi speed gate** | CPU scoring; the served pipeline end to end; a Kimsufi benchmark | Brain scoring on the CPU: DONE (`v1_gate2.py:287-307`). But: (a) **test embeddings came from the GPU**: `v1_data.cmd_embed` defaults to `mps` (`v1_data.py:143`, used by `v1_gate2_data.py:135`; `v1_gate3_data.py:101`; `v1_gate4_harm_tone.py:679`), while the service embeds on the CPU (`core.py:74`), so published ≠ served; (b) gates 2 and 3 took the temperature from **GPU** validation logits (`v1_gate2.py:306-307`; that cost intent its ECE); (c) the parity test checks 12 items per task at `abs=2e-3` (`tests/test_service_parity.py:15,39`), not bitwise; (d) **Kimsufi: not measured, no benchmark script** (SERVICE.md "Not yet") | PARTIAL; Kimsufi NOT BUILT | numbers change slightly; weights do not |
| 8 | Faithful but extended: each extension points at something real and is **written down before use** | a written reason per extension | Written: antenna, design B (27). Unwritten: the 0.5 constant drive on all 46 glomeruli as "rest" (`core.py:39`, `export_family.py:172`), the trained read scale, and the option-word-as-smell T-maze as a fly analogue | PARTIAL | — |
| 9 | No pair encoder as a sense | encoder sees one text at a time | nomic encodes item and option word separately (`core.py:160-163`) | DONE | — |
| 10 | Provenance ledger per dataset and **per encoder** before training | a ledger | dataset manifests exist (`docs/clean-data-manifest.json`, `gate4-data-manifest-*.json`). Encoder: a docstring plus the provenance audit (`encoders.py:8-11`), no ledger entry. **FlyWire annotations still fetched from GitHub `main`** (unlicensed, unpinned; `ops/fetch_data.sh:10-11`), though the audit said Zenodo | PARTIAL | an unpinned annotation table can silently change neuron types, DN groups and regions |
| 11 | **Eyes option (c)**: real eyes, photoreceptors → optic lobes → VPNs, with the encoder at the optic-lobe handoff; (b) stand-in as fallback | optic lobes, a picture pipeline, a vision encoder | **Nothing built.** No optic lobes (2); `encoders.VISION` is pinned (`encoders.py:21-26`) but **never loaded by any v1 code**. The stand-in (b) is not used either: no gate trains on pictures, and the API has no `state.image` or `/v1/uploads` (`app.py:52-53`). Anatomy-audit fixes are **undone**: ocellar OCG cells still driven (`senses.py:81` takes all `visual_projection`); 30 misfiled visual types not driven; the **VPN→DN bypass control** (PLAN-V1 A6) does not exist | **NOT BUILT** | under 39, fallback (b) is the eyes; any picture specialist needs a new family |
| 12 | Own encoders allowed (route to clean provenance) | track C2 on the 3080 | no code, script or doc of results | NOT BUILT | a family change → retrain all |
| 13 | Self-hosting ships MaleCNS attribution + licence link + **list of changes**, the **Shiu MIT notice**, only commercially redistributable encoders/weights | a bundle builder with NOTICE/LICENSE | **No bundle builder anywhere** (grep: no NOTICE/bundle code in `src/`, `scripts/`). REGISTRY.md:32-33 says bundles "are built" and include NOTICE, which is **false**. jina-clip dropped: DONE | NOT BUILT (doc overclaims) | none |
| 14 | nomic text + **vision**, pinned; our own in parallel; **encoder gate before launch** | both pinned and in use; a gate | text: DONE (`encoders.py:14-20`). Vision: pinned, unused. Parallel own encoders: none. Encoder gate: none. PLAN-V1 C1 "re-fit the eyes on them": not done | PARTIAL | — |
| 15 | Fleet of 15–20 at launch | — | 8 exported (now deleted); gate 4 halted | PARTIAL | — |
| 16 | Teach-your-own via the fly's local rule on KC→MBON, **measured in the specialist pilot** against gradient | a local-rule arm and a few-shot arm (5/10/20, PLAN-V1 D) in the pilot | the pilot pre-registration moved both to "Next pilot, not this one" (`SPECIALIST-PILOT.md`, Rules). **No decision amendment.** No code | CONTRADICTED (silently deferred) / NOT BUILT | none to the public specialists |
| 17 | Nick's server runs only our specialists | loader restricted | `Fleet` loads only `status: shipped` with a matching sha256 (`core.py:250-265`) | DONE | — |
| 18 | Halteres SaaS + self-hosted Bosco | a self-host package | service code only; no package or bundle (13) | PARTIAL | — |
| 19 | No community specialists | none | none | DONE | — |
| 21 | Personality is the playback: the trace names real cell types (glomeruli, KCs, DANs, LH, DNs such as MDN) | named cells in the trace | `top` gives the 3 most-changed cells per region with type (`core.py:220-229`). **Glomeruli and ORNs are excluded** from the narration (`NARRATE`, `core.py:28`), though decision 21 names "the glomeruli". No MBON stage series (see 4). `kc_active` (Halteres R2) is absent | PARTIAL | — |
| 22 | No shuffled twin per specialist; **the fair real-vs-shuffle taken once for the family**, and again when the brain changes; a model-card footnote | a family-level comparison on the family as served; a footnote on the cards | taken once, in the pilot (layered only; hash never run), on the pilot recipe, not the shipped recipe. **No card carries the footnote**: the `evidence.report_only` field has plain, nose and prototype only (`export_family.py:214`). CLAUDE.md not amended (N4) | PARTIAL | — |
| 23 | Bar 0.80 + ECE ≤ 0.10; disputed labels 0.9 × ceiling, flagged before test | in the runner | `v1_gate2.py:31-43, 403`; `export_family.gate_verdict` re-checks | DONE | — |
| 24 | Caller combines; nothing automatic picks | no router | none; each question names its specialist (`app.py:79-88`) | DONE | — |
| 26 | DynaSent round 2 only; prompts never read | builder | `v1_gate2_data.py:83-93` (round02 only; `sentence`/`gold_label`) | DONE (mood not shipped) | — |
| 27 | Many options: one memory per option (way B) | design B | `enable_bank`, `logits_B` (`ratebrain3.py:162-165`, `v1_gate2.py:99-107`) | DONE | — |
| 28 | Licence per specialist; BY-SA manifest with revision URLs; no training text shipped; NOTICE credits | card licence + attribution file + NOTICE | card `licence` and `attribution_file` DONE (`export_family.py:38-139, 257-268`). NOTICE credits: no builder (13). **The politeness card was typed "Stack Exchange text CC BY-SA" but carries no attribution manifest** (only the wiki cards get one, `:257`), which decision 28 asks for on every BY-SA source. The **topic** card (DBpedia BY-SA) has none either | PARTIAL | none to weights |
| 29 | Generalists in their fields: taxonomy first, pooled sources, **breadth test**; today's 8 **cards say they are single-source** | breadth in gates; a card line | gate 4 held-out sources: DONE (SPECIALIST-GATE-4.md). **The 8 cards carry no single-source statement** (their `limits` checked in `789f77e:registry/specialists/*`) | PARTIAL | — |
| 30 | Civil Comments accepted; **two ethics gaps on every card trained on it** | card text | `export_family.SPECIALISTS` has no gate-4 or 4b entries at all (`export_family.py:51-139`), so the gap wording exists nowhere in code | NOT BUILT | — |
| 31 | Stack Exchange: 2024-04 dump, posts < 2022-11; LLM-terms note on every card | filter + card note | filter DONE (`v1_gate4_intent_finance.py:120, 1629`). Card note: no gate-4 cards; the politeness card (SE text via ConvoKit) has no note | PARTIAL | — |
| 32 | Language: 30–40 languages, out-of-scope scripts named on the card | card text | gate 4 language task built (36 options); no card text | PARTIAL | — |
| 33 | Live-source text < 2022-11-01; cards say "text up to 2022" | filters + card line | filters DONE (`v1_gate4_topic_language_danger.py:45-46`, `v1_gate4_intent_finance.py:120`, gate 3 fetch). Card line: only inside the Wikipedia source name | PARTIAL | — |
| 34 | Calibration first-class; components with a **component bar pre-registered with Gate 4** | a bar in GATE-4 | GATE-4 contains no component bar (grep "component": none); tone was dropped by 36, but 34 required the bar in Gate 4 | NOT BUILT | — |
| 35 | Harm = content moderation; separate experts; **tested across platforms with one source held back at a time**; HateCheck unscored | a rotating hold-out | one fixed held-out source (Aegis) for every harm expert (GATE-4 table); no rotation. HateCheck fetched (`v1_gate4_harm_tone.py:95-96`), no scoring found | PARTIAL | the breadth claim rests on one platform |
| 36 | Sharp yes/no questions | gate-4 task list | `v1_gate2.py:36-43` matches | DONE | — |
| 37 | Jev parity: score type, several questions per request, batched options | service | `rate` and ≤16 questions, one pass (`core.py:158-201`, `app.py:27`). But **`rate` accepts only the specialist's own taught options, reordered** (`core.py:143-144`), not 2–10 descriptive levels; `approach` works only on 2-option specialists and is a two-sniff T-maze (see API rows) | PARTIAL | — |
| 38 | Every sense with something honest to carry; taste = form; **first an anatomy check** (gustatory reach within 80 steps) | anatomy check, then a taste spike | none (dated 09-28; written as "after gate 4") | NOT BUILT (new) | future family change |
| 39 | No added neurons; timed sniffs → taste → **question and item apart** → wake idle senses, each tested on dev | four dev tests | none. The ordering puts decision 3 **third**, after two new ideas, although 39 itself says decision 3 "stands" | NOT BUILT (new) | each is a new family → retrain all |
| Open | Labelling budget (sentiment) | Nick's call | open; no mood/sentiment specialist | OPEN (correctly) | — |
| Open | Where the question enters | anatomy-based choice | anatomy audit §2 answered it (only vision reaches KCs; question × item meets only at γ MBONs). **No decision recorded since**; PLAN-V1 parked it | OPEN, drifted to "parked" | — |
| Open | Plasticity beyond KC→MBON (PN→KC) | a proposal with evidence, then a CLAUDE.md amendment | no code; `kp_*` only | NOT BUILT (correctly held) | — |

### PLAN-V1 (track items not already above)

| item | code | status |
|---|---|---|
| A1 graded unit | `ratebrain3.py:187-188` | DONE |
| A2 start: KCs 2–10%, **input similarity kept (fly_init_keep)**, DNs in a working range, **per-region liveness in the preflight** | homeostatic start has **no similarity criterion** (`v1.py:93-122`); liveness is computed but **not a preflight check** (`v1_gate2.py:207-213`) | PARTIAL |
| A3 controls rebuilt fairly | `v1.build` with its own input totals and duplicates kept (`v1.py:48-52`) | DONE (hash never run) |
| A4 sparseness on the fraction, "with a v2 reason for its size" | fraction: DONE (`ratebrain3.py:252-257`); size 10.0 has no reason | PARTIAL |
| A5 "run grows until the DNs settle, measured" | STEPS 80 / READ 8 fixed (`v1_pilot.py:32`); no settle measurement on v1 recorded | PARTIAL |
| A6 eyes; ocellar fix; misfiled types; MB-silenced picture control | none | NOT BUILT |
| A7 skip never-firing neurons (exact); CSR; batch; lazy record; **Kimsufi gate** | CSR and batch DONE (`core.py:55-59, 101-129`); lazy record partial (`RECORD_ALL`); skip-neurons and Kimsufi NOT BUILT | PARTIAL |
| A8 no sealed data in smoke runs | refusals at `v1_gate2.py:288-289`, `v1_pilot.py:426-427` | DONE |
| B pictures from commercially licensed images; **FlyWire from Zenodo** | none; still GitHub `main` | NOT BUILT |
| D few-shot arm (5/10/20) | none | NOT BUILT |
| E `/v1/decide`, `/v1/traces`, `/v1/models`, per-specialist versions, 422 not_taught | `app.py` | DONE |
| Code audit I4 τ clamp kills its gradient (use a smooth bound) | `ratebrain3.py:199` still `.clamp(*TAU_RANGE)` | NOT FIXED (retrain) |
| Code audit M5 hard-coded channel counts | `core.py:39` `np.full(46, …)`; `:135` | NOT FIXED |
| Code audit M7 wiring not in the state/hash | the card pins the neuron order and antenna only (`export_family.py:244-245`); **no connectome/sign/cut hash**; `mcns_v2.npz` is rebuilt silently if missing (`model2.py:109-114`) | NOT FIXED |
| Anatomy audit §5.6: state glutamate = −1 on the card (it sets DN approach/avoid meaning) | not on any card | NOT BUILT |
| Anatomy audit §5.10: R1–R6 9%/17% and OL asymmetry on the card | moot under 39 | — |

### API.md / SERVICE-R10 / BRIEF

| item | requires | code | status |
|---|---|---|---|
| `state.text`, **`state.image`**, both; objects/arrays flattened | three input shapes | `State.text: str` only (`app.py:52-53`); image silently ignored | NOT BUILT |
| **`smell` path parameter** | point the nose at part of the state | absent; pydantic drops it silently (`app.py:46-49`) | NOT BUILT |
| `instructions`, `toward`, `away` | accepted (instructions flattened to text) | absent, silently dropped | NOT BUILT; conflicts with decision 3 |
| Types approach / choose / rate / **familiar** | four types | three (`app.py:81`); `familiar` absent (SERVICE.md "Not yet") | PARTIAL |
| `approach` = one sniff, yes = he approaches; `lean` ∈ [−1, 1] = approach − avoid output | one-sniff DN lean | a **two-sniff softmax over option words** ("threat" vs "not a threat"); `lean` = difference of two scaled logits, unbounded (`core.py:189-193`) | CONTRADICTED |
| `rate`: 2–10 descriptive levels, each smelled | caller-supplied levels | only the taught options, reordered (`core.py:143-144`) | PARTIAL |
| `sure` formula | clamp((n·p−1)/(n−1)) | `core.py:186` | DONE |
| determinism "verified on the server's CPU and stated per version" | a Kimsufi check; a card field | Mac only; batch composition changes answers below 1e-7 (code audit), and the service batches every question of a request together (`core.py:165-167`) | NOT BUILT |
| calibration: **ECE and reliability per family, published per version** | ECE plus reliability curves | ECE on the card; no reliability data | PARTIAL |
| **limits page per version** | a page | a `limits` list per card; no family limits page (e.g. two-text relations, one-sense input) | PARTIAL |
| models endpoint card: training questions, calibration, limits, **wiring**, controls; R10 family: `build_id`, **`regions`**, DN groups | fields | `family.json` has no `regions`, `build_id` or connectome hash (`export_family.py:169-175`); cards have no controls or wiring | PARTIAL |
| errors: 400/422 name the field; **429 with retry-after; 503 busy** | codes | 422/404 name the field; no 429 or 503 (SERVICE.md defers them to Ask Bosco) | PARTIAL, undocumented deviation from API.md |
| `x-bosco-request-id` | header | `app.py:63-68` | DONE |
| `usage.sniffs` | count | `app.py:105` | DONE |
| `timing_ms` senses / brain | split | `total` only (`app.py:112`) | PARTIAL |
| response names its version | resolved version | top-level `version` echoes the request (may be null); per-answer `specialist` is resolved (`app.py:106-108`) | PARTIAL |
| `taught` / `422 not_taught` / `answered_as` | R10 §5 | `core.py:139-156`, `app.py:91-96` | DONE |
| Halteres R3 uploads, R4 NDJSON, R2 kc_active, R7 familiar | per SERVICE-REQUESTS | none | NOT BUILT |
| "Answering never changes him" | no learning at inference | `torch.no_grad`, no state writes (`core.py:118`) | DONE |
| Claims-audit corrections (C4 "one frozen embedding model", C5 MBON read, C6 "1 s measured", I1 determinism promise, I3 "its own learning rule") | docs corrected | API.md:18-19, 27, 132-133; BRIEF.md:15-18, 35-43; CLAUDE.md:34-36, 63-72; HANDOFF-HALTERES.md:63, 70, 161: **all unchanged** | NOT FIXED |

### Memory notes

| note | requires | reality | status |
|---|---|---|---|
| feedback-encoding-is-the-novelty: "treat the encoding as the first design question … test the encoding levers before breadth or tuning" | encoding levers (question channel, senses) before a fleet | three gates of breadth (2, 3, 4) plus 4b on the unchanged single-smell encoding; the levers (39) are listed only now | CONTRADICTED |
| feedback-audit-anatomy: an anatomy section in every pre-registration | per gate | see N2 | PARTIAL |
| feedback-corroborate | outside facts verified | not assessed in code | — |
| halteres-app divergences: the UI reads the MBON lean; the trace assumed 40 steps | the service tells Halteres | family.json gives steps=80; the MBON stage is not exposed, so a UI that reads `mbon_approach − mbon_avoid` has nothing to read | PARTIAL |

## 2. Contradictions and silent drift

1. **Decision 3 (question and item apart) was reversed by PLAN-V1 the same day it was locked.** PLAN-V1 says
   "Specialists need no question channel … the two-sense question channel is parked". PLAN-V1 calls itself "a plan,
   not a pre-registration"; it is not an amendment and Nick did not sign it. Every later pre-registration
   (SPECIALIST-PILOT "the question is fixed per specialist, so it needs no channel of its own", GATE-2, -3, -4, -4B)
   builds on that parking. The T-maze `clip(item + option − 0.5)` (`a4b_dev.py:315`) is the same mechanism code-audit
   I2 had named as contradicting decision 3. It was renamed "design A", not removed. Decision 6 ("the mashed question
   is fixed in the rebuild") is therefore also unmet.
2. **Decision 2 (optic lobes in The Model; no deferral to "later") was ignored for three days of gates.**
   `model2.py:5-6` still says "The optic lobes are still out (Doom phase)". Decision 37 withdrew Doom, and the reason
   in the code went with it. No gate between 09-25 and 09-28 mentions the optic lobes. Decision 39 amended decision 2
   only on 09-28, after gate 4 had been built and 13 of its sealed tests scored.
3. **Decision 11 (eyes, option c) and the stand-in fallback (b) are both absent from every v1 build.** The vision
   encoder was pinned (decision 14) and never loaded. The API dropped `state.image` without a written change.
   PLAN-V1's "Order" put A6 (optic lobes) before D (the pilot), but D ran first and A6 never started.
4. **Decision 16 (local rule measured in the specialist pilot)** was moved to "next pilot, not this one" inside
   SPECIALIST-PILOT.md. That is not a decision amendment. PLAN-V1 track D's few-shot arm went the same way.
5. **CLAUDE.md "controls run first … never drop a control arm"** stands unamended, while decision 22 and every gate
   since run the real brain only. The hash control was kept in `v1.ARMS` but has never run on v1.
6. **The API contract changed without an amendment to API.md.** `approach` became a two-arm T-maze; `lean` became an
   unbounded logit difference; `rate` lost free descriptive levels; `smell`, `instructions`, `toward`, `away` and
   `image` were dropped. The pydantic models **silently ignore** these fields, so a caller following `docs/API.md`
   gets an answer to a question it never asked, with no error.
7. **Decision 4's "MBON groups become one stage shown in the trace"** was dropped from the trace, and the DN groups
   went into service unchecked against the literature ("not yet checked", v1-preflight). DNa02 and DNa03 (steering)
   count as "avoid".
8. **"Published numbers are the served numbers" (7)** does not hold. Gate test embeddings were made on MPS, and gate
   2/3 temperatures came from GPU logits. The service does both on the CPU. The parity test is 12 items at 2e-3.
9. **Decision 09-24 #3's behaviour record and agreement number** vanished after A5. `RateBrain3` still loads
   v1's **account-action** groups (`config/readout_populations.yaml`: engage/like/leave/groom) as "behaviour". That is a
   v1 setting with no v2 reason.
10. **REGISTRY.md states that self-host bundles "are built" with LICENSE and NOTICE files.** No such code exists.
    Decision 13 (MaleCNS change list, Shiu MIT notice) is unmet.
11. **The earlier audit's fixes were only partly applied.** Fixed: the controls' normalisation, liveness, the
    sparseness penalty, log_k, smoke/sealed. Not fixed: τ clamp (I4), logit ×10 (I1), behaviour groups (I1), OCG
    ocellar drive (anatomy §5.1), the 30 misfiled visual types (§5.3), the VPN→DN bypass control (§5.2), the
    glutamate card note (§5.6), the wiring hash (M7), hard-coded channel counts (M5), FlyWire from Zenodo
    (provenance §6), and every claims-audit wording fix (BRIEF, API, CLAUDE.md, README, HANDOFF).
12. **Process: the harm experts were trained (harm-dev) before GATE-4 existed**, and GATE-4 adopted those
    checkpoints. It is disclosed, but it inverts "written down first".
13. **Decision 34's component bar "pre-registered with Gate 4"** is missing from GATE-4.
14. **Decision 35's "one source held back at a time"** became one fixed held-out source for all harm experts.
15. **Decision 29's card line ("each trained on one source")** and **decision 1's card line ("he learns by gradient
    descent")** were never written onto the 8 cards that shipped.
16. **The "Open: where the question enters" item was answered by the anatomy audit** (only vision has a KC route;
    question × item can meet only at γ MBONs or downstream). No decision followed; PLAN-V1 parked it instead.

## 3. Top 10 by consequence

1. **Decision 3 never built; the question is dropped.** Every specialist smells `item + option word` on one sense
   (`a4b_dev.py:315`, `core.py:153-154`). The API silently discards `instructions` (`app.py:46-49`). Fixing it
   changes the input of every specialist: retrain all.
2. **Eyes (decision 11) absent end to end.** There are no optic lobes, no vision encoder in use, no picture in any
   gate, and no `state.image` in the API. Even the stand-in drives the ocellar OCG cells, and the VPN→DN bypass has
   no control.
3. **Decision 2 contradicted for three days of gates (2, 3, 4, 4b)** with no amendment. The code still gives "Doom"
   as the reason. Decision 39 amended it after the fact. SERVICE-R10 and REGISTRY still promise the optic lobes.
4. **Controls dropped against an unamended CLAUDE.md.** No shuffle ran on the shipped recipe; hash never ran on v1;
   no card carries the decision-22 footnote.
5. **The API contract broken without amendment:** `approach` is a two-option T-maze, `lean` is not approach − avoid
   in [−1, 1], `rate` takes no custom levels, and `smell`, `familiar`, `image`, 429/503 are missing. Unknown fields
   are silently ignored.
6. **"Published = served" does not hold (decision 7).** Test embeddings came from MPS and gate 2/3 temperatures from
   GPU logits. Parity is checked on 12 items at 2e-3. There is no Kimsufi speed gate and no server-CPU determinism
   check.
7. **The DN read (decision 4) is shipped unchecked:** groups from an arbitrary 1e-4 threshold, "not checked against
   the literature", steering DNs labelled avoid, no MB-silenced control, and the MBON stage missing from the trace. A
   literature-based grouping means retrain all.
8. **Decision 16 (teach-your-own by the local rule) and PLAN-V1's few-shot arm were deferred silently** in the pilot
   pre-registration.
9. **Self-host and licence obligations are unbuilt while a doc says they are built:** no bundle builder and no
   NOTICE, no Shiu MIT notice, no MaleCNS change list (13, 28). BY-SA topic and politeness have no attribution
   manifest. FlyWire annotations still come from unpinned, unlicensed GitHub `main`.
10. **Earlier audit fixes left undone, and docs left false:** τ clamp, ×10, v1 behaviour groups, wiring hash; the
    claims audit's corrections to BRIEF, API, CLAUDE.md, README and HANDOFF were never applied. CLAUDE.md still
    describes a LIF model at 0.1 ms that nothing runs.
