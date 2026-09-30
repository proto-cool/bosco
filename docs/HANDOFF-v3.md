# Handoff to the v3 session (written 2026-09-28, at the end of v2)

> **CLOSED 2026-09-30 (tag `end-v3`).** Read `docs/END-v3.md` first. The fly's own learning cannot reach his
> behaviour in this connectome model; the verified minimum smell and taste brains are in `src/bosco/minibrain.py` and
> `src/bosco/tastebrain.py`. Do not resume without Nick.

Read this first, then `CLAUDE.md`, `docs/BRAIN-SPEC.md` and `docs/audit-2026-09-28/`. Nick restarts here because
v2 lost two weeks. Most of what went wrong was process, not difficulty. The requirements were clear.

## 1. What went wrong in v2 (so it is not repeated)
- **The fly was not deciding** (`audit-2026-09-28/model.md`):
  - the item and the option word were blended on one nose, so a yes/no's two sniffs differed by a constant and
    about 2% of the signal decided;
  - only about 20% of the Kenyon cells were ever used, because every glomerulus rested at 0.5;
  - every sniff booted from r = 0;
  - the answer rode on rate differences of about 0.001, scaled up about 10⁴;
  - a plain classifier on the encoder matched or beat every specialist, and the scrambled control tied the
    real wiring.

  Three gates were built on top of this.
- **Locked decisions were never built, and nobody checked** (`audit-2026-09-28/decisions.md`):
  - decision 3, question and item through separate senses;
  - decision 11, eyes;
  - controls;
  - decision 16, teach-your-own by the fly's local rule;
  - decision 7, published numbers equal served numbers;
  - the API contract.

  The assistant then re-pitched locked decisions as new ideas.
- **Evaluation holes** (`audit-2026-09-28/evaluation.md`):
  - a rounded bar let politeness through;
  - danger measured "what kind of thing", not danger;
  - three passes were within noise;
  - the near-duplicate drop ignored labels;
  - pilot test items were reused;
  - gate-4 procedural leaks.
- **Overclaims and data gaps** (`audit-2026-09-28/data-claims.md`):
  - "full brain" and "all 50,000 neurons" are false;
  - the encoder's training data (scraped Reddit and Amazon; MS MARCO is non-commercial) was not disclosed;
  - the SMS scrub was incomplete;
  - there is no LICENSE or NOTICE file;
  - the FlyWire annotation table is unlicensed;
  - ConvAbuse was used without a decision.
- **Working habits Nick called out.** Do not repeat any of these:
  - running things without telling him;
  - re-running known-broken baselines;
  - launching overnight runs before a design was proven;
  - reporting "nothing is running" without checking every machine;
  - apologising instead of changing.

**Nick had every v2 result and trained model deleted.** The record is in git: branch `v2`, tag `end-v2`, and the
tags `specialist-gate-*-run`.

## 2. Requirements (the source of truth)
- **Locked decisions:** `docs/DECISIONS-2026-09-24.md` (1–4) and `docs/DECISIONS-2026-09-25.md` (1–39; note
  that 39 amends 2).
- **Nick's stated requirements** from the last days of v2, not all yet written as decisions:
  - **The brain must work first.** "If it's not actually working correctly the experiment is moot."
    `docs/BRAIN-SPEC.md`.
  - **The fly is kept.** Without him Bosco is "a worse Jev". Accuracy is not the novelty. What the fly adds must
    be real (brain-spec T2 and T3), not decoration.
  - **No added neurons and no slower** (decision 39). Use the 50,140 we have properly.
  - **Use every sense reasonably** (decision 38):
    - smell = the item's meaning;
    - **taste may carry a judgement:** "there's no reason a fly expert can't taste if text is gross or not".
      Fly taste is wired for good versus bitter. This reverses decision 38's "taste = form only" and is the
      candidate for decision 3, with the question through taste and the item through smell. Confirm with Nick
      and write it as an amendment;
    - sight = pictures, through the visual projection neurons, with no optic lobes. **"We need picture
      handling."**
  - **Sharp yes/no questions over broad menus** (decision 36). Specialists are **generalists in their field**
    (decision 29), tested on a held-out source.
  - **Harm is the priority:** p(harm) is the headline, content moderation (decision 35). Follow-up experts are
    threat, sexual, hate and p(harassment) ("is this text harassing someone").
  - **The consensus is the human's** (decision 34). Each expert answers alone, calibrated.
  - **Jev parity is the yardstick** (decision 37): choose / approach / rate, several questions per request, speed.
    Pictures are where Bosco can pass Jev.
  - **Launch fleet about 15–20,** growing toward about 100 (decision 15).
  - **Marketing:** "a swarm" or "a panel of expert flies", where each answers on its own. Not "ensemble".
    Nick worries about novelty ("RPA with a brain you can watch"). Revisit once the brain works.
  - **Budget about $30 a month;** no paid datasets.

## 3. Decisions waiting on Nick (from the audit; ask, do not assume)
1. **Decision 3:** the question through taste and the item through smell (his taste remark), or "the question is
   the learned memory".
2. **Controls on every card** (layered and hash shuffle, silenced mushroom body)?
3. **The encoder:** keep nomic, with its scraped and MS MARCO training data disclosed, or make our own clean
   encoder a launch blocker (decision 14, second track)?
4. **SMS spam:** the ledger said DROP (its upstream sources are unlicensed). Drop junk, or accept the risk
   explicitly.
5. **Statistics:** a sampling-error rule in every bar, e.g. the lower 95% bound clears the bar, or test sets
   sized so noise cannot flip a verdict.
6. **Public wording:** adopt the audit's corrected pitch (`data-claims.md`).
7. **FlyWire annotation table** (used in `populations.py`): replace it with the CC BY 4.0 Nature supplement.

## 4. What this branch carries (and what it does not)
- **Code:**
  - the connectome build (`data`, `model`, `model2`, `populations`, `senses`, `controls`);
  - the rate brain (`ratebrain2` constants, `ratebrain3`);
  - `v1.py` (build, homeostatic start, DN groups);
  - encoder pins.

  It builds standalone (checked: 50,140 neurons, 4,064 KCs, a DN read of 16 approach and 30 avoid).
  `ratebrain3.run` has new options from the last day of v2: `r0` (start state), timed smell windows
  `(B, W, 46)` and a fixed per-cell offset `b_cell`. **They are regression-identical when off, and untested when
  on.**
- **`scripts/brain_check.py`:** a draft of the label-free checks L1–L6. **It has never been run. Review it
  against BRAIN-SPEC before use.** Its per-KC threshold routine (`kc_per_cell`) is crude.
- **Not carried:**
  - the training stack (designs A and B, `v1_pilot`, `a4*`, the gate runner);
  - the service (`src/bosco/service`) and the registry;
  - every result;
  - the docs that overclaimed (BRIEF, PLAN-V1, ASK-BOSCO-PRIORITIES, SERVICE.md, REGISTRY.md).

  All are on `v2` if a piece is needed. Rewrite, do not resurrect.
- **Data docs:** `clean-data.md` and its manifest, `wiki-data.md`, `free-datasets.md`, `research-gate4/`,
  `gate4-data-*.md` with manifests, and `harassment-dev.md`. The builders are in `reference/v2-data-builders/`.
  They import removed v2 modules, so read them; do not run them.
- **Brain views:** `runs/brain-map/`, `docs/BRAIN-VIEWS.md`, `scripts/brain_render.py`. The maps are in
  `service/families/v1/{anatomy,wave}.json` (gitignored, local).

## 5. Local state and known issues
- **Data, local on the Mac and gitignored:**
  - `data/raw/`: the MaleCNS feathers and `data/raw/clean/*` (every fetched dataset, with `FETCH.json` and
    licences);
  - `data/cache/*`: v2's embedding caches, including X, the raw 768-d nomic embeddings. Reusable as encoder
    output. The Z/ZL there are the old antenna; do not use them.
- **The gate-4 held-out test splits were scored once on the old brain** (13 of 14) and deleted unread. Any
  future use of those test parts must be declared as a second use.
- **`service/families/v1/antenna.npz` is the old nose** (rest 0.5, fit half on label words). Do not use it.
  Nick declined deleting it in session; ask.
- **The encoder does not load with the locked `transformers` 5.x.** Embed in an isolated env:
  `uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops python …`. The
  nomic remote code (`nomic-bert-2048`) is not pinned by revision. Pin it.
- **magnetar** may still have a `brain_check.py` process from the last minutes of v2; the check was incomplete.
  First action in v3: `ssh magnetar 'pgrep -af brain_check'`, show Nick the output, and kill anything found.
  Its `~/bosco` holds v2-era code and data; rsync what v3 needs.

## 6. The plan (each step ends with showing Nick and waiting for his go)
1. **Confirm nothing is running** on either machine, and show the output.
2. **Settle decision 3** (taste for the question?) and any audit decisions Nick wants settled before the brain
   work. Write them as amendments.
3. **Fix the brain in code**, per BRAIN-SPEC's allowed changes:
   - start from rest;
   - a quiet resting drive (± split, rest near spontaneous);
   - per-KC label-free thresholds;
   - one-sniff yes/no read;
   - fix the DN read groups' steering neurons (audit F10) with Nick's agreement.

   Show the diff.
4. **Run the label-free checks L1–L6** (one run, the 3080, about 15 minutes, no labels). Show the table.
5. **Run the one small trained check T1–T6** (p(harm), train and val only, KC→MBON-only main arm, layered
   control beside it). Show the table.
6. **Only if everything passes:** the full spec (`docs/SPEC.md`) for sign-off, conformance tests that gate
   training, and then specialists, pictures and service.
