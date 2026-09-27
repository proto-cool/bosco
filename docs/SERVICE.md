# The Bosco service (first build, 2026-09-26)

Code: `src/bosco/service/` (`core.py`: family, specialists, decide, trace; `app.py`: HTTP API).
Contract: `docs/API.md`, `docs/SERVICE-R10.md`, `docs/BRAIN-VIEWS.md`.

## Run

```
uv run python scripts/export_family.py family                             # family v1 -> service/families/v1
uv run python scripts/export_family.py specialist --gate 2 --task topic   # gate-passed only (docs/REGISTRY.md)
uv run --group service --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    uvicorn bosco.service.app:app --host 127.0.0.1 --port 8420
```

`service/` is not in git: it is regenerated from the runs. Specialist weights carry their training data's
licence (e.g. topic is trained on DBpedia, CC BY-SA), which matters before anything is published.

## What works (tested end to end on the Mac's CPU, 4 threads)
- `GET /v1/models`: the family (neurons, order hash, steps, DN groups, encoder, antenna) and the specialist
  cards; `topic-latest` aliases.
- `POST /v1/decide` (choose): "The Eiffel Tower is a wrought-iron lattice tower…" → building, sure 0.96;
  "Messi scored twice…" → artist, sure 0.37 (wrong, and honestly unsure). **2.0 s for a 14-option question**
  (the first request also loads the encoder). **Deterministic:** repeat requests give identical answers.
- Out of a specialist's task → `422 not_taught`; `allow_untaught: true` answers anyway, with
  `answered_as`. Unknown version → 404. Every response carries `x-bosco-request-id`.
- `GET /v1/traces/{id}`: about 0.7 MB (JSON) per answer, kept in memory only (the last 256).
  - Per step and per view (anatomy, wave): 3,072 dots as int8 base64 (value = q / 127 × `scale`),
    row-major.
  - `named`: the avoid and approach DN groups' mean activity per step.
  - `top`: per step, the 3 most-changed cells in each processing region (projection neurons, KCs,
    MBONs, DANs, lateral horn, central complex, DNs), each as [neuron, cell type, region, delta].
- `GET /v1/families/{id}/maps/{anatomy|wave}`: the dot maps.
- Activity is relative to **the specialist's own rest** (its brain with every glomerulus at rest),
  computed once per specialist.

## As built, 2026-09-27 (Jev parity, decision 37)
- **Three question types:** `choose` (pick one taught option), `approach` (yes/no on a two-option specialist;
  its first option is "yes"; returns `p`, `lean` and `sure`) and `rate` (the caller orders the taught options
  low → high; returns `score` = 1 + Σ level × p, the `legend` and the per-level `p`).
- **Several questions per request, each naming its own specialist** (`questions.<id>.specialist`; the
  request's `version` is the default). Up to 16 questions.
- **One brain pass per request.** The connectome is shared, and every sniff column carries its own
  specialist's gains, thresholds, time constants and KC→MBON memory (`Family.run_cols`). There is one encoder
  call per request. Every question is validated before any brain time is spent. `tests/test_service_parity.py`
  still matches the gates' sealed CPU scores.
- **Warm CPU latency** (Mac, 4 threads): 1.1 s for one yes/no, 1.5 s for 14–18 options, and 4.0 s for all 8
  specialists in one request (49 sniffs; about 9 s one at a time).
  - The sparse connectome step is the cost. It is memory-bound up to about 16 columns (batching is almost
    free) and grows beyond that.
  - The remaining lever is fewer steps: the effort levels (`docs/EFFORT-PILOT.md`, to be pre-registered).
  - Jev quotes 70–500 ms.
- **Traces:** recorded for every sniff when a request has ≤ 48; above that, only the picked sniffs, in a
  second pass.

## Not yet
- `familiar` is not built; `approach` and `rate` are (above).
- No auth, rate limits or queue (Ask Bosco's server does those); no batching across requests.
- Only shipped, hash-pinned specialists load (docs/REGISTRY.md): 8 as of 2026-09-27. Design B specialists
  (support) answer only their own options, or a subset of them.
- **The family antenna is fixed from the pilot's data.** Future specialists must train through this
  same antenna (the gap round refits its own; that is fine for development only).
- The Kimsufi speed is not measured.

## Speed (Mac CPU, 4 threads, 80 steps; the Kimsufi is not measured)
- The cost is dominated by 80 steps × one sparse product over about 2.4M connections.
- **The serving fast path (`RateBrain3.freeze`)** folds each neuron's gain and the learned
  KC→MBON multipliers into one CSR matrix. It is 1.1–1.3× faster, with identical picks (logits
  within 6e-5) and repeatable.
- **Measured:** about 1.0 s of brain for a 2-option question and about 1.3 s for 14 options. The
  whole request, with the encoder, is about 2 s.
- **Tried and not taken:** SciPy CSR (faster at 2 sniffs, 0.69 s, but single-threaded and slower at
  14); bfloat16 (2.2× slower on CPU, per the code audit).
- **Further speed would change the model,** so each option needs its own check against the bar:
  fewer steps (the DN read settles at about 60–80), or a higher synapse cutoff (fewer connections).
