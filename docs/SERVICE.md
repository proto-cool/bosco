# The Bosco service (first build, 2026-09-26)

Code: `src/bosco/service/` (`core.py`: family, specialists, decide, trace; `app.py`: HTTP API).
Contract: `docs/API.md`, `docs/SERVICE-R10.md`, `docs/BRAIN-VIEWS.md`.

## Run

```
uv run python scripts/export_family.py                                    # family v1 -> service/families/v1
uv run python scripts/export_family.py --specialist topic-real --name topic  # -> service/specialists/topic/<version>
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

## Not yet
- Only `choose` questions; `approach`, `rate` and `familiar` come later.
- No auth, rate limits or queue (Ask Bosco's server does those); no batching across requests.
- Specialists are development-grade until they pass the production bar.
- **The family antenna is fixed from the pilot's data.** Future specialists must train through this
  same antenna (the gap round refits its own; that is fine for development only).
- The Kimsufi speed is not measured.
