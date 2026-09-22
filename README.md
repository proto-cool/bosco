# bosco v2

A fruit fly's brain as a decision model.

The MaleCNS v1.0 connectome, simulated as leaky integrate-and-fire neurons
with mushroom-body plasticity, asked one question: is a real brain's wiring
diagram a useful prior for learning decisions? Typed decisions with a
probability, no text.

v1 put the same brain on a Bluesky account for nine days (2026-09-13 to
2026-09-22). It is archived whole on `main` (tag `end-v1`); its closing note
is `docs/CLOSING-v1.md` there. This branch starts over with only the parts
that stood on their own.

- `CLAUDE.md` — the rules.
- `docs/PLAN.md` — what is being done, in what order, with what gates.
- `docs/GATE-B.md` — the first pre-registered gate.
- `docs/phase*-*.md`, `docs/plasticity-v*.md` — v1 reproductions and
  measurements the model rests on.

## Running

```
uv sync
make -C kernel
uv run pytest tests/test_kernel.py     # kernel only, no data needed
ops/fetch_data.sh                      # ~1.1 GB of MaleCNS feathers
uv run pytest                          # builds the central-brain model on first run
```
