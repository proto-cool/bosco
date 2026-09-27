# The specialist registry (2026-09-27)

**What a specialist is:** a specialist is one brain family (`service/families/<id>/`: connectome build,
fixed antenna, DN read, brain maps) plus its own trained weights and a card. The registry is files, not a
database.

## Layout
- `registry/specialists/<name>/<version>.json`: the card (in git). A card holds:
  - the question, the options and the design;
  - the family, pinned by the antenna's sha256 and the neuron order;
  - the encoder pin and the temperature;
  - the weights' sha256;
  - the gate evidence (sealed CPU balanced accuracy, ECE, bar, validation, report-only comparisons, CPU time);
  - the sources and the licence (decision 28), the disputed-labels note, and the limits in plain words.
- `registry/specialists/<name>/<version>.attribution.jsonl`: for share-alike Wikipedia training data, every
  training and validation row's exact revision URL.
- `registry/index.json`: one line per version.
- `service/specialists/<name>/<version>/{weights.pt, card.json}`: the weights (not in git; they are rsynced
  to the server).

## Rules
- **Only gate-passed specialists are exported.** `scripts/export_family.py specialist --gate G --task T`
  recomputes the ship verdict from the sealed CPU score. It refuses unless that verdict agrees with the
  gate's results doc.
- **Versions are immutable** (`<name>-<date>`). A fix is always a new version.
- **The server loads only shipped cards whose weights match their sha256** (`Fleet`; `dev=True` is for local
  work only).
- **Callers pin exact versions.** `<name>-latest` is a convenience for exploring, not for Ask Bosco
  production.
- **Parity:** `tests/test_service_parity.py` checks that the service gives the same pick and confidence as
  the gate's sealed CPU score (designs A and B).
- **Self-host bundles** are built from the registry: the family, the chosen cards and weights, and LICENSE
  and NOTICE files per decision 28. Training text is never included.
- **A new family** (eyes, own encoders) means retraining every specialist into it.

## Shipped (2026-09-27)
topic, junk, support, hate, politeness (gate 2); kind, food, danger (gate 3).
