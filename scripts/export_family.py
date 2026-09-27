"""Export a brain family and its gate-passed specialists (docs/REGISTRY.md, docs/SERVICE-R10.md).

uv run python scripts/export_family.py family                    # family v1 from the specialist pilot's data
uv run python scripts/export_family.py specialist --gate 2 --task topic
uv run python scripts/export_family.py index                     # rebuild registry/index.json

A family is shared by every specialist: the connectome build, the nose's antenna (fit once, label-free),
the answer's DN groups, steps, the brain maps and the resting trace. A specialist is its learned weights
plus a card. Only a specialist its gate's results doc lists under "Ships" can be exported, and the ship
verdict is recomputed from the sealed CPU scores; the two must agree. A version is immutable: it is never
overwritten. Cards (with the evidence, licence and limits) go to `registry/` (in git); weights go to
`service/specialists/<name>/<version>/` (not in git) and are checked against the card's sha256 on load.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys

import numpy as np
import torch

from bosco import encoders, paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import v1_gate2 as G  # noqa: E402
import v1_pilot as V  # noqa: E402

SERVICE = paths.ROOT / "service"
REGISTRY = paths.ROOT / "registry"
WIKI = paths.ROOT / "data" / "raw" / "clean" / "wiki"

# Per decision 28: CC BY / CC0 data -> our own licence with credits; CC BY-SA text -> the weights are CC BY-SA 4.0.
OWN = "Bosco specialist licence (the self-host bundle's LICENSE), crediting the sources listed"
BY_SA = "CC BY-SA 4.0 (trained on share-alike text; decision 28)"
WIKIPEDIA = {
    "name": "English Wikipedia, lead paragraphs, last revision on or before 2022-11-01",
    "licence": "CC BY-SA 4.0 (older text also GFDL)",
    "credit": "Wikipedia contributors; each training row's revision is listed in the attribution file",
}
WIKIDATA = {"name": "Wikidata class labels", "licence": "CC0 1.0", "credit": "Wikidata contributors"}
WIKI_LIMIT = (
    "Tested on Wikipedia lead paragraphs, which often state the answer outright; expect lower accuracy on "
    "everyday text."
)
SPECIALISTS = {
    "topic": {
        "question": "What kind of thing is this text about?",
        "sources": [
            {
                "name": "DBpedia-14 (Zhang et al. 2015), version 2",
                "licence": "CC BY-SA 3.0 and GFDL (dual)",
                "credit": "DBpedia, from Wikipedia abstracts by Wikipedia contributors",
            }
        ],
        "licence": BY_SA,
        "limits": [
            "Trained on encyclopedia abstracts in 14 classes; it always picks one of the 14, even when none fits.",
        ],
    },
    "junk": {
        "question": "Is this message junk?",
        "sources": [
            {
                "name": "SMS Spam Collection v.1 (UCI 228)",
                "licence": "CC BY 4.0",
                "credit": "Almeida & Gomez Hidalgo; phone numbers were scrubbed before training",
            }
        ],
        "licence": OWN,
        "limits": ["Trained on text messages from the 2000s; email and modern scams may look different."],
    },
    "support": {
        "question": "Which area is this request about?",
        "sources": [
            {"name": "MASSIVE 1.1, en-US scenarios", "licence": "CC BY 4.0", "credit": "Amazon (FitzGerald et al. 2022)"}
        ],
        "licence": OWN,
        "limits": ["Trained on short voice-assistant requests (18 areas); long or written messages may be mis-sorted."],
    },
    "hate": {
        "question": "Is this hate speech?",
        "sources": [
            {"name": "DynaHate v0.2.3, all rounds", "licence": "CC BY 4.0", "credit": "Vidgen et al. 2021"}
        ],
        "licence": OWN,
        "disputed_labels": "People often disagree on these labels; its bar is 0.9 x the encoder's ceiling (0.731).",
        "limits": [
            "Catches blatant hate; often flags texts that merely mention a group.",
            "Misses coded language and dog whistles more often than blatant hate.",
        ],
    },
    "politeness": {
        "question": "Is this request polite?",
        "sources": [
            {
                "name": "Stack Exchange Politeness corpus (ConvoKit), top vs bottom quartile",
                "licence": "CC BY 4.0 (corpus); Stack Exchange text CC BY-SA",
                "credit": "Danescu-Niculescu-Mizil et al. 2013; Stack Exchange contributors",
            }
        ],
        "licence": BY_SA,
        "disputed_labels": "People often disagree on politeness; its bar is 0.9 x the encoder's ceiling (0.666).",
        "limits": [
            "Passed its bar by 0.0006: barely above it.",
            "Trained on the clearly polite and clearly rude quarters of requests; middling tone is a coin toss.",
        ],
    },
    "kind": {
        "question": "What kind of thing is this?",
        "sources": [WIKIPEDIA, WIKIDATA],
        "licence": BY_SA,
        "attribution": "kind",
        "limits": [WIKI_LIMIT],
    },
    "food": {
        "question": "Is this food or drink?",
        "sources": [WIKIPEDIA, WIKIDATA],
        "licence": BY_SA,
        "attribution": "food",
        "limits": [WIKI_LIMIT],
    },
    "danger": {
        "question": "Is this dangerous?",
        "sources": [WIKIPEDIA, WIKIDATA],
        "licence": BY_SA,
        "attribution": "danger",
        "limits": [
            WIKI_LIMIT,
            "Its labels come from our own rule over Wikidata classes (disasters, infectious diseases, carcinogens "
            "and toxins other than medicines, venomous animals, weapons); some members are debatable.",
        ],
    },
}


def sha256(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def antenna(meta, X, L):
    """Exactly v1_pilot.load's antenna (seeded), returned as parameters."""
    tr = np.where([it["split"] == "train" for it in meta["items"]])[0]
    Xf = X[tr[np.random.default_rng(V.SEED).permutation(len(tr))[:4000]]]
    F = np.concatenate([Xf, np.repeat(L, max(1, len(Xf) // len(L)), 0)])
    mu = F.mean(0)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / s[:46, None]
    norm = float(np.percentile(np.abs((F - mu) @ W.T), 99))
    return mu.astype(np.float32), W.astype(np.float32), norm


def cmd_family(a) -> int:
    meta = json.load(open(V.OUT / "items.json"))
    e = np.load(V.OUT / "emb.npz")
    mu, W, norm = antenna(meta, e["X"], e["L"])
    m, info = V.make_brain("real", device="cpu")
    order = hashlib.sha256(np.ascontiguousarray(M2ids()).tobytes()).hexdigest()[:16]
    d = SERVICE / "families" / a.family
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "antenna.npz", mu=mu, W=W, norm=np.float32(norm))
    for src, dst in (("runs/brain-map/anatomy-map-mix0.3.json", "anatomy.json"), ("runs/brain-map/map.json", "wave.json")):
        shutil.copy(paths.ROOT / src, d / dst)
    json.dump({
        "id": a.family, "neuron_count": m.n, "neuron_order_hash": order, "steps": m.steps, "dt_ms": 5.0,
        "read_steps": m.read_steps, "dn_groups": info, "gain": V.GAIN,
        "encoder": encoders.TEXT, "antenna": "bi46: 46 whitened components, resting rate 0.5 (docs/SPECIALIST-PILOT.md)",
        "connectome": "MaleCNS v1.0 (Janelia FlyEM et al., CC BY 4.0), cut at 5 synapses, all KC->MBON kept",
        "created": datetime.date.today().isoformat(),
    }, open(d / "family.json", "w"), indent=1)
    print(f"family {a.family}: {m.n} neurons, order {order}")
    return 0


def M2ids():
    from bosco import model2 as M2

    return M2.load_or_build().brain.ids


def gate_verdict(gate: str, task: str) -> dict:
    """The sealed CPU score, recomputed exactly as the gate's report did, and the report's own Ships line."""
    data, runs, _, bars = G.GATES[gate]
    s = json.load(open(runs / f"score-test-{task}.json"))
    bal = G.D.macro(G.B.balanced([(x["task"], x["gold"], x["pick"]) for x in s["rows"]]))
    ece = V.ece(s["rows"])
    ok = bal >= bars[task] and ece <= 0.10
    doc = paths.DOCS / f"specialist-gate-{gate}-results.md"
    m = re.search(r"\*\*Ships: ([^*]*)\.\*\*", doc.read_text())
    listed = task in [x.strip() for x in m.group(1).split(",")] if m else False
    if ok != listed:
        raise SystemExit(f"{task}: recomputed verdict {ok} disagrees with {doc.name} ({listed})")
    base = json.load(open(runs / "baselines.json"))[task]
    tj = json.load(open(runs / f"{task}.json"))
    return {
        "ships": ok,
        "gate": gate,
        "prereg": f"docs/SPECIALIST-GATE-{gate}.md",
        "results": f"docs/specialist-gate-{gate}-results.md",
        "test_items": len(s["rows"]),
        "balanced_accuracy": round(bal, 4),
        "ece": round(ece, 4),
        "bar": round(bars[task], 4),
        "ece_limit": 0.10,
        "best_val": round(tj["best_val"], 4),
        "epochs": [h["epoch"] for h in tj["hist"]],
        "cpu_s_per_question": round(s["cpu_s_per_question"], 3),
        "temperature": s["temperature"],
        "report_only": {k: round(v, 4) for k, v in base.items()},
        "data": str(data.relative_to(paths.ROOT)),
        "runs": str(runs.relative_to(paths.ROOT)),
    }


def cmd_specialist(a) -> int:
    name, spec = a.task, SPECIALISTS[a.task]
    ev = gate_verdict(a.gate, a.task)
    if not ev["ships"]:
        raise SystemExit(f"refusing: {name} did not pass gate {a.gate} (the production bar)")
    data, runs, _, _ = G.GATES[a.gate]
    meta = json.load(open(data / "items.json"))
    ck = torch.load(runs / f"{a.task}.pt", weights_only=True)
    fam = SERVICE / "families" / a.family
    version = f"{name}-{a.date or datetime.date.today().isoformat()}"
    d = SERVICE / "specialists" / name / version
    card_path = REGISTRY / "specialists" / name / f"{version}.json"
    if card_path.exists():
        raise SystemExit(f"refusing: {version} exists; versions are immutable (export a new date)")
    d.mkdir(parents=True, exist_ok=True)
    torch.save(ck["state"], d / "weights.pt")  # every trained parameter (per-type b, g, tau; KC->MBON; read scale)
    card = {
        "specialist": name,
        "version": version,
        "status": "shipped",
        "question": spec["question"],
        "options": meta["tasks"][a.task]["options"],
        "design": ck["design"],
        "family": a.family,
        "family_antenna_sha256": sha256(fam / "antenna.npz"),
        "family_neuron_order": json.load(open(fam / "family.json"))["neuron_order_hash"],
        "encoder": encoders.TEXT,
        "task": a.task,
        "temperature": ev["temperature"],
        "weights_sha256": sha256(d / "weights.pt"),
        "evidence": ev,
        "sources": spec["sources"],
        "licence": spec["licence"],
        "disputed_labels": spec.get("disputed_labels"),
        "limits": spec["limits"],
        "exported": datetime.date.today().isoformat(),
    }
    if spec.get("attribution"):  # share-alike text: every training/validation row's exact revision
        rows = [
            json.loads(line)
            for s in ("train", "val")
            for line in open(WIKI / spec["attribution"] / f"{s}.jsonl")
        ]
        att = card_path.with_suffix(".attribution.jsonl")
        att.parent.mkdir(parents=True, exist_ok=True)
        with open(att, "w") as f:
            for r in rows:
                f.write(json.dumps({"title": r["title"], "url": r["url"], "wiki": r["wiki"]}) + "\n")
        card["attribution_file"] = str(att.relative_to(REGISTRY))
    card_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(card, open(card_path, "w"), indent=1)
    shutil.copy(card_path, d / "card.json")
    print(f"{version}: {ev['balanced_accuracy']:.3f} (bar {ev['bar']:.3f}), ECE {ev['ece']:.3f}, {card['licence']}")
    return cmd_index(a)


def cmd_index(a) -> int:
    rows = []
    for p in sorted((REGISTRY / "specialists").glob("*/*.json")):
        c = json.load(open(p))
        rows.append({
            "version": c["version"], "specialist": c["specialist"], "status": c["status"], "family": c["family"],
            "options": len(c["options"]), "balanced_accuracy": c["evidence"]["balanced_accuracy"],
            "ece": c["evidence"]["ece"], "licence": c["licence"], "card": str(p.relative_to(REGISTRY)),
        })
    json.dump({"specialists": rows}, open(REGISTRY / "index.json", "w"), indent=1)
    print(f"index: {len(rows)} versions")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("family").set_defaults(fn=cmd_family)
    p = sub.add_parser("specialist")
    p.add_argument("--gate", choices=list(G.GATES), required=True)
    p.add_argument("--task", choices=list(SPECIALISTS), required=True)
    p.add_argument("--date", help="version date (default today)")
    p.set_defaults(fn=cmd_specialist)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
