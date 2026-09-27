"""The service answers exactly as the gate scored: for each exported specialist, the first test items through
`Family.decide` give the same pick and confidence as the gate's sealed CPU score (docs/REGISTRY.md).
Needs the data (ops/fetch_data.sh), the gate runs and an export (scripts/export_family.py)."""

from __future__ import annotations

import json

import pytest

from bosco import paths

ROOT = paths.ROOT / "service"
N = 12
CASES = [("2", "topic"), ("2", "support"), ("2", "junk"), ("3", "kind")]


@pytest.fixture(scope="module")
def fleet():
    if not (ROOT / "families" / "v1").exists():
        pytest.skip("no exported family")
    from bosco.service.core import Fleet

    return Fleet(ROOT)


@pytest.mark.parametrize("gate,task", CASES)
def test_matches_gate_score(fleet, gate, task):
    d = paths.CACHE / ("v1-gate2" if gate == "2" else "v1-gate3")
    runs = paths.ROOT / "runs" / f"specialist-gate-{gate}"
    if not (runs / f"score-test-{task}.json").exists():
        pytest.skip("no gate score")
    rows = json.load(open(runs / f"score-test-{task}.json"))["rows"][:N]
    items = [it for it in json.load(open(d / "items.json"))["items"] if it["task"] == task and it["split"] == "test"]
    spec = fleet.get(f"{task}-latest")
    for it, r in zip(items, rows, strict=False):
        assert it["gold"] == r["gold"]
        out = fleet.family.decide(spec, it["text"], None)
        assert spec.card["options"].index(out["pick"]) == r["pick"]
        assert max(out["p"].values()) == pytest.approx(r["pmax"], abs=2e-3)
