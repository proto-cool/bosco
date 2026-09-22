"""Gate B runner (docs/GATE-B.md).  Nothing to set here; the pre-registration is in bosco.gateb.

  learn   one arm, one seed, one learning task (t1 acquisition, t2 reversal, t3 pictures)
  after   t4 (pictures, no rewards), t5 (mixtures), t6 (the probe sheet) from a saved t1 state
  report  every run under runs/gate-b/ -> docs/gate-b-results.md

    uv run python scripts/make_dunce.py --seed 1 --out data/cache/dunce_v1.npz
    uv run python scripts/make_hash.py  --seed 1 --out data/cache/hash_v1.npz
    for arm in real shuffle hash logistic; do for s in 1 2 3 4 5; do
      uv run python scripts/gate_b.py learn --task t1 --arm $arm --seed $s
    done; done

`--smoke N` runs the first N items only and writes under runs/gate-b/smoke/; it is for checking
the plumbing and is never a result.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import yaml

from bosco import gateb as G
from bosco import paths

BRAINS = {"real": None, "shuffle": str(paths.CACHE / "dunce_v1.npz"), "hash": str(paths.CACHE / "hash_v1.npz")}


def make_arm(name: str, seed: int, drive: G.Drive | None):
    if name == "logistic":
        return G.LogisticArm(seed)
    return G.FlyArm(BRAINS[name], drive)


def drive_for(arm_name: str, e_calib: np.ndarray | None) -> G.Drive | None:
    """The drive is calibrated once on the real wiring and shared (docs/GATE-B.md)."""
    if arm_name == "logistic":
        return None
    from bosco.sim import Fly

    f = G.DRIVE_FILE
    if not f.exists():
        assert e_calib is not None
        return G.calibrate_drive(Fly(), e_calib)
    return G.Drive.from_json(json.load(open(f)))


def learn(a) -> int:
    if a.task in ("t1", "t2"):
        items, calib = G.sst_items()
        e = G.embed(items, "sst-pool")
        e_calib = G.embed(calib, "sst-calib")
        eval_k, window = G.EVAL_K, G.EVAL_WINDOW
    elif a.task == "t3":
        items = G.oasis_items()
        e = G.embed(items, "oasis")
        _, calib = G.sst_items()
        e_calib = G.embed(calib, "sst-calib")
        eval_k, window = G.EVAL_K_IMAGE, G.EVAL_WINDOW_IMAGE
    else:
        raise SystemExit("learn takes t1, t2 or t3")
    if a.smoke:
        items, e = items[: a.smoke], e[: a.smoke]
        eval_k, window = (2, 4), 5
    proto = G.Protocol.make(len(items), a.seed, flip_at=(G.FLIP_AT if a.task == "t2" else None))
    if a.smoke and a.task == "t2":
        proto.flip_at = a.smoke // 2
    drive = drive_for(a.arm, e_calib)
    arm = make_arm(a.arm, a.seed, drive)
    out = (G.RUNS / "smoke" if a.smoke else G.RUNS) / f"{a.task}-{a.arm}-s{a.seed}"
    run = G.Run(a.task, a.arm, a.seed, items, e, proto, eval_k, window, out)
    run.log(f"{len(items)} items, {int(proto.rewarded.sum())} rewards, drive {drive.to_json() if drive else None}")
    G.run_learning(run, arm)
    summary = G.summarize(run)
    summary["recall"] = G.recall(run, arm)
    if a.task == "t2" and isinstance(arm, G.FlyArm):
        summary["retention"] = G.retention(run, arm)
    G.save_run(run, summary, arm)
    run.log(json.dumps(summary, indent=1))
    return 0


def after(a) -> int:
    """t4/t5/t6 on the state a t1 run left behind, no exposure, no rewards."""
    src = G.RUNS / f"t1-{a.arm}-s{a.seed}"
    if a.smoke:
        src = G.RUNS / "smoke" / f"t1-{a.arm}-s{a.seed}"
    if a.arm == "logistic":
        raise SystemExit("the logistic arm has no saved state yet; run t4 for it inside learn (todo)")
    drive = drive_for(a.arm, None)
    arm = make_arm(a.arm, a.seed, drive)
    arm.load_state(dict(np.load(src / "state.npz")))
    out = {}
    if a.task == "t4":
        items = G.oasis_items()
        if a.smoke:
            items = items[: a.smoke]
        e = G.embed(items, "oasis")
        sc = [arm.score(e[i], G.h32(a.seed, it.id, "t4")) for i, it in enumerate(items)]
        y = [it.label for it in items]
        out = {
            "n": len(items),
            "accuracy": G.accuracy([s for s, _ in sc], y),
            "spearman": G.spearman([s for s, _ in sc], y),
            "rows": [
                {"id": it.id, "label": it.label, "score": s, "sure": u} for it, (s, u) in zip(items, sc, strict=True)
            ],
        }
    elif a.task in ("t5", "t6"):
        probes = yaml.safe_load(open(paths.DOCS / "gate-b-probes.yaml"))
        rows = []
        for p in probes.get("text", []):
            it = G.Item(f"probe-{p['text']}", "text", p["text"], float("nan"))
            e = G.embed([it], f"probe-{G.h32(p['text'])}")[0]
            s, u = arm.score(e, G.h32(a.seed, it.id, "t6"))
            rows.append({"kind": "text", "text": p["text"], "expect": p.get("expect"), "score": s, "sure": u})
        for p in probes.get("image", []):
            path = paths.ROOT / p["path"]
            if not path.exists():
                rows.append({"kind": "image", "about": p.get("about"), "missing": str(path)})
                continue
            it = G.Item(f"probe-{p['path']}", "image", str(path), float("nan"))
            e = G.embed([it], f"probe-{G.h32(p['path'])}")[0]
            s, u = arm.score(e, G.h32(a.seed, it.id, "t6"))
            rows.append({"kind": "image", "about": p.get("about"), "expect": p.get("expect"), "score": s, "sure": u})
        if a.task == "t5":
            for p in probes.get("mixture", []):
                path = paths.ROOT / p["path"]
                if not path.exists():
                    rows.append({"kind": "mixture", "about": p.get("about"), "missing": str(path)})
                    continue
                et = G.embed(
                    [G.Item(f"probe-{p['text']}", "text", p["text"], float("nan"))], f"probe-{G.h32(p['text'])}"
                )[0]
                ei = G.embed(
                    [G.Item(f"probe-{p['path']}", "image", str(path), float("nan"))], f"probe-{G.h32(p['path'])}"
                )[0]
                # the two drives summed at the nose: score with a stimulus whose rates are the sum
                z = drive.rates(et) + drive.rates(ei)
                vs = []
                for k in range(G.N_SCORE_SEEDS):
                    r = arm.fly.run_episode(
                        G.stimulus_of(arm.fly, drive, z, arm.orn_idx),
                        G.h32(a.seed, p["text"], p["path"], k),
                        ms=G.PRESENT_MS,
                    )
                    vs.append(arm.valence(r.counts[arm.fly.kc]))
                m = float(np.mean(vs))
                rows.append(
                    {
                        "kind": "mixture",
                        "text": p["text"],
                        "about": p.get("about"),
                        "score": (m + 1) / 2,
                        "sure": float(np.mean(np.array(vs) > 0)) if m > 0 else float(np.mean(np.array(vs) < 0)),
                    }
                )
        out = {"rows": rows}
    else:
        raise SystemExit("after takes t4, t5 or t6")
    dst = src / f"{a.task}.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"} | {"rows": out.get("rows", [])[:12]}, indent=1))
    print("wrote", dst)
    return 0


def report(a) -> int:
    runs = sorted(p for p in G.RUNS.glob("t*-*-s*") if (p / "summary.json").exists())
    by: dict[tuple[str, str], list[dict]] = {}
    for p in runs:
        s = json.load(open(p / "summary.json"))
        by.setdefault((s["task"], s["arm"]), []).append(s)
    gate = G.GATE.upper()
    lines = [
        f"# Gate {gate} results\n",
        f"{len(runs)} runs under `runs/gate-{G.GATE}/`. Pre-registration: `GATE-{gate}.md`.\n",
    ]
    for task in sorted({t for t, _ in by}):
        lines.append(f"\n## {task}\n")
        keys = sorted(
            {k for (t, _), ss in by.items() if t == task for s in ss for k in s["windows"]},
            key=lambda k: (k.startswith("flip"), int(k.split("k")[1])),
        )
        lines.append("| arm | seeds | " + " | ".join(f"{k} acc | {k} rho" for k in keys) + " | ECE raw | ECE iso |")
        lines.append("|---|---|" + "---|---|" * len(keys) + "---|---|")
        for arm in ("real", "shuffle", "hash", "logistic"):
            ss = by.get((task, arm))
            if not ss:
                continue
            cells = []
            for k in keys:
                acc = [s["windows"][k]["accuracy"] for s in ss if k in s["windows"]]
                rho = [s["windows"][k]["spearman"] for s in ss if k in s["windows"]]
                cells.append(f"{np.mean(acc):.3f} ± {np.std(acc):.3f}" if acc else "—")
                cells.append(f"{np.nanmean(rho):.2f}" if rho else "—")
            er = [s.get("ece_raw_k100plus") for s in ss if s.get("ece_raw_k100plus") is not None]
            ei = [s.get("ece_isotonic_k100plus") for s in ss if s.get("ece_isotonic_k100plus") is not None]
            lines.append(
                f"| {arm} | {len(ss)} | " + " | ".join(cells) + f" | {np.nanmean(er):.3f} | {np.nanmean(ei):.3f} |"
                if er
                else f"| {arm} | {len(ss)} | " + " | ".join(cells) + " | — | — |"
            )
        rec = []
        for arm in ("real", "shuffle", "hash", "logistic"):
            rs = [s["recall"] for s in by.get((task, arm), []) if s.get("recall")]
            if rs:
                rec.append(
                    f"{arm} sweet-trained {np.nanmean([r['sweet_trained'] for r in rs]):.3f} / "
                    f"bitter-trained {np.nanmean([r['bitter_trained'] for r in rs]):.3f} "
                    f"(gap {np.nanmean([r['gap'] for r in rs]):+.3f})"
                )
        if rec:
            lines.append("\nRecall of the last 30 rewarded items of each taste, at the end: " + "; ".join(rec))
        ret = [(arm, s["retention"]) for (t, arm), ss in by.items() if t == task for s in ss if s.get("retention")]
        if ret:
            lines.append(
                "\nRetention 24 h after the last reward (T2): "
                + "; ".join(f"{arm} {r['accuracy_24h']:.3f}" for arm, r in ret)
            )
    # the decision rule, applied
    lines.append(f"\n## Decision rule (GATE-{gate}.md)\n")
    verdicts = []
    for task in ("t1", "t2"):
        for k in ("k50", "k200") if task == "t1" else ("flip-k50", "flip-k200"):
            r = [s["windows"][k]["accuracy"] for s in by.get((task, "real"), []) if k in s["windows"]]
            if not r:
                continue
            line = f"- {task} {k}: real {np.mean(r):.3f} (sd {np.std(r):.3f})"
            ok = True
            for other in ("shuffle", "hash"):
                o = [s["windows"][k]["accuracy"] for s in by.get((task, other), []) if k in s["windows"]]
                if not o:
                    line += f"; {other} —"
                    ok = False
                    continue
                pooled = float(np.sqrt((np.var(r) + np.var(o)) / 2))
                beat = np.mean(r) - np.mean(o) > pooled
                ok &= bool(beat)
                line += f"; {other} {np.mean(o):.3f} ({'beaten' if beat else 'not beaten'} by more than pooled sd {pooled:.3f})"
            verdicts.append(ok)
            lines.append(line)
    if verdicts:
        lines.append(
            f"\n**Real wiring earns its place: {'YES' if all(verdicts) else 'NO'}** ({sum(verdicts)} of {len(verdicts)} comparisons)."
        )
    drv = G.DRIVE_FILE
    if drv.exists():
        d = json.load(open(drv))
        lines.append(
            f"\nDrive: scale {d['scale_hz']:.1f} Hz, {'centred, ' if d.get('mu') else ''}KC fraction over {d['n']} "
            f"calibration sentences mean {d['kc_mean']:.4f} (target {d['target']})."
        )
    txt = "\n".join(lines) + "\n"
    (paths.DOCS / f"gate-{G.GATE}-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("learn")
    p.add_argument("--task", required=True, choices=["t1", "t2", "t3"])
    p.add_argument("--arm", required=True, choices=list(BRAINS) + ["logistic"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", type=int, default=0)
    p.set_defaults(fn=learn)
    p = sub.add_parser("after")
    p.add_argument("--task", required=True, choices=["t4", "t5", "t6"])
    p.add_argument("--arm", required=True, choices=list(BRAINS) + ["logistic"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", type=int, default=0)
    p.set_defaults(fn=after)
    p = sub.add_parser("report")
    p.set_defaults(fn=report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
