"""Freeze the artifacts at `freeze-v1` (EXPERIMENT.md 3).

Two jobs, both mechanical, so the tag is not a hand-typed table:

1. `config/frozen_digests.json` -- the content digests of the things he says with
   (corpus, phrasebook, topics, identity).  Once it exists, `tests/test_corpus.py`
   pins them, and any later edit to those artifacts fails the suite.
2. The hash table for EXPERIMENT.md 3, printed as markdown to paste in.

Digests match how the code reads each artifact (Generator/Phrasebook/TopicMap/
IdentityReflex `.digest()`), not the bytes on disk, so a reformatting that changes
nothing he can say is not a false alarm.  The table alongside them is sha256 of the
file, which is what an auditor can check with `sha256sum`.

  uv run python scripts/freeze.py            # print; change nothing
  uv run python scripts/freeze.py --write    # write config/frozen_digests.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from bosco import paths

# (row label in EXPERIMENT.md 3, paths under the repo root)
ARTIFACTS: list[tuple[str, list[str]]] = [
    ("Phrasebook", ["phrasebook.yaml"]),
    ("Encoder spec", ["docs/encoder.md"]),
    ("Readout thresholds", ["config/thresholds.json"]),
    ("Threshold policy", ["config/thresholds_policy.yaml"]),
    ("Appetite", ["config/appetite_v1.yaml"]),
    ("Words as smells", ["config/words_v1.yaml"]),
    ("Learning rule", ["config/plasticity_v1.yaml", "config/mb_compartments.yaml"]),
    ("Retina", ["config/retina_v1.yaml"]),
    ("Association memory", ["config/associations_v1.yaml"]),
    ("Readout populations", ["config/readout_populations.yaml"]),
    ("Innate smells", ["config/innate_v1.yaml"]),
    ("Topics", ["config/topics_v1.yaml"]),
    ("Feeds", ["config/feeds_v1.yaml"]),
    ("Identity", ["config/identity_v1.yaml"]),
    ("Caps", ["config/caps_v1.yaml"]),
    ("Model", ["config/model_v1.yaml"]),
    ("Encoder config", ["config/encoder_v1.yaml"]),
    ("Circadian", ["config/circadian_v1.yaml"]),
    ("Moderation", ["config/moderation_v1.yaml"]),
]


def sha256_file(rel: str) -> str:
    h = hashlib.sha256()
    h.update((paths.ROOT / rel).read_bytes())
    return h.hexdigest()


def sha256_tree(rel: str, glob: str) -> str:
    """A directory, as the sha256 over each file's name and sha256, sorted: order-independent
    of the filesystem, and it changes if any file is added, removed or edited."""
    h = hashlib.sha256()
    for f in sorted((paths.ROOT / rel).glob(glob)):
        h.update(f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def content_digests() -> dict[str, str]:
    """What he can say, as the code reads it.  Mirrors tests/test_corpus.py exactly."""
    from bosco.identity import IdentityReflex
    from bosco.phrasebook import Phrasebook
    from bosco.textgen import Generator
    from bosco.topics import TopicMap

    pb = Phrasebook()
    return {
        "corpus": Generator(phrasebook_lines=[ln.text for ln in pb.lines]).digest(),
        "phrasebook": pb.digest(),
        "topics": TopicMap().digest(),
        "identity": IdentityReflex().digest(),
    }


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write config/frozen_digests.json")
    a = ap.parse_args(argv)

    digests = content_digests()
    out = paths.CONFIG / "frozen_digests.json"
    if a.write:
        if out.exists():
            print(f"{out} already exists; the artifacts are pinned. Delete it deliberately to re-freeze.")
            return 1
        out.write_text(json.dumps(digests, indent=1) + "\n")
        print(f"wrote {out.relative_to(paths.ROOT)}")
    else:
        print("(dry run; --write to pin)")
    for k, v in digests.items():
        print(f"  {k:10} {v}")

    rev, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    print("\n## EXPERIMENT.md 3 -- frozen artifacts\n")
    print("| Artifact | Location | Hash |")
    print("|---|---|---|")
    print(f"| Code | proto-cool/bosco @ `freeze-v1` | `{rev}` |")
    print("| Weights snapshot | `snapshots/freeze-v1/` | (sha256 of `brain_state.npz` at the tag) |")
    for label, rels in ARTIFACTS:
        loc = ", ".join(f"`{r}`" for r in rels)
        print(f"| {label} | {loc} | {' '.join('`' + sha256_file(r)[:16] + '`' for r in rels)} |")
    print(f"| Corpus | `corpus/` | `{sha256_tree('corpus', '*.txt')[:16]}` |")
    print("\nsha256, first 16 hex; `sha256sum <file>` to check. Content digests above are the")
    print("code's reading of the same artifacts and are what the test suite pins.")
    if dirty:
        print(f"\n!! working tree is dirty ({len(dirty.splitlines())} paths); the tag must be clean")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
