"""Does this box compute his numbers the way the last one did?  Prints the CPU, numpy's code
path (Agent.numerics_level), glibc, and the digest of a fixed synthetic run of the whole loop
(four windows on a fresh brain).  Run it on both boxes (or on one box with and without
NPY_DISABLE_CPU_FEATURES) and compare the last line: equal digests mean spans replay bit for
bit across the two; different digests mean the move is a `numerics` boundary in the record,
which the agent logs by itself on its first start (docs/MIGRATE.md).

    podman exec bosco /app/.venv/bin/python /app/scripts/cpu_determinism.py
"""

from __future__ import annotations

import os
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from bosco.agent import Agent
from bosco.encoder import Features
from bosco.ledger import Ledger
from bosco.sim import Fly


def cpu_model() -> str:
    try:
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        return m.group(1).strip() if m else platform.processor()
    except OSError:
        return platform.processor()


def main() -> int:
    print("cpu      ", cpu_model())
    print("numpy    ", np.__version__, "level", Agent.numerics_level())
    print("glibc    ", platform.libc_ver()[1] or "?")
    print("disabled ", os.environ.get("NPY_DISABLE_CPU_FEATURES", "-"))
    tmp = Path(tempfile.mkdtemp())
    L = Ledger(tmp / "l.sqlite")
    ag = Agent(L, Fly(), state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    t0 = 1_800_000_000.0
    windows = [
        Features("did:plc:a", 0.5, True, 0, False, ("fruit",), True, ("banana",)),
        Features("did:plc:b", -0.6, False, 0, False, ("politics",), False, ("wall",), feed="discover"),
        Features("did:plc:c", 0.0, False, 1, False, (), False, ("sun", "leaf"), feed="science"),
        Features("did:plc:a", 0.3, True, 1, False, ("fruit",), False, ("grape",)),
    ]
    for i, f in enumerate(windows):
        ag.run(f, t0 + 400 * i, f"at://x/{i}", fast=True)
    print("digest   ", ag.digest(), "mb", ag.mb.digest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
