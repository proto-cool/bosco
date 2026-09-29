"""Load a design as a frozen RateBrain3 plus its rest smell and explore smells. Read-only."""
from __future__ import annotations

import sys

import numpy as np
import torch

TMP = "/Users/nickd/.claude/jobs/4059153c/tmp"


def load(design: str, G: float = 4.0):
    """design: C0 (run under the v30 venv), C1, C3 (saved candidates), C2 (osc_state.pt s_in/b_cell on v3.1)."""
    if design == "C0":
        sys.path.insert(0, f"{TMP}/v30/scripts")
        import brain_check as B
        from bosco import model2 as M2
        b2 = M2.load_or_build()
        m = B.brain("real", b2)
        B.load_start(m, "real")
    else:
        sys.path.insert(0, "/Users/nickd/projects/bosco/scripts")
        sys.path.insert(0, TMP)
        import brain_check as B
        from bosco import model2 as M2
        b2 = M2.load_or_build()
        if design in ("C1", "C3", "C1c", "C3c"):
            import candidates as C
            m = C.load(design[:2], G, b2=b2)
            if design.endswith("c"):  # read-cell scales clipped to [0.1, 100], as rest_start31 clips
                with torch.no_grad():
                    m.s_in.clamp_(0.1, 100.0)
                m.freeze()
        elif design == "C2":
            from bosco import v1
            st = torch.load(f"{TMP}/osc_state.pt")
            m = v1.build("real", device="cpu", b2=b2)
            with torch.no_grad():
                m.log_g.zero_(); m.b.zero_()
                m.s_in.copy_(st["s_in"]); m.b_cell.copy_(st["b_cell"])
            m.freeze()
            m.r_rest = None
        else:
            raise ValueError(design)
    ant = B.antenna()
    rest = torch.tensor(ant.resting()).float()
    X = np.load(f"{TMP}/explore_items.npz")["X"]
    smells = torch.tensor(ant(X)).float()
    from bosco import data
    ann = data.annotations().reindex(b2.brain.ids)
    return m, rest, smells, ann, b2
