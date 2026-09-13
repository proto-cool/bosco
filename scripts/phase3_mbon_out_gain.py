"""Choose mbon_out_gain: MBON output synapses x gain so that learned MBON
changes reach the action populations.  Criteria: after 3 reward pairings on
an account odor, approach populations (engage+like) rise relative to naive
and after 3 punishment pairings leave rises; unpaired account unchanged;
sugar reflex intact; no persistent activity."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.encoder import Encoder, Features
from bosco.kernel import Net
from bosco.model import load_or_build
from bosco.plasticity import MushroomBody
from bosco.readout import Readout
from bosco.sim import Drive, Fly, Stimulus, load_params


def main() -> int:
    b = load_or_build()
    p, cfg = load_params()
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    e_pre = b.pre_of_edges()
    mbon_out = np.isin(e_pre, mbon)
    sugar = b.index_of_present(pop.grns("sugar/water"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    for gain in [float(x) for x in sys.argv[1:]] or [1, 3, 5, 10]:
        fly = Fly(b)
        w = fly.base_w.copy()
        w[mbon_out] *= gain
        fly.base_w = w
        fly.net = Net(b.indptr, b.indices, w, p)
        enc = Encoder(b)
        ro = Readout(b)
        mb = MushroomBody(fly)
        A = Features("did:plc:alice", 0.0, True, 0)
        B = Features("did:plc:bob", 0.0, True, 0)

        def sc(f, seed=1):
            res = fly.run_episode(enc.encode(f), seed)
            d = ro.scores(res, fly.episode_ms)
            return d, int((res.counts > 0).sum())

        out = []
        for val in ("reward", "punishment"):
            fly.set_multiplier(np.ones_like(fly.multiplier))
            mb.t_last = 0.0
            a0, act0 = sc(A)
            b0, _ = sc(B)
            for n in range(3):
                mb.pair(enc.encode(A), val, seed=100 + n, t_hours=0.0)
            a1, act1 = sc(A)
            b1, _ = sc(B)
            out.append(
                f"{val[:3]}: A engage {a0['engage']:.1f}->{a1['engage']:.1f} like {a0['like']:.1f}->{a1['like']:.1f} "
                f"leave {a0['leave']:.1f}->{a1['leave']:.1f} | B engage {b0['engage']:.1f}->{b1['engage']:.1f} leave {b0['leave']:.1f}->{b1['leave']:.1f} | act {act0}->{act1}"
            )
        fly.set_multiplier(np.ones_like(fly.multiplier))
        res = fly.run_episode(Stimulus([Drive(sugar, 100.0)]), 1)
        mn9r = res.rate(mn9, fly.episode_ms)
        # persistence after odor off
        fly.net.set_inputs(*enc.encode(A).merged(), p.input_jump_mv)
        fly.net.reset(1)
        fly.net.run_ms(500)
        fly.net.clear_inputs()
        fly.net.run_ms(300)
        c2 = fly.net.spike_counts().copy()
        fly.net.run_ms(200)
        post = int((fly.net.spike_counts() - c2).sum())
        print(f"mbon_out_gain {gain:4.1f} | " + " | ".join(out) + f" | MN9@100 {mn9r:.0f}Hz post {post}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
