"""Simulation: the network running continuously, one biological second per wall
second, with state carried across everything that happens.

There are no episodes from rest.  Time advances in slices; stimuli are
presented for a duration on top of the base drive (clock + bristle debris);
decisions are read from spike counts over a presentation window or over
idle one-second slices; plasticity acts on the counts of the window.

Determinism: the kernel state (voltages, conductances, adaptation, delay
ring, RNG) plus the mushroom-body state fully determine the future given
the input timeline.  `snapshot()`/`restore()` capture both, so any span can
be replayed from the snapshot before it and the logged inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from bosco.sim import Drive, Fly

SLICE_MS = 1000.0


@dataclass
class Window:
    """Spike counts accumulated over a span of the run."""

    t0_ms: int
    t1_ms: int
    counts: np.ndarray

    @property
    def ms(self) -> float:
        return float(self.t1_ms - self.t0_ms)

    def rate(self, idx: np.ndarray) -> float:
        if len(idx) == 0 or self.ms <= 0:
            return 0.0
        return float(self.counts[idx].mean() * 1000.0 / self.ms)


class Simulation:
    """Bosco's network, running continuously.  Shares the Fly's Net, so the mushroom body's
    weight updates (fly.set_multiplier) apply to the simulation."""

    def __init__(self, fly: Fly, seed: int = 0) -> None:
        self.fly = fly
        self.brain = fly.brain
        self.params = fly.params
        self.net = fly.net
        self.net.reset(seed=seed)
        self.t_ms = 0  # biological time since brain start
        self.base: dict[str, Drive] = {}

    # ---- inputs -----------------------------------------------------------------
    def set_base(self, drives: dict[str, Drive]) -> None:
        """Background inputs that persist across slices (clock, bristle debris)."""
        self.base = dict(drives)
        self._apply_inputs([])

    def _apply_inputs(self, extra: list[Drive]) -> None:
        drives = list(self.base.values()) + list(extra)
        if not drives:
            self.net.clear_inputs()
            return
        idx = np.concatenate([d.idx for d in drives]).astype(np.int32)
        rate = np.concatenate([np.full(len(d.idx), d.rate_hz) for d in drives])
        order = np.argsort(idx, kind="stable")
        self.net.set_inputs(idx[order], rate[order], self.params.input_jump_mv)

    # ---- time -------------------------------------------------------------------
    def _run(self, ms: float) -> None:
        self.net.run(int(round(ms / self.params.dt_ms)))
        self.t_ms += int(round(ms))

    def present(self, drives: list[Drive], ms: float = SLICE_MS) -> Window:
        """Present drives on top of the base for ms; return the window's counts."""
        c0 = self.net.spike_counts()
        t0 = self.t_ms
        self._apply_inputs(drives)
        self._run(ms)
        self._apply_inputs([])
        return Window(t0, self.t_ms, self.net.spike_counts() - c0)

    def idle(self, ms: float = SLICE_MS) -> Window:
        """Advance with only the base drive."""
        c0 = self.net.spike_counts()
        t0 = self.t_ms
        self._run(ms)
        return Window(t0, self.t_ms, self.net.spike_counts() - c0)

    # ---- state ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"kernel": np.frombuffer(self.net.get_state(), dtype=np.uint8).copy(), "t_ms": np.array([self.t_ms])}

    def restore(self, snap: dict) -> None:
        self.net.set_state(np.asarray(snap["kernel"], dtype=np.uint8).tobytes())
        self.t_ms = int(np.asarray(snap["t_ms"]).ravel()[0])

    def digest(self) -> str:
        return hashlib.blake2b(self.net.get_state(), digest_size=16).hexdigest()
