"""ctypes binding for kernel/liblif.so.

The kernel is single-threaded and deterministic.  Build with `make -C kernel`.
"""

from __future__ import annotations

import ctypes as C
import subprocess
from dataclasses import dataclass

import numpy as np

from bosco import paths

_LIB_PATH = paths.KERNEL_DIR / "liblif.so"


class _Params(C.Structure):
    _fields_ = [
        ("dt_ms", C.c_double),
        ("tau_m", C.c_double),
        ("tau_s", C.c_double),
        ("v0", C.c_double),
        ("v_rst", C.c_double),
        ("v_th", C.c_double),
        ("t_rfc", C.c_double),
        ("t_dly", C.c_double),
        ("std_u", C.c_double),
        ("std_tau_rec", C.c_double),
    ]


@dataclass(frozen=True)
class LifParams:
    """Shiu et al. 2024 defaults (Kakaria & de Bivort 2017; Jürgensen 2021; Lazar 2021; Paul 2015)."""

    dt_ms: float = 0.1
    tau_m: float = 20.0
    tau_s: float = 5.0
    v0: float = -52.0
    v_rst: float = -52.0
    v_th: float = -45.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275  # mV per synapse; applied in Python when building weights
    f_poi: float = 250.0  # Poisson input jump = w_syn * f_poi mV
    # Short-term depression (Tsodyks & Markram 1997), off by default (Shiu has none).
    std_u: float = 0.0
    std_tau_rec: float = 0.0

    @property
    def input_jump_mv(self) -> float:
        return self.w_syn * self.f_poi


_lib: C.CDLL | None = None


def _load() -> C.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        subprocess.run(["make", "-C", str(paths.KERNEL_DIR)], check=True)
    lib = C.CDLL(str(_LIB_PATH))
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.lif_create.restype = C.c_void_p
    lib.lif_create.argtypes = [C.c_int32, i64p, i32p, f64p, C.POINTER(_Params)]
    lib.lif_free.argtypes = [C.c_void_p]
    lib.lif_reset.argtypes = [C.c_void_p, C.c_uint64]
    lib.lif_set_weights.argtypes = [C.c_void_p, i64p, f64p, C.c_int64]
    lib.lif_get_weights.argtypes = [C.c_void_p, f64p]
    lib.lif_nnz.restype = C.c_int64
    lib.lif_nnz.argtypes = [C.c_void_p]
    lib.lif_set_inputs.argtypes = [C.c_void_p, C.c_int32, i32p, f64p, f64p]
    lib.lif_run.restype = C.c_int64
    lib.lif_run.argtypes = [C.c_void_p, C.c_int64, i32p, i32p, C.c_int64]
    lib.lif_spike_counts.argtypes = [C.c_void_p, i64p]
    lib.lif_get_v.argtypes = [C.c_void_p, f64p]
    lib.lif_get_x.argtypes = [C.c_void_p, f64p]
    lib.lif_set_std_u.argtypes = [C.c_void_p, f64p]
    lib.lif_step.restype = C.c_int64
    lib.lif_step.argtypes = [C.c_void_p]
    _lib = lib
    return lib


class Net:
    """A LIF network over a CSR (by presynaptic index) weight matrix in mV."""

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        w_mv: np.ndarray,
        params: LifParams = LifParams(),
    ) -> None:
        self.lib = _load()
        self.params = params
        self.n = int(len(indptr) - 1)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        w = np.ascontiguousarray(w_mv, dtype=np.float64)
        assert len(self.indices) == len(w) == self.indptr[-1]
        p = _Params(
            params.dt_ms, params.tau_m, params.tau_s, params.v0, params.v_rst,
            params.v_th, params.t_rfc, params.t_dly, params.std_u, params.std_tau_rec,
        )
        self._h = self.lib.lif_create(self.n, self.indptr, self.indices, w, C.byref(p))
        if not self._h:
            raise MemoryError("lif_create failed")

    def __del__(self) -> None:
        h = getattr(self, "_h", None)
        if h:
            self.lib.lif_free(h)
            self._h = None

    @property
    def nnz(self) -> int:
        return int(self.lib.lif_nnz(self._h))

    def reset(self, seed: int) -> None:
        self.lib.lif_reset(self._h, C.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))

    def set_weights(self, edge_idx: np.ndarray, w_mv: np.ndarray) -> None:
        e = np.ascontiguousarray(edge_idx, dtype=np.int64)
        w = np.ascontiguousarray(w_mv, dtype=np.float64)
        self.lib.lif_set_weights(self._h, e, w, len(e))

    def get_weights(self) -> np.ndarray:
        out = np.empty(self.nnz, dtype=np.float64)
        self.lib.lif_get_weights(self._h, out)
        return out

    def set_inputs(self, idx: np.ndarray, rate_hz: np.ndarray, jump_mv: np.ndarray | float | None = None) -> None:
        idx = np.ascontiguousarray(idx, dtype=np.int32)
        rate = np.ascontiguousarray(rate_hz, dtype=np.float64)
        if jump_mv is None:
            jump_mv = self.params.input_jump_mv
        jump = np.ascontiguousarray(np.broadcast_to(np.asarray(jump_mv, dtype=np.float64), idx.shape))
        self.lib.lif_set_inputs(self._h, len(idx), idx, rate, jump)

    def clear_inputs(self) -> None:
        self.set_inputs(np.zeros(0, np.int32), np.zeros(0), np.zeros(0))

    def run(self, n_steps: int, max_spikes: int = 5_000_000) -> tuple[np.ndarray, np.ndarray]:
        """Advance n_steps; return (spike_step, spike_neuron) arrays."""
        t = np.empty(max_spikes, dtype=np.int32)
        i = np.empty(max_spikes, dtype=np.int32)
        k = int(self.lib.lif_run(self._h, n_steps, t, i, max_spikes))
        if k > max_spikes:
            raise RuntimeError(f"spike buffer overflow: {k} > {max_spikes}")
        return t[:k].copy(), i[:k].copy()

    def run_ms(self, ms: float, **kw) -> tuple[np.ndarray, np.ndarray]:
        return self.run(int(round(ms / self.params.dt_ms)), **kw)

    def spike_counts(self) -> np.ndarray:
        out = np.empty(self.n, dtype=np.int64)
        self.lib.lif_spike_counts(self._h, out)
        return out

    def set_std_u(self, u: np.ndarray) -> None:
        """Per-presynaptic-neuron depression utilisation. Requires params.std_u > 0 to enable STD at all;
        std_u then acts as the default and this array overrides it per neuron."""
        self.lib.lif_set_std_u(self._h, np.ascontiguousarray(u, dtype=np.float64))

    def x(self) -> np.ndarray:
        out = np.empty(self.n, dtype=np.float64)
        self.lib.lif_get_x(self._h, out)
        return out

    def v(self) -> np.ndarray:
        out = np.empty(self.n, dtype=np.float64)
        self.lib.lif_get_v(self._h, out)
        return out

    @property
    def step(self) -> int:
        return int(self.lib.lif_step(self._h))


def csr_from_edges(n: int, pre: np.ndarray, post: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR (by pre) with rows sorted by (pre, post). Duplicate edges are summed."""
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order]
    # merge duplicates
    key = pre * n + post
    uniq, start = np.unique(key, return_index=True)
    if len(uniq) != len(key):
        w = np.add.reduceat(w, start)
        pre, post = pre[start], post[start]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, post.astype(np.int32), w
