"""Where each synapse lands: dendrite-side input vs input onto a neuron's own output terminals (Nick, 2026-09-29).

A one-unit rate neuron cannot tell a synapse on its dendrites from one on its axon terminals, yet the second mostly
modulates release rather than driving the cell (e.g. lateral-horn inhibition onto PN axon terminals; KC->KC in the
lobes, Manoim et al. 2022). From MaleCNS's synapse-level table (syn-partners, the region of each synapse,
`primary_post`), per neuron and region:
- an INPUT REGION is one holding at least as large a share of the neuron's input synapses as of its output synapses
  (a first rule, "receives >= sends there", wrongly dropped 72% of PN->LH drive and all of an ORN's input);
  synapses outside any named region ("<unspecified>") count as input;
- a synapse DRIVES its target only if it lands in one of the target's input regions.
Output: per connection onto a model neuron, its driving synapse count and its total (data/cache/mcns_v1_drive.parquet).
MaleCNS v1.0, CC BY 4.0 (data/raw/synapses/FETCH.log has the checksums).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from bosco import model2 as M2
from bosco import paths

SYN = paths.RAW / "synapses" / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
CACHE = paths.CACHE / "mcns_v1_drive.parquet"
NODE_CACHE = paths.CACHE / "mcns_v1_regions.npz"
CHUNK = 64  # record batches per chunk


def _batches(reader, rois: dict):
    """Yield (pre bodies, post bodies, global region ids) per chunk of record batches."""
    n = reader.num_record_batches
    for s in range(0, n, CHUNK):
        tbl = pa.Table.from_batches([reader.get_batch(i) for i in range(s, min(s + CHUNK, n))],
                                    schema=reader.schema)
        col = tbl.column("primary_post").combine_chunks()
        names = col.dictionary.to_pylist()
        remap = np.array([rois.setdefault(x, len(rois)) for x in names], dtype=np.int32)
        yield (tbl.column("body_pre").to_numpy(), tbl.column("body_post").to_numpy(),
               remap[col.indices.to_numpy(zero_copy_only=False)])


def regions_per_neuron(max_roi: int = 512):
    """Pass 1: synapses received and sent per model neuron and region. Returns (ids, rois, n_in, n_out)."""
    if NODE_CACHE.exists():
        z = np.load(NODE_CACHE, allow_pickle=False)
        return z["ids"], list(z["rois"]), z["n_in"], z["n_out"]
    ids = M2.load_or_build().brain.ids
    rois: dict[str, int] = {}
    n_in = np.zeros((len(ids), max_roi), np.int32)
    n_out = np.zeros((len(ids), max_roi), np.int32)
    r = ipc.open_file(SYN)
    t0 = time.time()
    for pre, post, roi in _batches(r, rois):
        pi = np.searchsorted(ids, pre)
        pi[pi >= len(ids)] = 0
        okp = ids[pi] == pre
        qi = np.searchsorted(ids, post)
        qi[qi >= len(ids)] = 0
        okq = ids[qi] == post
        n_in += np.bincount(qi[okq] * max_roi + roi[okq], minlength=n_in.size).reshape(n_in.shape).astype(np.int32)
        n_out += np.bincount(pi[okp] * max_roi + roi[okp], minlength=n_out.size).reshape(n_out.shape).astype(np.int32)
    names = [k for k, _ in sorted(rois.items(), key=lambda kv: kv[1])]
    n_in, n_out = n_in[:, : len(names)], n_out[:, : len(names)]
    np.savez(NODE_CACHE, ids=ids, rois=np.array(names), n_in=n_in, n_out=n_out)
    print(f"pass 1: {time.time() - t0:.0f}s, {len(names)} regions")
    return ids, names, n_in, n_out


def input_regions(n_in, n_out, rois) -> np.ndarray:
    """(neurons, regions) bool: a region is an input region if it holds at least as large a share of the neuron's
    input synapses as of its output synapses (so a neuron that sends more than it receives overall keeps its
    dendritic field, and a neuron with one output region, like an ORN, keeps every input there); unnamed = input."""
    in_share = n_in / np.maximum(n_in.sum(1, keepdims=True), 1)
    out_share = n_out / np.maximum(n_out.sum(1, keepdims=True), 1)
    inp = in_share >= out_share
    if "<unspecified>" in rois:
        inp[:, rois.index("<unspecified>")] = True
    return inp


def drive_table() -> pd.DataFrame:
    """Pass 2: per connection onto a model neuron, synapses in total and in the target's input regions."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    ids, names, n_in, n_out = regions_per_neuron()
    inp = input_regions(n_in, n_out, names)
    rois = {k: i for i, k in enumerate(names)}
    r = ipc.open_file(SYN)
    parts = []
    t0 = time.time()
    for pre, post, roi in _batches(r, rois):
        qi = np.searchsorted(ids, post)
        qi[qi >= len(ids)] = 0
        ok = ids[qi] == post
        pre, qi, roi = pre[ok], qi[ok], roi[ok]
        roi = np.minimum(roi, inp.shape[1] - 1)
        drive = inp[qi, roi]
        df = pd.DataFrame({"body_pre": pre, "post": qi, "drive": drive.astype(np.int32)})
        g = df.groupby(["body_pre", "post"], sort=False).drive.agg(["size", "sum"])
        parts.append(g)
    g = pd.concat(parts).groupby(level=[0, 1]).sum()
    out = g.reset_index().rename(columns={"size": "total", "sum": "drive"})
    out["body_post"] = ids[out.post.to_numpy()]
    out = out[["body_pre", "body_post", "total", "drive"]]
    out.to_parquet(CACHE)
    print(f"pass 2: {time.time() - t0:.0f}s, {len(out)} connections")
    return out
