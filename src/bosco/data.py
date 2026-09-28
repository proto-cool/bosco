"""Loaders for the raw connectome tables.

Nothing here is model-specific; it only reads the published feathers and
returns pandas frames with stable column names.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pyarrow.feather as pf

from bosco import paths

# Neurotransmitter -> synaptic sign.  Same convention as Shiu et al. 2024:
# acetylcholine excitatory; GABA and glutamate inhibitory.  Monoamines are
# modelled as excitatory at the fast-synapse level (they are also the
# modulatory signal in the plasticity rule, handled separately).
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "dopamine": +1,
    "serotonin": +1,
    "octopamine": +1,
    "histamine": -1,
    "unclear": +1,
    "unknown": +1,
}


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    """MaleCNS body annotations, indexed by bodyId."""
    df = pf.read_feather(paths.MCNS_ANNOTATIONS)
    df = df.set_index("bodyId", drop=False)
    return df


@lru_cache(maxsize=1)
def neurotransmitters() -> pd.DataFrame:
    """Per-body NT predictions, indexed by body."""
    df = pf.read_feather(paths.MCNS_NT)
    return df.set_index("body", drop=False)


def weights() -> pd.DataFrame:
    """Full segment-to-segment synapse counts: columns body_pre, body_post, weight."""
    return pf.read_feather(paths.MCNS_WEIGHTS)


@lru_cache(maxsize=1)
def flywire_annotations() -> pd.DataFrame:
    return pd.read_csv(paths.FLYWIRE_ANNOTATIONS, sep="\t", low_memory=False)


def nt_sign_for_bodies(body_ids: pd.Index | list[int]) -> pd.Series:
    """Signed (+1/-1) transmitter sign per body, using consensus_nt with fallbacks.

    Bodies with no prediction at all default to excitatory (+1); the count of
    such bodies is reported in the phase-0 coverage report.
    """
    nt = neurotransmitters()
    idx = pd.Index(body_ids)
    cons = nt.reindex(idx)["consensus_nt"]
    pred = nt.reindex(idx)["predicted_nt"]
    label = cons.where(cons.notna(), pred).fillna("unknown").str.lower()
    return label.map(NT_SIGN).fillna(1).astype("int8")
