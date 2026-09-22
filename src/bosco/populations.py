"""Named neuron populations drawn from MaleCNS annotations.

Every selector returns a sorted list of bodyIds.  Selection is by annotated
type/class, never by single hand-picked body, except where a population is
literally one cell type (APL, DPM).
"""

from __future__ import annotations

import pandas as pd

from bosco.data import annotations, flywire_annotations

# ---- Mushroom body ---------------------------------------------------------

KC_MAIN_LOBES = ("ab", "a'b'", "g")


def kenyon_cells() -> list[int]:
    a = annotations()
    return sorted(a.index[a["class"] == "Kenyon_Cell"])


def kc_by_lobe() -> dict[str, list[int]]:
    a = annotations()
    kc = a[a["class"] == "Kenyon_Cell"]
    out: dict[str, list[int]] = {"ab": [], "a'b'": [], "g": [], "untyped": []}
    for bid, t in kc["type"].items():
        t = t or ""
        if t.startswith("KCab"):
            out["ab"].append(bid)
        elif t.startswith("KCa'b'"):
            out["a'b'"].append(bid)
        elif t.startswith("KCg"):
            out["g"].append(bid)
        else:
            out["untyped"].append(bid)
    return {k: sorted(v) for k, v in out.items()}


def visual_kcs(types: tuple[str, ...] = ("KCg-d", "KCab-p")) -> list[int]:
    """The Kenyon cells that receive visual projection neurons in the fly (Vogt et al. 2016;
    Li et al. 2020): gamma-d and alpha/beta-posterior.  The retina drives these directly
    (config/retina_v1.yaml), their afferents having gone with the optic lobes."""
    a = annotations()
    return sorted(a.index[(a["class"] == "Kenyon_Cell") & a["type"].isin(list(types))])


def mbons() -> pd.DataFrame:
    a = annotations()
    m = a[a["class"] == "MBON"]
    return m[["bodyId", "type", "somaSide"]].sort_values(["type", "somaSide"])


def dans() -> pd.DataFrame:
    a = annotations()
    d = a[a["class"] == "DAN"]
    return d[["bodyId", "type", "somaSide"]].sort_values(["type", "somaSide"])


def apl() -> list[int]:
    a = annotations()
    return sorted(a.index[a["type"] == "APL"])


def dpm() -> list[int]:
    a = annotations()
    return sorted(a.index[a["type"] == "DPM"])


# ---- Sensory inputs --------------------------------------------------------


def orns() -> pd.DataFrame:
    """Olfactory receptor neurons with glomerulus label (type 'ORN_<glom>')."""
    a = annotations()
    o = a[(a["class"] == "olfactory") & a["type"].fillna("").str.startswith("ORN_")]
    df = o[["bodyId", "type", "somaSide"]].copy()
    df["glomerulus"] = df["type"].str[4:]
    return df.sort_values(["glomerulus", "somaSide"])


# Labellar GRN modality.  MaleCNS types labellar GRNs by bristle (LB1a..LB4b)
# and does not state modality; FlyWire (Schlegel et al. 2024) annotates the
# same cell types with cell_sub_class in {bitter, sugar/water, low-salt}.
# The mapping is verified at runtime in grn_modality_map() and reported in
# the phase-0 coverage report.
GRN_MODALITY_BY_TYPE: dict[str, str] = {
    "LB1a": "bitter",
    "LB1b": "bitter",
    "LB1c": "bitter",
    "LB1d": "bitter",
    "LB1e": "bitter",
    "LB2a": "low-salt",
    "LB2b": "low-salt",
    "LB2c": "low-salt",
    "LB2d": "sugar/water",
    "LB3": "sugar/water",
    "LB3a": "sugar/water",
    "LB3b": "sugar/water",
    "LB3c": "sugar/water",
    "LB3d": "sugar/water",
    "LB4a": "low-salt",
    # LB4b cross-references to FlyWire "LB3" but is a distinct MaleCNS type;
    # left unassigned rather than guessed.
}


def grn_modality_map() -> pd.DataFrame:
    """FlyWire cell_type -> cell_sub_class for gustatory neurons (for verification)."""
    fw = flywire_annotations()
    g = fw[fw["cell_class"] == "gustatory"]
    return (
        g.groupby("cell_type")["cell_sub_class"]
        .agg(lambda s: ",".join(sorted(set(s.dropna()))))
        .rename("flywire_sub_class")
        .reset_index()
    )


def labellar_grns() -> pd.DataFrame:
    a = annotations()
    g = a[(a["class"] == "gustatory") & (a["superclass"] == "cb_sensory")]
    g = g[g["type"].fillna("").str.startswith("LB")]
    df = g[["bodyId", "type", "flywireType", "somaSide"]].copy()
    df["modality"] = df["type"].map(GRN_MODALITY_BY_TYPE).fillna("unassigned")
    return df.sort_values(["modality", "type"])


def grns(modality: str) -> list[int]:
    df = labellar_grns()
    return sorted(df.loc[df["modality"] == modality, "bodyId"])


def johnston_organ() -> pd.DataFrame:
    """Antennal mechanosensory (JO) neurons, by JO subgroup letter."""
    a = annotations()
    j = a[a["type"].fillna("").str.startswith("JO-")]
    df = j[["bodyId", "type", "somaSide"]].copy()
    df["group"] = df["type"].str.extract(r"^JO-([A-F])")[0].fillna("unclear")
    return df.sort_values(["group", "type"])


# ---- Outputs ---------------------------------------------------------------


def descending_neurons() -> pd.DataFrame:
    a = annotations()
    d = a[a["superclass"].fillna("").str.startswith("descending_neuron")]
    return d[["bodyId", "type", "somaSide", "subclass"]].sort_values(["type", "somaSide"])


def pc1() -> list[int]:
    """pC1 (P1) neurons: the male courtship command hub (types beginning with pC1).  Their
    excitability tracks internal state; the song descending neurons sit downstream."""
    a = annotations()
    return sorted(a.index[a["type"].fillna("").str.startswith("pC1")])


def bodies_of_types(types: list[str]) -> list[int]:
    a = annotations()
    return sorted(a.index[a["type"].isin(types)])


# ---- Clock -----------------------------------------------------------------

CLOCK_TYPES = (
    "s-LNv",
    "l-LNv",
    "5thsLNv_LNd6",
    "LNd_b",
    "LNd_c",
    "DN1a",
    "DN1pA",
    "DN1pB",
    "LPN_a",
    "LPN_b",
)


def clock_neurons() -> pd.DataFrame:
    a = annotations()
    c = a[a["type"].isin(CLOCK_TYPES)]
    return c[["bodyId", "type", "somaSide"]].sort_values(["type", "somaSide"])
