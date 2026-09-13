"""Filesystem layout. Everything derives from the repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
CONFIG = ROOT / "config"
DOCS = ROOT / "docs"
SNAPSHOTS = ROOT / "snapshots"
STATE = ROOT / "state"
KERNEL_DIR = ROOT / "kernel"

# MaleCNS v1.0 flat connectome (gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/)
MCNS_ANNOTATIONS = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
MCNS_NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
MCNS_WEIGHTS = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# FlyWire (Schlegel et al. 2024) annotation table, used only to cross-reference
# cell types that MaleCNS names but does not describe (e.g. GRN modality).
FLYWIRE_ANNOTATIONS = RAW / "flywire" / "Supplemental_file1_neuron_annotations.tsv"

# Shiu et al. 2024 model repo (philshiu/Drosophila_brain_model), for the phase-1 gate.
SHIU = RAW / "shiu"
