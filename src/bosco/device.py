"""The torch device: BOSCO_DEVICE if set, else CUDA (the 3080), else MPS (the Mac), else CPU."""

from __future__ import annotations

import os

import torch


def default() -> str:
    if os.environ.get("BOSCO_DEVICE"):
        return os.environ["BOSCO_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
