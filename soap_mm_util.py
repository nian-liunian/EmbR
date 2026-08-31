"""Small helpers for soap_mm_* (no gly_water_descriptor / train_stub)."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """
    Seed Python / NumPy / PyTorch RNGs for reproducible train from scratch.

    Call once before ``build_mmh_model`` when ``--seed`` should fix init + split + batches.
    """
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    key = name.strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but torch.cuda.is_available() is False")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if key == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {name!r}; use auto, cuda, or cpu")


def print_device(device: torch.device) -> None:
    if device.type == "cuda":
        print(f"device=cuda  {torch.cuda.get_device_name(device)}")
    else:
        print("device=cpu")
