"""Override / clamp per-MM amplitudes (O/H sites in precompute order)."""

from __future__ import annotations

import numpy as np

MM_EL_O = 0
MM_EL_H = 1


def apply_amp_overrides(
    amp_mm: np.ndarray,
    mm_element: np.ndarray | None,
    *,
    set_a_o: float | None = None,
    set_a_h: float | None = None,
    floor_a_o: float | None = None,
    floor_a_h: float | None = None,
    scale_a_o: float | None = None,
    scale_a_h: float | None = None,
) -> np.ndarray:
    """
    Return a copy of ``amp_mm`` with optional per-element overrides.

    ``mm_element``: 0 = O, 1 = H (same layout as ``pert.npz`` / ``pert_peratom.npz``).
    """
    out = np.asarray(amp_mm, dtype=np.float64).copy()
    if mm_element is None:
        if any(
            x is not None
            for x in (set_a_o, set_a_h, floor_a_o, floor_a_h, scale_a_o, scale_a_h)
        ):
            raise ValueError("mm_element required for per-element amp overrides")
        return out

    el = np.asarray(mm_element, dtype=np.int8).reshape(-1)
    if el.size != out.size:
        raise ValueError(f"mm_element length {el.size} != amp_mm {out.size}")

    o = el == MM_EL_O
    h = el == MM_EL_H
    if set_a_o is not None:
        out[o] = float(set_a_o)
    if set_a_h is not None:
        out[h] = float(set_a_h)
    if scale_a_o is not None:
        out[o] *= float(scale_a_o)
    if scale_a_h is not None:
        out[h] *= float(scale_a_h)
    if floor_a_o is not None:
        out[o] = np.maximum(out[o], float(floor_a_o))
    if floor_a_h is not None:
        out[h] = np.maximum(out[h], float(floor_a_h))
    return out
