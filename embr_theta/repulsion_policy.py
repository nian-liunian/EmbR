"""Gaussian V_rep site policy: H-only repulsion (A_O = 0) by default."""

from __future__ import annotations

import numpy as np

from embr_theta.amp_override import MM_EL_H, MM_EL_O, apply_amp_overrides

DEFAULT_H_ONLY = True
REPULSION_POLICY_H_ONLY = "h_only"
REPULSION_POLICY_BOTH = "o_and_h"


def resolve_h_only(
    *,
    h_only: bool | None = None,
    allow_o_repulsion: bool = False,
    meta: dict | None = None,
) -> bool:
    """Return whether MM O sites should have zero repulsion amplitude."""
    if h_only is not None:
        return bool(h_only)
    if allow_o_repulsion:
        return False
    if meta is not None:
        if meta.get("h_only_repulsion") is not None:
            return bool(meta["h_only_repulsion"])
        pol = meta.get("repulsion_policy")
        if pol == REPULSION_POLICY_H_ONLY:
            return True
        if pol == REPULSION_POLICY_BOTH:
            return False
    return DEFAULT_H_ONLY


def amplitude_bounds(
    n: int,
    mm_element: np.ndarray | None,
    *,
    h_only: bool,
    a_max: float | None,
    a_o_min: float = 0.0,
    a_h_min: float = 0.0,
    a_o_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lb, ub) for per-MM amplitude least-squares fit."""
    lb = np.zeros(n, dtype=np.float64)
    if a_max is not None and float(a_max) > 0.0:
        ub = np.full(n, float(a_max), dtype=np.float64)
    else:
        ub = np.full(n, np.inf, dtype=np.float64)

    if mm_element is not None:
        el = np.asarray(mm_element, dtype=np.int8).reshape(-1)
        if el.size != n:
            raise ValueError(f"mm_element length {el.size} != {n}")
        lb[el == MM_EL_O] = float(a_o_min)
        lb[el == MM_EL_H] = float(a_h_min)
        if a_o_max is not None:
            ub[el == MM_EL_O] = float(a_o_max)

    if h_only:
        if mm_element is None:
            raise ValueError("mm_element required for h_only repulsion fit")

    if np.any(lb > ub):
        raise ValueError("infeasible bounds: some lb > ub")
    if np.any(lb >= ub):
        raise ValueError(
            "infeasible bounds: scipy requires lb < ub on every site; "
            "use h_only fit (O excluded from LSQ) or widen bounds"
        )

    return lb, ub


def enforce_h_only_amplitudes(
    amp_mm: np.ndarray,
    mm_element: np.ndarray | None,
    *,
    h_only: bool,
) -> np.ndarray:
    if not h_only or mm_element is None:
        return np.asarray(amp_mm, dtype=np.float64)
    return apply_amp_overrides(np.asarray(amp_mm, dtype=np.float64), mm_element, set_a_o=0.0)


def meta_h_only_fields(h_only: bool) -> dict:
    return {
        "h_only_repulsion": bool(h_only),
        "repulsion_policy": REPULSION_POLICY_H_ONLY if h_only else REPULSION_POLICY_BOTH,
    }
