"""Load SOAP + theta caches for the standalone theta pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_soap_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = ("desc_flat", "qm_fit_flat", "ptr", "meta_json")
    for k in required:
        if k not in z:
            raise ValueError(f"{path}: missing {k!r}")
    meta = json.loads(str(z["meta_json"]))
    if meta.get("pipeline") not in ("soap_theta", "soap_theta_mix"):
        raise ValueError(
            f"{path}: expected pipeline soap_theta, got {meta.get('pipeline')!r}. "
            "Use theta/precompute_soap.py — not precompute_soap_e0_mix.py / mix.npz"
        )
    return {
        "desc_flat": np.asarray(z["desc_flat"], dtype=np.float32),
        "qm_fit_flat": np.asarray(z["qm_fit_flat"], dtype=np.int8),
        "ptr": np.asarray(z["ptr"], dtype=np.int64),
        "residue_id": np.asarray(z["residue_id"], dtype=np.int16) if "residue_id" in z else None,
        "meta": meta,
    }


def load_theta_labels(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = ("amp_o", "amp_h", "alpha", "e0", "e_high", "err_kcal", "meta_json")
    for k in required:
        if k not in z:
            raise ValueError(f"{path}: missing {k!r}")
    meta = json.loads(str(z["meta_json"]))
    return {
        "amp_o": np.asarray(z["amp_o"], dtype=np.float64),
        "amp_h": np.asarray(z["amp_h"], dtype=np.float64),
        "alpha": np.asarray(z["alpha"], dtype=np.float64),
        "e0": np.asarray(z["e0"], dtype=np.float64),
        "e_int_bg": np.asarray(z["e_int_bg"], dtype=np.float64) if "e_int_bg" in z else None,
        "e_high": np.asarray(z["e_high"], dtype=np.float64),
        "err_kcal": np.asarray(z["err_kcal"], dtype=np.float64),
        "rho_o_mean": np.asarray(z["rho_o_mean"], dtype=np.float64) if "rho_o_mean" in z else None,
        "residue_id": np.asarray(z["residue_id"], dtype=np.int16) if "residue_id" in z else None,
        "meta": meta,
    }


def n_frames(cache: dict) -> int:
    return int(cache["ptr"].shape[0]) - 1


def frame_slice(cache: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    ptr = cache["ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    return cache["desc_flat"][s:e], cache["qm_fit_flat"][s:e]


def theta_row(labels: dict, k: int) -> tuple[float, float, float]:
    return float(labels["amp_o"][k]), float(labels["amp_h"][k]), float(labels["alpha"][k])


def assert_aligned(soap_cache: dict, theta_labels: dict) -> None:
    n_soap = n_frames(soap_cache)
    n_theta = int(theta_labels["amp_o"].shape[0])
    if n_soap != n_theta:
        raise ValueError(
            f"SOAP cache has {n_soap} frames but theta labels have {n_theta}. "
            "Use the same theta/manifest.json for precompute_soap and optimize_labels."
        )


def load_pert_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = ("e0", "kernel_o", "kernel_h", "meta_json")
    for k in required:
        if k not in z:
            raise ValueError(f"{path}: missing {k!r}")
    meta = json.loads(str(z["meta_json"]))
    pl = meta.get("pipeline")
    if pl not in (
        "pert_kernels",
        "pert_kernels_rep",
        "pert_kernels_smear",
        "pert_kernels_ohbond",
    ):
        raise ValueError(
            f"{path}: expected pipeline pert_kernels, pert_kernels_rep, "
            f"pert_kernels_smear, or pert_kernels_ohbond, got {pl!r}"
        )
    out = {
        "e0": np.asarray(z["e0"], dtype=np.float64),
        "e_int_bg": np.asarray(z["e_int_bg"], dtype=np.float64) if "e_int_bg" in z else None,
        "kernel_o": np.asarray(z["kernel_o"], dtype=np.float64),
        "kernel_h": np.asarray(z["kernel_h"], dtype=np.float64),
        "rho_o_mean": np.asarray(z["rho_o_mean"], dtype=np.float64) if "rho_o_mean" in z else None,
        "residue_id": np.asarray(z["residue_id"], dtype=np.int16) if "residue_id" in z else None,
        "meta": meta,
    }
    if "kernel_mm_flat" in z and "mm_ptr" in z:
        out["kernel_mm_flat"] = np.asarray(z["kernel_mm_flat"], dtype=np.float64)
        out["mm_ptr"] = np.asarray(z["mm_ptr"], dtype=np.int64)
        out["mm_element"] = np.asarray(z["mm_element"], dtype=np.int8) if "mm_element" in z else None
    return out


def n_pert_frames(pert: dict) -> int:
    return int(pert["e0"].shape[0])


def require_per_mm_kernels(pert: dict) -> None:
    if "kernel_mm_flat" not in pert or "mm_ptr" not in pert:
        raise ValueError(
            "pert cache has no per-MM kernels (kernel_mm_flat / mm_ptr). "
            "Re-run: python -m embr_theta.precompute_pert --manifest ... --out theta/pert.npz "
            "(or precompute_pert_ohbond / precompute_pert_smear)"
        )


def mm_frame_slice(pert: dict, k: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (kernel_mm, mm_element) for frame k."""
    require_per_mm_kernels(pert)
    ptr = pert["mm_ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    k_mm = pert["kernel_mm_flat"][s:e]
    el = pert["mm_element"][s:e] if pert.get("mm_element") is not None else None
    return np.asarray(k_mm, dtype=np.float64), None if el is None else np.asarray(el, dtype=np.int8)


def assert_pert_aligned(soap_cache: dict, pert_cache: dict) -> None:
    n_soap = n_frames(soap_cache)
    n_pert = int(pert_cache["e0"].shape[0])
    if n_soap != n_pert:
        raise ValueError(
            f"SOAP cache has {n_soap} frames but pert cache has {n_pert}. "
            "Use the same theta/manifest.json for precompute_soap and precompute_pert."
        )


def load_pert_peratom_labels(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    required = ("e0", "amp_mm_flat", "mm_ptr", "meta_json")
    for k in required:
        if k not in z:
            raise ValueError(f"{path}: missing {k!r}")
    meta = json.loads(str(z["meta_json"]))
    out = {
        "e0": np.asarray(z["e0"], dtype=np.float64),
        "err_kcal": np.asarray(z["err_kcal"], dtype=np.float64) if "err_kcal" in z else None,
        "amp_mm_flat": np.asarray(z["amp_mm_flat"], dtype=np.float64),
        "mm_ptr": np.asarray(z["mm_ptr"], dtype=np.int64),
        "mm_element": np.asarray(z["mm_element"], dtype=np.int8) if "mm_element" in z else None,
        "meta": meta,
    }
    if "kernel_mm_flat" in z:
        out["kernel_mm_flat"] = np.asarray(z["kernel_mm_flat"], dtype=np.float64)
    if "e_j_flat" in z:
        out["e_j_flat"] = np.asarray(z["e_j_flat"], dtype=np.float64)
    if "delta_e_kcal" in z:
        out["delta_e_kcal"] = np.asarray(z["delta_e_kcal"], dtype=np.float64)
    return out


def amp_mm_frame_slice(labels: dict, k: int) -> np.ndarray:
    ptr = labels["mm_ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    return np.asarray(labels["amp_mm_flat"][s:e], dtype=np.float64)
