"""
Map per-MM-site ML outputs e_i [kcal/mol] to repulsive amplitudes A_i^ML [Hartree].

Requires Emb0 kernels k_i (``kernel_mm`` from precompute) and Σ_i e_i = ΔE_ML [kcal/mol].

    A_i^ML = e_i / (627.5 · k_i)

Then 627.5 · Σ_i A_i^ML k_i = Σ_i e_i = ΔE_ML  (first-order, fixed Emb0 density).

EmbR repulsion uses atom-centered exponential envelopes (see ``embr_envelope``); internal
code may still use legacy names ``A_j``, ``k_j`` for checkpoint / NPZ compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.5094740631

# Re-export element ids (canonical definitions live in embr_elements).
from embr_elements import (
    MMH_ELEM_C,
    MMH_ELEM_CL,
    MMH_ELEM_H,
    MMH_ELEM_K,
    MMH_ELEM_N,
    MMH_ELEM_NA,
    MMH_ELEM_O,
    MMH_ELEM_NAMES,
    mmh_element_from_symbol,
)

def amp_h_flat_from_ej_kernel(
    e_j_kcal: np.ndarray,
    kernel_mm: np.ndarray,
    *,
    k_min: float = 1e-30,
) -> np.ndarray:
    """Per-site A [Hartree] from partition e_j and aligned k_j."""
    return ei_kcal_to_amp_hartree(e_j_kcal, kernel_mm, k_min=float(k_min))


def load_frame_amp_labels(path: Path, *, n_frames: int) -> np.ndarray:
    """One scalar A [Hartree] per frame — npz ``amp``/``A``/``amp_frame`` or whitespace txt."""
    path = Path(path)
    if path.suffix.lower() in (".txt", ".dat", ".csv"):
        vals: list[float] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(float(s.split()[0]))
        out = np.asarray(vals, dtype=np.float64)
        if int(out.size) != int(n_frames):
            raise ValueError(f"{path}: {out.size} lines != n_frames={n_frames}")
        return out
    z = np.load(path, allow_pickle=False)
    for key in ("amp", "A", "amp_frame", "amp_scalar", "A_frame", "A_scalar"):
        if key in z:
            out = np.asarray(z[key], dtype=np.float64).reshape(-1)
            if int(out.size) != int(n_frames):
                raise ValueError(f"{path}: {key} length {out.size} != n_frames={n_frames}")
            return out
    raise ValueError(f"{path}: need frame amp array (got {list(z.files)})")


def amp_labels_from_cache(cache: dict) -> np.ndarray:
    """Per-frame scalar from ``cache['e0']`` (manifest ``e0_file`` at precompute).

    For ``--model amp``: values are global A [Hartree].
    For ``--model amp-c``: values are coefficient c (your script in e0_file).
    """
    return np.asarray(cache["e0"], dtype=np.float64).reshape(-1)


def attach_frame_amp_labels(cache: dict, amp_frame: np.ndarray) -> dict:
    """Shallow copy with ``amp_frame`` shape (n_frames,)."""
    n = int(cache["e0"].shape[0])
    amp = np.asarray(amp_frame, dtype=np.float64).reshape(-1)
    if int(amp.size) != n:
        raise ValueError(f"amp_frame length {amp.size} != n_frames {n}")
    out = dict(cache)
    out["amp_frame"] = amp
    return out


def build_coeff_h_flat_from_amp_frame(
    cache: dict,
    *,
    k_min: float = 1e-30,
) -> np.ndarray:
    """
    Deprecated: do not use for amp-c training.

    Old helper: c_j = A_frame · sqrt(k_j). amp-c now reads c directly from e0_file.
    """
    if cache.get("kernel_h_flat") is None:
        raise ValueError("coeff labels need kernel_h_flat in cache")
    if cache.get("amp_frame") is not None:
        amp_frame = np.asarray(cache["amp_frame"], dtype=np.float64)
    else:
        amp_frame = amp_labels_from_cache(cache)
    ptr = cache["mm_ptr"]
    ker_all = np.asarray(cache["kernel_h_flat"], dtype=np.float64)
    blocks: list[np.ndarray] = []
    for k in range(int(amp_frame.shape[0])):
        s, e = int(ptr[k]), int(ptr[k + 1])
        ker = np.maximum(ker_all[s:e], float(k_min))
        a = float(amp_frame[k])
        blocks.append(a * np.sqrt(ker))
    return np.concatenate(blocks, axis=0)


def attach_coeff_h_flat(cache: dict, coeff_flat: np.ndarray) -> dict:
    n_rows = int(cache["feat_h_flat"].shape[0])
    c = np.asarray(coeff_flat, dtype=np.float64).reshape(-1)
    if int(c.size) != n_rows:
        raise ValueError(f"coeff_flat length {c.size} != soap rows {n_rows}")
    out = dict(cache)
    out["coeff_h_flat"] = c
    return out


def load_amp_label_flat(path: Path, *, n_rows: int) -> np.ndarray:
    """``amp_mm_flat`` or ``amp_h_flat`` from mix_mmh_val_peratom export."""
    z = np.load(path, allow_pickle=False)
    for key in ("amp_mm_flat", "amp_h_flat", "amp_flat"):
        if key in z:
            out = np.asarray(z[key], dtype=np.float64).reshape(-1)
            if int(out.size) != int(n_rows):
                raise ValueError(f"{path}: {key} length {out.size} != soap rows {n_rows}")
            return out
    raise ValueError(f"{path}: need amp_mm_flat / amp_h_flat (got {list(z.files)})")


def ei_kcal_to_amp_hartree(
    e_i_kcal: np.ndarray,
    kernel_mm: np.ndarray,
    *,
    k_min: float = 1e-30,
) -> np.ndarray:
    """
    e_i_kcal, kernel_mm : aligned per-site arrays.
    Returns A_i^ML [Hartree]; the spatial envelope is defined separately by ``embr_envelope``.
    """
    e = np.asarray(e_i_kcal, dtype=np.float64).reshape(-1)
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    if e.shape != k.shape:
        raise ValueError(f"e_i shape {e.shape} != kernel {k.shape}")
    # Negative e_i → zero amplitude (no attractive site repulsion).
    e = np.maximum(e, 0.0)
    k_safe = np.maximum(np.abs(k), float(k_min))
    return e / (float(HARTREE_TO_KCAL) * k_safe)


def amp_repulsion_from_partition(
    e_i_kcal: np.ndarray,
    kernel_mm: np.ndarray,
    mm_element: np.ndarray,
    *,
    k_min: float = 1e-30,
    repulsion_policy: str = "all",
) -> np.ndarray:
    """
    A_i^ML from the ML partition: A_i^ML = e_i / (627.5·k_i) on active sites.

    In the published path ``repulsion_policy='all'`` so all partition sites are active;
    legacy policies can zero selected sites for compatibility/diagnostics.
    """
    from embr_envelope import is_active_repulsion_mmh_element

    e = np.asarray(e_i_kcal, dtype=np.float64).reshape(-1)
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    el = np.asarray(mm_element, dtype=np.int8).reshape(-1)
    if e.shape != k.shape or e.shape != el.shape:
        raise ValueError(f"shape mismatch e={e.shape} k={k.shape} el={el.shape}")
    amp = np.zeros_like(e)
    mask = np.asarray(
        [is_active_repulsion_mmh_element(int(x), policy=str(repulsion_policy)) for x in el],
        dtype=bool,
    )
    if np.any(mask):
        amp[mask] = ei_kcal_to_amp_hartree(e[mask], k[mask], k_min=float(k_min))
    return amp


def first_order_de_from_partition(
    e_i_kcal: np.ndarray,
    kernel_mm: np.ndarray,
    mm_element: np.ndarray,
    *,
    repulsion_policy: str = "all",
) -> float:
    """627.5·Σ A_i^ML k_i on active sites (published path uses all sites)."""
    amp = amp_repulsion_from_partition(
        e_i_kcal,
        kernel_mm,
        mm_element,
        repulsion_policy=str(repulsion_policy),
    )
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    return float(HARTREE_TO_KCAL * np.sum(amp * k))


def first_order_de_kcal(e_i_kcal: np.ndarray, kernel_mm: np.ndarray) -> float:
    """627.5 · Σ A_i^ML k_i with amplitudes from ``ei_kcal_to_amp_hartree``."""
    amp = ei_kcal_to_amp_hartree(e_i_kcal, kernel_mm)
    return float(HARTREE_TO_KCAL * np.sum(amp * np.asarray(kernel_mm, dtype=np.float64)))


def de1_dot_kcal(kernel_mm: np.ndarray, amp_hartree: np.ndarray) -> float:
    """627.5 · Σ_j A_j k_j (aligned compact vectors)."""
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    a = np.asarray(amp_hartree, dtype=np.float64).reshape(-1)
    if k.shape != a.shape:
        raise ValueError(f"kernel {k.shape} != amp {a.shape}")
    return float(HARTREE_TO_KCAL * np.dot(k, a))


def de1_partition_audit(
    e_j: np.ndarray,
    kernel_mm: np.ndarray,
    mm_element_k: np.ndarray,
    amp: np.ndarray | None = None,
    *,
    e0_label: float,
    repulsion_policy: str = "all",
) -> dict:
    """
    Verify first-order closure on compact partition rows (same order as precompute k_j).

    When ``repulsion_policy=all``: Σ A_j k_j = Σ e_j = E0.
    """
    from embr_features import mmh_element_for_kernel_codes
    from embr_envelope import MM_KERNEL_EL_NAMES, is_active_repulsion_mmh_element

    e = np.asarray(e_j, dtype=np.float64).reshape(-1)
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    el_k = np.asarray(mm_element_k, dtype=np.int8).reshape(-1)
    if e.shape != k.shape or e.shape != el_k.shape:
        raise ValueError(f"shape mismatch e={e.shape} k={k.shape} el={el_k.shape}")
    el_mmh = mmh_element_for_kernel_codes(el_k)

    amp_all = amp_repulsion_from_partition(e, k, el_mmh, repulsion_policy="all")
    if amp is None:
        amp_rep = amp_repulsion_from_partition(
            e, k, el_mmh, repulsion_policy=str(repulsion_policy)
        )
    else:
        amp_rep = np.asarray(amp, dtype=np.float64).reshape(-1)
        if amp_rep.shape != k.shape:
            raise ValueError(f"amp {amp_rep.shape} != kernel {k.shape}")

    de1_all = de1_dot_kcal(k, amp_all)
    de1_rep = de1_dot_kcal(k, amp_rep)
    sum_e = float(np.sum(e))

    per_elem: dict[str, dict[str, float]] = {}
    for code, sym in MM_KERNEL_EL_NAMES.items():
        mask = el_k == int(code)
        if not np.any(mask):
            continue
        per_elem[sym] = {
            "n": float(np.sum(mask)),
            "sum_e_j": float(np.sum(e[mask])),
            "sum_k": float(np.sum(k[mask])),
            "de1_all": de1_dot_kcal(k[mask], amp_all[mask]),
            "de1_rep": de1_dot_kcal(k[mask], amp_rep[mask]),
        }

    rep_mask = np.asarray(
        [is_active_repulsion_mmh_element(int(x), policy=str(repulsion_policy)) for x in el_mmh],
        dtype=bool,
    )
    max_closure = 0.0
    if np.any(rep_mask):
        pred = float(HARTREE_TO_KCAL) * amp_rep[rep_mask] * k[rep_mask]
        max_closure = float(np.max(np.abs(pred - e[rep_mask])))

    return {
        "sum_e_j": sum_e,
        "e0_label": float(e0_label),
        "de1_all": float(de1_all),
        "de1_rep": float(de1_rep),
        "err_sum_e_vs_e0": float(sum_e - float(e0_label)),
        "err_de1_all_vs_e0": float(de1_all - float(e0_label)),
        "err_de1_rep_vs_e0": float(de1_rep - float(e0_label)),
        "repulsion_policy": str(repulsion_policy),
        "per_element": per_elem,
        "n_sites": int(k.size),
        "max_amp_closure_err": float(max_closure),
    }


def format_de1_audit_block(audit: dict) -> str:
    """Parseable E0 / Σ A_j k_j closure block for console audit."""
    lines = [
        "",
        "========== [E0 一阶闭合 Σ A_j k_j] (compact k_j, Emb0 ρ) ==========",
        f"  repulsion_policy = {audit.get('repulsion_policy', 'all')}",
        f"  n_partition_sites = {int(audit.get('n_sites', 0))}",
        f"  Σ e_j              = {float(audit['sum_e_j']):10.4f} kcal/mol  "
        f"vs E0_label |err|={abs(float(audit['err_sum_e_vs_e0'])):.6f}",
        f"  Σ A_j k_j (all)    = {float(audit['de1_all']):10.4f} kcal/mol  "
        f"vs E0_label |err|={abs(float(audit['err_de1_all_vs_e0'])):.6f}",
        f"  Σ A_j k_j (rep)    = {float(audit['de1_rep']):10.4f} kcal/mol  "
        f"vs E0_label |err|={abs(float(audit['err_de1_rep_vs_e0'])):.6f}",
    ]
    pol = str(audit.get("repulsion_policy", "all"))
    if pol != "all":
        gap = float(audit["de1_all"]) - float(audit["de1_rep"])
        lines.append(
            f"  (all−rep) de1 gap  = {gap:10.4f} kcal/mol  "
            "(partition 有 e_j 但 policy 下 A=0 的份额)"
        )
    if audit.get("max_amp_closure_err") is not None:
        lines.append(
            f"  max|627.5·A_j·k_j − e_j|  = {float(audit['max_amp_closure_err']):.3e} kcal/mol  "
            "(rep sites, per-site closure)"
        )
    lines.append("  --- per element (compact partition rows) ---")
    lines.append("  elem   n      Σe_j      Σ(Ak)_all   Σ(Ak)_rep")
    for sym in ("O", "H", "C", "N", "Na", "K", "Cl"):
        row = (audit.get("per_element") or {}).get(sym)
        if row is None:
            continue
        lines.append(
            f"  {sym:<4} {int(row['n']):4d}  {float(row['sum_e_j']):9.4f}  "
            f"{float(row['de1_all']):11.4f}  {float(row['de1_rep']):11.4f}"
        )
    if audit.get("de1_scf_frame") is not None and np.isfinite(float(audit["de1_scf_frame"])):
        lines.append(
            f"  Σ A_j k_j (SCF frame) = {float(audit['de1_scf_frame']):10.4f} kcal/mol  "
            f"vs E0_label |err|={abs(float(audit.get('err_de1_scf_vs_e0', float('nan')))):.6f}  "
            f"(n_finite_k={int(audit.get('n_scf_kernel_sites', 0))})"
        )
    lines.append("")
    return "\n".join(lines)
