"""
Shared helpers: ML e_i → repulsive amplitudes A_i^ML → EmbR SCF labels.

Emb0 kernels k_i live in the mix_mmh cache when built with ``--ref-dir`` precompute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scf_embed_io import load_scf_frame
from embr_cache import load_mmh_cache, mmh_frame_slice, n_frames as n_mmh_frames
from embr_features import (
    mm_kernel_atom_indices_for_scf_mm,
)
from embr_io import load_coo_frame_atoms
from embr_partition import (
    amp_repulsion_from_partition,
    ei_kcal_to_amp_hartree,
    first_order_de_from_partition,
    first_order_de_kcal,
)
from embr_envelope import (
    is_active_kernel_element,
    is_active_repulsion_symbol,
    is_partition_kernel_element,
)
from embr_model import forward_e_j, load_mmh_model_from_ckpt
from soap_mm_util import resolve_device


def load_mmh_model(
    ckpt_path: Path,
    cache: dict,
    *,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    dev = resolve_device(str(device or "auto"))
    model, _ = load_mmh_model_from_ckpt(ckpt_path, cache, device=dev)
    return model


@torch.no_grad()
def infer_e_j(
    model: torch.nn.Module,
    cache: dict,
    k: int,
    *,
    kernel_src: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    feat, ker_cache, el, dist, e0_label = mmh_frame_slice(cache, k)
    dev = next(model.parameters()).device
    feat_t = torch.tensor(feat, dtype=torch.float32, device=dev).unsqueeze(0)
    el_t = torch.tensor(el, dtype=torch.long, device=dev).unsqueeze(0)
    ker_t = None
    dist_t = None
    from embr_model import SoapE0MmhKModel

    if isinstance(model, SoapE0MmhKModel):
        if kernel_src is not None:
            ker = kernel_h_for_frame(kernel_src, k)
        elif ker_cache is not None:
            ker = ker_cache
        else:
            raise ValueError("k model infer needs kernel in cache or kernel_src")
        ker_t = torch.tensor(ker, dtype=torch.float32, device=dev).unsqueeze(0)
        if dist is not None:
            dist_t = torch.tensor(dist, dtype=torch.float32, device=dev).unsqueeze(0)
    ker_np = None if ker_t is None else ker_t.squeeze(0).cpu().numpy()
    e_j = (
        forward_e_j(
            model,
            feat_t=feat_t,
            el_t=el_t,
            ker_t=ker_t,
            e0_t=None,
            dist_t=dist_t,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )
    e0_pred = float(np.sum(e_j))
    return np.asarray(e_j, dtype=np.float64), e0_pred, ker_np


def _load_kernel_source(path: Path) -> dict:
    cache = load_mmh_cache(path, normalize_elements=True)
    if cache.get("kernel_h_flat") is None:
        raise ValueError(
            f"{path}: mix_mmh cache has no kernel_h_flat; re-run precompute with --ref-dir"
        )
    kind = "mmh"
    if cache.get("kernel_mm_flat") is not None:
        kind = "mmh_full"
    return {"kind": kind, "cache": cache, "path": path}


def resolve_kernel_source(mmh_cache_path: Path, kernel_cache_path: Path | None) -> dict:
    """Prefer explicit kernel cache; else use mix_mmh npz if it has kernels."""
    if kernel_cache_path is not None:
        return _load_kernel_source(Path(kernel_cache_path))
    return _load_kernel_source(Path(mmh_cache_path))


def _cache_partition_oh(cache: dict) -> bool:
    meta = cache.get("meta") or {}
    return bool(meta.get("partition_oh", False))


def kernel_partition_for_frame(kernel_src: dict, k: int) -> np.ndarray:
    """Emb0 k_j for k-partition sites (O+H+ions when ``partition_oh``)."""
    cache = kernel_src["cache"]
    kind = kernel_src["kind"]
    ptr = cache["mm_ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    if kind in ("mmh", "mmh_full") and cache.get("kernel_h_flat") is not None:
        ker = np.asarray(cache["kernel_h_flat"][s:e], dtype=np.float64)
        if _cache_partition_oh(cache):
            return ker
        return ker
    if (
        kind in ("mmh", "mmh_full")
        and cache.get("kernel_mm_flat") is not None
        and cache.get("mm_ptr_k") is not None
        and cache.get("mm_element_k_flat") is not None
    ):
        ptr_k = cache["mm_ptr_k"]
        sk, ek = int(ptr_k[k]), int(ptr_k[k + 1])
        ker = np.asarray(cache["kernel_mm_flat"][sk:ek], dtype=np.float64)
        el = np.asarray(cache["mm_element_k_flat"][sk:ek], dtype=np.int8)
        filt = np.asarray(
            [
                is_partition_kernel_element(int(x))
                if _cache_partition_oh(cache)
                else is_active_kernel_element(int(x))
            ],
            dtype=bool,
        )
        if not np.any(filt):
            raise RuntimeError(f"frame {k}: no partition kernels in mmh_full cache")
        return ker[filt]
    raise RuntimeError(f"frame {k}: kernel source missing kernel_h_flat / kernel_mm_flat")


def kernel_active_for_frame(kernel_src: dict, k: int) -> np.ndarray:
    """Per-MM H/Na/K Emb0 kernel k_j aligned with mix_mmh active row order."""
    if _cache_partition_oh(kernel_src["cache"]):
        return kernel_partition_for_frame(kernel_src, k)
    cache = kernel_src["cache"]
    kind = kernel_src["kind"]
    ptr = cache["mm_ptr"]
    s, e = int(ptr[k]), int(ptr[k + 1])
    if kind in ("mmh", "mmh_full") and cache.get("kernel_h_flat") is not None:
        return np.asarray(cache["kernel_h_flat"][s:e], dtype=np.float64)
    if (
        kind in ("mmh", "mmh_full")
        and cache.get("kernel_mm_flat") is not None
        and cache.get("mm_ptr_k") is not None
        and cache.get("mm_element_k_flat") is not None
    ):
        ptr_k = cache["mm_ptr_k"]
        sk, ek = int(ptr_k[k]), int(ptr_k[k + 1])
        ker = np.asarray(cache["kernel_mm_flat"][sk:ek], dtype=np.float64)
        el = np.asarray(cache["mm_element_k_flat"][sk:ek], dtype=np.int8)
        active = np.asarray([is_active_kernel_element(int(x)) for x in el], dtype=bool)
        if not np.any(active):
            raise RuntimeError(f"frame {k}: no MM H/Na/K kernels in mmh_full cache")
        return ker[active]
    raise RuntimeError(f"frame {k}: kernel source missing kernel_h_flat / kernel_mm_flat")


def kernel_h_for_frame(kernel_src: dict, k: int) -> np.ndarray:
    """Backward-compatible alias for active-site kernels."""
    return kernel_active_for_frame(kernel_src, k)


def kernel_mm_full_for_frame(kernel_src: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Full MM O+H kernel vector and element ids (theta / SCF order)."""
    kind = kernel_src["kind"]
    if kind in ("mmh", "mmh_full"):
        cache = kernel_src["cache"]
        if cache.get("kernel_mm_flat") is not None and cache.get("mm_ptr_k") is not None:
            ptr = cache["mm_ptr_k"]
            s, e = int(ptr[k]), int(ptr[k + 1])
            ker = np.asarray(cache["kernel_mm_flat"][s:e], dtype=np.float64)
            el = np.asarray(cache["mm_element_k_flat"][s:e], dtype=np.int8)
            return ker, el
        if kind == "mmh_full":
            raise ValueError("mmh_full kernel source missing kernel_mm_flat")
    cache = kernel_src["cache"]
    mm_ptr = cache["mm_ptr"]
    s, e = int(mm_ptr[k]), int(mm_ptr[k + 1])
    ker = np.asarray(cache["kernel_mm_flat"][s:e], dtype=np.float64)
    el = np.asarray(cache["mm_element"][s:e], dtype=np.int8)
    return ker, el


def amp_h_from_e_j(e_j: np.ndarray, kernel_h: np.ndarray) -> np.ndarray:
    return ei_kcal_to_amp_hartree(np.asarray(e_j, dtype=np.float64), kernel_h)


def expand_kernel_amp_to_scf_frame(frame, amp_kernel: np.ndarray) -> np.ndarray:
    """Compact kernel-row amplitudes → ``len(frame.mm_symbols)`` (same as ``amp_mm_for_o_h_sites``)."""
    from scf_embed_pyscf import amp_mm_for_o_h_sites

    return amp_mm_for_o_h_sites(frame, amp_kernel)


def _count_kernel_mm_sites(frame) -> int:
    from embr_envelope import is_kernel_mm_symbol

    return sum(1 for sym in frame.mm_symbols if is_kernel_mm_symbol(sym))


def _kernel_site_mismatch_error(
    frame,
    n_compact: int,
    *,
    el_k: np.ndarray | None = None,
) -> ValueError:
    from embr_envelope import MM_KERNEL_EL_NAMES, is_kernel_mm_symbol

    n_frame = _count_kernel_mm_sites(frame)
    frame_syms = [str(s).strip() for s in frame.mm_symbols if is_kernel_mm_symbol(s)]
    cache_syms: list[str] = []
    if el_k is not None:
        for code in np.asarray(el_k, dtype=np.int8).reshape(-1):
            cache_syms.append(MM_KERNEL_EL_NAMES.get(int(code), "?"))
    msg = (
        f"MM kernel site count mismatch: cache/compact={int(n_compact)} "
        f"vs SCF frame={int(n_frame)} "
        f"(frame kernel symbols={frame_syms}"
    )
    if cache_syms:
        msg += f", cache kernel symbols={cache_syms}"
    msg += (
        "). Re-run precompute_soap_e0_mix_mmh.py with current manifest/envelope "
        "(uses dm_emb from ref npz; no HF re-run needed)."
    )
    return ValueError(msg)


def scf_frame_from_cache_meta(mmh_cache: dict, fm: dict):
    """Load Coo frame with the same MM distance cut used at precompute time."""
    from scf_embed_io import filter_mm_by_distance, load_scf_frame

    coo_path = Path(str(fm["coo_path"]))
    n_qm = int(fm.get("n_qm", 10))
    frame = load_scf_frame(coo_path, n_qm=n_qm)
    meta = mmh_cache.get("meta") or {}
    r_cut = fm.get("r_cut_mm", meta.get("r_cut_mm"))
    if r_cut is not None and float(r_cut) > 0:
        frame = filter_mm_by_distance(frame, r_cut_ang=float(r_cut))
    return frame, coo_path, n_qm


def expand_kernel_k_to_scf_frame(frame, ker_kernel: np.ndarray) -> np.ndarray:
    """Compact kernel-row k_j → ``len(frame.mm_symbols)``."""
    from embr_envelope import is_kernel_mm_symbol

    k_in = np.asarray(ker_kernel, dtype=np.float64).reshape(-1)
    out = np.full(len(frame.mm_symbols), np.nan, dtype=np.float64)
    j = 0
    for i, sym in enumerate(frame.mm_symbols):
        if not is_kernel_mm_symbol(sym):
            continue
        if j >= k_in.size:
            raise ValueError("kernel_mm shorter than number of MM kernel sites")
        out[i] = float(k_in[j])
        j += 1
    if j != k_in.size:
        raise ValueError(f"kernel_mm length {k_in.size} != n MM kernel sites {j}")
    return out


def de1_audit_for_labels_slice(
    labels: dict,
    slice_index: int,
    e0_label: float,
    *,
    repulsion_policy: str = "all",
    amp_scf: np.ndarray | None = None,
    k_scf: np.ndarray | None = None,
) -> dict:
    """
    Compact-partition audit: Σ e_j and Σ A_j k_j vs ``e0_label`` (HF E0 from ref).

    Optional ``amp_scf``/``k_scf`` are full-frame vectors after ``expand_kernel_*`` for cross-check.
    """
    from embr_partition import de1_dot_kcal, de1_partition_audit

    ptr = labels["mm_ptr"]
    s, e = int(ptr[slice_index]), int(ptr[slice_index + 1])
    k = np.asarray(labels["kernel_mm_flat"][s:e], dtype=np.float64)
    el = np.asarray(labels["mm_element"][s:e], dtype=np.int8)
    if "e_j_flat" not in labels:
        raise KeyError("labels missing e_j_flat; re-export mix_mmh_val_peratom.npz")
    e_j = np.asarray(labels["e_j_flat"][s:e], dtype=np.float64)
    amp = np.asarray(labels["amp_mm_flat"][s:e], dtype=np.float64)
    audit = de1_partition_audit(
        e_j,
        k,
        el,
        amp=amp,
        e0_label=float(e0_label),
        repulsion_policy=str(repulsion_policy),
    )
    if amp_scf is not None and k_scf is not None:
        a = np.asarray(amp_scf, dtype=np.float64).reshape(-1)
        kk = np.asarray(k_scf, dtype=np.float64).reshape(-1)
        if a.shape == kk.shape:
            fin = np.isfinite(kk) & np.isfinite(a) & (np.abs(a) + np.abs(kk) > 0)
            if np.any(fin):
                de1_scf = de1_dot_kcal(kk[fin], a[fin])
                audit["de1_scf_frame"] = float(de1_scf)
                audit["err_de1_scf_vs_e0"] = float(de1_scf - float(e0_label))
                audit["n_scf_kernel_sites"] = int(np.sum(fin))
    return audit


def amp_mm_for_e_j(
    frame,
    coo_path: Path,
    n_qm: int,
    e_j: np.ndarray,
    kernel: np.ndarray,
    mm_element: np.ndarray,
    *,
    partition_oh: bool,
    kernel_src: dict | None = None,
    frame_k: int | None = None,
    repulsion_policy: str = "all",
) -> np.ndarray:
    """
    Map per-site e_i to full-frame repulsive amplitudes A_i^ML.

    ``partition_oh``: e_j and A_j on O+H+ions (all partition sites carry repulsion).
    """
    from embr_features import mmh_element_for_kernel_codes

    e_j = np.asarray(e_j, dtype=np.float64).reshape(-1)
    kernel = np.asarray(kernel, dtype=np.float64).reshape(-1)
    el = np.asarray(mm_element, dtype=np.int8).reshape(-1)
    if e_j.shape != kernel.shape or e_j.shape != el.shape:
        raise ValueError(f"shape mismatch e={e_j.shape} k={kernel.shape} el={el.shape}")

    if bool(partition_oh) and kernel_src is not None and frame_k is not None:
        ker_mm, el_k = kernel_mm_full_for_frame(kernel_src, int(frame_k))
        if e_j.shape == ker_mm.shape:
            el_mmh = mmh_element_for_kernel_codes(el_k)
            amp_part = amp_repulsion_from_partition(
                e_j, ker_mm, el_mmh, repulsion_policy=str(repulsion_policy)
            )
            n_k_frame = _count_kernel_mm_sites(frame)
            if int(amp_part.size) != int(n_k_frame):
                raise _kernel_site_mismatch_error(frame, int(amp_part.size), el_k=el_k)
            return amp_part

    if bool(partition_oh):
        amp_rep = amp_repulsion_from_partition(
            e_j, kernel, el, repulsion_policy=str(repulsion_policy)
        )
    else:
        amp_rep = amp_h_from_e_j(e_j, kernel)
    return amp_mm_on_full_frame(frame, coo_path, n_qm, amp_rep)


def print_mmh_h_e_k_a_table(
    *,
    dist: np.ndarray | None,
    ker: np.ndarray,
    e_j: np.ndarray,
    amp_h: np.ndarray,
    e0_label: float,
    e0_pred: float,
    cache_k: int,
) -> None:
    """Per-MM-H table sorted by dist_qm (cache row order); summary A_j spread."""
    ker = np.asarray(ker, dtype=np.float64).reshape(-1)
    e_j = np.asarray(e_j, dtype=np.float64).reshape(-1)
    amp_h = np.asarray(amp_h, dtype=np.float64).reshape(-1)
    if ker.shape != e_j.shape or e_j.shape != amp_h.shape:
        raise ValueError(f"shape mismatch ker={ker.shape} e_j={e_j.shape} amp={amp_h.shape}")
    n = int(e_j.size)
    if dist is not None:
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        if d.shape != e_j.shape:
            raise ValueError(f"dist shape {d.shape} != e_j {e_j.shape}")
        order = np.argsort(d)
    else:
        d = np.full(n, float("nan"))
        order = np.arange(n)

    de1 = first_order_de_kcal(e_j, ker)
    print(
        f"\n[MM-H e_j → A_j]  cache_k={cache_k}  n_h={n}  "
        f"E0_label={float(e0_label):+.4f}  E0_pred={float(e0_pred):+.4f}  "
        f"err={float(e0_pred - e0_label):+.4f}  de1_check={de1:+.4f} kcal/mol"
    )
    print("  site   dist(Å)      k_j          e_j(kcal)    A_j(Ha)    e_j/k")
    for rank, j in enumerate(order):
        ratio = float(e_j[j] / ker[j]) if abs(ker[j]) > 0 else float("nan")
        print(
            f"  {rank:3d}   {d[j]:7.3f}  {ker[j]:11.4e}  {e_j[j]:+11.4f}  "
            f"{amp_h[j]:+10.6f}  {ratio:11.4e}"
        )

    finite_a = amp_h[np.isfinite(amp_h)]
    if finite_a.size:
        a_mean = float(np.mean(finite_a))
        a_std = float(np.std(finite_a))
        a_min = float(np.min(finite_a))
        a_max = float(np.max(finite_a))
        cv = 100.0 * a_std / abs(a_mean) if abs(a_mean) > 1e-30 else float("nan")
        print(
            f"\n  A_j summary:  mean={a_mean:+.6f} Ha  std={a_std:.6e}  "
            f"min={a_min:+.6f}  max={a_max:+.6f}  cv={cv:.2f}%"
        )
        print(f"  A_j range (max/min) = {a_max / a_min:.3f}x" if abs(a_min) > 1e-30 else "")
    if n >= 2 and float(np.std(ker)) > 1e-30 and float(np.std(e_j)) > 1e-30:
        r_ek = float(np.corrcoef(ker, e_j)[0, 1])
        print(f"  pearson(e_j, k_j) = {r_ek:+.4f}")
    print()


def amp_mm_on_full_frame(
    frame,
    coo_path: Path,
    n_qm: int,
    amp_active: np.ndarray,
) -> np.ndarray:
    """Place per-kernel-site amplitudes on full MM vector (all O/H/Na/K/Cl kernel rows)."""
    # PySCF is needed only when amplitudes are expanded for an actual EmbR SCF.
    # Keep this import local so cache-only ML inference does not require PySCF/SciPy.
    from scf_embed_pyscf import amp_mm_for_o_h_sites
    pos, syms = load_coo_frame_atoms(coo_path, n_qm=n_qm)
    full_idx = mm_kernel_atom_indices_for_scf_mm(
        pos,
        syms,
        n_qm,
        frame.mm_coords_ang,
        frame.mm_symbols,
    )
    amp_full = np.zeros(len(full_idx), dtype=np.float64)
    active_j = 0
    amp_active = np.asarray(amp_active, dtype=np.float64).reshape(-1)
    for j, ia in enumerate(full_idx):
        if not is_active_repulsion_symbol(syms[ia]):
            continue
        if active_j >= amp_active.size:
            break
        amp_full[j] = float(amp_active[active_j])
        active_j += 1
    if active_j != amp_active.size:
        raise RuntimeError(
            f"MM active amp count {amp_active.size} != mapped O/H/Na/K/Cl sites {active_j}"
        )
    return amp_mm_for_o_h_sites(frame, amp_full)


def frame_meta_entry(cache: dict, k: int) -> dict:
    meta = cache.get("meta") or {}
    frames = meta.get("frames") or []
    if k < len(frames):
        return dict(frames[k])
    raise KeyError(f"cache frame {k}: no meta.frames[{k}] (re-run precompute with frame metadata)")


def build_val_peratom_labels(
    *,
    mmh_cache: dict,
    kernel_cache_path: Path | None,
    ckpt_path: Path,
    val_indices: list[int],
    fix_alpha: float,
    device: torch.device | str | None = None,
    h_only: bool = True,
    scf_meta: dict | None = None,
    envelope_meta: dict | None = None,
) -> dict:
    """
    Build theta-compatible peratom labels for validation frames only.

    Local slice index ``j`` maps to global cache row ``val_indices[j]``.
    """
    mmh_path = Path(str(mmh_cache.get("_path", "")))
    if kernel_cache_path is None and mmh_cache.get("kernel_h_flat") is None:
        raise ValueError("mix_mmh cache has no kernel_h_flat; pass --kernel-cache or use --ref-dir precompute")
    kernel_src = resolve_kernel_source(mmh_path, kernel_cache_path)
    partition_oh = _cache_partition_oh(mmh_cache)
    n = n_mmh_frames(mmh_cache)
    for k in val_indices:
        if k < 0 or k >= n:
            raise ValueError(f"val index {k} out of range [0, {n})")

    model = load_mmh_model(ckpt_path, mmh_cache, device=device)
    amp_blocks: list[np.ndarray] = []
    ker_blocks: list[np.ndarray] = []
    el_blocks: list[np.ndarray] = []
    e0_list: list[float] = []
    err_list: list[float] = []
    frame_meta: list[dict] = []
    mm_ptr = [0]

    for j, k in enumerate(val_indices):
        e_j, e0_pred, ker_h_infer = infer_e_j(model, mmh_cache, k, kernel_src=kernel_src)
        _, _, el_part, _, e0_label = mmh_frame_slice(mmh_cache, k)
        ker_h = ker_h_infer if ker_h_infer is not None else kernel_h_for_frame(kernel_src, k)
        if ker_h.shape[0] != e_j.shape[0]:
            raise RuntimeError(
                f"frame {k}: n_mm_h e_j={e_j.shape[0]} != kernel_h={ker_h.shape[0]}"
            )
        fm = frame_meta_entry(mmh_cache, k)
        frame, coo_path, n_qm = scf_frame_from_cache_meta(mmh_cache, fm)
        amp_mm = amp_mm_for_e_j(
            frame,
            coo_path,
            n_qm,
            e_j,
            ker_h,
            el_part,
            partition_oh=partition_oh,
            kernel_src=kernel_src,
            frame_k=int(k),
        )
        ker_mm, el_mm = kernel_mm_full_for_frame(kernel_src, k)
        if amp_mm.shape != ker_mm.shape:
            raise RuntimeError(
                f"frame {k}: amp_mm {amp_mm.shape} != kernel_mm {ker_mm.shape}"
            )

        amp_blocks.append(np.asarray(amp_mm, dtype=np.float64))
        ker_blocks.append(ker_mm)
        el_blocks.append(el_mm)
        e0_list.append(float(e0_pred))
        de1 = (
            first_order_de_from_partition(e_j, ker_h, el_part)
            if partition_oh
            else first_order_de_kcal(e_j, ker_h)
        )
        err_list.append(float(e0_pred - e0_label))
        mm_ptr.append(mm_ptr[-1] + int(amp_mm.size))
        frame_meta.append(
            {
                "slice_index": int(j),
                "val_run_id": int(fm.get("file_index", k)),
                "global_cache_k": int(k),
                "file_index": int(fm.get("file_index", k)),
                "coo_path": str(coo_path.resolve()),
                "n_qm": n_qm,
                "residue": str(fm.get("residue", meta_residue(mmh_cache, k))),
                "E0_pred": float(e0_pred),
                "E0_label": float(e0_label),
                "sum_e_j": float(np.sum(e_j)),
                "de1_kcal": float(de1),
            }
        )

    meta_out: dict = {
        "pipeline": "soap_e0_mix_mmh_scf",
        "fix_alpha": float(fix_alpha),
        "h_only": bool(h_only),
        "partition_oh": bool(partition_oh),
        "ckpt": str(Path(ckpt_path).resolve()),
        "soap_cache": str(mmh_cache.get("_path", "")),
        "kernel_cache": None if kernel_cache_path is None else str(Path(kernel_cache_path).resolve()),
        "val_indices": [int(x) for x in val_indices],
        "frames": frame_meta,
    }
    if envelope_meta:
        meta_out.update(dict(envelope_meta))
    if scf_meta:
        meta_out["scf_config"] = dict(scf_meta)

    return {
        "e0": np.asarray(e0_list, dtype=np.float64),
        "err_kcal": np.asarray(err_list, dtype=np.float64),
        "amp_mm_flat": np.concatenate(amp_blocks, axis=0),
        "mm_ptr": np.asarray(mm_ptr, dtype=np.int64),
        "mm_element": np.concatenate(el_blocks, axis=0),
        "kernel_mm_flat": np.concatenate(ker_blocks, axis=0),
        "meta": meta_out,
    }


def meta_residue(mmh_cache: dict, k: int) -> str:
    """Frame system label from cache meta (no amino-acid registry)."""
    meta = mmh_cache.get("meta") or {}
    frames = meta.get("frames") or []
    if 0 <= int(k) < len(frames):
        fm = frames[int(k)]
        if isinstance(fm, dict):
            for key in ("residue", "prefix", "label"):
                v = fm.get(key)
                if v is not None and str(v).strip():
                    return str(v).strip().lower()
    rid = mmh_cache.get("residue_id")
    if rid is not None:
        r2id = meta.get("residue_to_id") or {}
        id2name = {int(v): str(name).strip().lower() for name, v in r2id.items()}
        r = int(rid[int(k)])
        if r in id2name:
            return id2name[r]
    return "unknown"


def save_peratom_npz(labels: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw: dict = {
        "e0": np.asarray(labels["e0"], dtype=np.float64),
        "err_kcal": np.asarray(labels["err_kcal"], dtype=np.float64),
        "amp_mm_flat": np.asarray(labels["amp_mm_flat"], dtype=np.float64),
        "mm_ptr": np.asarray(labels["mm_ptr"], dtype=np.int64),
        "mm_element": np.asarray(labels["mm_element"], dtype=np.int8),
        "kernel_mm_flat": np.asarray(labels["kernel_mm_flat"], dtype=np.float64),
        "meta_json": np.array(json.dumps(labels["meta"])),
    }
    if labels.get("e_j_flat") is not None:
        kw["e_j_flat"] = np.asarray(labels["e_j_flat"], dtype=np.float64)
    np.savez_compressed(out_path, **kw)


def attach_cache_path(cache: dict, path: Path) -> dict:
    out = dict(cache)
    out["_path"] = str(path.resolve())
    return out
