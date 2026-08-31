"""
Precompute MM-H SOAP for train_soap_e0_mix_mmh.py (pure E0 fit and/or Emb0 k_j).

Per MM site (O/H/C/N/Na/K/Cl present in Coo):
  · ``--mm-env full`` (default): SOAP center on each MM kernel site, **QM + MM** full Coo environment
  · ``--mm-env qm``: same centers, **QM-only** SOAP environment (mix_ai / Emb0 path)

  · ``--partition-oh``: when Emb0/k_j on, O+H partition for e_j; **does not remove O from pure SOAP**

Examples
--------
  # Pure fit (recommended first):
  python precompute_soap_e0_mix_mmh.py --manifest my_mix.json --out mix_mmh_full.npz \\
    --no-emb0 --mm-env full

  # Reuse batch_hf_emb0_cp ref npz (E0 + k_j from saved dm_emb; no Emb0 SCF):
  python precompute_soap_e0_mix_mmh.py --manifest my_mix.json --out mix_mmh_ref.npz \\
    --ref-dir ala_hf_+ --mm-env full --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from embr_features import SoapE0Hyper, build_soap_calculator, soap_feature_dim
from embr_features import (
    FEAT_MODE_MM_FULL_ENV_SOAP,
    FEAT_MODE_MM_QM_ENV_SOAP,
    FEAT_MODE_QM_NEAR_MM_SOAP,
    FEAT_MODE_ALL_ATOM_SOAP,
    compute_soap_mm_qm_env_centers,
    feat_dim_for_mode,
    min_dist_qm_per_mm,
)
from embr_io import COO_NAME_FMT, coo_path, load_coo_frame_atoms, load_e0_txt
from embr_scf_manifest import build_emb0_scf_config, isotropic_repulsion_cone, scf_settings_from_manifest
from embr_envelope import MmhEnvelopeConfig
from embr_dataset import resolve_dataset_label, resolve_dataset_n_qm


_G_SOAP = None
_G_HYPER = None
_G_SCF_CFG = None
_G_ENVELOPE_CFG: MmhEnvelopeConfig | None = None
_G_R_CUT_MM: float | None = None
_G_EMB0_CACHE_DIR: Path | None = None
_G_FORCE_EMB0 = False
_G_NO_EMB0 = False
_G_MM_ENV = "full"
_G_REF_DIR: Path | None = None
_G_REF_PATTERN = "ref_{}.npz"
_G_SCF_THREADS = 1
_G_POOL_WORKERS = 1
_G_PARTITION_OH = True
_G_AMP_QM_NEAR_CUT: float | None = None
_G_AMP_ALL_ATOMS: bool = False


def _init_worker(
    hyper_kw: dict,
    scf_cfg_kw: dict,
    envelope_cfg_kw: dict,
    r_cut_mm: float | None,
    emb0_cache_dir: str | None,
    force_emb0: bool,
    no_emb0: bool,
    mm_env: str,
    ref_dir: str | None,
    ref_pattern: str,
    kj_threads: int,
    pool_workers: int,
    partition_oh: bool,
    amp_qm_near_cut: float | None = None,
    amp_all_atoms: bool = False,
) -> None:
    global _G_SOAP, _G_HYPER, _G_SCF_CFG, _G_ENVELOPE_CFG, _G_R_CUT_MM
    global _G_EMB0_CACHE_DIR, _G_FORCE_EMB0, _G_NO_EMB0, _G_MM_ENV
    global _G_REF_DIR, _G_REF_PATTERN, _G_SCF_THREADS, _G_POOL_WORKERS, _G_PARTITION_OH
    global _G_AMP_QM_NEAR_CUT, _G_AMP_ALL_ATOMS
    if int(pool_workers) > 1:
        _pin_pool_blas_threads()
    hyper = SoapE0Hyper(**hyper_kw)
    _G_SOAP, _G_HYPER = build_soap_calculator(hyper)
    _G_SCF_CFG = dict(scf_cfg_kw)
    _G_ENVELOPE_CFG = MmhEnvelopeConfig(
        kind=str(envelope_cfg_kw["kind"]),
        width_by_element=dict(envelope_cfg_kw["width_by_element"]),
        lnC_by_element=dict(envelope_cfg_kw["lnC_by_element"]),
        exp_sum_by_element=dict(envelope_cfg_kw.get("exp_sum_by_element") or {}),
    )
    _G_R_CUT_MM = r_cut_mm
    _G_EMB0_CACHE_DIR = None if emb0_cache_dir is None else Path(emb0_cache_dir)
    _G_FORCE_EMB0 = bool(force_emb0)
    _G_NO_EMB0 = bool(no_emb0)
    _G_MM_ENV = str(mm_env)
    _G_REF_DIR = None if ref_dir is None else Path(ref_dir)
    _G_REF_PATTERN = str(ref_pattern)
    _G_SCF_THREADS = max(1, int(kj_threads))
    _G_POOL_WORKERS = max(1, int(pool_workers))
    _G_PARTITION_OH = bool(partition_oh)
    _G_AMP_QM_NEAR_CUT = None if amp_qm_near_cut is None else float(amp_qm_near_cut)
    _G_AMP_ALL_ATOMS = bool(amp_all_atoms)


def _compute_emb0_kernels(path_str: str, n_qm: int) -> tuple[np.ndarray, np.ndarray]:
    from scf_embed_io import filter_mm_by_distance, load_scf_frame
    from scf_embed_perturb import precompute_emb0_pert_frame

    frame = load_scf_frame(Path(path_str), n_qm=int(n_qm))
    frame = filter_mm_by_distance(frame, r_cut_ang=_G_R_CUT_MM)
    pert = precompute_emb0_pert_frame(
        frame,
        build_emb0_scf_config(_G_SCF_CFG),
        envelope_cfg=_G_ENVELOPE_CFG,
        cone=isotropic_repulsion_cone(),
    )
    km = np.asarray(pert.kernels_per_mm.kernel_mm, dtype=np.float64).reshape(-1)
    el = np.asarray(pert.kernels_per_mm.mm_element, dtype=np.int8).reshape(-1)
    return km, el


def _kernels_for_frame(path_str: str, n_qm: int) -> tuple[np.ndarray, np.ndarray, bool]:
    from soap_e0_emb0_cache_ai import load_or_compute_emb0_kernels

    return load_or_compute_emb0_kernels(
        path_str,
        int(n_qm),
        cache_dir=_G_EMB0_CACHE_DIR,
        fix_alpha=float(_G_ENVELOPE_CFG.legacy_fix_alpha()),
        envelope_tag=_G_ENVELOPE_CFG.sidecar_tag(),
        scf_cfg_kw=_G_SCF_CFG,
        r_cut_mm=_G_R_CUT_MM,
        force=bool(_G_FORCE_EMB0),
        compute_fn=lambda: _compute_emb0_kernels(path_str, int(n_qm)),
    )


def _compute_feat_mm(
    pos: np.ndarray,
    syms: list[str],
    *,
    hyper_frame: SoapE0Hyper,
    n_qm: int,
    scf_mm_coords: np.ndarray,
    scf_mm_symbols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    from embr_features import compute_soap_mm_partition_centers

    if _G_NO_EMB0 or _G_PARTITION_OH:
        return compute_soap_mm_partition_centers(
            pos,
            syms,
            soap=_G_SOAP,
            hyper=hyper_frame,
            n_qm=int(n_qm),
            scf_mm_coords=scf_mm_coords,
            scf_mm_symbols=scf_mm_symbols,
            full_env=(_G_MM_ENV == "full"),
        )
    if _G_MM_ENV == "full":
        from embr_features import compute_soap_mm_active_centers

        return compute_soap_mm_active_centers(
            pos,
            syms,
            soap=_G_SOAP,
            hyper=hyper_frame,
            n_qm=int(n_qm),
            scf_mm_coords=scf_mm_coords,
            scf_mm_symbols=scf_mm_symbols,
            full_env=True,
        )
    if _G_MM_ENV == "qm":
        return compute_soap_mm_qm_env_centers(
            pos,
            syms,
            soap=_G_SOAP,
            hyper=hyper_frame,
            n_qm=int(n_qm),
            scf_mm_coords=scf_mm_coords,
            scf_mm_symbols=scf_mm_symbols,
        )
    raise ValueError(f"unknown mm_env {_G_MM_ENV!r} (use full or qm)")


def _select_partition_from_soap(
    feat_mm: np.ndarray,
    mm_el_soap: np.ndarray,
    dist_qm: np.ndarray,
    *,
    kernel_mm: np.ndarray | None = None,
    mm_el_k: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    from embr_features import mmh_element_for_kernel_codes
    from embr_envelope import is_partition_kernel_element

    if mm_el_k is not None:
        el_k = np.asarray(mm_el_k, dtype=np.int8).reshape(-1)
        part = np.asarray([is_partition_kernel_element(int(x)) for x in el_k], dtype=bool)
        if not np.any(part):
            raise RuntimeError("frame has no MM O/H partition sites")
        ker_a = None if kernel_mm is None else np.asarray(kernel_mm, dtype=np.float64).reshape(-1)[part]
        mmh_el = mmh_element_for_kernel_codes(el_k[part])
        if feat_mm.shape[0] == el_k.shape[0]:
            feat_a = feat_mm[part]
            dist_a = dist_qm[part] if dist_qm.shape[0] == el_k.shape[0] else dist_qm
        elif feat_mm.shape[0] == int(part.sum()):
            feat_a = feat_mm
            dist_a = dist_qm
        else:
            raise RuntimeError(
                f"feat rows {feat_mm.shape[0]} != kernel rows {el_k.shape[0]} "
                f"or n_partition {int(part.sum())}"
            )
        return feat_a, ker_a, mmh_el, dist_a

    feat_a = feat_mm
    dist_a = dist_qm
    mmh_el = np.asarray(mm_el_soap, dtype=np.int8).reshape(-1)
    if mmh_el.size != feat_a.shape[0]:
        raise RuntimeError(
            f"mm_element SOAP rows {mmh_el.size} != feat rows {feat_a.shape[0]}"
        )
    if feat_a.shape[0] == 0:
        raise RuntimeError("frame has no MM O/H partition sites")
    return feat_a, None, mmh_el, dist_a


def _select_active_from_soap(
    feat_mm: np.ndarray,
    mm_el_soap: np.ndarray,
    dist_qm: np.ndarray,
    *,
    kernel_mm: np.ndarray | None = None,
    mm_el_k: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    from embr_features import mmh_element_for_kernel_codes
    from embr_envelope import is_active_kernel_element

    if mm_el_k is not None:
        el_k = np.asarray(mm_el_k, dtype=np.int8).reshape(-1)
        active = np.asarray([is_active_kernel_element(int(x)) for x in el_k], dtype=bool)
        if not np.any(active):
            raise RuntimeError("frame has no MM H/Na/K active sites")
        ker_a = None if kernel_mm is None else np.asarray(kernel_mm, dtype=np.float64).reshape(-1)[active]
        mmh_el = mmh_element_for_kernel_codes(el_k[active])
        if feat_mm.shape[0] == el_k.shape[0]:
            feat_a = feat_mm[active]
            dist_a = dist_qm[active] if dist_qm.shape[0] == el_k.shape[0] else dist_qm
        elif feat_mm.shape[0] == int(active.sum()):
            feat_a = feat_mm
            dist_a = dist_qm
        else:
            raise RuntimeError(
                f"feat rows {feat_mm.shape[0]} != kernel rows {el_k.shape[0]} "
                f"or n_active {int(active.sum())}"
            )
        return feat_a, ker_a, mmh_el, dist_a

    feat_a = feat_mm
    dist_a = dist_qm
    mmh_el = np.asarray(mm_el_soap, dtype=np.int8).reshape(-1)
    if mmh_el.size != feat_a.shape[0]:
        raise RuntimeError(
            f"mm_element SOAP rows {mmh_el.size} != feat rows {feat_a.shape[0]}"
        )
    if feat_a.shape[0] == 0:
        raise RuntimeError("frame has no MM H/Na/K active sites")
    return feat_a, None, mmh_el, dist_a


def _kernels_from_ref(file_index: int, scf_frame, path_str: str, n_qm: int) -> tuple[np.ndarray, np.ndarray, float, str, str]:
    from embr_ref_kernels import kernels_per_mm_from_ref_npz, ref_npz_path

    if _G_REF_DIR is None:
        raise RuntimeError("ref kernels requested but _G_REF_DIR is unset")
    ref_path = ref_npz_path(_G_REF_DIR, int(file_index), pattern=_G_REF_PATTERN)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    per_mm, e0_kcal, _meta, kj_src = kernels_per_mm_from_ref_npz(
        ref_path,
        scf_frame,
        alpha_cfg=_G_ENVELOPE_CFG,
        r_cut_mm=_G_R_CUT_MM,
        num_threads=int(_G_SCF_THREADS),
    )
    km = np.asarray(per_mm.kernel_mm, dtype=np.float64).reshape(-1)
    el = np.asarray(per_mm.mm_element, dtype=np.int8).reshape(-1)
    return km, el, float(e0_kcal), str(ref_path.resolve()), str(kj_src)


def _one(task: tuple[int, str, int, int, int]) -> dict:
    global _G_SOAP, _G_HYPER
    from scf_embed_io import filter_mm_by_distance, load_scf_frame

    frame_k, path_str, n_qm, _rid, file_index = task
    path = Path(path_str)
    scf_frame = filter_mm_by_distance(load_scf_frame(path, n_qm=int(n_qm)), r_cut_ang=_G_R_CUT_MM)

    ker_h: np.ndarray | None = None
    emb0_cached = False
    e0_kcal: float | None = None
    ref_npz: str | None = None
    kernel_mm_full: np.ndarray | None = None
    mm_el_k_full: np.ndarray | None = None
    kj_source: str | None = None
    t_soap_s: float | None = None
    t_kj_s: float | None = None

    pos, syms = load_coo_frame_atoms(
        path,
        n_qm=int(n_qm),
        qm_symbols_fallback=(),
    )
    hyper_frame = SoapE0Hyper(
        r_cut=float(_G_HYPER.r_cut),
        n_max=int(_G_HYPER.n_max),
        l_max=int(_G_HYPER.l_max),
        sigma=float(_G_HYPER.sigma),
        species=tuple(_G_HYPER.species),
        n_qm=int(n_qm),
    )
    t0 = time.perf_counter()
    feat_mm, mm_el_soap = _compute_feat_mm(
        pos,
        syms,
        hyper_frame=hyper_frame,
        n_qm=int(n_qm),
        scf_mm_coords=scf_frame.mm_coords_ang,
        scf_mm_symbols=list(scf_frame.mm_symbols),
    )
    feat_qm_near = None
    if _G_AMP_QM_NEAR_CUT is not None:
        from embr_features import compute_soap_qm_near_mm_centers

        feat_qm_near = compute_soap_qm_near_mm_centers(
            pos,
            syms,
            soap=_G_SOAP,
            hyper=hyper_frame,
            n_qm=int(n_qm),
            near_mm_cut_ang=float(_G_AMP_QM_NEAR_CUT),
        )
    feat_all_atom = None
    if _G_AMP_ALL_ATOMS:
        from embr_features import compute_soap_all_atom_centers

        feat_all_atom = compute_soap_all_atom_centers(
            pos,
            syms,
            soap=_G_SOAP,
            hyper=hyper_frame,
        )
    t_soap_s = time.perf_counter() - t0
    dist_qm = min_dist_qm_per_mm(
        pos,
        syms,
        int(n_qm),
        scf_mm_coords=scf_frame.mm_coords_ang,
        scf_mm_symbols=scf_frame.mm_symbols,
        active_only=(_G_MM_ENV == "full" and not _G_PARTITION_OH),
    )

    if _G_REF_DIR is not None and not _G_NO_EMB0:
        t1 = time.perf_counter()
        kernel_mm, mm_el_k, e0_kcal, ref_npz, kj_source = _kernels_from_ref(
            int(file_index), scf_frame, path_str, int(n_qm)
        )
        t_kj_s = time.perf_counter() - t1
        emb0_cached = str(kj_source) in ("npz", "sidecar")
        kernel_mm_full = kernel_mm
        mm_el_k_full = mm_el_k

    select_fn = _select_partition_from_soap if _G_PARTITION_OH else _select_active_from_soap

    if _G_NO_EMB0:
        feat_h, ker_h, mmh_el, dist_h = select_fn(feat_mm, mm_el_soap, dist_qm)
    elif _G_REF_DIR is not None:
        if kernel_mm_full is None or mm_el_k_full is None:
            raise RuntimeError("ref k_j missing after _kernels_from_ref")
        feat_h, ker_h, mmh_el, dist_h = select_fn(
            feat_mm, mm_el_soap, dist_qm, kernel_mm=kernel_mm_full, mm_el_k=mm_el_k_full
        )
    else:
        kernel_mm, mm_el_k, emb0_cached = _kernels_for_frame(path_str, int(n_qm))
        feat_h, ker_h, mmh_el, dist_h = select_fn(
            feat_mm, mm_el_soap, dist_qm, kernel_mm=kernel_mm, mm_el_k=mm_el_k
        )

    out: dict = {
        "frame_k": int(frame_k),
        "feat_h": feat_h.astype(np.float32, copy=False),
        "mm_element": mmh_el,
        "dist_qm": dist_h,
        "n_mm_h": int(feat_h.shape[0]),
        "n_qm": int(n_qm),
    }
    if feat_qm_near is not None:
        out["feat_qm_near"] = feat_qm_near.astype(np.float32, copy=False)
    if feat_all_atom is not None:
        out["feat_all_atom"] = feat_all_atom.astype(np.float32, copy=False)
        out["n_atoms"] = int(feat_all_atom.shape[0])
    if ker_h is not None:
        out["kernel_h"] = ker_h
    if e0_kcal is not None:
        out["e0_kcal"] = float(e0_kcal)
    if ref_npz is not None:
        out["ref_npz"] = ref_npz
    if kernel_mm_full is not None:
        out["kernel_mm_full"] = kernel_mm_full
        out["mm_el_k_full"] = mm_el_k_full
    if not _G_NO_EMB0:
        out["emb0_cached"] = bool(emb0_cached)
        if _G_REF_DIR is not None and kj_source is not None:
            out["kj_source"] = str(kj_source)
    if t_soap_s is not None:
        out["t_soap_s"] = float(t_soap_s)
    if t_kj_s is not None:
        out["t_kj_s"] = float(t_kj_s)
    return out


def resolve_workers(requested: int, n_tasks: int) -> int:
    if int(requested) > 0:
        n = int(requested)
    else:
        n = int(os.cpu_count() or 1)
    return max(1, min(n, int(n_tasks)))


def resolve_kj_threads(
    requested: int | None,
    manifest_threads: int,
    pool_workers: int,
) -> int:
    """PySCF grid k_j threads per worker (cap when using ProcessPoolExecutor)."""
    cpus = max(1, int(os.cpu_count() or 1))
    base = max(1, int(manifest_threads if requested is None else requested))
    if int(pool_workers) <= 1:
        return base
    return max(1, min(base, cpus // int(pool_workers)))


def _pin_pool_blas_threads() -> None:
    """One BLAS/OpenMP layer per process when frames run in parallel."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _resolve_manifest_path(manifest: Path, p: Path | str) -> Path:
    """Resolve dataset paths: try as-is, then relative to manifest.json directory."""
    manifest = Path(manifest).resolve()
    path = Path(p)
    if path.is_file() or path.is_dir():
        return path.resolve()
    cand = (manifest.parent / path).resolve()
    if cand.is_file() or cand.is_dir():
        return cand
    return path.resolve() if path.is_absolute() else cand


def _parse_only_file_indices(raw: str | None) -> set[int] | None:
    if raw is None or not str(raw).strip():
        return None
    out: set[int] = set()
    for tok in str(raw).replace(",", " ").split():
        t = tok.strip()
        if t:
            out.add(int(t))
    return out if out else None


def _parse_exclude_frames(ds: dict) -> set[int]:
    """manifest datasets[] 项里的 exclude_frames / exclude_file_indices。"""
    raw = ds.get("exclude_frames")
    if raw is None:
        raw = ds.get("exclude_file_indices")
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {int(x) for x in raw}
    raise ValueError(
        f'{ds.get("prefix") or ds.get("residue") or "dataset"}: exclude_frames must be a JSON array of Coo indices'
    )


def _ref_meta_light(ref_path: Path) -> dict:
    with np.load(ref_path, allow_pickle=False) as z:
        if "meta_json" not in z:
            return {}
        return json.loads(str(z["meta_json"]))


def _resolve_n_qm_for_frame(
    *,
    ds: dict,
    file_index: int,
    default_n_qm: int,
    use_ref: bool,
    ref_dir: Path | None,
    ref_pattern: str,
) -> int:
    """Per-Coo n_qm: manifest map → ref npz meta → dataset default."""
    by_idx = ds.get("n_qm_by_file_index") or ds.get("n_qm_by_frame")
    if by_idx is not None:
        fi = int(file_index)
        if str(fi) in by_idx:
            return int(by_idx[str(fi)])
        if fi in by_idx:
            return int(by_idx[fi])
    if use_ref and ref_dir is not None:
        from embr_ref_kernels import ref_npz_path

        fi = int(file_index)
        ref_path = ref_npz_path(ref_dir, fi, pattern=str(ref_pattern))
        if ref_path.is_file():
            meta = _ref_meta_light(ref_path)
            if meta.get("n_qm") is not None:
                return int(meta["n_qm"])
    return int(default_n_qm)


def _feature_mode(mm_env: str) -> str:
    if mm_env == "full":
        return FEAT_MODE_MM_FULL_ENV_SOAP
    if mm_env == "qm":
        return FEAT_MODE_MM_QM_ENV_SOAP
    raise ValueError(f"unknown mm_env {mm_env!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute MM-H SOAP (+ optional Emb0 k_j) for mix_mmh")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("soap_e0_mix_mmh.npz"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--mm-env",
        type=str,
        default="full",
        choices=("full", "qm"),
        help="full=MM H center + QM+MM Coo env (pure fit); qm=QM-only env (mix_ai / Emb0)",
    )
    ap.add_argument("--fix-alpha", type=float, default=None)
    ap.add_argument(
        "--multi-alpha",
        action="store_true",
        help="Approach A: alpha_elem = alpha_H*(R_H/R_elem)^2 for H/Na/K (O uses alpha_H)",
    )
    ap.add_argument(
        "--alpha-by-element",
        type=str,
        default=None,
        help='JSON map e.g. \'{"H":3,"O":0.94,"Na":0.353,"K":0.194,"Cl":0.42}\' (overrides manifest multi_alpha)',
    )
    ap.add_argument("--alpha-na", type=float, default=None)
    ap.add_argument("--alpha-k", type=float, default=None)
    ap.add_argument("--r-cut-mm", type=float, default=None)
    ap.add_argument(
        "--emb0-cache-dir",
        type=Path,
        default=None,
        help="reuse Emb0 k_j cache (only without --ref-dir)",
    )
    ap.add_argument("--force-emb0", action="store_true")
    ap.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="batch_hf_emb0_cp out dir with ref_*.npz: E0 + k_j from saved dm_emb (no Emb0 SCF)",
    )
    ap.add_argument(
        "--ref-pattern",
        type=str,
        default="ref_{}.npz",
        help="ref npz filename pattern; {} = Coo file_index",
    )
    ap.add_argument(
        "--kj-threads",
        type=int,
        default=None,
        help="PySCF threads per worker for k_j grid (default: manifest scf.threads, "
        "capped to cpus/workers when --workers>1)",
    )
    ap.add_argument(
        "--partition-oh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="k-partition e_j on O+H+ions; EmbR A_j on all partition sites (default: on)",
    )
    ap.add_argument(
        "--only-file-indices",
        type=str,
        default=None,
        help="run only these Coo file_index values (e.g. 5 or 1,2,5); ignores manifest n_frames span",
    )
    ap.add_argument(
        "--e0-from-txt",
        action="store_true",
        help="E0 labels from manifest e0_file (col 1), not ref_*.npz e0_kcal "
        "(k_j still from --ref-dir when set; also manifest e0_source=txt)",
    )
    ap.add_argument(
        "--amp-qm-near-cut",
        type=float,
        default=None,
        help="optional: SOAP on MM sites within this Ang of QM (feat_qm_near_flat; off by default)",
    )
    ap.add_argument(
        "--amp-all-atoms",
        action="store_true",
        help="optional: SOAP at every Coo atom (feat_all_atom_flat; off by default)",
    )
    args = ap.parse_args()

    use_ref = args.ref_dir is not None
    cfg = json.loads(args.manifest.read_text(encoding="utf-8"))
    e0_from_txt = bool(args.e0_from_txt) or str(cfg.get("e0_source", "")).strip().lower() == "txt"
    manifest_path = Path(args.manifest).resolve()
    datasets = cfg.get("datasets") or []
    if not datasets:
        raise ValueError("manifest has no datasets")

    r_cut = float(cfg.get("r_cut", 5.0))
    n_max = int(cfg.get("n_max", 8))
    l_max = int(cfg.get("l_max", 6))
    sigma = float(cfg.get("sigma", 0.5))
    workers_cfg = int(cfg.get("workers", 0) if args.workers is None else args.workers)
    fix_alpha_cli = args.fix_alpha if args.fix_alpha is not None else cfg.get("fix_alpha", 3.0)
    alpha_cfg = MmhEnvelopeConfig.parse(
        fix_alpha=float(fix_alpha_cli),
        multi_alpha=bool(args.multi_alpha) or bool(cfg.get("multi_alpha", False)),
        alpha_by_element=args.alpha_by_element,
        alpha_na=args.alpha_na,
        alpha_k=args.alpha_k,
        envelope_kind=cfg.get("envelope_kind"),
        manifest=cfg,
    )
    envelope_cfg_kw = {
        "kind": alpha_cfg.kind,
        "width_by_element": dict(alpha_cfg.width_by_element),
        "lnC_by_element": dict(alpha_cfg.lnC_by_element),
        "exp_sum_by_element": dict(alpha_cfg.exp_sum_by_element),
    }
    fix_alpha = float(alpha_cfg.legacy_fix_alpha())
    r_cut_mm = cfg.get("r_cut_mm") if args.r_cut_mm is None else args.r_cut_mm
    if r_cut_mm is not None:
        r_cut_mm = float(r_cut_mm)
    scf_cfg_kw = scf_settings_from_manifest(cfg)
    mm_env = str(args.mm_env)
    partition_oh = bool(args.partition_oh)
    if not use_ref:
        raise ValueError("--ref-dir is required (EmbR pipeline reads k_i from ref_*.npz)")
    no_emb0 = False
    feature_mode = _feature_mode(mm_env)

    emb0_cache_dir: Path | None = None

    hyper_kw = {
        "r_cut": r_cut,
        "n_max": n_max,
        "l_max": l_max,
        "sigma": sigma,
        "species": ("C", "H", "O", "N", "Na", "Cl", "K"),
        "n_qm": 0,
    }
    hyper = SoapE0Hyper(**hyper_kw)
    _, hyper = build_soap_calculator(hyper)
    d_soap = int(soap_feature_dim(hyper))
    d_feat = int(feat_dim_for_mode(feature_mode, d_soap))

    residue_names = sorted({resolve_dataset_label(ds) for ds in datasets})
    # Keep registry names stable when present; append any extra labels (asp/lys/…).
    residue_to_id: dict[str, int] = {}
    for name in residue_names:
        if name not in residue_to_id:
            residue_to_id[name] = len(residue_to_id)
    tasks: list[tuple[int, str, int, int, int]] = []
    e0_vals: list[float] = []
    frame_meta: list[dict] = []
    global_k = 0

    only_file_indices = _parse_only_file_indices(args.only_file_indices)

    for ds in datasets:
        label = resolve_dataset_label(ds)
        n_qm_default = resolve_dataset_n_qm(ds)
        rid = int(residue_to_id[label])
        coo_dir = _resolve_manifest_path(manifest_path, ds["coo_dir"])
        fmt = str(ds.get("coo_name_fmt", COO_NAME_FMT))
        i0 = int(ds.get("i0", 0))
        n_frames = int(ds["n_frames"])
        e0_i0 = int(ds.get("e0_i0", 1))
        exclude_frames = _parse_exclude_frames(ds)
        if exclude_frames:
            print(
                f"  [{label}] exclude_frames: {len(exclude_frames)} Coo indices "
                f"(first few: {sorted(exclude_frames)[:8]}…)",
                flush=True,
            )
        ds_e0_from_txt = e0_from_txt or str(ds.get("e0_source", "")).strip().lower() == "txt"
        load_e0_txt_file = (not use_ref) or ds_e0_from_txt
        e0_arr = None
        if load_e0_txt_file:
            if ds.get("e0_file") is None:
                raise ValueError(
                    f"{label}: need manifest \"e0_file\" when not using ref npz E0 "
                    f"(omit --ref-dir or pass --e0-from-txt / e0_source=txt with e0_file set)"
                )
            e0_path = _resolve_manifest_path(manifest_path, ds["e0_file"])
            e0_arr = load_e0_txt(e0_path)
            e0_end = int(i0 + n_frames - e0_i0)
            if e0_end > int(e0_arr.shape[0]):
                raise ValueError(
                    f"{label}: Coo {i0}..{i0 + n_frames - 1} needs E0 lines "
                    f"{i0 - e0_i0 + 1}..{e0_end} but {e0_path} has {e0_arr.shape[0]} lines "
                    f"(e0_i0={e0_i0}; E0 line 1 = Coo{e0_i0})"
                )

        for k in range(n_frames):
            file_index = int(i0 + k)
            if file_index in exclude_frames:
                continue
            if only_file_indices is not None and file_index not in only_file_indices:
                continue
            p = coo_path(coo_dir, fmt, file_index)
            if not p.is_file():
                last_idx = int(i0 + n_frames - 1)
                raise FileNotFoundError(
                    f"{p}  (manifest file_index={file_index} = i0({i0})+local_k({k}); "
                    f"dataset span Coo{i0}..Coo{last_idx} from n_frames={n_frames}; "
                    f"for one config only use i0=N, n_frames=1 or --only-file-indices N; "
                    f"or add {file_index} to exclude_frames)"
                )
            n_qm = _resolve_n_qm_for_frame(
                ds=ds,
                file_index=file_index,
                default_n_qm=n_qm_default,
                use_ref=use_ref,
                ref_dir=Path(args.ref_dir) if use_ref else None,
                ref_pattern=str(args.ref_pattern),
            )
            e0_line = int(file_index - e0_i0)
            if load_e0_txt_file:
                assert e0_arr is not None
                e0_val = float(e0_arr[e0_line])
                if not np.isfinite(e0_val):
                    print(
                        f"  [skip] {label} Coo{file_index}: e0_file line {e0_line + 1} "
                        f"is non-finite ({e0_val}); add to exclude_frames?",
                        flush=True,
                    )
                    continue
            tasks.append((global_k, str(p.resolve()), n_qm, rid, file_index))
            if load_e0_txt_file:
                e0_vals.append(e0_val)
            else:
                e0_vals.append(float("nan"))
            fm: dict = {
                "global_index": global_k,
                "residue": label,
                "prefix": str(ds.get("prefix") or label),
                "residue_id": rid,
                "local_k": k,
                "file_index": file_index,
                "e0_line": e0_line + 1,
                "coo_path": str(p.resolve()),
                "n_qm": n_qm,
                "e0_source": "txt" if load_e0_txt_file else "ref_npz",
            }
            if use_ref:
                from embr_ref_kernels import ref_npz_path

                fm["ref_npz"] = str(
                    ref_npz_path(args.ref_dir, file_index, pattern=str(args.ref_pattern)).resolve()
                )
            frame_meta.append(fm)
            global_k += 1

    n_total = int(global_k)
    if n_total == 0:
        hint = ""
        if only_file_indices is not None:
            hint = f" (--only-file-indices={sorted(only_file_indices)} matched no manifest frames)"
        raise ValueError(f"no frames to precompute{hint}")
    workers = resolve_workers(workers_cfg, n_total)
    manifest_threads = int(scf_cfg_kw.get("num_threads", 4))
    kj_threads = resolve_kj_threads(args.kj_threads, manifest_threads, workers)
    cpus = max(1, int(os.cpu_count() or 1))
    if use_ref:
        emb0_tag = f"ref:{Path(args.ref_dir).resolve()}"
    else:
        emb0_tag = "off" if no_emb0 else str(emb0_cache_dir)
    print(
        f"[precompute] frames={n_total} d_feat={d_feat} workers={workers}  "
        f"emb0={emb0_tag}"
    )
    if only_file_indices is not None:
        print(f"  only-file-indices={sorted(only_file_indices)}", flush=True)
    if use_ref:
        print(
            f"  ref-dir={Path(args.ref_dir).resolve()}  "
            f"kj_threads={kj_threads}  workers={workers}",
            flush=True,
        )
    elif not no_emb0:
        print(
            f"  scf={scf_cfg_kw['method']}/{scf_cfg_kw['basis']} "
            f"d3bj={scf_cfg_kw['use_d3bj']}",
            flush=True,
        )

    results: dict[int, dict] = {}
    t0 = time.perf_counter()
    done = 0
    n_emb0_hit = 0
    n_kj_npz = 0
    n_kj_sidecar = 0
    n_kj_grid = 0
    initargs = (
        hyper_kw,
        scf_cfg_kw,
        envelope_cfg_kw,
        r_cut_mm,
        None if emb0_cache_dir is None else str(emb0_cache_dir.resolve()),
        bool(args.force_emb0),
        no_emb0,
        mm_env,
        None if not use_ref else str(Path(args.ref_dir).resolve()),
        str(args.ref_pattern),
        int(kj_threads),
        int(workers),
        bool(partition_oh),
        args.amp_qm_near_cut,
        bool(args.amp_all_atoms),
    )

    def _report_progress(r: dict) -> None:
        nonlocal done, n_emb0_hit, n_kj_npz, n_kj_sidecar, n_kj_grid
        src = str(r.get("kj_source", ""))
        if src == "npz":
            n_kj_npz += 1
            n_emb0_hit += 1
        elif src == "sidecar":
            n_kj_sidecar += 1
            n_emb0_hit += 1
        elif src == "grid":
            n_kj_grid += 1
        elif r.get("emb0_cached"):
            n_emb0_hit += 1
        done += 1
        if done == 1 or done % 10 == 0 or done == n_total:
            print(f"  {done}/{n_total} frames", flush=True)

    if workers == 1:
        _init_worker(*initargs)
        for t in tasks:
            r = _one(t)
            results[t[0]] = r
            _report_progress(r)
    else:
        print(f"  spawning {workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=initargs) as ex:
            futs = [ex.submit(_one, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                results[int(r["frame_k"])] = r
                _report_progress(r)

    dt = time.perf_counter() - t0
    if no_emb0:
        print(f"done {dt:.1f}s  (SOAP only, no Emb0)", flush=True)
    elif use_ref:
        print(f"done {dt:.1f}s  ({n_total} frames)", flush=True)
    else:
        print(f"done {dt:.1f}s  emb0 fresh={n_total - n_emb0_hit}  hit={n_emb0_hit}", flush=True)

    feat_blocks: list[np.ndarray] = []
    ker_blocks: list[np.ndarray] = []
    ker_mm_blocks: list[np.ndarray] = []
    el_k_blocks: list[np.ndarray] = []
    el_blocks: list[np.ndarray] = []
    dist_blocks: list[np.ndarray] = []
    mm_ptr = [0]
    mm_ptr_k = [0]
    residue_id = np.zeros((n_total,), dtype=np.int16)
    has_ker = True
    has_ker_mm = True
    has_qm_near = args.amp_qm_near_cut is not None
    qm_near_blocks: list[np.ndarray] = []
    qm_near_ptr = [0]
    has_all_atom = bool(args.amp_all_atoms)
    all_atom_blocks: list[np.ndarray] = []
    all_atom_ptr = [0]

    for k in range(n_total):
        r = results[k]
        feat_blocks.append(r["feat_h"])
        if has_qm_near:
            if "feat_qm_near" not in r:
                raise RuntimeError(f"frame {k}: --amp-qm-near-cut set but feat_qm_near missing")
            qm_near_blocks.append(r["feat_qm_near"])
            qm_near_ptr.append(qm_near_ptr[-1] + int(r["feat_qm_near"].shape[0]))
        if has_all_atom:
            if "feat_all_atom" not in r:
                raise RuntimeError(f"frame {k}: --amp-all-atoms set but feat_all_atom missing")
            all_atom_blocks.append(r["feat_all_atom"])
            all_atom_ptr.append(all_atom_ptr[-1] + int(r["feat_all_atom"].shape[0]))
        if has_ker:
            ker_blocks.append(r["kernel_h"])
        if has_ker_mm:
            ker_mm_blocks.append(r["kernel_mm_full"])
            el_k_blocks.append(r["mm_el_k_full"])
            mm_ptr_k.append(mm_ptr_k[-1] + int(r["kernel_mm_full"].shape[0]))
        el_blocks.append(r["mm_element"])
        dist_blocks.append(r["dist_qm"])
        mm_ptr.append(mm_ptr[-1] + int(r["feat_h"].shape[0]))
        residue_id[k] = int(frame_meta[k]["residue_id"])
        if frame_meta[k].get("e0_source") == "ref_npz":
            e0_vals[k] = float(r["e0_kcal"])
        if "ref_npz" in r:
            frame_meta[k]["ref_npz"] = r["ref_npz"]

    meta = {
        "pipeline": "soap_e0_mix_mmh_ref" if use_ref else "soap_e0_mix_mmh",
        "n_frames": n_total,
        "d_feat": d_feat,
        "d_soap": d_soap,
        "feature_mode": feature_mode,
        "mm_env": mm_env,
        "partition_oh": bool(partition_oh),
        "repulsion_sites": "O_H_Na_K_Cl",
        "model_sites": "MM_kernel_OH_CN_Na_K_Cl" if (partition_oh or no_emb0) else "MM_H_Na_K",
        "with_emb0": not no_emb0,
        "residue_to_id": residue_to_id,
        "hyper": {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "sigma": sigma,
            "d_feat": d_feat,
            "d_soap": d_soap,
        },
        "r_cut_mm": r_cut_mm,
        "frames": frame_meta,
        "manifest": str(manifest_path),
    }
    if args.amp_qm_near_cut is not None:
        meta["amp_qm_near_cut"] = float(args.amp_qm_near_cut)
        meta["amp_qm_near_feature_mode"] = FEAT_MODE_QM_NEAR_MM_SOAP
    if args.amp_all_atoms:
        meta["amp_all_atom_feature_mode"] = FEAT_MODE_ALL_ATOM_SOAP
    env_meta = alpha_cfg.to_full_meta()
    if use_ref:
        meta.update(env_meta)
        meta["ref_dir"] = str(Path(args.ref_dir).resolve())
        meta["ref_pattern"] = str(args.ref_pattern)
        meta["e0_source"] = "E_txt" if e0_from_txt else "batch_hf_ref_npz"
    elif not no_emb0:
        meta.update(env_meta)
        meta["scf"] = scf_cfg_kw
        meta["emb0_cache_dir"] = str(emb0_cache_dir.resolve())
        meta["e0_source"] = "E_txt"
    else:
        meta["e0_source"] = "E_txt"

    out_kw: dict = {
        "feat_h_flat": np.concatenate(feat_blocks, axis=0),
        "mm_element_flat": np.concatenate(el_blocks, axis=0),
        "dist_qm_flat": np.concatenate(dist_blocks, axis=0),
        "mm_ptr": np.asarray(mm_ptr, dtype=np.int64),
        "e0": np.asarray(e0_vals, dtype=np.float64),
        "residue_id": residue_id,
        "meta_json": np.array(json.dumps(meta)),
    }
    if has_ker:
        out_kw["kernel_h_flat"] = np.concatenate(ker_blocks, axis=0)
    if has_ker_mm:
        out_kw["kernel_mm_flat"] = np.concatenate(ker_mm_blocks, axis=0)
        out_kw["mm_element_k_flat"] = np.concatenate(el_k_blocks, axis=0)
        out_kw["mm_ptr_k"] = np.asarray(mm_ptr_k, dtype=np.int64)
    if has_qm_near:
        out_kw["feat_qm_near_flat"] = np.concatenate(qm_near_blocks, axis=0)
        out_kw["qm_near_ptr"] = np.asarray(qm_near_ptr, dtype=np.int64)
    if has_all_atom:
        out_kw["feat_all_atom_flat"] = np.concatenate(all_atom_blocks, axis=0)
        out_kw["all_atom_ptr"] = np.asarray(all_atom_ptr, dtype=np.int64)

    np.savez_compressed(args.out, **out_kw)
    print("wrote", args.out.resolve(), "MM-site rows", int(mm_ptr[-1]), "frames", n_total)


if __name__ == "__main__":
    main()

