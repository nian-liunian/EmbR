"""
Reuse ``batch_hf_emb0_cp`` / ``run_hf_emb0_cp_frame`` ``ref_*.npz`` in the mix_mmh pipeline.

Each ref npz already contains Emb0 ``dm_emb`` and PySCF ``e0_kcal``. This module extracts
per-MM density-overlap kernels k_i **without rerunning Emb0 SCF** (grid quadrature on saved dm).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pyscf import qmmm

from scf_embed_cluster import build_cluster_supermol_mol
from scf_embed_io import ScfFrame, filter_mm_by_distance, load_scf_frame
from embr_scf_manifest import isotropic_repulsion_cone
from scf_embed_pyscf import (
    ScfEmbedConfig,
    _build_mol,
    hartree_to_kcal,
    make_mean_field,
    scf_embed_config_from_cli,
)
from scf_embed_perturb import (
    GaussRhoKernelsPerMm,
    _mean_line_profiles_at_r_grid,
    all_cl_qm_line_pairs,
    all_ion_qm_line_pairs,
    enumerate_qm_mm_line_pairs,
    gauss_rho_kernels_per_mm,
    line_rho_profile_on_qm_mm_axis,
    nearest_cl_qm_line_pair,
    nearest_ion_qm_line_pair,
)
from embr_envelope import ION_MM_KEYS, _norm_elem_key


def ref_npz_path(ref_dir: Path, file_index: int, *, pattern: str = "ref_{}.npz") -> Path:
    return Path(ref_dir) / pattern.format(int(file_index))


def coo_file_index(coo_path: Path) -> int | None:
    m = re.search(r"(\d+)", Path(coo_path).stem)
    return int(m.group(1)) if m else None


def _ref_scalar_kcal(z, meta: dict, key: str, *, ha_key: str | None = None) -> float:
    if key in z:
        return float(z[key])
    if key in meta:
        return float(meta[key])
    if ha_key and ha_key in z:
        return float(hartree_to_kcal(float(z[ha_key])))
    if ha_key and ha_key.replace("_hartree", "_kcal") in meta:
        return float(meta[ha_key.replace("_hartree", "_kcal")])
    return float("nan")


def qmmm_interaction_ref_kcal(
    *,
    e_int_raw_kcal: float,
    e_int_cp4_kcal: float,
) -> float:
    """
    Gaussian **3834 corrected** complexation from batch_hf npz fields.

    npz ``e_int_cp`` is the 4-term CP sum (~BSSE scale, Gaussian 3831);
    ``e_int_raw − e_int_cp`` matches Gaussian 3834 (QM/MM interaction ref).
    """
    if not np.isfinite(float(e_int_raw_kcal)) or not np.isfinite(float(e_int_cp4_kcal)):
        return float("nan")
    return float(e_int_raw_kcal - e_int_cp4_kcal)


def e_int_pred_from_e0_emb0(*, e0_pred_kcal: float, e_int_emb0_kcal: float) -> float:
    """Fast QM/MM interaction from mix_mmh: E0_pred + E_int^Emb0."""
    if not np.isfinite(float(e0_pred_kcal)) or not np.isfinite(float(e_int_emb0_kcal)):
        return float("nan")
    return float(e0_pred_kcal + e_int_emb0_kcal)


def compute_mix_err_metrics(
    *,
    e0_pred_kcal: float,
    e0_label_kcal: float,
    e_int_emb0_kcal: float,
    e_int_qmmm_ref_kcal: float,
    e_int_embtheta_kcal: float | None = None,
) -> dict[str, float]:
    """Absolute mix / SCF errors vs labels, pred target, and QM/MM ref (kcal/mol)."""
    e0_pred = float(e0_pred_kcal)
    e0_label = float(e0_label_kcal)
    e_emb0 = float(e_int_emb0_kcal)
    e_qmmm = float(e_int_qmmm_ref_kcal)
    e_pred = e_int_pred_from_e0_emb0(e0_pred_kcal=e0_pred, e_int_emb0_kcal=e_emb0)
    out: dict[str, float] = {
        "e0_pred": e0_pred,
        "e0_label": e0_label,
        "e_int_emb0": e_emb0,
        "e_int_pred": e_pred,
        "e_int_qmmm_ref": e_qmmm,
        "abs_e0_err": float("nan"),
        "abs_e_int_pred_err": float("nan"),
        "abs_embtheta_err": float("nan"),
        "abs_embtheta_vs_pred": float("nan"),
        "signed_embtheta_vs_pred": float("nan"),
    }
    if np.isfinite(e0_pred) and np.isfinite(e0_label):
        out["abs_e0_err"] = abs(e0_pred - e0_label)
    if np.isfinite(e_pred) and np.isfinite(e_qmmm):
        out["abs_e_int_pred_err"] = abs(e_pred - e_qmmm)
    if e_int_embtheta_kcal is not None and np.isfinite(float(e_int_embtheta_kcal)):
        e_th = float(e_int_embtheta_kcal)
        out["e_int_embtheta"] = e_th
        if np.isfinite(e_qmmm):
            out["abs_embtheta_err"] = abs(e_th - e_qmmm)
        if np.isfinite(e_pred):
            out["signed_embtheta_vs_pred"] = float(e_th - e_pred)
            out["abs_embtheta_vs_pred"] = abs(e_th - e_pred)
    return out


def format_mix_err_one_line(metrics: dict[str, float]) -> str:
    e0_pred = float(metrics.get("e0_pred", float("nan")))
    e0_label = float(metrics.get("e0_label", float("nan")))
    signed_e0 = float(metrics.get("signed_e0_err", float("nan")))
    if not np.isfinite(signed_e0) and np.isfinite(e0_pred) and np.isfinite(e0_label):
        signed_e0 = float(e0_pred - e0_label)
    signed_scf = float(metrics.get("signed_embtheta_err", float("nan")))
    if not np.isfinite(signed_scf):
        signed_scf = float(metrics.get("signed_embtheta_vs_pred", float("nan")))
    parts: list[str] = []
    if np.isfinite(signed_e0):
        parts.append(f"dE_ML - dE_ref={signed_e0:+.4f}")
    if np.isfinite(signed_scf):
        parts.append(f"dE_SCF(n) - dE_ML={signed_scf:+.4f}")
    return "  err: " + "  ".join(parts) if parts else ""


def load_ref_hf_npz(path: Path, *, load_dm: bool = True) -> dict:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    if load_dm and "dm_emb" not in z:
        raise ValueError(f"{path}: missing dm_emb (not a batch_hf ref npz?)")
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
    e0 = float(z["e0_kcal"]) if "e0_kcal" in z else float(meta.get("e0_kcal", float("nan")))
    e_emb = _ref_scalar_kcal(z, meta, "e_int_emb_kcal", ha_key="e_int_emb_hartree")
    e_raw = _ref_scalar_kcal(z, meta, "e_int_raw_kcal", ha_key="e_int_raw_hartree")
    e_cp4 = _ref_scalar_kcal(z, meta, "e_int_cp_kcal", ha_key="e_int_cp_hartree")
    e_qmmm = qmmm_interaction_ref_kcal(e_int_raw_kcal=e_raw, e_int_cp4_kcal=e_cp4)
    out: dict = {
        "path": path.resolve(),
        "meta": meta,
        "e0_kcal": e0,
        "e_int_emb_kcal": e_emb,
        "e_int_raw_kcal": e_raw,
        "e_int_cp4_kcal": e_cp4,
        "e_int_cp_kcal": e_cp4,
        "e_int_qmmm_ref_kcal": e_qmmm,
    }
    if "e_gas_hartree" in z:
        out["e_gas_hartree"] = float(z["e_gas_hartree"])
    elif meta.get("e_gas_hartree") is not None:
        out["e_gas_hartree"] = float(meta["e_gas_hartree"])
    if load_dm and "dm_emb" in z:
        out["dm_emb"] = np.asarray(z["dm_emb"], dtype=np.float64)
    if "dm_a" in z:
        out["dm_a"] = np.asarray(z["dm_a"], dtype=np.float64)
    if "dm_cluster_qm" in z:
        out["dm_cluster_qm"] = np.asarray(z["dm_cluster_qm"], dtype=np.float64)
    if "dm_cluster_tot" in z:
        out["dm_cluster_tot"] = np.asarray(z["dm_cluster_tot"], dtype=np.float64)
    if "r_axis_ang" in z:
        out["r_axis_ang"] = np.asarray(z["r_axis_ang"], dtype=np.float64)
        out["rho_emb_h"] = np.asarray(z["rho_emb_h"], dtype=np.float64)
        out["rho_cluster_qm_h"] = np.asarray(z["rho_cluster_qm_h"], dtype=np.float64)
    if "kernel_mm" in z and "mm_element_k" in z:
        out["kernel_mm"] = np.asarray(z["kernel_mm"], dtype=np.float64).reshape(-1)
        out["mm_element_k"] = np.asarray(z["mm_element_k"], dtype=np.int8).reshape(-1)
        if "fix_alpha_bohr2" in z:
            out["fix_alpha_bohr2"] = float(z["fix_alpha_bohr2"])
        elif "fix_alpha_bohr2" in meta:
            out["fix_alpha_bohr2"] = float(meta["fix_alpha_bohr2"])
        if "fix_alpha_by_element_json" in z:
            out["fix_alpha_by_element"] = json.loads(str(z["fix_alpha_by_element_json"]))
        elif meta.get("fix_alpha_by_element") is not None:
            out["fix_alpha_by_element"] = dict(meta["fix_alpha_by_element"])
        if "lnC_by_element_json" in z:
            out["lnC_by_element"] = json.loads(str(z["lnC_by_element_json"]))
        elif meta.get("lnC_by_element") is not None:
            out["lnC_by_element"] = dict(meta["lnC_by_element"])
        if "exp_sum_by_element_json" in z:
            out["exp_sum_by_element"] = json.loads(str(z["exp_sum_by_element_json"]))
        elif meta.get("exp_sum_by_element") is not None:
            out["exp_sum_by_element"] = dict(meta["exp_sum_by_element"])
        if "envelope_kind" in z:
            out["envelope_kind"] = str(z["envelope_kind"])
        elif meta.get("envelope_kind") is not None:
            out["envelope_kind"] = str(meta["envelope_kind"])
        if "rho_o_mean_k" in z:
            out["rho_o_mean_k"] = float(z["rho_o_mean_k"])
    return out


def ref_kj_sidecar_path(ref_path: Path, alpha_cfg) -> Path:
    from embr_envelope import MmhAlphaConfig

    ref_path = Path(ref_path)
    if isinstance(alpha_cfg, (int, float)):
        alpha_cfg = MmhAlphaConfig.uniform(float(alpha_cfg))
    tag = alpha_cfg.sidecar_tag()
    return ref_path.parent / "kj_cache" / f"{ref_path.stem}_alpha{tag}.npz"


def _per_mm_from_arrays(
    kernel_mm: np.ndarray,
    mm_element: np.ndarray,
    *,
    alpha_bohr2: float,
    rho_o_mean: float = float("nan"),
) -> GaussRhoKernelsPerMm:
    return GaussRhoKernelsPerMm(
        kernel_mm=np.asarray(kernel_mm, dtype=np.float64).reshape(-1),
        mm_element=np.asarray(mm_element, dtype=np.int8).reshape(-1),
        alpha_bohr2=float(alpha_bohr2),
        rho_o_mean=float(rho_o_mean),
    )


def _alpha_matches(stored: float | None, requested: float, *, tol: float = 1e-8) -> bool:
    if stored is None:
        return False
    return abs(float(stored) - float(requested)) <= float(tol)


def _envelope_sidecar_meta(z, alpha_cfg) -> dict:
    stored_kind = str(z["envelope_kind"]) if "envelope_kind" in z else "gauss"
    stored_lnC = None
    if "lnC_by_element_json" in z:
        stored_lnC = json.loads(str(z["lnC_by_element_json"]))
    stored_exp = None
    if "exp_sum_by_element_json" in z:
        stored_exp = json.loads(str(z["exp_sum_by_element_json"]))
    return {
        "envelope_kind": stored_kind,
        "lnC_by_element": stored_lnC,
        "exp_sum_by_element": stored_exp,
    }


def _load_kj_sidecar(path: Path, alpha_cfg) -> GaussRhoKernelsPerMm | None:
    from embr_envelope import MmhEnvelopeConfig

    if isinstance(alpha_cfg, (int, float)):
        alpha_cfg = MmhEnvelopeConfig.uniform(float(alpha_cfg))
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    if "kernel_mm" not in z or "mm_element_k" not in z:
        return None
    stored_uniform = float(z["fix_alpha_bohr2"]) if "fix_alpha_bohr2" in z else None
    stored_map = None
    if "fix_alpha_by_element_json" in z:
        stored_map = json.loads(str(z["fix_alpha_by_element_json"]))
    side = _envelope_sidecar_meta(z, alpha_cfg)
    if not alpha_cfg.matches_stored(
        fix_alpha_bohr2=stored_uniform,
        fix_alpha_by_element=stored_map,
        envelope_kind=side["envelope_kind"],
        lnC_by_element=side["lnC_by_element"],
        exp_sum_by_element=side["exp_sum_by_element"],
    ):
        return None
    rho_o = float(z["rho_o_mean_k"]) if "rho_o_mean_k" in z else float("nan")
    return _per_mm_from_arrays(
        z["kernel_mm"],
        z["mm_element_k"],
        alpha_bohr2=float(alpha_cfg.legacy_fix_alpha()),
        rho_o_mean=rho_o,
    )


def _save_kj_sidecar(path: Path, per_mm: GaussRhoKernelsPerMm, *, alpha_cfg) -> None:
    from embr_envelope import MmhEnvelopeConfig

    if isinstance(alpha_cfg, (int, float)):
        alpha_cfg = MmhEnvelopeConfig.uniform(float(alpha_cfg))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw: dict = {
        "kernel_mm": np.asarray(per_mm.kernel_mm, dtype=np.float64),
        "mm_element_k": np.asarray(per_mm.mm_element, dtype=np.int8),
        "fix_alpha_bohr2": np.float64(float(alpha_cfg.legacy_fix_alpha())),
        "rho_o_mean_k": np.float64(float(per_mm.rho_o_mean)),
        "envelope_kind": np.array(str(alpha_cfg.kind)),
    }
    if not alpha_cfg.is_uniform_width():
        kw["fix_alpha_by_element_json"] = np.array(json.dumps(alpha_cfg.width_meta()))
    if not alpha_cfg.is_legacy_gaussian():
        kw["lnC_by_element_json"] = np.array(json.dumps(alpha_cfg.lnC_by_element))
    if alpha_cfg.exp_sum_by_element:
        kw["exp_sum_by_element_json"] = np.array(
            json.dumps(
                {
                    str(sym): [t.to_manifest_dict() for t in terms]
                    for sym, terms in alpha_cfg.exp_sum_by_element.items()
                }
            )
        )
    np.savez_compressed(path, **kw)


def scf_cfg_from_ref_meta(meta: dict, *, num_threads: int = 1) -> ScfEmbedConfig:
    xc = str(meta.get("xc", "B3LYP")).strip().upper()
    method = "hf" if xc == "HF" else "b3lyp"
    from scf_embed_pyscf import resolve_qm_charge

    return scf_embed_config_from_cli(
        method=method,
        basis=str(meta.get("basis", "6-31g*")),
        use_d3bj=bool(meta.get("use_d3bj", False)),
        num_threads=int(num_threads),
        verbose=0,
        cart=bool(meta.get("cart", False)),
        qm_charge=resolve_qm_charge(meta, default=0),
    )


def build_emb0_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    """Lightweight Emb0 mean-field (AO basis only; no SCF) for ρ-kernel quadrature."""
    mol = _build_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return qmmm.mm_charge(mf, frame.mm_coords_ang, frame.mm_charges, unit=cfg.unit)


def build_cluster_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    mol = build_cluster_supermol_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return mf


def build_gas_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    mol = _build_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return mf


_AU_TO_DEBYE = 2.5417461928


def _qm_com_bohr(frame: ScfFrame) -> np.ndarray:
    from pyscf import lib

    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    return np.mean(qm, axis=0) / lib.param.BOHR


def _dipole_solute_debye(mf, dm: np.ndarray, frame: ScfFrame) -> np.ndarray:
    """QM solute dipole (Debye) with origin at QM geometric center."""
    mol = mf.mol
    dm = np.asarray(dm, dtype=np.float64)
    com = tuple(float(x) for x in _qm_com_bohr(frame))
    old_verbose = int(getattr(mol, "verbose", 0))
    mol.verbose = 0
    try:
        with mol.with_common_origin(com):
            try:
                mu = mf.dip_moment(unit="AU", dm=dm, verbose=0)
            except TypeError:
                mu = mf.dip_moment(unit="AU", dm=dm)
    finally:
        mol.verbose = old_verbose
    return np.asarray(mu, dtype=np.float64).reshape(3) * _AU_TO_DEBYE


def _solute_dipoles_ref_embtheta(
    frame: ScfFrame,
    *,
    mf_emb,
    dm_emb: np.ndarray,
    dm_cl_qm: np.ndarray,
    mf_scf,
    dm_scf: np.ndarray,
    ref: dict,
    cfg: ScfEmbedConfig,
    cfg_scf: ScfEmbedConfig,
) -> dict:
    """Gas / Emb0 / Cluster^QM from ref npz DM; EmbR from this SCF (no Cluster rerun)."""
    from scf_embed_pyscf import run_gas_mf

    mf_gas = build_gas_mf_stub(frame, cfg)
    dm_gas_src = "ref.npz dm_a"
    if "dm_a" in ref:
        dm_gas = np.asarray(ref["dm_a"], dtype=np.float64)
    else:
        dm_gas_src = "run_gas_mf"
        mf_gas = run_gas_mf(frame, cfg_scf)
        dm_gas = np.asarray(mf_gas.make_rdm1(), dtype=np.float64)

    mu_gas = _dipole_solute_debye(mf_gas, dm_gas, frame)
    mu_emb = _dipole_solute_debye(mf_emb, dm_emb, frame)
    mu_cl_qm = _dipole_solute_debye(mf_emb, dm_cl_qm, frame)
    mu_embtheta = _dipole_solute_debye(mf_scf, dm_scf, frame)
    return {
        "mu_gas": mu_gas,
        "mu_emb": mu_emb,
        "mu_cl_qm": mu_cl_qm,
        "mu_embtheta": mu_embtheta,
        "dm_gas_src": dm_gas_src,
    }


def _fmt_kcal(x: float) -> str:
    return f"{float(x):+.4f}" if np.isfinite(x) else "   n/a"


def _fmt_rho(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if abs(float(x)) >= 1e-3:
        return f"{float(x):.5f}"
    return f"{float(x):+.3e}"


def _pair_for_mm_index(hi: int, pairs: list):
    hits = [p for p in pairs if int(p.jm) == int(hi)]
    if not hits:
        return None
    return min(hits, key=lambda p: float(p.dist_ang))


def _pair_for_mm_h(hi: int, pairs: list):
    return _pair_for_mm_index(hi, pairs)


def _mm_element_key(sym: str) -> str:
    return _norm_elem_key(sym)


ION_PROFILE_ORDER: tuple[str, ...] = ("Na", "K", "Cl")
MM_PAIR_AXIS_ORDER: tuple[str, ...] = ("H", "O", "C", "N")


def _mm_pair_axis_elements_in_frame(frame: ScfFrame) -> tuple[str, ...]:
    """H/O/C/N present in MM list (pair_max axis tables)."""
    present = {_mm_element_key(str(sym)) for sym in frame.mm_symbols}
    return tuple(el for el in MM_PAIR_AXIS_ORDER if el in present)


def _mm_ion_elements_in_frame(frame: ScfFrame) -> tuple[str, ...]:
    """Ion species present in MM list (Na/K/Cl), fixed display order."""
    present = {_mm_element_key(str(sym)) for sym in frame.mm_symbols}
    return tuple(el for el in ION_PROFILE_ORDER if el in present and el in ION_MM_KEYS)


def _optional_plane_mm_ion_index(frame: ScfFrame, element: str) -> int | None:
    el = _mm_element_key(str(element))
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    best: tuple[int, float] | None = None
    for i, sym in enumerate(frame.mm_symbols):
        if _mm_element_key(str(sym)) != el:
            continue
        rc = np.asarray(frame.mm_coords_ang[int(i)], dtype=np.float64).reshape(3)
        d = float(np.min(np.linalg.norm(qm - rc.reshape(1, 3), axis=1)))
        if best is None or d < best[1]:
            best = (int(i), d)
    return None if best is None else int(best[0])


def _optional_plane_mm_cl_index(frame: ScfFrame) -> int | None:
    return _optional_plane_mm_ion_index(frame, "Cl")


def _axis_profiles_for_mm_element(
    pairs: list,
    element: str,
    *,
    frame: ScfFrame | None = None,
    mf_emb,
    dm_emb,
    mf_cl_rho,
    dm_cl_qm: np.ndarray,
    line_dr: float,
    line_max: float,
    dm_scf: np.ndarray | None = None,
    plane_mm_index: int | None = None,
) -> tuple[dict, dict | None, int, bool]:
    """
    Axis-averaged ρ_emb / ρ_Cluster^QM (+ optional EmbR ρ_scf) for MM-H, MM-O, or MM-Cl pairs.

    Uses ``pair_max``-filtered ``pairs`` first; if none match ``element`` and ``frame`` is
    given, falls back to each MM site → nearest QM (same as ion spillover tables).

    Returns (avg_dict, nearest_pair_dict|None, n_pairs, used_nearest_qm_fallback).
    """
    key = _mm_element_key(element)
    sub = [p for p in pairs if _mm_element_key(p.mm_symbol) == key]
    used_fallback = False
    if not sub and frame is not None:
        sub = all_ion_qm_line_pairs(frame, element=key)
        used_fallback = bool(sub)
    d_max = max((float(p.dist_ang) for p in sub), default=0.0)
    r_end = max(float(line_max), d_max + 0.01) if used_fallback else float(line_max)
    r_grid = np.arange(0.0, r_end + 0.5 * float(line_dr), float(line_dr))
    nan = np.full(int(r_grid.size), np.nan, dtype=np.float64)
    if not sub:
        avg: dict = {"r": r_grid, "rho_emb": nan.copy(), "rho_cl_qm": nan.copy()}
        if dm_scf is not None:
            avg["rho_scf"] = nan.copy()
            avg["fix_s"] = nan.copy()
        return avg, None, 0, False

    emb_samples: list[tuple[np.ndarray, np.ndarray]] = []
    cl_samples: list[tuple[np.ndarray, np.ndarray]] = []
    scf_samples: list[tuple[np.ndarray, np.ndarray]] = []
    for p in sub:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cl_rho, dm_cl_qm, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
        )
        emb_samples.append((r_e, rho_e))
        cl_samples.append((r_c, rho_c))
        if dm_scf is not None:
            r_s, rho_s = line_rho_profile_on_qm_mm_axis(
                mf_emb, dm_scf, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
            )
            scf_samples.append((r_s, rho_s))

    rho_emb_avg = _mean_line_profiles_at_r_grid(emb_samples, r_grid)
    rho_cl_avg = _mean_line_profiles_at_r_grid(cl_samples, r_grid)
    avg = {
        "r": r_grid,
        "rho_emb": rho_emb_avg,
        "rho_cl_qm": rho_cl_avg,
    }
    if dm_scf is not None:
        rho_scf_avg = _mean_line_profiles_at_r_grid(scf_samples, r_grid)
        fix_avg = np.asarray(
            [
                _fix_s_scalar(rho_emb_avg[i], rho_cl_avg[i], rho_scf_avg[i])
                for i in range(int(r_grid.size))
            ],
            dtype=np.float64,
        )
        avg["rho_scf"] = rho_scf_avg
        avg["fix_s"] = fix_avg

    nearest: dict | None = None
    if plane_mm_index is not None:
        pair = _pair_for_mm_index(int(plane_mm_index), pairs)
        if pair is None or _mm_element_key(pair.mm_symbol) != key:
            if frame is not None:
                pair = nearest_ion_qm_line_pair(frame, element=key, mm_index=int(plane_mm_index))
            else:
                pair = None
        if pair is not None and _mm_element_key(pair.mm_symbol) == key:
            r_e, rho_e = line_rho_profile_on_qm_mm_axis(
                mf_emb, dm_emb, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
            )
            r_c, rho_c = line_rho_profile_on_qm_mm_axis(
                mf_cl_rho, dm_cl_qm, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
            )
            nearest = {
                "hi": int(pair.jm),
                "iqm": int(pair.iq),
                "qm_symbol": str(pair.qm_symbol).strip(),
                "dist_ang": float(pair.dist_ang),
                "r": r_e,
                "rho_emb": rho_e,
                "rho_cl_qm": rho_c,
            }
            if dm_scf is not None:
                r_s, rho_s = line_rho_profile_on_qm_mm_axis(
                    mf_emb, dm_scf, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
                )
                fix_n = np.asarray(
                    [_fix_s_scalar(rho_e[i], rho_c[i], rho_s[i]) for i in range(r_e.size)],
                    dtype=np.float64,
                )
                nearest["rho_scf"] = rho_s
                nearest["fix_s"] = fix_n

    return avg, nearest, int(len(sub)), bool(used_fallback)


def _collect_mm_pair_axis_profiles(
    frame: ScfFrame,
    pairs: list,
    *,
    mf_emb,
    dm_emb,
    mf_cl_rho,
    dm_cl_qm: np.ndarray,
    line_dr: float,
    line_max: float,
    dm_scf: np.ndarray | None = None,
    plane_mm_h_index: int | None = None,
    pair_max_ang: float,
) -> dict:
    """Avg/nearest axis ρ for MM-H/O/C/N (pair_max with nearest-QM fallback)."""
    elements = _mm_pair_axis_elements_in_frame(frame)
    out: dict = {
        "pair_axis_elements": elements,
        "pair_max_ang": float(pair_max_ang),
    }
    for el in elements:
        tag = el.lower()
        if el == "H" and plane_mm_h_index is not None:
            plane_ix: int | None = int(plane_mm_h_index)
        elif el == "O":
            plane_ix = _optional_plane_mm_o_index(frame)
        else:
            plane_ix = _optional_plane_mm_ion_index(frame, el)
        avg, nearest, n_pairs, fallback = _axis_profiles_for_mm_element(
            pairs,
            el,
            frame=frame,
            mf_emb=mf_emb,
            dm_emb=dm_emb,
            mf_cl_rho=mf_cl_rho,
            dm_cl_qm=dm_cl_qm,
            dm_scf=dm_scf,
            line_dr=float(line_dr),
            line_max=float(line_max),
            plane_mm_index=plane_ix,
        )
        out[f"avg_{tag}"] = avg
        out[f"nearest_{tag}"] = nearest
        out[f"n_{tag}_pairs"] = int(n_pairs)
        out[f"{tag}_axis_nearest_qm_fallback"] = bool(fallback)
        if plane_ix is not None:
            out[f"plane_mm_{tag}_index"] = int(plane_ix)
    return out


def _axis_profiles_for_ion_sites(
    frame: ScfFrame,
    element: str,
    *,
    mf_emb,
    dm_emb,
    mf_cl_rho,
    dm_cl_qm: np.ndarray,
    line_dr: float,
    line_max: float,
    dm_scf: np.ndarray | None = None,
    plane_mm_ion_index: int | None = None,
) -> tuple[dict, dict | None, int, float]:
    """
    Ion axis profiles (Na/K/Cl): each site → nearest QM (no pair_max cutoff).

    Returns (avg_dict, nearest_pair_dict|None, n_sites, r_print_max).
    """
    el = _mm_element_key(str(element))
    ion_pairs = all_ion_qm_line_pairs(frame, element=el)
    d_max = max((float(p.dist_ang) for p in ion_pairs), default=0.0)
    r_end = max(float(line_max), d_max + 0.01)
    r_grid = np.arange(0.0, r_end + 0.5 * float(line_dr), float(line_dr))
    nan = np.full(int(r_grid.size), np.nan, dtype=np.float64)
    if not ion_pairs:
        avg: dict = {"r": r_grid, "rho_emb": nan.copy(), "rho_cl_qm": nan.copy()}
        if dm_scf is not None:
            avg["rho_scf"] = nan.copy()
            avg["fix_s"] = nan.copy()
        return avg, None, 0, float(r_end)

    emb_samples: list[tuple[np.ndarray, np.ndarray]] = []
    cl_samples: list[tuple[np.ndarray, np.ndarray]] = []
    scf_samples: list[tuple[np.ndarray, np.ndarray]] = []
    for p in ion_pairs:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cl_rho, dm_cl_qm, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
        )
        emb_samples.append((r_e, rho_e))
        cl_samples.append((r_c, rho_c))
        if dm_scf is not None:
            r_s, rho_s = line_rho_profile_on_qm_mm_axis(
                mf_emb, dm_scf, p.r_mm_ang, p.r_qm_ang, dr=float(line_dr)
            )
            scf_samples.append((r_s, rho_s))

    rho_emb_avg = _mean_line_profiles_at_r_grid(emb_samples, r_grid)
    rho_cl_avg = _mean_line_profiles_at_r_grid(cl_samples, r_grid)
    avg = {
        "r": r_grid,
        "rho_emb": rho_emb_avg,
        "rho_cl_qm": rho_cl_avg,
    }
    if dm_scf is not None:
        rho_scf_avg = _mean_line_profiles_at_r_grid(scf_samples, r_grid)
        fix_avg = np.asarray(
            [
                _fix_s_scalar(rho_emb_avg[i], rho_cl_avg[i], rho_scf_avg[i])
                for i in range(int(r_grid.size))
            ],
            dtype=np.float64,
        )
        avg["rho_scf"] = rho_scf_avg
        avg["fix_s"] = fix_avg

    nearest: dict | None = None
    pair = nearest_ion_qm_line_pair(frame, element=el, mm_index=plane_mm_ion_index)
    if pair is not None:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cl_rho, dm_cl_qm, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
        )
        nearest = {
            "hi": int(pair.jm),
            "iqm": int(pair.iq),
            "qm_symbol": str(pair.qm_symbol).strip(),
            "dist_ang": float(pair.dist_ang),
            "r": r_e,
            "rho_emb": rho_e,
            "rho_cl_qm": rho_c,
        }
        if dm_scf is not None:
            r_s, rho_s = line_rho_profile_on_qm_mm_axis(
                mf_emb, dm_scf, pair.r_mm_ang, pair.r_qm_ang, dr=float(line_dr)
            )
            fix_n = np.asarray(
                [_fix_s_scalar(rho_e[i], rho_c[i], rho_s[i]) for i in range(r_e.size)],
                dtype=np.float64,
            )
            nearest["rho_scf"] = rho_s
            nearest["fix_s"] = fix_n

    return avg, nearest, int(len(ion_pairs)), float(r_end)


def _axis_profiles_for_cl_sites(
    frame: ScfFrame,
    *,
    mf_emb,
    dm_emb,
    mf_cl_rho,
    dm_cl_qm: np.ndarray,
    line_dr: float,
    line_max: float,
    dm_scf: np.ndarray | None = None,
    plane_mm_cl_index: int | None = None,
) -> tuple[dict, dict | None, int, float]:
    """Backward-compatible Cl⁻ wrapper."""
    return _axis_profiles_for_ion_sites(
        frame,
        "Cl",
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        mf_cl_rho=mf_cl_rho,
        dm_cl_qm=dm_cl_qm,
        line_dr=float(line_dr),
        line_max=float(line_max),
        dm_scf=dm_scf,
        plane_mm_ion_index=plane_mm_cl_index,
    )


def _collect_ion_axis_profiles(
    frame: ScfFrame,
    *,
    mf_emb,
    dm_emb,
    mf_cl_rho,
    dm_cl_qm: np.ndarray,
    line_dr: float,
    line_max: float,
    dm_scf: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per present ion (Na/K/Cl): avg/nearest axis ρ profiles."""
    ions: dict[str, dict] = {}
    for el in _mm_ion_elements_in_frame(frame):
        plane_ix = _optional_plane_mm_ion_index(frame, el)
        avg, nearest, n_sites, r_print = _axis_profiles_for_ion_sites(
            frame,
            el,
            mf_emb=mf_emb,
            dm_emb=dm_emb,
            mf_cl_rho=mf_cl_rho,
            dm_cl_qm=dm_cl_qm,
            dm_scf=dm_scf,
            line_dr=float(line_dr),
            line_max=float(line_max),
            plane_mm_ion_index=plane_ix,
        )
        ions[el] = {
            "avg": avg,
            "nearest": nearest,
            "n_sites": int(n_sites),
            "r_print_max": float(r_print),
            "plane_mm_index": plane_ix,
        }
    return ions


def _merge_ion_profiles_into_dict(out: dict, ions: dict[str, dict]) -> None:
    out["ion_elements"] = tuple(ions.keys())
    out["ions"] = ions
    for el, block in ions.items():
        tag = el.lower()
        out[f"avg_{tag}"] = block["avg"]
        out[f"nearest_{tag}"] = block["nearest"]
        out[f"n_{tag}_sites"] = int(block["n_sites"])
        out[f"{tag}_r_print_max"] = float(block["r_print_max"])
        if block["plane_mm_index"] is not None:
            out[f"plane_mm_{tag}_index"] = int(block["plane_mm_index"])


def cluster_dm_on_emb_basis(
    ref_dm: dict,
    mf_emb,
    frame: ScfFrame,
    cfg,
) -> tuple[np.ndarray, str]:
    """Load DFT cluster QM DM compacted to Emb0 mf AO dimension."""
    from scf_embed_cluster import compact_cluster_qm_dm

    n_emb = int(mf_emb.mol.nao)
    mf_cl = build_cluster_mf_stub(frame, cfg)
    if "dm_cluster_qm" in ref_dm:
        dm = np.asarray(ref_dm["dm_cluster_qm"], dtype=np.float64)
        if dm.shape == (n_emb, n_emb):
            return dm, f"dm_cluster_qm {dm.shape}"
        if dm.shape[0] == mf_cl.mol.nao:
            out = compact_cluster_qm_dm(dm, mf_cl, frame, n_emb_nao=n_emb)
            return out, f"compact dm_cluster_qm {dm.shape} -> {out.shape}"
    if "dm_cluster_tot" in ref_dm:
        dm = np.asarray(ref_dm["dm_cluster_tot"], dtype=np.float64)
        if dm.shape[0] == mf_cl.mol.nao:
            out = compact_cluster_qm_dm(dm, mf_cl, frame, n_emb_nao=n_emb)
            return out, f"compact dm_cluster_tot {dm.shape} -> {out.shape}"
    raise ValueError(
        "ref npz has no usable dm_cluster_qm/dm_cluster_tot for Emb0 AO basis "
        f"(emb nao={n_emb}, cluster nao={mf_cl.mol.nao})"
    )


def mf_for_qm_dm_profile(mf_emb, mf_cl, dm_qm: np.ndarray):
    """Mean-field whose AO count matches ``dm_qm`` (Emb0-compact vs full cluster QM block)."""
    dm_qm = np.asarray(dm_qm)
    n_emb = int(mf_emb.mol.nao)
    if dm_qm.shape[0] == n_emb:
        return mf_emb
    n_cl = int(mf_cl.mol.nao)
    if dm_qm.shape[0] == n_cl:
        return mf_cl
    raise ValueError(
        f"dm_qm shape {dm_qm.shape} != emb nao {n_emb} or cluster nao {n_cl}"
    )


def scf_meta_from_ref_npz(ref_path: Path) -> dict[str, object]:
    """xc/basis/d3bj recorded in batch_hf ``ref_*.npz`` meta."""
    ref = load_ref_hf_npz(ref_path, load_dm=False)
    cfg = scf_cfg_from_ref_meta(ref["meta"])
    return {"xc": cfg.xc, "basis": cfg.basis, "use_d3bj": bool(cfg.use_d3bj)}


def spillover_profiles_from_ref_npz(
    ref_path: Path,
    frame: ScfFrame,
    *,
    plane_mm_h_index: int,
    pair_max_ang: float = 2.0,
    line_dr: float = 0.1,
    line_max: float = 2.0,
    r_cut_mm: float | None = None,
) -> dict:
    """
    Axis ρ_emb vs ρ_Cluster^QM from saved ref dm (no SCF).

    Returns nearest-H single-pair and MM-H averaged profiles.
    """
    ref = load_ref_hf_npz(ref_path)
    if "dm_cluster_qm" not in ref:
        raise ValueError(f"{ref_path}: missing dm_cluster_qm")
    if r_cut_mm is not None:
        frame = filter_mm_by_distance(frame, r_cut_ang=float(r_cut_mm))
    cfg = scf_cfg_from_ref_meta(ref["meta"])
    mf_emb = build_emb0_mf_stub(frame, cfg)
    mf_cl = build_cluster_mf_stub(frame, cfg)
    dm_emb = np.asarray(ref["dm_emb"], dtype=np.float64)
    dm_cl_qm, _cl_src = cluster_dm_on_emb_basis(ref, mf_emb, frame, cfg)
    if dm_emb.shape[0] != mf_emb.mol.nao:
        raise ValueError(f"{ref_path}: dm_emb nao mismatch")
    if dm_cl_qm.shape[0] != mf_emb.mol.nao:
        raise ValueError(f"{ref_path}: compact dm_cluster_qm nao mismatch")
    mf_cl_rho = mf_for_qm_dm_profile(mf_emb, mf_cl, dm_cl_qm)

    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    pair_axis = _collect_mm_pair_axis_profiles(
        frame,
        pairs,
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        mf_cl_rho=mf_cl_rho,
        dm_cl_qm=dm_cl_qm,
        line_dr=float(line_dr),
        line_max=float(line_max),
        plane_mm_h_index=int(plane_mm_h_index),
        pair_max_ang=float(pair_max_ang),
    )
    ion_profiles = _collect_ion_axis_profiles(
        frame,
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        mf_cl_rho=mf_cl_rho,
        dm_cl_qm=dm_cl_qm,
        line_dr=float(line_dr),
        line_max=float(line_max),
    )

    out = {
        "e0_kcal": float(ref["e0_kcal"]),
        "meta": ref["meta"],
        "ref_path": ref["path"],
        **pair_axis,
    }
    _merge_ion_profiles_into_dict(out, ion_profiles)
    return out


def _optional_plane_mm_o_index(frame: ScfFrame) -> int | None:
    try:
        from embr_theta.rho_plane_geom import nearest_mm_o_index_to_qm

        return int(nearest_mm_o_index_to_qm(frame))
    except ValueError:
        return None


def _fix_s_scalar(rho_emb: float, rho_cl: float, rho_scf: float, *, eps: float = 1e-18) -> float:
    de = float(rho_emb) - float(rho_cl)
    ds = float(rho_scf) - float(rho_cl)
    if not np.isfinite(de) or not np.isfinite(ds) or abs(de) < float(eps):
        return float("nan")
    return 1.0 - ds / de


def _embed_theta_rep_kwargs(
    frame: ScfFrame,
    alpha_cfg,
    *,
    rep_centers: np.ndarray,
    cone,
) -> dict:
    """Build kwargs for ``run_embed_theta_mf_rep`` (new envelope_cfg or legacy α per site)."""
    import inspect

    from scf_embed_rep_center import run_embed_theta_mf_rep
    from embr_envelope import is_kernel_mm_symbol

    kw: dict = {
        "alpha_bohr2": float(alpha_cfg.legacy_fix_alpha()),
        "rep_centers_ang": rep_centers,
        "cone": cone,
    }
    sig = inspect.signature(run_embed_theta_mf_rep)
    # Legacy gauss + lnC=0: keep original EmbR path (per-site α, no envelope_cfg).
    if alpha_cfg.is_legacy_gaussian():
        if not alpha_cfg.is_uniform_width() and "alpha_per_center" in sig.parameters:
            kw["alpha_per_center"] = np.asarray(
                [
                    float(alpha_cfg.alpha_for_symbol(sym))
                    if is_kernel_mm_symbol(sym)
                    else float(alpha_cfg.legacy_fix_alpha())
                    for sym in frame.mm_symbols
                ],
                dtype=np.float64,
            )
        return kw
    if "envelope_cfg" not in sig.parameters:
        raise RuntimeError(
            "EmbR envelope/lnC needs updated scf_embed_rep_center.py and scf_embed_pyscf.py "
            "(run_embed_theta_mf_rep must accept envelope_cfg=). "
            "Sync scf_embed_rep_center.py and scf_embed_pyscf.py from this repo, then re-run."
        )
    kw["envelope_cfg"] = alpha_cfg
    return kw


def spillover_profiles_ref_embtheta(
    ref_path: Path,
    frame: ScfFrame,
    amp_mm: np.ndarray,
    *,
    fix_alpha: float,
    alpha_cfg=None,
    plane_mm_h_index: int,
    pair_max_ang: float = 2.0,
    line_dr: float = 0.1,
    line_max: float = 2.0,
    r_cut_mm: float | None = None,
    embed_scf_conv_tol: float = 1e-4,
    cone_theta1_deg: float = 180.0,
    cone_theta2_deg: float = 180.0,
    rep_center: str = "on_nucleus",
    rep_oh_frac: float = 0.5,
    num_threads: int = 4,
) -> dict:
    """
    ρ_emb / ρ_Cluster^QM from ``ref_*.npz``; ρ_scf from EmbR SCF with ``amp_mm`` (no Cluster rerun).
    """
    from scf_embed_pyscf import ConeRepAng, _total_energy, hartree_to_kcal, run_gas_mf, scf_embed_config_from_cli
    from scf_embed_rep_center import build_repulsion_center_coords_ang, run_embed_theta_mf_rep
    from embr_envelope import MmhAlphaConfig

    if alpha_cfg is None:
        alpha_cfg = MmhAlphaConfig.uniform(float(fix_alpha))

    ref = load_ref_hf_npz(ref_path)
    if "dm_cluster_qm" not in ref:
        raise ValueError(f"{ref_path}: missing dm_cluster_qm")
    if r_cut_mm is not None:
        frame = filter_mm_by_distance(frame, r_cut_ang=float(r_cut_mm))

    cfg = scf_cfg_from_ref_meta(ref["meta"], num_threads=int(num_threads))
    cfg_scf = scf_embed_config_from_cli(
        method="hf" if str(cfg.xc).upper() == "HF" else "b3lyp",
        basis=str(cfg.basis),
        use_d3bj=bool(cfg.use_d3bj),
        num_threads=int(num_threads),
        verbose=0,
        conv_tol=float(embed_scf_conv_tol),
        cart=bool(cfg.cart),
        qm_charge=int(getattr(cfg, "qm_charge", 0) or 0),
    )
    cone = ConeRepAng(theta1_deg=float(cone_theta1_deg), theta2_deg=float(cone_theta2_deg))
    rep_centers = build_repulsion_center_coords_ang(
        frame, str(rep_center), oh_frac=float(rep_oh_frac)
    )

    mf_emb = build_emb0_mf_stub(frame, cfg)
    mf_cl = build_cluster_mf_stub(frame, cfg)
    dm_emb = np.asarray(ref["dm_emb"], dtype=np.float64)
    dm_cl_qm, _cl_src = cluster_dm_on_emb_basis(ref, mf_emb, frame, cfg)
    if dm_emb.shape[0] != mf_emb.mol.nao:
        raise ValueError(f"{ref_path}: dm_emb nao mismatch")
    if dm_cl_qm.shape[0] != mf_emb.mol.nao:
        raise ValueError(f"{ref_path}: compact dm_cluster_qm nao mismatch")
    mf_cl_rho = mf_for_qm_dm_profile(mf_emb, mf_cl, dm_cl_qm)

    print(
        f"  EmbR conv_tol={float(embed_scf_conv_tol):g}",
        flush=True,
    )
    embed_rep_kw = _embed_theta_rep_kwargs(
        frame,
        alpha_cfg,
        rep_centers=rep_centers,
        cone=cone,
    )
    mf_scf = run_embed_theta_mf_rep(
        frame,
        cfg_scf,
        np.asarray(amp_mm, dtype=np.float64),
        **embed_rep_kw,
    )
    dm_scf = np.asarray(mf_scf.make_rdm1(), dtype=np.float64)
    dipole = _solute_dipoles_ref_embtheta(
        frame,
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        dm_cl_qm=dm_cl_qm,
        mf_scf=mf_scf,
        dm_scf=dm_scf,
        ref=ref,
        cfg=cfg,
        cfg_scf=cfg_scf,
    )
    e_gas_h = ref.get("e_gas_hartree")
    if e_gas_h is None or not np.isfinite(float(e_gas_h)):
        mf_gas = run_gas_mf(frame, cfg_scf)
        e_gas_h, _, _ = _total_energy(mf_gas, use_d3bj=bool(cfg_scf.use_d3bj))
    else:
        e_gas_h = float(e_gas_h)
    e_emb_h, _, _ = _total_energy(mf_scf, use_d3bj=bool(cfg_scf.use_d3bj))
    e_int_embtheta_kcal = float(hartree_to_kcal(float(e_emb_h) - float(e_gas_h)))
    e_int_emb0_kcal = float(ref.get("e_int_emb_kcal", float("nan")))
    e_int_cp4_kcal = float(ref.get("e_int_cp4_kcal", ref.get("e_int_cp_kcal", float("nan"))))
    e_int_raw_kcal = float(ref.get("e_int_raw_kcal", float("nan")))
    e_int_qmmm_ref_kcal = float(
        ref.get(
            "e_int_qmmm_ref_kcal",
            qmmm_interaction_ref_kcal(e_int_raw_kcal=e_int_raw_kcal, e_int_cp4_kcal=e_int_cp4_kcal),
        )
    )
    e0_label_kcal = float(ref["e0_kcal"])

    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    pair_axis = _collect_mm_pair_axis_profiles(
        frame,
        pairs,
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        mf_cl_rho=mf_cl_rho,
        dm_cl_qm=dm_cl_qm,
        dm_scf=dm_scf,
        line_dr=float(line_dr),
        line_max=float(line_max),
        plane_mm_h_index=int(plane_mm_h_index),
        pair_max_ang=float(pair_max_ang),
    )
    ion_profiles = _collect_ion_axis_profiles(
        frame,
        mf_emb=mf_emb,
        dm_emb=dm_emb,
        mf_cl_rho=mf_cl_rho,
        dm_cl_qm=dm_cl_qm,
        dm_scf=dm_scf,
        line_dr=float(line_dr),
        line_max=float(line_max),
    )

    out = {
        "e0_kcal": e0_label_kcal,
        "meta": ref["meta"],
        "ref_path": ref["path"],
        "e_int_embtheta_kcal": e_int_embtheta_kcal,
        "e_int_emb0_kcal": e_int_emb0_kcal,
        "e_int_cp4_kcal": e_int_cp4_kcal,
        "e_int_raw_kcal": e_int_raw_kcal,
        "e_int_qmmm_ref_kcal": e_int_qmmm_ref_kcal,
        "e_int_cp_kcal": e_int_cp4_kcal,
        "embed_scf_converged": bool(getattr(mf_scf, "converged", True)),
        "dipole": dipole,
        # Densities already computed for fix tables — persist for cube / Δρ export
        "dm_emb": dm_emb,
        "dm_scf": dm_scf,
        "dm_cluster_qm": np.asarray(dm_cl_qm, dtype=np.float64),
        "e_gas_hartree": float(e_gas_h),
        "xc": str(cfg.xc),
        "basis": str(cfg.basis),
        **pair_axis,
    }
    _merge_ion_profiles_into_dict(out, ion_profiles)
    return out


def kernels_per_mm_from_ref_npz(
    ref_path: Path,
    frame: ScfFrame,
    *,
    fix_alpha: float | None = None,
    alpha_cfg=None,
    r_cut_mm: float | None = None,
    num_threads: int = 1,
    write_sidecar: bool = True,
) -> tuple[GaussRhoKernelsPerMm, float, dict, str]:
    """
    Return (kernels_per_mm, e0_kcal, ref_meta, kj_source).

    kj_source: ``npz`` | ``sidecar`` | ``grid`` (quadrature on saved dm_emb, no SCF).
    """
    from embr_envelope import MmhAlphaConfig

    ref_path = Path(ref_path)
    ref = load_ref_hf_npz(ref_path, load_dm=False)
    meta = dict(ref["meta"])
    e0 = float(ref["e0_kcal"])
    if alpha_cfg is None:
        alpha_cfg = MmhAlphaConfig.uniform(float(3.0 if fix_alpha is None else fix_alpha))

    if "kernel_mm" in ref and "mm_element_k" in ref:
        stored_a = ref.get("fix_alpha_bohr2")
        stored_map = ref.get("fix_alpha_by_element")
        stored_kind = str(ref.get("envelope_kind", "gauss"))
        stored_lnC = ref.get("lnC_by_element")
        trust_legacy = (
            stored_a is None
            and stored_map is None
            and alpha_cfg.is_legacy_gaussian()
            and alpha_cfg.is_uniform_width()
        )
        if trust_legacy or alpha_cfg.matches_stored(
            fix_alpha_bohr2=stored_a,
            fix_alpha_by_element=stored_map,
            envelope_kind=stored_kind,
            lnC_by_element=stored_lnC,
            exp_sum_by_element=ref.get("exp_sum_by_element"),
        ):
            per_mm = _per_mm_from_arrays(
                ref["kernel_mm"],
                ref["mm_element_k"],
                alpha_bohr2=float(stored_a if stored_a is not None else alpha_cfg.legacy_fix_alpha()),
                rho_o_mean=float(ref.get("rho_o_mean_k", float("nan"))),
            )
            return per_mm, e0, meta, "npz"
        # kernel_mm present but envelope mismatch — recompute from dm_emb.

    sidecar = ref_kj_sidecar_path(ref_path, alpha_cfg)
    hit = _load_kj_sidecar(sidecar, alpha_cfg)
    if hit is not None:
        return hit, e0, meta, "sidecar"

    ref_dm = load_ref_hf_npz(ref_path, load_dm=True)
    if r_cut_mm is not None:
        frame = filter_mm_by_distance(frame, r_cut_ang=float(r_cut_mm))
    cfg = scf_cfg_from_ref_meta(ref_dm["meta"], num_threads=int(num_threads))
    mf = build_emb0_mf_stub(frame, cfg)
    dm_emb = np.asarray(ref_dm["dm_emb"], dtype=np.float64)
    if dm_emb.shape[0] != mf.mol.nao:
        raise ValueError(
            f"{ref_path}: dm_emb nao {dm_emb.shape[0]} != frame stub nao {mf.mol.nao} "
            f"(SCF settings / geometry mismatch?)"
        )
    from scf_embed_perturb import rho_kernels_per_mm

    per_mm = rho_kernels_per_mm(
        mf,
        frame,
        envelope_cfg=alpha_cfg,
        cone=isotropic_repulsion_cone(),
        dm=dm_emb,
    )
    if write_sidecar:
        _save_kj_sidecar(sidecar, per_mm, alpha_cfg=alpha_cfg)
    return per_mm, e0, meta, "grid"
