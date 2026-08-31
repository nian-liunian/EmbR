"""
Cache converged DFT / embedding SCF for ``line_rho_pert_cluster``.

First run with ``--scf-cache PATH`` computes and writes the cache.
Later runs with the same path + matching geometry/settings reload DMs and
rebuild lightweight mf stubs (no SCF) for ρ evaluation / plane export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyscf import dft, qmmm

from scf_embed_cluster import build_cluster_supermol_mol
from scf_embed_io import ScfFrame
from scf_embed_pyscf import (
    ConeRepAng,
    ScfEmbedConfig,
    _attach_gaussian_repulsion_amps,
    _build_mol,
    amp_mm_for_o_h_sites,
    cone_axes_h_to_nearest_qm,
    make_mean_field,
)


def geom_hash(frame: ScfFrame) -> str:
    blob = np.asarray(
        np.concatenate([frame.qm_coords_ang, frame.mm_coords_ang], axis=0),
        dtype=np.float64,
    ).tobytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def _frame_to_npz(frame: ScfFrame) -> dict[str, np.ndarray]:
    return {
        "qm_coords_ang": np.asarray(frame.qm_coords_ang, dtype=np.float64),
        "mm_coords_ang": np.asarray(frame.mm_coords_ang, dtype=np.float64),
        "mm_charges": np.asarray(frame.mm_charges, dtype=np.float64),
        "qm_symbols": np.asarray(frame.qm_symbols, dtype=object),
        "mm_symbols": np.asarray(frame.mm_symbols, dtype=object),
    }


def _frame_from_npz(z) -> ScfFrame:
    return ScfFrame(
        qm_symbols=tuple(str(s) for s in z["qm_symbols"]),
        qm_coords_ang=np.asarray(z["qm_coords_ang"], dtype=np.float64),
        mm_symbols=tuple(str(s) for s in z["mm_symbols"]),
        mm_coords_ang=np.asarray(z["mm_coords_ang"], dtype=np.float64),
        mm_charges=np.asarray(z["mm_charges"], dtype=np.float64),
    )


@dataclass
class LineScfCacheBundle:
    frame: ScfFrame
    mf_emb: dft.rks.RKS
    mf_cl: dft.rks.RKS
    mf_gas: dft.rks.RKS
    mf_scf: dft.rks.RKS | None
    dm_emb: np.ndarray
    dm_cl_tot: np.ndarray
    dm_cl_qm: np.ndarray
    dm_pert: np.ndarray
    dm_gas: np.ndarray
    dm_scf: np.ndarray | None
    pert_res: SimpleNamespace
    de_kernel_kcal: float
    de_lin_kcal: float
    de_rho1_kcal: float
    meta: dict


@dataclass
class LineScfCacheReference:
    """Emb0 + Cluster + gas reference (independent of A_H scale)."""

    frame: ScfFrame
    mf_emb: dft.rks.RKS
    mf_cl: dft.rks.RKS
    mf_gas: dft.rks.RKS
    dm_emb: np.ndarray
    dm_cl_tot: np.ndarray
    dm_cl_qm: np.ndarray
    dm_gas: np.ndarray
    meta: dict


AMP_SCALE_META_KEYS = frozenset({"scale_a_h", "scale_a_o"})
# Reference reload: reuse Emb0/Cluster/gas when only A_H fit (peratom) or scale changed.
CLUSTER_REFERENCE_RELAX_META_KEYS = AMP_SCALE_META_KEYS | frozenset({"peratom"})
PARTIAL_VREP_REUSE_RELAX_META_KEYS = CLUSTER_REFERENCE_RELAX_META_KEYS


def cache_allows_partial_vrep_reuse(cached: dict, expect: dict) -> bool:
    """
    True when geometry/α/d3bj/etc. match but peratom (A_H fit) or scale changed.

    Reuse DFT Cluster/Emb0/gas from cache; rerun pert + EmbR only.
    """
    try:
        _check_cache_meta(cached, expect, ignore=PARTIAL_VREP_REUSE_RELAX_META_KEYS)
    except ValueError:
        return False
    for key in PARTIAL_VREP_REUSE_RELAX_META_KEYS:
        if key not in expect:
            continue
        if not _meta_values_equal(cached.get(key), expect.get(key)):
            return True
    return False


def cache_expect_without_amp_scale(expect: dict) -> dict:
    return {k: v for k, v in expect.items() if k not in AMP_SCALE_META_KEYS}


def amp_scale_meta_matches(cached: dict, expect: dict) -> bool:
    for key in AMP_SCALE_META_KEYS:
        if key not in expect:
            continue
        if not _meta_values_equal(cached.get(key), expect[key]):
            return False
    return True


def build_emb0_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    mol = _build_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return qmmm.mm_charge(mf, frame.mm_coords_ang, frame.mm_charges, unit=cfg.unit)


def build_gas_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    mol = _build_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return mf


def build_cluster_mf_stub(frame: ScfFrame, cfg: ScfEmbedConfig):
    mol = build_cluster_supermol_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return mf


def restore_mo_energy(mf) -> np.ndarray:
    """Set ``mf.mo_energy`` after ``mo_coeff``/``mo_occ`` were loaded from cache."""
    en = getattr(mf, "mo_energy", None)
    if en is not None:
        arr = np.asarray(en, dtype=np.float64)
        if arr.ndim >= 1 and arr.size > 0:
            mf.mo_energy = np.real(arr)
            return mf.mo_energy
    get_e = getattr(mf, "get_mo_energy", None)
    if callable(get_e):
        mf.mo_energy = np.real(np.asarray(get_e(), dtype=np.float64))
        return mf.mo_energy
    mo = np.asarray(mf.mo_coeff, dtype=np.float64)
    occ = np.asarray(mf.mo_occ, dtype=np.float64)
    dm = mf.make_rdm1() if hasattr(mf, "make_rdm1") else mo @ np.diag(occ) @ mo.conj().T
    fock = mf.get_fock(dm=dm)
    mf.mo_energy = np.real(np.einsum("pi,pi->p", mo.conj(), fock @ mo))
    return mf.mo_energy


def build_emb0_mf_from_cache(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    z,
) -> dft.rks.RKS:
    """Emb0 mf stub; restore MO if present in cache (skip Emb0 SCF on A_H rescans)."""
    mf = build_emb0_mf_stub(frame, cfg)
    if "mo_coeff_emb" in z and "mo_occ_emb" in z:
        mf.mo_coeff = np.asarray(z["mo_coeff_emb"], dtype=np.float64)
        mf.mo_occ = np.asarray(z["mo_occ_emb"], dtype=np.float64)
        mf.converged = True
        if "e_emb0_hartree" in z:
            mf.e_tot = float(z["e_emb0_hartree"])
        if "cycles_emb0" in z and int(z["cycles_emb0"]) >= 0:
            mf.cycles = int(z["cycles_emb0"])
        if "mo_energy_emb" in z:
            mf.mo_energy = np.asarray(z["mo_energy_emb"], dtype=np.float64)
        else:
            restore_mo_energy(mf)
    return mf


def build_emb_theta_mf_stub(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    amp_mm: np.ndarray,
    *,
    alpha_bohr2: float,
    rep_centers_ang: np.ndarray,
    cone: ConeRepAng | None = None,
) -> dft.rks.RKS:
    amp_full = amp_mm_for_o_h_sites(frame, amp_mm)
    rep_centers = np.asarray(rep_centers_ang, dtype=np.float64).reshape(-1, 3)
    mf = build_emb0_mf_stub(frame, cfg)
    if float(np.max(np.abs(amp_full))) > 0.0:
        cone_kw: dict = {}
        if cone is not None and not cone.is_isotropic():
            cone_kw = {
                "cone": cone,
                "cone_axes_ang": cone_axes_h_to_nearest_qm(frame),
                "mm_symbols": frame.mm_symbols,
            }
        mf = _attach_gaussian_repulsion_amps(
            mf, rep_centers, amp_full, alpha_bohr2=float(alpha_bohr2), **cone_kw
        )
    return mf


def _attach_mf_energies(
    mf_emb,
    mf_cl,
    mf_gas,
    mf_scf,
    *,
    e_emb0: float,
    e_cl: float,
    e_gas: float,
    e_scf: float | None,
    conv_emb0: bool,
    conv_cl: bool,
    conv_scf: bool | None,
    cycles_emb0: int | None,
    cycles_cl: int | None,
    cycles_scf: int | None,
) -> None:
    mf_emb.e_tot = float(e_emb0)
    mf_emb.converged = bool(conv_emb0)
    if cycles_emb0 is not None:
        mf_emb.cycles = int(cycles_emb0)
    mf_cl.e_tot = float(e_cl)
    mf_cl.converged = bool(conv_cl)
    if cycles_cl is not None:
        mf_cl.cycles = int(cycles_cl)
    mf_gas.e_tot = float(e_gas)
    mf_gas.converged = True
    if mf_scf is not None and e_scf is not None:
        mf_scf.e_tot = float(e_scf)
        mf_scf.converged = bool(conv_scf)
        if cycles_scf is not None:
            mf_scf.cycles = int(cycles_scf)


def save_line_scf_cache(
    path: Path,
    *,
    frame: ScfFrame,
    dm_emb: np.ndarray,
    dm_cl_tot: np.ndarray,
    dm_cl_qm: np.ndarray,
    dm_pert: np.ndarray,
    dm_gas: np.ndarray,
    dm_scf: np.ndarray | None,
    mf_emb,
    mf_cl,
    mf_gas,
    mf_scf,
    de_kernel_kcal: float,
    de_lin_kcal: float,
    de_rho1_kcal: float,
    cache_meta: dict,
) -> None:
    meta = dict(cache_meta)
    meta["pipeline"] = "line_rho_pert_cluster_scf_cache"
    meta["geom_hash"] = geom_hash(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta_json": np.array(json.dumps(meta)),
        "e_emb0_hartree": np.float64(float(mf_emb.e_tot)),
        "e_cl_hartree": np.float64(float(mf_cl.e_tot)),
        "e_gas_hartree": np.float64(float(mf_gas.e_tot)),
        "de_kernel_kcal": np.float64(float(de_kernel_kcal)),
        "de_lin_kcal": np.float64(float(de_lin_kcal)),
        "de_rho1_kcal": np.float64(float(de_rho1_kcal)),
        "dm_emb": np.asarray(dm_emb, dtype=np.float64),
        "dm_cl_tot": np.asarray(dm_cl_tot, dtype=np.float64),
        "dm_cl_qm": np.asarray(dm_cl_qm, dtype=np.float64),
        "dm_pert": np.asarray(dm_pert, dtype=np.float64),
        "dm_gas": np.asarray(dm_gas, dtype=np.float64),
        "conv_emb0": np.int8(1 if mf_emb.converged else 0),
        "conv_cl": np.int8(1 if mf_cl.converged else 0),
        "cycles_emb0": np.int32(int(getattr(mf_emb, "cycles", -1))),
        "cycles_cl": np.int32(int(getattr(mf_cl, "cycles", -1))),
    }
    if dm_scf is not None and mf_scf is not None:
        payload["dm_scf"] = np.asarray(dm_scf, dtype=np.float64)
        payload["e_scf_hartree"] = np.float64(float(mf_scf.e_tot))
        payload["conv_scf"] = np.int8(1 if mf_scf.converged else 0)
        payload["cycles_scf"] = np.int32(int(getattr(mf_scf, "cycles", -1)))
    mo = getattr(mf_emb, "mo_coeff", None)
    occ = getattr(mf_emb, "mo_occ", None)
    if mo is not None and occ is not None:
        payload["mo_coeff_emb"] = np.asarray(mo, dtype=np.float64)
        payload["mo_occ_emb"] = np.asarray(occ, dtype=np.float64)
        mo_e = getattr(mf_emb, "mo_energy", None)
        if mo_e is not None:
            payload["mo_energy_emb"] = np.asarray(mo_e, dtype=np.float64)
    payload.update(_frame_to_npz(frame))
    np.savez_compressed(path, **payload)
    print(f"  [scf-cache] wrote → {path.resolve()}")


def _meta_values_equal(got, val) -> bool:
    if val is None and got is None:
        return True
    if val is None or got is None:
        return False
    if isinstance(val, bool):
        return bool(got) == bool(val)
    if isinstance(val, (int, float, np.floating, np.integer)):
        if not np.isfinite(float(val)) and not np.isfinite(float(got)):
            return True
        return abs(float(got) - float(val)) <= 1e-9
    return got == val


def _check_cache_meta(meta: dict, expect: dict, *, ignore: frozenset[str] = frozenset()) -> None:
    for key, val in expect.items():
        if key in ignore:
            continue
        if key not in meta:
            if val is None:
                continue
            raise ValueError(
                f"scf-cache missing meta[{key!r}] (旧缓存？当前需要 {val!r}；请 --force-scf 重算)"
            )
        got = meta[key]
        if not _meta_values_equal(got, val):
            raise ValueError(
                f"scf-cache meta {key}: cached {got!r} ≠ current {val!r}  "
                f"(改了几何/α/r_cut/幅度等 → --force-scf 或换 cache 文件名)"
            )


def load_line_scf_cache_reference(
    path: Path,
    *,
    cfg: ScfEmbedConfig,
    expect_meta: dict,
) -> LineScfCacheReference:
    """Load Emb0/Cluster/gas only; ``expect_meta`` must exclude ``scale_a_h/o``."""
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    _check_cache_meta(meta, expect_meta, ignore=CLUSTER_REFERENCE_RELAX_META_KEYS)

    frame = _frame_from_npz(z)
    if meta.get("geom_hash") != geom_hash(frame):
        raise ValueError("scf-cache geometry hash mismatch (Coo / r_cut_mm changed?)")

    dm_emb = np.asarray(z["dm_emb"], dtype=np.float64)
    dm_cl_tot = np.asarray(z["dm_cl_tot"], dtype=np.float64)
    dm_cl_qm = np.asarray(z["dm_cl_qm"], dtype=np.float64)
    dm_gas = np.asarray(z["dm_gas"], dtype=np.float64)

    mf_emb = build_emb0_mf_from_cache(frame, cfg, z)
    mf_cl = build_cluster_mf_stub(frame, cfg)
    mf_gas = build_gas_mf_stub(frame, cfg)
    _attach_mf_energies(
        mf_emb,
        mf_cl,
        mf_gas,
        None,
        e_emb0=float(z["e_emb0_hartree"]),
        e_cl=float(z["e_cl_hartree"]),
        e_gas=float(z["e_gas_hartree"]),
        e_scf=None,
        conv_emb0=bool(int(z["conv_emb0"])),
        conv_cl=bool(int(z["conv_cl"])),
        conv_scf=None,
        cycles_emb0=int(z["cycles_emb0"]) if int(z["cycles_emb0"]) >= 0 else None,
        cycles_cl=int(z["cycles_cl"]) if int(z["cycles_cl"]) >= 0 else None,
        cycles_scf=None,
    )

    print(
        f"  [scf-cache] reference ← {path.resolve()}  "
        f"(Emb0/Cluster/gas 复用；仅重算 V_rep 相关 pert+EmbR)"
    )
    return LineScfCacheReference(
        frame=frame,
        mf_emb=mf_emb,
        mf_cl=mf_cl,
        mf_gas=mf_gas,
        dm_emb=dm_emb,
        dm_cl_tot=dm_cl_tot,
        dm_cl_qm=dm_cl_qm,
        dm_gas=dm_gas,
        meta=meta,
    )


def load_line_scf_cache(
    path: Path,
    *,
    cfg: ScfEmbedConfig,
    cfg_scf: ScfEmbedConfig | None,
    amp_mm: np.ndarray,
    alpha_bohr2: float,
    rep_centers_ang: np.ndarray,
    embed_scf: bool,
    expect_meta: dict,
    cone: ConeRepAng | None = None,
) -> LineScfCacheBundle:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    _check_cache_meta(meta, expect_meta)

    frame = _frame_from_npz(z)
    if meta.get("geom_hash") != geom_hash(frame):
        raise ValueError("scf-cache geometry hash mismatch (Coo / r_cut_mm changed?)")

    dm_emb = np.asarray(z["dm_emb"], dtype=np.float64)
    dm_cl_tot = np.asarray(z["dm_cl_tot"], dtype=np.float64)
    dm_cl_qm = np.asarray(z["dm_cl_qm"], dtype=np.float64)
    dm_pert = np.asarray(z["dm_pert"], dtype=np.float64)
    dm_gas = np.asarray(z["dm_gas"], dtype=np.float64)
    dm_scf = None
    if "dm_scf" in z:
        dm_scf = np.asarray(z["dm_scf"], dtype=np.float64)

    if embed_scf and dm_scf is None:
        raise ValueError("scf-cache has no EmbR DM; rerun with --force-scf or omit --no-embed-scf")

    mf_emb = build_emb0_mf_stub(frame, cfg)
    mf_cl = build_cluster_mf_stub(frame, cfg)
    mf_gas = build_gas_mf_stub(frame, cfg)
    mf_scf = None
    e_scf = None
    conv_scf = None
    cycles_scf = None
    if embed_scf and dm_scf is not None:
        scf_cfg = cfg_scf or cfg
        mf_scf = build_emb_theta_mf_stub(
            frame,
            scf_cfg,
            amp_mm,
            alpha_bohr2=float(alpha_bohr2),
            rep_centers_ang=rep_centers_ang,
            cone=cone,
        )
        e_scf = float(z["e_scf_hartree"])
        conv_scf = bool(int(z["conv_scf"]))
        cycles_scf = int(z["cycles_scf"]) if int(z["cycles_scf"]) >= 0 else None

    _attach_mf_energies(
        mf_emb,
        mf_cl,
        mf_gas,
        mf_scf,
        e_emb0=float(z["e_emb0_hartree"]),
        e_cl=float(z["e_cl_hartree"]),
        e_gas=float(z["e_gas_hartree"]),
        e_scf=e_scf,
        conv_emb0=bool(int(z["conv_emb0"])),
        conv_cl=bool(int(z["conv_cl"])),
        conv_scf=conv_scf,
        cycles_emb0=int(z["cycles_emb0"]) if int(z["cycles_emb0"]) >= 0 else None,
        cycles_cl=int(z["cycles_cl"]) if int(z["cycles_cl"]) >= 0 else None,
        cycles_scf=cycles_scf,
    )

    de_lin = float(z["de_lin_kcal"])
    de_rho1 = float(z["de_rho1_kcal"])
    de_kernel = float(z["de_kernel_kcal"])
    pert_res = SimpleNamespace(
        de1_kcal=de_lin,
        de_rho1_kcal=de_rho1,
        de_kernel_kcal=de_kernel,
    )

    print(
        f"  [scf-cache] loaded ← {path.resolve()}  "
        f"(Emb0/Cluster/EmbR DFT **跳过**；可直接 --plane-npz / 改 plot 参数)"
    )
    return LineScfCacheBundle(
        frame=frame,
        mf_emb=mf_emb,
        mf_cl=mf_cl,
        mf_gas=mf_gas,
        mf_scf=mf_scf,
        dm_emb=dm_emb,
        dm_cl_tot=dm_cl_tot,
        dm_cl_qm=dm_cl_qm,
        dm_pert=dm_pert,
        dm_gas=dm_gas,
        dm_scf=dm_scf,
        pert_res=pert_res,
        de_kernel_kcal=de_kernel,
        de_lin_kcal=de_lin,
        de_rho1_kcal=de_rho1,
        meta=meta,
    )
