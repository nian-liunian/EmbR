"""
First-order density-overlap helpers for Emb0 and the site-centered EmbR repulsive envelope.

For a fixed Emb0 density, the per-site quantity used by the paper workflow is

    k_i = ∫ ρ_Emb0(r) C_i exp(-ζ_i |r-R_i|) dr,

with the exponential envelope supplied through ``MmhEnvelopeConfig``. Several class/function names
still contain ``Gauss`` for historical checkpoint/API compatibility; those names do not imply that
the published workflow uses a Gaussian envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scf_embed_io import ScfFrame
from scf_embed_pyscf import (
    ConeRepAng,
    GaussRepParams,
    ScfEmbedConfig,
    amp_mm_for_o_h_sites,
    cone_angular_weight_from_cos,
    cone_axes_h_to_nearest_qm,
    eval_density_at_points,
    eval_density_from_dm,
    gaussian_repulsion_ao,
    hartree_to_kcal,
    kcal_to_hartree,
    mf_lebedev_grid,
    mf_numint,
    mm_buffer_n_electrons,
    run_embedding_scf,
    set_pyscf_threads,
)


@dataclass(frozen=True)
class GaussRhoKernelsPerMm:
    """Per MM site j: k_j = ∫ ρ^(0) exp(-α|r-R_j|²) dr (multiply by A_j [Hartree] → ΔE in Hartree)."""

    kernel_mm: np.ndarray  # (n_mm,) float64
    mm_element: np.ndarray  # (n_mm,) int8 — 0=O, 1=H
    alpha_bohr2: float
    rho_o_mean: float


@dataclass(frozen=True)
class GaussRhoKernelSums:
    """Per-frame ∫ρ^(0) exp(-α|r-R_j|²) dr summed over MM O / H sites (Hartree·Bohr³ units cancel → Hartree when × amplitude)."""

    s_o: float
    s_h: float
    alpha_bohr2: float
    rho_o_mean: float
    n_mm_o: int
    n_mm_h: int


@dataclass(frozen=True)
class Emb0PertFrame:
    e_int_bg_kcal: float
    e_int_bg_hartree: float
    kernels: GaussRhoKernelSums
    kernels_per_mm: GaussRhoKernelsPerMm
    converged: bool


def run_emb0_mf(frame: ScfFrame, cfg: ScfEmbedConfig):
    """Emb0 SCF: configured QM method + MM point charges, no V_rep."""
    from scf_embed_pyscf import _build_mol, _run_rks

    mol = _build_mol(frame, cfg)
    mf = _run_rks(
        mol,
        cfg,
        mm_coords_ang=frame.mm_coords_ang,
        mm_charges=frame.mm_charges,
        rep=None,
        mm_symbols=frame.mm_symbols,
    )
    if not mf.converged:
        raise RuntimeError("Emb0 SCF did not converge")
    return mf


def _mf_cphf_fvind(mf):
    """Response function for CP-KS (PySCF version tolerant)."""
    gen = mf.gen_response
    try:
        return gen(switch_off=0)
    except TypeError:
        return gen()


def _cphf_fvind_mo_basis(mf, mo_coeff: np.ndarray, mo_occ: np.ndarray):
    """
    Wrap RKS/RHF ``gen_response`` (AO dm1) for ``cphf.kernel`` (MO x1 block).

    cphf passes x with shape (nvir, nocc); gen_response expects (nao, nao).
    """
    occidx = mo_occ > 0
    mocc = mo_coeff[:, occidx]
    mvir = mo_coeff[:, ~occidx]
    vind_dm = _mf_cphf_fvind(mf)

    nvir, nocc = mvir.shape[1], mocc.shape[1]

    def fvind(mo1: np.ndarray) -> np.ndarray:
        x = np.asarray(mo1, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(nvir, nocc)
        if x.ndim == 2:
            dm1 = mvir @ x @ mocc.T
            dm1 = dm1 + dm1.T
            v1 = vind_dm(dm1)
            return mvir.conj().T @ v1 @ mocc
        if x.ndim == 3:
            out = []
            for xi in x:
                dm1 = mvir @ xi @ mocc.T
                dm1 = dm1 + dm1.T
                v1 = vind_dm(dm1)
                out.append(mvir.conj().T @ v1 @ mocc)
            return np.stack(out, axis=0)
        raise ValueError(f"CPHF fvind: bad mo1 shape {x.shape}")

    return fvind


def compute_cphf_dm1(mf, h1_ao: np.ndarray) -> np.ndarray:
    """First-order density matrix δP from static perturbation ``h1_ao`` (AO basis)."""
    from pyscf.scf import cphf

    h1_ao = np.asarray(h1_ao, dtype=np.float64)
    mo_coeff = np.asarray(mf.mo_coeff, dtype=np.float64)
    mo_occ = np.asarray(mf.mo_occ, dtype=np.float64)
    mo_energy = np.asarray(mf.mo_energy, dtype=np.float64)
    occidx = mo_occ > 0
    nocc = int(np.sum(occidx))
    mocc = mo_coeff[:, occidx]
    mvir = mo_coeff[:, ~occidx]

    # AO (i,j) -> MO (a,i) vir-occ block; cphf.solve expects (nvir, nocc)
    h1_mo = mvir.conj().T @ h1_ao @ mocc

    fvind = _cphf_fvind_mo_basis(mf, mo_coeff, mo_occ)
    last_err: Exception | None = None
    for ls in (0.0, 0.2, 0.5):
        try:
            result = cphf.kernel(
                fvind,
                mo_energy,
                mo_occ,
                h1_mo,
                level_shift=float(ls),
            )
            last_err = None
            break
        except RuntimeError as exc:
            last_err = exc
            if "Krylov" not in str(exc):
                raise
    if last_err is not None:
        raise last_err
    z1 = result[0] if isinstance(result, tuple) else result
    z1 = np.asarray(z1, dtype=np.float64)
    if z1.ndim == 3:
        z1 = z1[0]

    if z1.shape != (mvir.shape[1], nocc):
        raise RuntimeError(f"CPHF mo1 shape {z1.shape} != ({mvir.shape[1]}, {nocc})")

    dm1 = mvir @ z1 @ mocc.T
    dm1 = dm1 + dm1.T
    return np.asarray(dm1, dtype=np.float64)


def delta_e_tr_ph0(mf, h1_ao: np.ndarray) -> float:
    """Tr(P0 h1) in Hartree — first-order energy shift."""
    dm0 = mf.make_rdm1()
    h1 = np.asarray(h1_ao, dtype=np.float64)
    return float(np.einsum("ij,ji->", dm0, h1).real)


def delta_e_tr_ph(mf, h1_ao: np.ndarray, dm: np.ndarray) -> float:
    """Tr(P h1) in Hartree for explicit density matrix P."""
    h1 = np.asarray(h1_ao, dtype=np.float64)
    dm = np.asarray(dm, dtype=np.float64)
    return float(np.einsum("ij,ji->", dm, h1).real)


def _sphere_unit_directions(n_sphere: int = 26) -> np.ndarray:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    dirs: list[np.ndarray] = []
    for k in range(int(n_sphere)):
        y = 1.0 - (2.0 * k + 1.0) / float(n_sphere)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        phi = golden * k
        dirs.append(np.array([np.cos(phi) * r, y, np.sin(phi) * r], dtype=np.float64))
    return np.asarray(dirs, dtype=np.float64)


def radial_rho_profile(
    mf,
    dm: np.ndarray,
    centers_ang: np.ndarray,
    *,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherically averaged ρ(r) around each center; return (r_grid [Å], profile).

    r = 0 uses the center only; r > 0 averages over ``n_sphere`` directions.
    Profile is the mean over all given centers.
    """
    centers = np.asarray(centers_ang, dtype=np.float64).reshape(-1, 3)
    if centers.size == 0:
        r_grid = np.arange(0.0, float(r_max) + 0.5 * float(dr), float(dr))
        return r_grid, np.zeros(r_grid.size, dtype=np.float64)

    dirs = _sphere_unit_directions(n_sphere)
    r_grid = np.arange(0.0, float(r_max) + 0.5 * float(dr), float(dr))
    prof = np.zeros(r_grid.size, dtype=np.float64)

    for ir, rad in enumerate(r_grid):
        pts: list[np.ndarray] = []
        for R in centers:
            if float(rad) < 1e-12:
                pts.append(R.copy())
            else:
                for u in dirs:
                    pts.append(R + float(rad) * u)
        points = np.asarray(pts, dtype=np.float64)
        rho = eval_density_from_dm(mf, dm, points)
        n_cent = int(centers.shape[0])
        if float(rad) < 1e-12:
            prof[ir] = float(np.mean(rho))
        else:
            prof[ir] = float(np.mean(rho.reshape(n_cent, n_sphere)))
    return r_grid, prof


@dataclass(frozen=True)
class QmMmLinePair:
    """One QM–MM contact within ``pair_max_ang`` (MM is line origin r=0)."""

    iq: int
    jm: int
    qm_symbol: str
    mm_symbol: str
    dist_ang: float
    r_mm_ang: np.ndarray  # (3,)
    r_qm_ang: np.ndarray  # (3,)


def enumerate_qm_mm_line_pairs(frame: ScfFrame, *, pair_max_ang: float = 2.0) -> list[QmMmLinePair]:
    """
    For each QM atom, list MM O/H within ``pair_max_ang`` (Å).

    Line axis for density: MM nucleus → QM atom (MM at r=0).
    """
    pairs: list[QmMmLinePair] = []
    qm_coords = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm_coords = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    rmax = float(pair_max_ang)
    for iq, sq in enumerate(frame.qm_symbols):
        rq = qm_coords[int(iq)]
        for jm, sm in enumerate(frame.mm_symbols):
            mel = sm.strip()[0].upper()
            if mel not in ("O", "H"):
                continue
            rm = mm_coords[int(jm)]
            d = float(np.linalg.norm(rq - rm))
            if d <= rmax + 1e-9:
                pairs.append(
                    QmMmLinePair(
                        iq=int(iq),
                        jm=int(jm),
                        qm_symbol=str(sq).strip(),
                        mm_symbol=str(sm).strip(),
                        dist_ang=d,
                        r_mm_ang=np.asarray(rm, dtype=np.float64),
                        r_qm_ang=np.asarray(rq, dtype=np.float64),
                    )
                )
    return pairs


def _norm_mm_elem_key(sym: str) -> str:
    from embr_envelope import _norm_elem_key

    return _norm_elem_key(sym)


def all_ion_qm_line_pairs(frame: ScfFrame, *, element: str) -> list[QmMmLinePair]:
    """
    For each MM ion site (Na/K/Cl), line axis to its nearest QM atom (no distance cutoff).

    Ions sit farther than solvent O/H; spillover diagnostics need the full segment.
    """
    el = _norm_mm_elem_key(str(element))
    pairs: list[QmMmLinePair] = []
    qm_coords = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm_coords = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    for jm, sm in enumerate(frame.mm_symbols):
        if _norm_mm_elem_key(str(sm)) != el:
            continue
        rm = mm_coords[int(jm)]
        dists = np.linalg.norm(qm_coords - rm.reshape(1, 3), axis=1)
        iq = int(np.argmin(dists))
        d = float(dists[iq])
        pairs.append(
            QmMmLinePair(
                iq=iq,
                jm=int(jm),
                qm_symbol=str(frame.qm_symbols[iq]).strip(),
                mm_symbol=str(sm).strip(),
                dist_ang=d,
                r_mm_ang=np.asarray(rm, dtype=np.float64),
                r_qm_ang=np.asarray(qm_coords[iq], dtype=np.float64),
            )
        )
    return pairs


def all_cl_qm_line_pairs(frame: ScfFrame) -> list[QmMmLinePair]:
    """Backward-compatible alias for Cl⁻ sites."""
    return all_ion_qm_line_pairs(frame, element="Cl")


def nearest_ion_qm_line_pair(
    frame: ScfFrame,
    *,
    element: str,
    mm_index: int | None = None,
) -> QmMmLinePair | None:
    """Globally nearest ion–QM pair, or nearest QM to a chosen MM ion index."""
    pairs = all_ion_qm_line_pairs(frame, element=str(element))
    if not pairs:
        return None
    if mm_index is not None:
        hits = [p for p in pairs if int(p.jm) == int(mm_index)]
        if hits:
            return min(hits, key=lambda p: float(p.dist_ang))
        return None
    return min(pairs, key=lambda p: float(p.dist_ang))


def nearest_cl_qm_line_pair(
    frame: ScfFrame,
    *,
    cl_mm_index: int | None = None,
) -> QmMmLinePair | None:
    """Globally nearest Cl–QM pair, or nearest QM to a chosen MM Cl index."""
    return nearest_ion_qm_line_pair(frame, element="Cl", mm_index=cl_mm_index)


def line_rho_profile_on_qm_mm_axis(
    mf,
    dm: np.ndarray,
    r_mm_ang: np.ndarray,
    r_qm_ang: np.ndarray,
    *,
    dr: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ρ(r) on the straight segment MM → QM; ``r=0`` at MM nucleus, ``r=d`` at QM.

    No spherical averaging — single axis only.
    """
    r_mm = np.asarray(r_mm_ang, dtype=np.float64).reshape(3)
    r_qm = np.asarray(r_qm_ang, dtype=np.float64).reshape(3)
    vec = r_qm - r_mm
    d = float(np.linalg.norm(vec))
    if d < 1e-10:
        raise ValueError("QM–MM distance ~0; cannot define axis")
    u = vec / d
    r_grid = np.arange(0.0, d + 0.5 * float(dr), float(dr))
    points = r_mm.reshape(1, 3) + r_grid.reshape(-1, 1) * u.reshape(1, 3)
    rho = eval_density_from_dm(mf, dm, points)
    return r_grid, np.asarray(rho, dtype=np.float64)


def _mean_line_profiles_at_r_grid(
    samples: list[tuple[np.ndarray, np.ndarray]],
    r_grid: np.ndarray,
) -> np.ndarray:
    """Average 1D line profiles; include pair at ``r`` only if ``r <=`` that pair's length."""
    r_grid = np.asarray(r_grid, dtype=np.float64)
    acc = np.zeros(r_grid.size, dtype=np.float64)
    cnt = np.zeros(r_grid.size, dtype=np.int64)
    for r_loc, rho in samples:
        r_loc = np.asarray(r_loc, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        d_end = float(r_loc[-1])
        for ir, rad in enumerate(r_grid):
            if float(rad) > d_end + 1e-9:
                continue
            acc[ir] += float(np.interp(float(rad), r_loc, rho))
            cnt[ir] += 1
    out = np.zeros_like(acc)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out


def compute_qm_mm_line_rho_profiles(
    frame: ScfFrame,
    mf_emb,
    mf_cp,
    dm_emb,
    dm_cp_qm,
    *,
    pair_max_ang: float = 2.0,
    dr: float = 0.1,
    r_line_max: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[QmMmLinePair]]:
    """
    Mean ρ_emb and ρ_QM^CP on QM→MM axes (MM at r=0), grouped by MM element.

    Returns (r_grid, emb_groups, cp_qm_groups, pairs).
    """
    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    r_grid = np.arange(0.0, float(r_line_max) + 0.5 * float(dr), float(dr))
    emb_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    cp_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    for p in pairs:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cp, dm_cp_qm, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        mel = p.mm_symbol.strip()[0].upper()
        emb_samples["all"].append((r_e, rho_e))
        cp_samples["all"].append((r_c, rho_c))
        key = "mm_O" if mel == "O" else "mm_H"
        emb_samples[key].append((r_e, rho_e))
        cp_samples[key].append((r_c, rho_c))

    emb_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in emb_samples.items()}
    cp_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in cp_samples.items()}
    return r_grid, emb_out, cp_out, pairs


def compute_qm_mm_line_rho_emb_cluster(
    frame: ScfFrame,
    mf_emb,
    mf_cl,
    dm_emb,
    dm_cl_qm,
    *,
    pair_max_ang: float = 2.0,
    dr: float = 0.1,
    r_line_max: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[QmMmLinePair]]:
    """
    Mean ρ_emb and ρ_Cluster^QM on QM→MM axes (MM at r=0), grouped by MM element.

    Same geometry as ``line_rho_pert_cluster`` / mix_mmh kernel k_j (axis ρ, not sphere).
    """
    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    r_grid = np.arange(0.0, float(r_line_max) + 0.5 * float(dr), float(dr))
    emb_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    cl_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    for p in pairs:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cl, dm_cl_qm, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        mel = p.mm_symbol.strip()[0].upper()
        key = "mm_O" if mel == "O" else "mm_H"
        emb_samples["all"].append((r_e, rho_e))
        cl_samples["all"].append((r_c, rho_c))
        emb_samples[key].append((r_e, rho_e))
        cl_samples[key].append((r_c, rho_c))

    emb_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in emb_samples.items()}
    cl_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in cl_samples.items()}
    return r_grid, emb_out, cl_out, pairs


def compute_qm_mm_line_rho_emb_cp_pert(
    frame: ScfFrame,
    mf_emb,
    mf_cp,
    dm_emb,
    dm_cp_qm,
    dm_pert,
    *,
    pair_max_ang: float = 2.0,
    dr: float = 0.1,
    r_line_max: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[QmMmLinePair]]:
    """
    Same QM→MM axis pairing as ``compute_qm_mm_line_rho_profiles``, plus ρ_pert
    (Emb0 + CP-KS δρ) evaluated on the Emb0 AO basis.

    Returns (r_grid, emb_groups, cp_qm_groups, pert_groups, pairs).
    """
    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    r_grid = np.arange(0.0, float(r_line_max) + 0.5 * float(dr), float(dr))
    emb_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    cp_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    pert_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "all": [],
        "mm_O": [],
        "mm_H": [],
    }
    for p in pairs:
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        r_c, rho_c = line_rho_profile_on_qm_mm_axis(
            mf_cp, dm_cp_qm, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        r_p, rho_p = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_pert, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        mel = p.mm_symbol.strip()[0].upper()
        emb_samples["all"].append((r_e, rho_e))
        cp_samples["all"].append((r_c, rho_c))
        pert_samples["all"].append((r_p, rho_p))
        key = "mm_O" if mel == "O" else "mm_H"
        emb_samples[key].append((r_e, rho_e))
        cp_samples[key].append((r_c, rho_c))
        pert_samples[key].append((r_p, rho_p))

    emb_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in emb_samples.items()}
    cp_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in cp_samples.items()}
    pert_out = {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in pert_samples.items()}
    return r_grid, emb_out, cp_out, pert_out, pairs


def _mm_centers(frame: ScfFrame, element: str) -> np.ndarray:
    el = element.strip()[0].upper()
    pts = [
        frame.mm_coords_ang[j]
        for j, s in enumerate(frame.mm_symbols)
        if s.strip()[0].upper() == el
    ]
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def mm_o_sites_qm_distance(frame: ScfFrame) -> tuple[np.ndarray, np.ndarray]:
    """MM water O coordinates and min distance to any QM atom (Å)."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    centers: list[np.ndarray] = []
    dists: list[float] = []
    for j, s in enumerate(frame.mm_symbols):
        if s.strip()[0].upper() != "O":
            continue
        r = np.asarray(frame.mm_coords_ang[j], dtype=np.float64)
        centers.append(r)
        dists.append(float(np.min(np.linalg.norm(qm - r.reshape(1, 3), axis=1))))
    if not centers:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.asarray(centers, dtype=np.float64), np.asarray(dists, dtype=np.float64)


def radial_rho_at_centers(
    mf,
    dm: np.ndarray,
    centers_ang: np.ndarray,
    *,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[np.ndarray, np.ndarray]:
    """ρ(r) spherically averaged on an explicit center list (subset of MM sites)."""
    return radial_rho_profile(
        mf, dm, centers_ang, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )


@dataclass(frozen=True)
class RadialRhoProfiles:
    r_grid_ang: np.ndarray
    rho0_o: np.ndarray
    rho1_o: np.ndarray
    drho_o: np.ndarray
    rho0_h: np.ndarray
    rho1_h: np.ndarray
    drho_h: np.ndarray


def compute_gas_radial_profiles(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf=None,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gas-phase QM ρ_gas(r) sampled at MM O/H centers (same RDF grid as Emb0).

    Measures QM electron tail at shell water sites with **no** MM embedding.
    """
    from scf_embed_pyscf import run_gas_mf

    if mf is None:
        mf = run_gas_mf(frame, cfg)
    dm = mf.make_rdm1()
    o_cent = _mm_centers(frame, "O")
    h_cent = _mm_centers(frame, "H")
    r_grid, rho_o = radial_rho_profile(
        mf, dm, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_h = radial_rho_profile(
        mf, dm, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    if o_cent.size == 0:
        rho_o = np.zeros_like(r_grid)
    if h_cent.size == 0:
        rho_h = np.zeros_like(r_grid)
    return mf, r_grid, np.asarray(rho_o, dtype=np.float64), np.asarray(rho_h, dtype=np.float64)


def compute_emb0_radial_profiles(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf=None,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[Any, RadialRhoProfiles]:
    """Emb0 SCF + spherically averaged ρ₀(r) at MM O/H (Coo coordinates)."""
    if mf is None:
        mf = run_emb0_mf(frame, cfg)
    dm0 = mf.make_rdm1()
    radial = compute_radial_rho_profiles(
        mf, dm0, dm0, frame, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    return mf, radial


def compute_cp_bsse_radial_profiles(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf_ab=None,
    mf_b_ga=None,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Boys–Bernardi-style **density** combination at MM O/H (linear subtraction).

    Returns (mf_ab, mf_b_ga, r, rho_ab_o, rho_b_o, rho_cp_lin_o, rho_ab_h, rho_b_h, rho_cp_lin_h) with::

      rho_cp_lin = rho_ab - rho_b_ga

    Fragment A with ghost B uses the same supermol SCF as AB (real QM + ghost MM);
    the standard ``rho_ab - rho_a_gb - rho_b_ga`` reduces to ``rho_ab - rho_b_ga``
    when ``rho_a_gb`` is taken from that same total density field.
    """
    from scf_embed_pyscf import run_cp_fragment_b_mf, run_cp_supermol_mf

    if mf_ab is None:
        mf_ab = run_cp_supermol_mf(frame, cfg)
    if mf_b_ga is None:
        mf_b_ga = run_cp_fragment_b_mf(frame, cfg)

    o_cent = _mm_centers(frame, "O")
    h_cent = _mm_centers(frame, "H")
    dm_ab = mf_ab.make_rdm1()
    dm_b = mf_b_ga.make_rdm1()
    r_grid, rho_ab_o = radial_rho_profile(
        mf_ab, dm_ab, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_b_o = radial_rho_profile(
        mf_b_ga, dm_b, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_ab_h = radial_rho_profile(
        mf_ab, dm_ab, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_b_h = radial_rho_profile(
        mf_b_ga, dm_b, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    if o_cent.size == 0:
        z = np.zeros_like(r_grid)
        rho_ab_o, rho_b_o = z, z
    if h_cent.size == 0:
        z = np.zeros_like(r_grid)
        rho_ab_h, rho_b_h = z, z
    rho_cp_lin_o = np.asarray(rho_ab_o - rho_b_o, dtype=np.float64)
    rho_cp_lin_h = np.asarray(rho_ab_h - rho_b_h, dtype=np.float64)
    return (
        mf_ab,
        mf_b_ga,
        r_grid,
        np.asarray(rho_ab_o, dtype=np.float64),
        np.asarray(rho_b_o, dtype=np.float64),
        rho_cp_lin_o,
        np.asarray(rho_ab_h, dtype=np.float64),
        np.asarray(rho_b_h, dtype=np.float64),
        rho_cp_lin_h,
    )


def compute_cp_qm_only_radial_profiles(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf=None,
    n_qm_atoms: int | None = None,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    CP supermol density from **QM atoms only** (no ghost MM AO contribution).

    Compare this to Emb0 ρ_emb for electron-penetration / spillover diagnostics.
    """
    from scf_embed_pyscf import cp_supermol_qm_dm, run_cp_supermol_mf

    if mf is None:
        mf = run_cp_supermol_mf(frame, cfg)
    _ = n_qm_atoms  # unused; mask uses frame.qm_coords
    dm_qm = cp_supermol_qm_dm(mf, frame)
    o_cent = _mm_centers(frame, "O")
    h_cent = _mm_centers(frame, "H")
    r_grid, rho_o = radial_rho_profile(
        mf, dm_qm, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_h = radial_rho_profile(
        mf, dm_qm, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    if o_cent.size == 0:
        rho_o = np.zeros_like(r_grid)
    if h_cent.size == 0:
        rho_h = np.zeros_like(r_grid)
    return mf, r_grid, np.asarray(rho_o, dtype=np.float64), np.asarray(rho_h, dtype=np.float64)


def compute_cp_supermol_radial_profiles(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf=None,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    CP supermol total density ρ_cp(r) at MM O/H (same RDF convention as Emb0 ρ₀).

    Uses ``run_cp_supermol_mf``: QM + ghost basis on shell waters, no MM charges.
    """
    from scf_embed_pyscf import run_cp_supermol_mf

    if mf is None:
        mf = run_cp_supermol_mf(frame, cfg)
    dm = mf.make_rdm1()
    o_cent = _mm_centers(frame, "O")
    h_cent = _mm_centers(frame, "H")
    r_grid, rho_o = radial_rho_profile(
        mf, dm, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_h = radial_rho_profile(
        mf, dm, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    if o_cent.size == 0:
        rho_o = np.zeros_like(r_grid)
    if h_cent.size == 0:
        rho_h = np.zeros_like(r_grid)
    return mf, r_grid, np.asarray(rho_o, dtype=np.float64), np.asarray(rho_h, dtype=np.float64)


def compute_radial_rho_profiles(
    mf,
    dm0: np.ndarray,
    dm1_tot: np.ndarray,
    frame: ScfFrame,
    *,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> RadialRhoProfiles:
    o_cent = _mm_centers(frame, "O")
    h_cent = _mm_centers(frame, "H")
    _, r0o = radial_rho_profile(mf, dm0, o_cent, r_max=r_max, dr=dr, n_sphere=n_sphere)
    _, r1o = radial_rho_profile(mf, dm1_tot, o_cent, r_max=r_max, dr=dr, n_sphere=n_sphere)
    r_grid, r0h = radial_rho_profile(mf, dm0, h_cent, r_max=r_max, dr=dr, n_sphere=n_sphere)
    _, r1h = radial_rho_profile(mf, dm1_tot, h_cent, r_max=r_max, dr=dr, n_sphere=n_sphere)
    if o_cent.size == 0:
        z = np.zeros_like(r_grid)
        r0o, r1o = z, z
    if h_cent.size == 0:
        z = np.zeros_like(r_grid)
        r0h, r1h = z, z
    return RadialRhoProfiles(
        r_grid_ang=r_grid,
        rho0_o=np.asarray(r0o, dtype=np.float64),
        rho1_o=np.asarray(r1o, dtype=np.float64),
        drho_o=np.asarray(r1o - r0o, dtype=np.float64),
        rho0_h=np.asarray(r0h, dtype=np.float64),
        rho1_h=np.asarray(r1h, dtype=np.float64),
        drho_h=np.asarray(r1h - r0h, dtype=np.float64),
    )


@dataclass(frozen=True)
class DensityPertResult:
    rho0_mm: np.ndarray
    drho_mm: np.ndarray
    rho1_mm: np.ndarray
    rho0_o_mean: float
    rho1_o_mean: float
    drho_o_mean: float
    rel_drho_o: float
    rho0_h_mean: float
    rho1_h_mean: float
    drho_h_mean: float
    rel_drho_h: float
    de1_kcal: float
    de_rho1_kcal: float
    de_kernel_kcal: float
    n_buffer0: float
    n_buffer1: float
    max_abs_amp_hartree: float
    radial: RadialRhoProfiles | None = None
    dm1_tot: np.ndarray | None = None


def run_density_pert_frame(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    amp_mm: np.ndarray,
    *,
    alpha_bohr2: float = 0.6,
    envelope_cfg=None,
    mf=None,
    kernel_mm: np.ndarray | None = None,
    cone: ConeRepAng | None = None,
    rdf_max: float = 3.0,
    rdf_dr: float = 0.1,
    rdf_n_sphere: int = 26,
) -> tuple[DensityPertResult, Any]:
    """
    Emb0 + CP-KS linear response with per-site ``amp_mm`` Gaussian repulsion.

    Returns (summary, mf) where mf is the converged Emb0 calculation object.
    """
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))

    if mf is None:
        mf = run_emb0_mf(frame, cfg)

    amps = amp_mm_for_o_h_sites(frame, amp_mm)
    cone_kw: dict = {}
    if cone is not None and not cone.is_isotropic():
        cone_kw = {
            "cone": cone,
            "cone_axes_ang": cone_axes_h_to_nearest_qm(frame),
            "mm_symbols": frame.mm_symbols,
        }
    h1 = gaussian_repulsion_ao(
        mf,
        frame.mm_coords_ang,
        amps,
        alpha_bohr2=float(alpha_bohr2),
        envelope_cfg=envelope_cfg,
        mm_symbols=frame.mm_symbols,
        **cone_kw,
    )
    dm0 = mf.make_rdm1()
    dm1 = compute_cphf_dm1(mf, h1)
    dm1_tot = dm0 + dm1

    pts = frame.mm_coords_ang
    rho0 = eval_density_at_points(mf, pts)
    rho1 = eval_density_from_dm(mf, dm1_tot, pts)
    drho = rho1 - rho0

    o_idx = [j for j, s in enumerate(frame.mm_symbols) if s.strip()[0].upper() == "O"]
    h_idx = [j for j, s in enumerate(frame.mm_symbols) if s.strip()[0].upper() == "H"]

    def _mean(arr: np.ndarray, idx: list[int]) -> float:
        return float(np.mean(arr[idx])) if idx else float("nan")

    rho0_o = _mean(rho0, o_idx)
    rho1_o = _mean(rho1, o_idx)
    drho_o = _mean(drho, o_idx)
    rel = float(drho_o / rho0_o) if rho0_o and abs(rho0_o) > 1e-15 else float("nan")

    rho0_h = _mean(rho0, h_idx)
    rho1_h = _mean(rho1, h_idx)
    drho_h = _mean(drho, h_idx)
    rel_h = float(drho_h / rho0_h) if rho0_h and abs(rho0_h) > 1e-15 else float("nan")

    n0 = mm_buffer_n_electrons(mf, frame)
    n1 = _buffer_n_electrons_from_dm(mf, frame, dm1_tot)

    de1_kcal = hartree_to_kcal(delta_e_tr_ph0(mf, h1))
    de_rho1_kcal = hartree_to_kcal(delta_e_tr_ph(mf, h1, dm1_tot))
    if kernel_mm is not None:
        de_kernel_kcal = delta_e_peratom_kcal(kernel_mm, amp_mm)
    else:
        de_kernel_kcal = float("nan")

    radial = compute_radial_rho_profiles(
        mf, dm0, dm1_tot, frame, r_max=float(rdf_max), dr=float(rdf_dr), n_sphere=int(rdf_n_sphere)
    )

    res = DensityPertResult(
        rho0_mm=rho0,
        drho_mm=drho,
        rho1_mm=rho1,
        rho0_o_mean=rho0_o,
        rho1_o_mean=rho1_o,
        drho_o_mean=drho_o,
        rel_drho_o=rel,
        rho0_h_mean=rho0_h,
        rho1_h_mean=rho1_h,
        drho_h_mean=drho_h,
        rel_drho_h=rel_h,
        de1_kcal=float(de1_kcal),
        de_rho1_kcal=float(de_rho1_kcal),
        de_kernel_kcal=float(de_kernel_kcal),
        n_buffer0=float(n0),
        n_buffer1=float(n1),
        max_abs_amp_hartree=float(np.max(np.abs(amps))) if amps.size else 0.0,
        radial=radial,
        dm1_tot=np.asarray(dm1_tot, dtype=np.float64),
    )
    return res, mf


def _buffer_n_electrons_from_dm(mf, frame: ScfFrame, dm: np.ndarray) -> float:
    """Same shell proxy as ``mm_buffer_n_electrons`` but with explicit ``dm``."""
    qm = frame.qm_coords_ang
    mm = frame.mm_coords_ang
    inner_ang, outer_ang, n_radial, n_sphere = 1.2, 5.0, 8, 26
    golden = np.pi * (3.0 - np.sqrt(5.0))
    dirs: list[np.ndarray] = []
    for k in range(int(n_sphere)):
        y = 1.0 - (2.0 * k + 1.0) / float(n_sphere)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        phi = golden * k
        dirs.append(np.array([np.cos(phi) * r, y, np.sin(phi) * r], dtype=np.float64))
    dirs_arr = np.asarray(dirs)
    radii = np.linspace(float(inner_ang), float(outer_ang), int(n_radial))

    pts: list[np.ndarray] = []
    for R in mm:
        if float(np.min(np.linalg.norm(qm - R, axis=1))) > float(outer_ang):
            continue
        for rad in radii:
            if rad < float(inner_ang):
                continue
            for u in dirs_arr:
                p = R + rad * u
                if float(np.min(np.linalg.norm(qm - p, axis=1))) >= float(inner_ang):
                    pts.append(p)
    if not pts:
        return 0.0
    points = np.asarray(pts, dtype=np.float64)
    rho = eval_density_from_dm(mf, dm, points)
    n_pts = max(1, points.shape[0])
    shell_vol = 4.0 / 3.0 * np.pi * (float(outer_ang) ** 3 - float(inner_ang) ** 3)
    return float(np.sum(rho) / n_pts * shell_vol * 0.01)


def rho_kernels_per_mm(
    mf,
    frame: ScfFrame,
    *,
    envelope_cfg=None,
    alpha_bohr2: float | None = None,
    alpha_by_element: dict[str, float] | None = None,
    cone: ConeRepAng | None = None,
    dm: np.ndarray | None = None,
) -> GaussRhoKernelsPerMm:
    """
    ∫ρ · C_j · envelope_j(|r-R_j|) dr for each MM O/H/Na/K/Cl site.

    ``envelope_cfg``: :class:`MmhEnvelopeConfig` (gauss or exp + lnC).
    Legacy: pass ``alpha_bohr2`` / ``alpha_by_element`` only (C=1, gauss).
    """
    from pyscf import lib

    from embr_envelope import (
        MmhEnvelopeConfig,
        is_kernel_mm_symbol,
        mm_kernel_element_code,
    )

    if envelope_cfg is None:
        if alpha_by_element is not None:
            envelope_cfg = MmhEnvelopeConfig.from_width_mapping(alpha_by_element)
        else:
            envelope_cfg = MmhEnvelopeConfig.uniform(float(3.0 if alpha_bohr2 is None else alpha_bohr2))

    mol = mf.mol
    grids = mf_lebedev_grid(mf)
    coords = grids.coords
    weights = grids.weights
    if dm is None:
        dm = mf.make_rdm1()
    else:
        dm = np.asarray(dm, dtype=np.float64)
    ao = mf_numint(mf).eval_ao(mol, coords, deriv=0)
    rho = np.einsum("ij,ki,kj->k", dm, ao, ao)

    centers_bohr = np.asarray(frame.mm_coords_ang, dtype=np.float64) / lib.param.BOHR
    use_cone = cone is not None and not cone.is_isotropic()
    cone_axes = cone_axes_h_to_nearest_qm(frame) if use_cone else None
    kernels: list[float] = []
    elements: list[int] = []
    for j, (R, sym) in enumerate(zip(centers_bohr, frame.mm_symbols)):
        if not is_kernel_mm_symbol(sym):
            continue
        el_key = sym.strip()[0].upper()
        dr = coords - R
        r2 = np.sum(dr * dr, axis=1)
        r = np.sqrt(np.maximum(r2, 0.0))
        kernel = envelope_cfg.envelope_on_grid(sym, r_bohr=r, r2_bohr=r2)
        if use_cone and el_key == "H" and cone_axes is not None:
            axis = np.asarray(cone_axes[j], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(axis)) > 0.5:
                dn = np.linalg.norm(dr, axis=1)
                dn = np.maximum(dn, 1e-30)
                cos_th = np.sum(dr * axis.reshape(1, 3), axis=1) / dn
                kernel = kernel * cone_angular_weight_from_cos(
                    cos_th,
                    theta1_rad=cone.theta1_rad,
                    theta2_rad=cone.theta2_rad,
                )
        integral = float(np.sum(weights * rho * kernel))
        kernels.append(integral)
        elements.append(int(mm_kernel_element_code(sym)))

    rho_mm = eval_density_from_dm(mf, dm, frame.mm_coords_ang)
    o_idx = [j for j, s in enumerate(frame.mm_symbols) if s.strip()[0].upper() == "O"]
    rho_o_mean = float(np.mean(rho_mm[o_idx])) if o_idx else float("nan")

    return GaussRhoKernelsPerMm(
        kernel_mm=np.asarray(kernels, dtype=np.float64),
        mm_element=np.asarray(elements, dtype=np.int8),
        alpha_bohr2=float(envelope_cfg.legacy_fix_alpha()),
        rho_o_mean=rho_o_mean,
    )


def gauss_rho_kernels_per_mm(
    mf,
    frame: ScfFrame,
    *,
    alpha_bohr2: float,
    alpha_by_element: dict[str, float] | None = None,
    envelope_cfg=None,
    cone: ConeRepAng | None = None,
    dm: np.ndarray | None = None,
) -> GaussRhoKernelsPerMm:
    """Legacy wrapper → :func:`rho_kernels_per_mm` (Gaussian, C=1 unless envelope_cfg set)."""
    return rho_kernels_per_mm(
        mf,
        frame,
        envelope_cfg=envelope_cfg,
        alpha_bohr2=float(alpha_bohr2),
        alpha_by_element=alpha_by_element,
        cone=cone,
        dm=dm,
    )


def gauss_rho_kernel_sums_from_per_mm(per: GaussRhoKernelsPerMm) -> GaussRhoKernelSums:
    el = per.mm_element
    k = per.kernel_mm
    s_o = float(np.sum(k[el == 0])) if k.size else 0.0
    s_h = float(np.sum(k[el == 1])) if k.size else 0.0
    return GaussRhoKernelSums(
        s_o=s_o,
        s_h=s_h,
        alpha_bohr2=float(per.alpha_bohr2),
        rho_o_mean=float(per.rho_o_mean),
        n_mm_o=int(np.sum(el == 0)),
        n_mm_h=int(np.sum(el == 1)),
    )


def gauss_rho_kernel_sums(
    mf,
    frame: ScfFrame,
    *,
    alpha_bohr2: float,
    cone: ConeRepAng | None = None,
) -> GaussRhoKernelSums:
    """Sum ∫ρ exp(-α|r-R_j|²) W_cone dr over MM O and H centers (grid quadrature)."""
    per = gauss_rho_kernels_per_mm(mf, frame, alpha_bohr2=float(alpha_bohr2), cone=cone)
    return gauss_rho_kernel_sums_from_per_mm(per)


def delta_e_peratom_hartree(kernel_mm: np.ndarray, amp_mm: np.ndarray) -> float:
    k = np.asarray(kernel_mm, dtype=np.float64).reshape(-1)
    a = np.asarray(amp_mm, dtype=np.float64).reshape(-1)
    if k.shape != a.shape:
        raise ValueError(f"kernel_mm {k.shape} != amp_mm {a.shape}")
    return float(np.dot(k, a))


def delta_e_peratom_kcal(kernel_mm: np.ndarray, amp_mm: np.ndarray) -> float:
    return hartree_to_kcal(delta_e_peratom_hartree(kernel_mm, amp_mm))


def delta_e_first_order_hartree(
    amp_o_hartree: float,
    amp_h_hartree: float,
    kernels: GaussRhoKernelSums,
) -> float:
    return float(amp_o_hartree) * float(kernels.s_o) + float(amp_h_hartree) * float(kernels.s_h)


def delta_e_first_order_kcal(
    amp_o_hartree: float,
    amp_h_hartree: float,
    kernels: GaussRhoKernelSums,
) -> float:
    return hartree_to_kcal(delta_e_first_order_hartree(amp_o_hartree, amp_h_hartree, kernels))


def precompute_emb0_pert_frame(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    alpha_bohr2: float = 0.6,
    alpha_by_element: dict[str, float] | None = None,
    envelope_cfg=None,
    e_int_bg_kcal: float | None = None,
    cone: ConeRepAng | None = None,
) -> Emb0PertFrame:
    """
    One Emb0 SCF + Gaussian ρ-kernels for first-order V_rep energy.

    If ``e_int_bg_kcal`` is supplied (E*.txt col2), skip gas-phase reference
    and use the provided label; otherwise run full ``run_embedding_scf`` once.
    """
    rep0 = GaussRepParams(alpha_bohr2=float(alpha_bohr2), amp_o_hartree=0.0, amp_h_hartree=0.0)
    if e_int_bg_kcal is None:
        res = run_embedding_scf(frame, cfg, rep=rep0)
        mf = res.mf
        e_int_bg_kcal = hartree_to_kcal(res.e_int_hartree)
        e_int_hartree = float(res.e_int_hartree)
    else:
        mf = run_emb0_mf(frame, cfg)
        e_int_hartree = kcal_to_hartree(float(e_int_bg_kcal))

    alpha_kw: dict = {}
    if envelope_cfg is not None:
        alpha_kw["envelope_cfg"] = envelope_cfg
    else:
        alpha_kw["alpha_bohr2"] = float(alpha_bohr2)
        if alpha_by_element is not None:
            alpha_kw["alpha_by_element"] = alpha_by_element
    per_mm = rho_kernels_per_mm(mf, frame, cone=cone, **alpha_kw)
    kernels = gauss_rho_kernel_sums_from_per_mm(per_mm)
    return Emb0PertFrame(
        e_int_bg_kcal=float(e_int_bg_kcal),
        e_int_bg_hartree=float(e_int_hartree),
        kernels=kernels,
        kernels_per_mm=per_mm,
        converged=True,
    )
