#!/usr/bin/env python3
"""
PySCF frozen-orbital dens (ALMO frozen / EDA Pauli-stage analogue).

Default construction (unchanged; analyze --pauli-source frz):

  1. SCF A = QM, SCF B = MM fragment (normally **all MM in Coo**)
  2. Embed dens into AB → P_A0, P_B0
  3. Löwdin-orthogonalize occupied MOs of A∪B → C_orth
  4. P_frz = C_orth n C_orth†
  hole = A0 + B0 − frz

Optional ``scf_mi=True`` (additive; does **not** replace FRZ):

  After fragment SCF, run a Gianinetti-style two-fragment SCF-MI:
  ALMOs stay absolutely localized (A MOs only on QM AOs, B on MM AOs),
  coefficients variationally relaxed in the full AB Fock field.
  Saves ``dm_almo`` for analyze ``--pauli-source almo``.
  Note: A0+B0−almo mixes frozen Pauli **and** ALMO polarization — not pure Pauli.

Cost: SCF-MI ≈ 10–30 extra Fock builds on AB (~2–5× the HL-only dens step).

Not bit-identical to Q-Chem ALMO-EDA2 (no FERF / EDA energy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from scf_embed_io import ScfFrame
from scf_embed_pyscf import (
    _aoslice_ao_range,
    _apply_mol_cfg,
    build_mm_mol,
    cp_mm_formal_charge,
    make_mean_field,
    set_pyscf_threads,
)
from pyscf import gto


@dataclass(frozen=True)
class HlPauliResult:
    mol_ab: Any
    mol_a: Any
    mol_b: Any
    dm_A0: np.ndarray  # (nao_ab, nao_ab) isolated A embedded
    dm_B0: np.ndarray  # (nao_ab, nao_ab) isolated B embedded
    dm_frz: np.ndarray  # (nao_ab, nao_ab) Löwdin-frozen total
    dm_A_tilde: np.ndarray  # A-assigned after Löwdin (full AB)
    dm_B_tilde: np.ndarray  # B-assigned after Löwdin
    dm_ClP: np.ndarray  # frz − B0  (= A0 − PauliHole in dens space)
    dm_vac: np.ndarray  # (nao_a, nao_a) vacuum QM (for ref)
    nocc_a: int
    nocc_b: int
    e_a: float
    e_b: float
    # optional SCF-MI (None / nan if not requested)
    dm_almo: np.ndarray | None = None
    e_almo: float = float("nan")
    scf_mi_converged: bool = False
    scf_mi_niter: int = 0


def _subframe_mm(frame: ScfFrame, mm_idx: np.ndarray) -> ScfFrame:
    ii = [int(j) for j in np.asarray(mm_idx, dtype=np.int64).reshape(-1)]
    return ScfFrame(
        qm_symbols=frame.qm_symbols,
        qm_coords_ang=np.asarray(frame.qm_coords_ang, dtype=np.float64),
        mm_symbols=tuple(frame.mm_symbols[j] for j in ii),
        mm_coords_ang=np.asarray(frame.mm_coords_ang[ii], dtype=np.float64),
        mm_charges=np.asarray(frame.mm_charges[ii], dtype=np.float64),
    )


def _build_qm_mol(frame: ScfFrame, cfg) -> gto.Mole:
    from scf_embed_pyscf import _build_mol

    return _build_mol(frame, cfg)


def _build_ab_mol(frame: ScfFrame, cfg) -> gto.Mole:
    """Real QM + real MM supermolecule."""
    qm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}"
        for s, (x, y, z) in zip(frame.qm_symbols, frame.qm_coords_ang)
    ]
    mm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}"
        for s, (x, y, z) in zip(frame.mm_symbols, frame.mm_coords_ang)
    ]
    mol = gto.Mole()
    mol.atom = "\n".join(qm_lines + mm_lines)
    q_qm = int(getattr(cfg, "qm_charge", 0) or 0)
    q = q_qm + int(cp_mm_formal_charge(frame))
    _apply_mol_cfg(mol, cfg, charge=q)
    mol.build()
    return mol


def _ao_indices_for_atoms(mol: gto.Mole, atom_ids: list[int] | range) -> np.ndarray:
    sl = mol.aoslice_by_atom()
    out: list[int] = []
    for ia in atom_ids:
        p0, p1 = _aoslice_ao_range(sl, int(ia))
        out.extend(range(p0, p1))
    return np.asarray(out, dtype=np.int64)


def _occupied_mo(mf) -> tuple[np.ndarray, np.ndarray]:
    mo = np.asarray(mf.mo_coeff, dtype=np.float64)
    occ = np.asarray(mf.mo_occ, dtype=np.float64)
    if mo.ndim == 3:
        raise RuntimeError("HL Pauli path expects closed-shell RHF/RKS (got UHF/UKS)")
    mask = occ > 1e-8
    return mo[:, mask], occ[mask]


def _lowdin_orthogonalize(C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """C_orth = C @ (C† S C)^{-1/2}"""
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Sot = C.T @ S @ C
    Sot = 0.5 * (Sot + Sot.T)
    evals, u = np.linalg.eigh(Sot)
    evals = np.clip(evals, 1e-12, None)
    X = (u * (1.0 / np.sqrt(evals))) @ u.T
    return C @ X


def _run_frag_scf(mol: gto.Mole, cfg, *, label: str):
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    e = mf.kernel()
    if not bool(getattr(mf, "converged", False)):
        mf.level_shift = 0.5
        e = mf.kernel()
    if not bool(getattr(mf, "converged", False)):
        raise RuntimeError(f"HL fragment SCF did not converge ({label})")
    return mf, float(e)


def _embed_dm(nao_ab: int, ao_idx: np.ndarray, dm_small: np.ndarray) -> np.ndarray:
    dm = np.zeros((nao_ab, nao_ab), dtype=np.float64)
    dm[np.ix_(ao_idx, ao_idx)] = np.asarray(dm_small, dtype=np.float64)
    return 0.5 * (dm + dm.T)


def _dm_from_nonorth_almos(
    C_a_full: np.ndarray,
    C_b_full: np.ndarray,
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Closed-shell dens from non-orthogonal ALMOs:
      dm = C @ diag(occ) @ (C† S C)^{-1} @ C†
    """
    C = np.hstack([C_a_full, C_b_full])
    occ = np.concatenate([np.asarray(occ_a), np.asarray(occ_b)]).astype(np.float64)
    Soo = C.T @ S @ C
    Soo = 0.5 * (Soo + Soo.T)
    try:
        Soo_inv = np.linalg.inv(Soo)
    except np.linalg.LinAlgError:
        Soo_inv = np.linalg.pinv(Soo, rcond=1e-10)
    dm = C @ (occ.reshape(-1, 1) * (Soo_inv @ C.T))
    return 0.5 * (dm + dm.T)


def _scf_mi_gianinetti(
    mol_ab,
    cfg,
    *,
    qm_ao: np.ndarray,
    mm_ao: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    max_iter: int = 50,
    conv_tol: float = 1e-7,
    damp: float = 0.3,
) -> tuple[np.ndarray, float, bool, int]:
    """
    Two-fragment SCF-MI (Gianinetti-style block diagonalization).

    ALMOs remain absolutely localized; coefficients relax under full AB Fock.
    Returns (dm_almo, E, converged, niter).
    """
    nao = int(mol_ab.nao)
    nocc_a = int(C_a.shape[1])
    nocc_b = int(C_b.shape[1])
    S = np.asarray(mol_ab.intor_symmetric("int1e_ovlp"), dtype=np.float64)

    C_a_full = np.zeros((nao, nocc_a), dtype=np.float64)
    C_b_full = np.zeros((nao, nocc_b), dtype=np.float64)
    C_a_full[qm_ao, :] = np.asarray(C_a, dtype=np.float64)
    C_b_full[mm_ao, :] = np.asarray(C_b, dtype=np.float64)

    mf = make_mean_field(mol_ab, cfg)
    mf.verbose = 0

    dm = _dm_from_nonorth_almos(C_a_full, C_b_full, occ_a, occ_b, S)
    e_old = float(mf.energy_tot(dm=dm))
    converged = False
    niter = 0

    for it in range(1, int(max_iter) + 1):
        niter = it
        F = np.asarray(mf.get_fock(dm=dm), dtype=np.float64)
        F = 0.5 * (F + F.T)

        Fa = F[np.ix_(qm_ao, qm_ao)]
        Sa = S[np.ix_(qm_ao, qm_ao)]
        Fb = F[np.ix_(mm_ao, mm_ao)]
        Sb = S[np.ix_(mm_ao, mm_ao)]
        _ea, ua = eigh(Fa, Sa)
        _eb, ub = eigh(Fb, Sb)
        C_a_new = ua[:, :nocc_a]
        C_b_new = ub[:, :nocc_b]

        C_a_full = np.zeros((nao, nocc_a), dtype=np.float64)
        C_b_full = np.zeros((nao, nocc_b), dtype=np.float64)
        C_a_full[qm_ao, :] = C_a_new
        C_b_full[mm_ao, :] = C_b_new

        dm_new = _dm_from_nonorth_almos(C_a_full, C_b_full, occ_a, occ_b, S)
        ddm = float(np.max(np.abs(dm_new - dm)))
        a = float(damp)
        dm = (1.0 - a) * dm_new + a * dm
        dm = 0.5 * (dm + dm.T)

        e = float(mf.energy_tot(dm=dm))
        de = abs(e - e_old)
        e_old = e
        if it == 1 or it % 5 == 0 or (de < float(conv_tol) and ddm < max(float(conv_tol) * 10, 1e-6)):
            print(
                f"  [scf-mi] iter={it:3d}  ΔE={de:.3e}  max|ΔP|={ddm:.3e}  E={e:.8f}",
                flush=True,
            )
        if de < float(conv_tol) and ddm < max(float(conv_tol) * 10, 1e-6):
            dm = dm_new
            e = float(mf.energy_tot(dm=dm))
            converged = True
            break

    if not converged:
        dm = _dm_from_nonorth_almos(C_a_full, C_b_full, occ_a, occ_b, S)
        e = float(mf.energy_tot(dm=dm))

    return dm, float(e), converged, niter


def compute_hl_pauli_qm_dm(
    frame: ScfFrame,
    cfg,
    *,
    threads: int = 4,
    scf_mi: bool = False,
    scf_mi_max_iter: int = 50,
    scf_mi_tol: float = 1e-7,
) -> HlPauliResult:
    """
    Heitler–London frozen dens on AB; optionally also SCF-MI dens.

    frame.mm_* should already be the MM subset used as fragment B.
    Default path (scf_mi=False) is bit-identical to the previous FRZ-only code.
    """
    if int(threads) > 0:
        set_pyscf_threads(int(threads))

    n_qm = len(frame.qm_symbols)
    if n_qm < 1:
        raise ValueError("empty QM")
    if len(frame.mm_symbols) < 1:
        raise ValueError("empty MM fragment B — widen --mm-cut or check Coo")

    print(
        f"  [hl] build mols  n_qm={n_qm}  n_mm={len(frame.mm_symbols)} ...",
        flush=True,
    )
    mol_a = _build_qm_mol(frame, cfg)
    mol_b = build_mm_mol(frame, cfg)
    mol_ab = _build_ab_mol(frame, cfg)

    print(f"  [hl] SCF A (QM, nao={mol_a.nao}) ...", flush=True)
    mf_a, e_a = _run_frag_scf(mol_a, cfg, label="QM/vacuum")
    print(f"  [hl] SCF A done  E={e_a:.8f}", flush=True)

    print(f"  [hl] SCF B (MM, nao={mol_b.nao}) ...", flush=True)
    mf_b, e_b = _run_frag_scf(mol_b, cfg, label="MM")
    print(f"  [hl] SCF B done  E={e_b:.8f}", flush=True)

    C_a, occ_a = _occupied_mo(mf_a)
    C_b, occ_b = _occupied_mo(mf_b)
    nocc_a = int(C_a.shape[1])
    nocc_b = int(C_b.shape[1])

    qm_ao = _ao_indices_for_atoms(mol_ab, range(n_qm))
    mm_ao = _ao_indices_for_atoms(mol_ab, range(n_qm, mol_ab.natm))
    if int(qm_ao.size) != int(mol_a.nao):
        raise RuntimeError(
            f"QM AO embed size mismatch: AB qm AOs={qm_ao.size} vs mol_a.nao={mol_a.nao}"
        )
    if int(mm_ao.size) != int(mol_b.nao):
        raise RuntimeError(
            f"MM AO embed size mismatch: AB mm AOs={mm_ao.size} vs mol_b.nao={mol_b.nao}"
        )

    dm_vac = np.asarray(mf_a.make_rdm1(), dtype=np.float64)
    if dm_vac.ndim == 3:
        dm_vac = dm_vac[0] + dm_vac[1]
    dm_b_small = np.asarray(mf_b.make_rdm1(), dtype=np.float64)
    if dm_b_small.ndim == 3:
        dm_b_small = dm_b_small[0] + dm_b_small[1]

    nao_ab = int(mol_ab.nao)
    dm_A0 = _embed_dm(nao_ab, qm_ao, dm_vac)
    dm_B0 = _embed_dm(nao_ab, mm_ao, dm_b_small)

    C_a_full = np.zeros((nao_ab, nocc_a), dtype=np.float64)
    C_b_full = np.zeros((nao_ab, nocc_b), dtype=np.float64)
    C_a_full[qm_ao, :] = C_a
    C_b_full[mm_ao, :] = C_b

    print(
        f"  [hl] Löwdin FRZ  nao_ab={nao_ab}  nocc_a={nocc_a}  nocc_b={nocc_b} ...",
        flush=True,
    )
    C_cat = np.hstack([C_a_full, C_b_full])
    occ_all = np.concatenate([occ_a, occ_b])
    S = mol_ab.intor_symmetric("int1e_ovlp")
    C_orth = _lowdin_orthogonalize(C_cat, S)

    dm_frz = (C_orth * occ_all.reshape(1, -1)) @ C_orth.T
    dm_frz = 0.5 * (dm_frz + dm_frz.T)

    C_a_orth = C_orth[:, :nocc_a]
    C_b_orth = C_orth[:, nocc_a:]
    dm_A_tilde = (C_a_orth * occ_a.reshape(1, -1)) @ C_a_orth.T
    dm_A_tilde = 0.5 * (dm_A_tilde + dm_A_tilde.T)
    dm_B_tilde = (C_b_orth * occ_b.reshape(1, -1)) @ C_b_orth.T
    dm_B_tilde = 0.5 * (dm_B_tilde + dm_B_tilde.T)

    # Cl_P := frz − B0  ⇒  A0 − Cl_P = A0 + B0 − frz  (Pauli hole)
    dm_ClP = dm_frz - dm_B0
    dm_ClP = 0.5 * (dm_ClP + dm_ClP.T)
    print("  [hl] FRZ dens ready", flush=True)

    dm_almo = None
    e_almo = float("nan")
    mi_ok = False
    mi_nit = 0
    if bool(scf_mi):
        print(
            f"  [scf-mi] start  max_iter={scf_mi_max_iter}  tol={scf_mi_tol:g} ...",
            flush=True,
        )
        dm_almo, e_almo, mi_ok, mi_nit = _scf_mi_gianinetti(
            mol_ab,
            cfg,
            qm_ao=qm_ao,
            mm_ao=mm_ao,
            C_a=C_a,
            C_b=C_b,
            occ_a=occ_a,
            occ_b=occ_b,
            max_iter=int(scf_mi_max_iter),
            conv_tol=float(scf_mi_tol),
        )
        print(
            f"  [scf-mi] converged={mi_ok}  niter={mi_nit}  "
            f"E_almo={e_almo:.8f}  E_A+E_B={e_a + e_b:.8f}",
            flush=True,
        )

    return HlPauliResult(
        mol_ab=mol_ab,
        mol_a=mol_a,
        mol_b=mol_b,
        dm_A0=dm_A0,
        dm_B0=dm_B0,
        dm_frz=dm_frz,
        dm_A_tilde=dm_A_tilde,
        dm_B_tilde=dm_B_tilde,
        dm_ClP=dm_ClP,
        dm_vac=dm_vac,
        nocc_a=nocc_a,
        nocc_b=nocc_b,
        e_a=e_a,
        e_b=e_b,
        dm_almo=dm_almo,
        e_almo=e_almo,
        scf_mi_converged=mi_ok,
        scf_mi_niter=mi_nit,
    )


def write_density_cube(
    mol,
    dm: np.ndarray,
    path: Path,
    *,
    n_grid: int = 80,
) -> Path:
    """Write Gaussian cube of ρ from dm on mol (PySCF cubegen)."""
    from pyscf.tools import cubegen

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cubegen.density(mol, str(path), np.asarray(dm, dtype=np.float64), nx=n_grid, ny=n_grid, nz=n_grid)
    return path


def build_ab_mf_stub(frame: ScfFrame, cfg):
    """Mean-field stub on AB mol (no SCF) for dens evaluation."""
    mol = _build_ab_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.verbose = 0
    return mf


def run_frame_to_cube(
    frame: ScfFrame,
    cfg,
    *,
    out_cube: Path,
    threads: int = 4,
    n_grid: int = 80,
    save_npz: Path | None = None,
    cube_from: str = "hole",
    scf_mi: bool = False,
    scf_mi_max_iter: int = 50,
    scf_mi_tol: float = 1e-7,
) -> HlPauliResult:
    """
    Compute HL dens; write cube; optionally save npz for analyze.

    cube_from:
      hole    — ρ_A0 + ρ_B0 − ρ_frz  (default; FRZ Pauli hole)
      ClP     — ρ_frz − ρ_B0
      Atilde  — A-assigned Löwdin dens
      almo_hole — ρ_A0 + ρ_B0 − ρ_almo (requires --scf-mi)
    """
    res = compute_hl_pauli_qm_dm(
        frame,
        cfg,
        threads=threads,
        scf_mi=scf_mi,
        scf_mi_max_iter=scf_mi_max_iter,
        scf_mi_tol=scf_mi_tol,
    )
    if cube_from == "Atilde":
        dm_cube = res.dm_A_tilde
    elif cube_from == "ClP":
        dm_cube = res.dm_ClP
    elif cube_from == "almo_hole":
        if res.dm_almo is None:
            raise RuntimeError("cube_from=almo_hole needs scf_mi=True")
        dm_cube = res.dm_A0 + res.dm_B0 - res.dm_almo
    else:
        dm_cube = res.dm_A0 + res.dm_B0 - res.dm_frz
    print(f"  [hl] write cube {out_cube.name} (n_grid={n_grid}) ...", flush=True)
    write_density_cube(res.mol_ab, dm_cube, out_cube, n_grid=n_grid)
    print("  [hl] cube done", flush=True)
    if save_npz is not None:
        save_npz = Path(save_npz)
        save_npz.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = dict(
            dm_A0=res.dm_A0,
            dm_B0=res.dm_B0,
            dm_frz=res.dm_frz,
            dm_A_tilde=res.dm_A_tilde,
            dm_B_tilde=res.dm_B_tilde,
            dm_ClP=res.dm_ClP,
            dm_vac=res.dm_vac,
            nocc_a=np.int32(res.nocc_a),
            nocc_b=np.int32(res.nocc_b),
            e_a=np.float64(res.e_a),
            e_b=np.float64(res.e_b),
            basis=np.array(str(cfg.basis)),
            xc=np.array(str(cfg.xc)),
            n_qm=np.int32(len(frame.qm_symbols)),
            n_mm=np.int32(len(frame.mm_symbols)),
            qm_symbols=np.asarray(frame.qm_symbols),
            qm_coords_ang=np.asarray(frame.qm_coords_ang, dtype=np.float64),
            mm_symbols=np.asarray(frame.mm_symbols),
            mm_coords_ang=np.asarray(frame.mm_coords_ang, dtype=np.float64),
            mm_charges=np.asarray(frame.mm_charges, dtype=np.float64),
            definition=np.array(
                "ClP=frz-B0; N_Pauli=A0+B0-frz=A0-ClP; ALMO-FRZ analogue; "
                "optional dm_almo from SCF-MI"
            ),
            scf_mi=np.bool_(bool(scf_mi)),
            scf_mi_converged=np.bool_(bool(res.scf_mi_converged)),
            scf_mi_niter=np.int32(res.scf_mi_niter),
            e_almo=np.float64(res.e_almo),
        )
        if res.dm_almo is not None:
            payload["dm_almo"] = res.dm_almo
        np.savez_compressed(save_npz, **payload)
    return res
