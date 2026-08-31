"""
Generic QM + retained-MM cluster supermolecule (full_QM reference; paper reproduction uses HF).

One SCF on the real QM region + real MM shell from the Coo frame gives ρ and the cluster
density matrix directly — no ghost atoms, no projection. Benchmark calculations in the
manuscript use HF/6-31G* or HF/6-31+G* via manifest ``calculate_parameter.scf``.
"""

from __future__ import annotations

import numpy as np

from pyscf import dft, gto

from scf_embed_io import ScfFrame
from scf_embed_pyscf import (
    ScfEmbedConfig,
    _apply_mol_cfg,
    cp_cluster_formal_charge,
    make_mean_field,
    project_dm_to_qm_aos,
    qm_ao_mask_for_cp_supermol,
    set_pyscf_threads,
)


def build_cluster_supermol_mol(frame: ScfFrame, cfg: ScfEmbedConfig) -> gto.Mole:
    """Real QM atoms + real MM shell (O/H/Na+/K+ …) at Coo positions."""
    qm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.qm_symbols, frame.qm_coords_ang)
    ]
    mm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.mm_symbols, frame.mm_coords_ang)
    ]
    mol = gto.Mole()
    mol.atom = "\n".join(qm_lines + mm_lines)
    q_qm = int(getattr(cfg, "qm_charge", 0) or 0)
    q_cl = cp_cluster_formal_charge(frame, qm_charge=q_qm)
    _apply_mol_cfg(mol, cfg, charge=q_cl)
    if q_cl != 0 or q_qm != 0:
        print(
            f"  [cluster] total charge={q_cl:+d}  "
            f"(QM qm_charge={q_qm:+d} + MM_formal={q_cl - q_qm:+d}; "
            f"closed-shell RKS; need even nelec)",
            flush=True,
        )
    mol.build()
    return mol


def run_cluster_supermol_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None):
    """Full SCF on QM + explicit shell waters; returns converged mean-field with total DM."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mol = build_cluster_supermol_mol(frame, cfg)
    mf = make_mean_field(mol, cfg)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("cluster supermolecule (full_QM) SCF did not converge")
    return mf


def cluster_qm_dm(mf: dft.rks.RKS, frame: ScfFrame) -> np.ndarray:
    """
    QM×QM block of cluster DM (water AO rows/cols zeroed).

    Same spillover observable class as Emb0 ρ: **only QM-region electrons**, but polarized
    in the presence of explicit waters. Use this — not total cluster DM — when comparing
    to ρ_emb on QM–MM axes near MM (r≈0–1 Å).
    """
    mask = qm_ao_mask_for_cp_supermol(mf, frame)
    return project_dm_to_qm_aos(mf.make_rdm1(), mask)


def compact_cluster_qm_dm(
    dm: np.ndarray,
    mf_cluster,
    frame: ScfFrame,
    *,
    n_emb_nao: int | None = None,
) -> np.ndarray:
    """
  Return cluster QM DM as (n_emb_nao, n_emb_nao) on the **same AO basis as Emb0** mf.

  Legacy ref npz may store ``dm_cluster_qm`` as full supermol (e.g. 278×278) with only
  the QM block nonzero; ``eval_density_from_dm`` on Emb0 mf needs the compact 98×98 slice.
    """
    dm = np.asarray(dm, dtype=np.float64)
    n_emb = int(n_emb_nao) if n_emb_nao is not None else int(np.sum(qm_ao_mask_for_cp_supermol(mf_cluster, frame)))
    if dm.shape == (n_emb, n_emb):
        return dm
    mask = qm_ao_mask_for_cp_supermol(mf_cluster, frame)
    if dm.shape[0] != mask.size:
        raise ValueError(f"cluster DM shape {dm.shape} != supermol nao {mask.size}")
    dm_proj = project_dm_to_qm_aos(dm, mask)
    idx = np.flatnonzero(mask)
    return dm_proj[np.ix_(idx, idx)]


def cluster_interaction_hartree(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    *,
    mf_cluster: dft.rks.RKS | None = None,
    mf_a: dft.rks.RKS | None = None,
    mf_b: dft.rks.RKS | None = None,
) -> tuple[float, dft.rks.RKS, dft.rks.RKS, dft.rks.RKS]:
    """
    Pure supermolecule interaction (no ghost, no BSSE):

    E_int^DFT = E(real QM + real MM shell) − E(QM monomer) − E(MM monomer)

    Same cluster reference as ``line_rho_pert_cluster`` / fix tables (ρ_cl^QM).
    """
    from scf_embed_pyscf import _mf_total_energy_hartree, run_gas_mf, run_mm_mf

    mf_cl = mf_cluster or run_cluster_supermol_mf(frame, cfg)
    mf_a = mf_a or run_gas_mf(frame, cfg)
    mf_b = mf_b or run_mm_mf(frame, cfg)
    e_cl = _mf_total_energy_hartree(mf_cl, cfg)
    e_a = _mf_total_energy_hartree(mf_a, cfg)
    e_b = _mf_total_energy_hartree(mf_b, cfg)
    return float(e_cl - e_a - e_b), mf_cl, mf_a, mf_b
