"""
EmbR SOAP features centered on MM sites; the active workflow can use the full_QM+MM environment.

Uses third-party packages: ``dscribe`` (SOAP), ``ase`` (structures), ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from embr_io import M_GLY, SOAP_SPECIES


def _require_dscribe():
    try:
        from dscribe.descriptors import SOAP  # noqa: F401
        from ase import Atoms  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SOAP pipeline needs DScribe + ASE. Install with:\n"
            "  pip install dscribe ase\n"
            "See requirements-soap.txt"
        ) from e


@dataclass(frozen=True)
class SoapE0Hyper:
    r_cut: float = 3.75
    n_max: int = 8
    l_max: int = 6
    sigma: float = 0.5
    species: tuple[str, ...] = SOAP_SPECIES
    n_qm: int = M_GLY

    def soap_kwargs(self) -> dict:
        return {
            "species": list(self.species),
            "periodic": False,
            "r_cut": float(self.r_cut),
            "n_max": int(self.n_max),
            "l_max": int(self.l_max),
            "sigma": float(self.sigma),
            "average": "off",
            "sparse": False,
        }


def build_soap_calculator(h: SoapE0Hyper | None = None):
    _require_dscribe()
    from dscribe.descriptors import SOAP

    hyper = h if h is not None else SoapE0Hyper()
    return SOAP(**hyper.soap_kwargs()), hyper


def soap_feature_dim(h: SoapE0Hyper | None = None) -> int:
    soap, _ = build_soap_calculator(h)
    return int(soap.get_number_of_features())


def sanitize_atoms_for_dscribe(atoms) -> "Atoms":
    """
    Minimal ASE Atoms for DScribe on ASE 3.13+.

    DScribe's ``System.from_atoms`` forwards both momenta and velocities; ASE 3.13
    raises if both are set. Strip those arrays (and calculator) before MBTR/SOAP.
    """
    from ase import Atoms

    nums = atoms.get_atomic_numbers()
    pos = np.ascontiguousarray(atoms.get_positions(), dtype=np.float64)
    clean = Atoms(numbers=nums, positions=pos)
    clean.set_pbc((False, False, False))
    for key in ("momenta", "velocities", "forces", "masses"):
        if key in clean.arrays:
            del clean.arrays[key]
    clean.calc = None
    clean.info = {}
    return clean


def positions_symbols_to_ase(positions: np.ndarray, symbols: Sequence[str]):
    """
    Build a non-periodic ASE Atoms for DScribe (SOAP / ACSF / MBTR).

    Uses atomic numbers (not ``symbols=``) to avoid ASE 3.13+ issues.
    """
    _require_dscribe()
    from ase import Atoms
    from ase.data import atomic_numbers

    pos = np.ascontiguousarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {pos.shape}")
    syms = [str(s).strip() for s in symbols]
    if len(syms) != int(pos.shape[0]):
        raise ValueError("len(symbols) must match number of atoms")

    numbers: list[int] = []
    for s in syms:
        key = s[0].upper() + (s[1:].lower() if len(s) > 1 else "")
        if key not in atomic_numbers:
            raise ValueError(f"unknown element {s!r} for ASE")
        numbers.append(int(atomic_numbers[key]))

    atoms = Atoms(numbers=numbers, positions=pos)
    return sanitize_atoms_for_dscribe(atoms)


def compute_soap_qm_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
) -> np.ndarray:
    """
    SOAP at each of the first ``n_qm`` atoms (glycine QM order in Coo).

    Returns
    -------
    (n_qm, d_soap) float64
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper()

    n_qm = int(hyper.n_qm)
    if int(positions.shape[0]) < n_qm:
        raise ValueError(f"need at least {n_qm} atoms for QM centers, got {positions.shape[0]}")

    atoms = positions_symbols_to_ase(positions, symbols)
    centers = list(range(n_qm))
    desc = soap.create(atoms, centers=centers)
    out = np.asarray(desc, dtype=np.float64)
    if out.shape != (n_qm, soap.get_number_of_features()):
        raise RuntimeError(f"unexpected SOAP shape {out.shape}")
    return out


def hyper_from_dict(d: dict) -> SoapE0Hyper:
    sp = d.get("species", SOAP_SPECIES)
    return SoapE0Hyper(
        r_cut=float(d.get("r_cut", 3.75)),
        n_max=int(d.get("n_max", 8)),
        l_max=int(d.get("l_max", 6)),
        sigma=float(d.get("sigma", 0.5)),
        species=tuple(sp),
        n_qm=int(d.get("n_qm", M_GLY)),
    )


# --- MM-site SOAP ---

from embr_envelope import (
    MM_KERNEL_EL_C,
    MM_KERNEL_EL_CL,
    MM_KERNEL_EL_H,
    MM_KERNEL_EL_K,
    MM_KERNEL_EL_N,
    MM_KERNEL_EL_NA,
    MM_KERNEL_EL_O,
    _norm_elem_key,
    is_active_repulsion_symbol,
    is_kernel_mm_symbol,
    mm_kernel_element_code,
)

MM_EL_O = 0
MM_EL_H = 1
FEAT_MODE_MM_QM_ENV_SOAP = "mm_center_qm_env_soap"
FEAT_MODE_MM_FULL_ENV_SOAP = "mm_center_full_env_soap"
FEAT_MODE_QM_NEAR_MM_SOAP = "qm_center_near_mm_soap"
FEAT_MODE_ALL_ATOM_SOAP = "all_atom_full_env_soap"
# Legacy (nearest-QM SOAP copy + dist); upgrade scripts rebuild to mm_center_qm_env_soap.
LEGACY_FEAT_MODE_QM_SOAP_DIST = "qm_soap_plus_dist"


def feat_dim_for_mode(mode: str, d_soap: int) -> int:
    if mode in (
        FEAT_MODE_MM_QM_ENV_SOAP,
        FEAT_MODE_MM_FULL_ENV_SOAP,
        FEAT_MODE_QM_NEAR_MM_SOAP,
        FEAT_MODE_ALL_ATOM_SOAP,
    ):
        return int(d_soap)
    if mode == LEGACY_FEAT_MODE_QM_SOAP_DIST:
        return int(d_soap) + 1
    raise ValueError(f"unknown feature_mode {mode!r}")


def feat_dim_mm_qm_env_soap(d_soap: int) -> int:
    return feat_dim_for_mode(FEAT_MODE_MM_QM_ENV_SOAP, int(d_soap))


def mm_oh_atom_indices(symbols: Sequence[str], n_qm: int) -> list[int]:
    """Full-frame indices of MM O/H sites (Coo order; all O/H when no MM distance cut)."""
    out: list[int] = []
    for i, sym in enumerate(symbols):
        if i < int(n_qm):
            continue
        el = str(sym).strip()[0].upper()
        if el in ("O", "H"):
            out.append(int(i))
    return out


def mm_oh_atom_indices_for_scf_mm(
    positions: np.ndarray,
    symbols: Sequence[str],
    n_qm: int,
    mm_coords_ang: np.ndarray,
    mm_symbols: Sequence[str],
    *,
    tol: float = 1e-3,
) -> list[int]:
    """
    Map Emb0 MM list (post ``filter_mm_by_distance``) to full Coo indices.

    Order matches ``gauss_rho_kernels_per_mm`` (O/H only, Coo traversal on filtered MM).
    """
    full_oh = mm_oh_atom_indices(symbols, int(n_qm))
    centers: list[int] = []
    used: set[int] = set()
    pos = np.asarray(positions, dtype=np.float64)
    for R, sym in zip(np.asarray(mm_coords_ang, dtype=np.float64), mm_symbols):
        el = str(sym).strip()[0].upper()
        if el not in ("O", "H"):
            continue
        best_ia: int | None = None
        best_d = float("inf")
        for ia in full_oh:
            if ia in used:
                continue
            if str(symbols[ia]).strip()[0].upper() != el:
                continue
            d = float(np.linalg.norm(pos[ia] - R))
            if d < best_d:
                best_d = d
                best_ia = int(ia)
        if best_ia is None or best_d > float(tol):
            raise RuntimeError(
                f"no Coo O/H match for MM {sym!r} at {R} (best_d={best_d:.4e}, tol={tol})"
            )
        centers.append(best_ia)
        used.add(best_ia)
    return centers




def mm_active_atom_indices(symbols: Sequence[str], n_qm: int) -> list[int]:
    """Full-frame indices of MM O/H/Na/K/Cl repulsion sites."""
    out: list[int] = []
    for i, sym in enumerate(symbols):
        if i < int(n_qm):
            continue
        if is_active_repulsion_symbol(sym):
            out.append(int(i))
    return out


def mm_kernel_atom_indices(symbols: Sequence[str], n_qm: int) -> list[int]:
    """Full-frame indices of MM O/H/Na/K sites (kernel / report order)."""
    out: list[int] = []
    for i, sym in enumerate(symbols):
        if i < int(n_qm):
            continue
        if is_kernel_mm_symbol(sym):
            out.append(int(i))
    return out


def mm_active_atom_indices_for_scf_mm(
    positions: np.ndarray,
    symbols: Sequence[str],
    n_qm: int,
    mm_coords_ang: np.ndarray,
    mm_symbols: Sequence[str],
    *,
    tol: float = 1e-3,
) -> list[int]:
    """Map filtered MM list to Coo indices for O/H/Na/K/Cl active sites."""
    full_active = mm_active_atom_indices(symbols, int(n_qm))
    centers: list[int] = []
    used: set[int] = set()
    pos = np.asarray(positions, dtype=np.float64)
    for R, sym in zip(np.asarray(mm_coords_ang, dtype=np.float64), mm_symbols):
        if not is_active_repulsion_symbol(sym):
            continue
        best_ia: int | None = None
        best_d = float("inf")
        want = _norm_elem_key(sym)
        for ia in full_active:
            if ia in used:
                continue
            if _norm_elem_key(symbols[ia]) != want:
                continue
            d = float(np.linalg.norm(pos[ia] - R))
            if d < best_d:
                best_d = d
                best_ia = int(ia)
        if best_ia is None or best_d > float(tol):
            raise RuntimeError(
                f"no Coo match for active MM {sym!r} at {R} (best_d={best_d:.4e}, tol={tol})"
            )
        centers.append(best_ia)
        used.add(best_ia)
    return centers


def mm_kernel_atom_indices_for_scf_mm(
    positions: np.ndarray,
    symbols: Sequence[str],
    n_qm: int,
    mm_coords_ang: np.ndarray,
    mm_symbols: Sequence[str],
    *,
    tol: float = 1e-3,
) -> list[int]:
    """Map filtered MM list to Coo indices for O/H/Na/K kernel sites."""
    full_k = mm_kernel_atom_indices(symbols, int(n_qm))
    centers: list[int] = []
    used: set[int] = set()
    pos = np.asarray(positions, dtype=np.float64)
    for R, sym in zip(np.asarray(mm_coords_ang, dtype=np.float64), mm_symbols):
        if not is_kernel_mm_symbol(sym):
            continue
        best_ia: int | None = None
        best_d = float("inf")
        want = _norm_elem_key(sym)
        for ia in full_k:
            if ia in used:
                continue
            if _norm_elem_key(symbols[ia]) != want:
                continue
            d = float(np.linalg.norm(pos[ia] - R))
            if d < best_d:
                best_d = d
                best_ia = int(ia)
        if best_ia is None or best_d > float(tol):
            raise RuntimeError(
                f"no Coo match for kernel MM {sym!r} at {R} (best_d={best_d:.4e}, tol={tol})"
            )
        centers.append(best_ia)
        used.add(best_ia)
    return centers


def mm_element_for_indices(symbols: Sequence[str], indices: list[int]) -> np.ndarray:
    el = np.zeros(len(indices), dtype=np.int8)
    for j, ia in enumerate(indices):
        key = _norm_elem_key(symbols[int(ia)])
        if key == "O":
            el[j] = MM_EL_O
        elif key == "H":
            el[j] = MM_EL_H
        elif key == "Na":
            el[j] = MM_KERNEL_EL_NA
        elif key == "K":
            el[j] = MM_KERNEL_EL_K
        elif key == "Cl":
            el[j] = MM_KERNEL_EL_CL
        elif key == "C":
            el[j] = MM_KERNEL_EL_C
        elif key == "N":
            el[j] = MM_KERNEL_EL_N
        else:
            raise ValueError(f"mm site {ia}: expected O/H/C/N/Na/K/Cl, got {symbols[int(ia)]!r}")
    return el


def mmh_element_for_kernel_codes(el: np.ndarray) -> np.ndarray:
    from embr_elements import (
        MMH_ELEM_C,
        MMH_ELEM_CL,
        MMH_ELEM_H,
        MMH_ELEM_K,
        MMH_ELEM_N,
        MMH_ELEM_NA,
        MMH_ELEM_O,
    )

    out = np.empty(np.asarray(el, dtype=np.int8).shape, dtype=np.int8)
    for j, code in enumerate(np.asarray(el, dtype=np.int8).reshape(-1)):
        c = int(code)
        if c == MM_KERNEL_EL_O:
            out[j] = MMH_ELEM_O
        elif c == MM_KERNEL_EL_H:
            out[j] = MMH_ELEM_H
        elif c == MM_KERNEL_EL_C:
            out[j] = MMH_ELEM_C
        elif c == MM_KERNEL_EL_N:
            out[j] = MMH_ELEM_N
        elif c == MM_KERNEL_EL_NA:
            out[j] = MMH_ELEM_NA
        elif c == MM_KERNEL_EL_K:
            out[j] = MMH_ELEM_K
        elif c == MM_KERNEL_EL_CL:
            out[j] = MMH_ELEM_CL
        else:
            raise ValueError(f"kernel element {c} is not a partition site (O/H/C/N/Na/K/Cl)")
    return out


def compute_soap_mm_partition_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
    full_env: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SOAP at each MM O/H/ion nucleus for k-partition + EmbR repulsion training.

    ``full_env=True``: full Coo environment; ``False``: QM-only environment.
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper(n_qm=int(n_qm))

    n_qm_i = int(n_qm)
    pos = np.asarray(positions, dtype=np.float64)
    center_pos, center_idx = _mm_center_positions(
        pos,
        symbols,
        n_qm_i,
        scf_mm_coords,
        scf_mm_symbols,
        active_only=False,
    )
    d_soap = int(soap.get_number_of_features())
    if center_pos.shape[0] == 0:
        return np.zeros((0, d_soap), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    if bool(full_env):
        atoms = positions_symbols_to_ase(pos, symbols)
    else:
        qm_pos = pos[:n_qm_i]
        qm_syms = [str(s) for s in symbols[:n_qm_i]]
        atoms = positions_symbols_to_ase(qm_pos, qm_syms)
    desc = soap.create(atoms, centers=center_pos)
    out = np.asarray(desc, dtype=np.float32)
    if out.shape != (center_pos.shape[0], d_soap):
        raise RuntimeError(f"MM partition SOAP shape {out.shape} != ({center_pos.shape[0]}, {d_soap})")
    kernel_el = mm_element_for_indices(symbols, center_idx)
    return out, mmh_element_for_kernel_codes(kernel_el)


def _mm_center_positions(
    positions: np.ndarray,
    symbols: Sequence[str],
    n_qm: int,
    scf_mm_coords: np.ndarray | None,
    scf_mm_symbols: Sequence[str] | None,
    *,
    active_only: bool = False,
) -> tuple[np.ndarray, list[int]]:
    pos = np.asarray(positions, dtype=np.float64)
    if scf_mm_coords is not None and scf_mm_symbols is not None:
        if active_only:
            center_idx = mm_active_atom_indices_for_scf_mm(
                pos,
                symbols,
                int(n_qm),
                scf_mm_coords,
                scf_mm_symbols,
            )
        else:
            center_idx = mm_kernel_atom_indices_for_scf_mm(
                pos,
                symbols,
                int(n_qm),
                scf_mm_coords,
                scf_mm_symbols,
            )
    else:
        center_idx = (
            mm_active_atom_indices(symbols, int(n_qm))
            if active_only
            else mm_kernel_atom_indices(symbols, int(n_qm))
        )
    if not center_idx:
        return np.zeros((0, 3), dtype=np.float64), center_idx
    return np.asarray(pos[center_idx], dtype=np.float64), center_idx


def compute_soap_mm_qm_env_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SOAP at each MM O/H nucleus; only QM atoms (first ``n_qm``) enter the environment.

    DScribe: ``Atoms`` = QM substructure; ``centers`` = MM O/H Cartesian positions.

    Returns
    -------
    desc : (n_mm, d_soap) float32
    mm_element : (n_mm,) int8 — 0=O, 1=H
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper(n_qm=int(n_qm))

    n_qm_i = int(n_qm)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] < n_qm_i:
        raise ValueError(f"need at least {n_qm_i} atoms, got {pos.shape[0]}")

    center_pos, center_idx = _mm_center_positions(
        pos,
        symbols,
        n_qm_i,
        scf_mm_coords,
        scf_mm_symbols,
    )
    d_soap = int(soap.get_number_of_features())
    if center_pos.shape[0] == 0:
        return np.zeros((0, d_soap), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    qm_pos = pos[:n_qm_i]
    qm_syms = [str(s) for s in symbols[:n_qm_i]]
    atoms_qm = positions_symbols_to_ase(qm_pos, qm_syms)
    desc = soap.create(atoms_qm, centers=center_pos)
    out = np.asarray(desc, dtype=np.float32)
    if out.shape != (center_pos.shape[0], d_soap):
        raise RuntimeError(f"MM QM-env SOAP shape {out.shape} != ({center_pos.shape[0]}, {d_soap})")
    return out, mm_element_for_indices(symbols, center_idx)


def compute_soap_mm_active_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
    full_env: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """SOAP at each MM H/Na/K/Cl (repulsion-active) nucleus."""
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper(n_qm=int(n_qm))

    n_qm_i = int(n_qm)
    center_pos, center_idx = _mm_center_positions(
        positions,
        symbols,
        n_qm_i,
        scf_mm_coords,
        scf_mm_symbols,
        active_only=True,
    )
    d_soap = int(soap.get_number_of_features())
    if center_pos.shape[0] == 0:
        return np.zeros((0, d_soap), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    if bool(full_env):
        atoms = positions_symbols_to_ase(positions, symbols)
    else:
        qm_pos = np.asarray(positions, dtype=np.float64)[:n_qm_i]
        qm_syms = [str(s) for s in symbols[:n_qm_i]]
        atoms = positions_symbols_to_ase(qm_pos, qm_syms)
    desc = soap.create(atoms, centers=center_pos)
    out = np.asarray(desc, dtype=np.float32)
    if out.shape != (center_pos.shape[0], d_soap):
        raise RuntimeError(f"MM active SOAP shape {out.shape} != ({center_pos.shape[0]}, {d_soap})")
    return out, mmh_element_for_kernel_codes(mm_element_for_indices(symbols, center_idx))



def compute_soap_mm_active_centers_old_qm_env(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward alias: active repulsion sites, QM-only SOAP environment."""
    return compute_soap_mm_active_centers(
        positions,
        symbols,
        soap=soap,
        hyper=hyper,
        n_qm=n_qm,
        scf_mm_coords=scf_mm_coords,
        scf_mm_symbols=scf_mm_symbols,
        full_env=False,
    )


def compute_soap_mm_oh_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy: SOAP at MM O/H with **full** Coo environment (QM + MM waters).

    Prefer ``compute_soap_mm_qm_env_centers`` for mix_ai A_j features.
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper(n_qm=int(n_qm))

    if scf_mm_coords is not None and scf_mm_symbols is not None:
        centers = mm_oh_atom_indices_for_scf_mm(
            positions,
            symbols,
            int(n_qm),
            scf_mm_coords,
            scf_mm_symbols,
        )
    else:
        centers = mm_oh_atom_indices(symbols, int(n_qm))
    d_soap = int(soap.get_number_of_features())
    if not centers:
        return np.zeros((0, d_soap), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    atoms = positions_symbols_to_ase(positions, symbols)
    desc = soap.create(atoms, centers=centers)
    out = np.asarray(desc, dtype=np.float32)
    if out.shape != (len(centers), d_soap):
        raise RuntimeError(f"MM O/H SOAP shape {out.shape} != ({len(centers)}, {d_soap})")
    return out, mm_element_for_indices(symbols, centers)


def compute_soap_qm_near_mm_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
    n_qm: int,
    near_mm_cut_ang: float,
) -> np.ndarray:
    """
    SOAP at each QM center; environment = all QM + MM within ``near_mm_cut_ang`` of any QM.

    Returns (n_qm, d_soap) float32.  Fixed ``d_soap`` regardless of how many MM are kept.
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper(n_qm=int(n_qm))

    n_qm_i = int(n_qm)
    cut = float(near_mm_cut_ang)
    if cut <= 0.0:
        raise ValueError(f"near_mm_cut_ang must be > 0, got {cut}")
    pos = np.asarray(positions, dtype=np.float64)
    if int(pos.shape[0]) < n_qm_i:
        raise ValueError(f"need at least {n_qm_i} atoms, got {pos.shape[0]}")

    qm_pos = pos[:n_qm_i]
    keep: list[int] = list(range(n_qm_i))
    for ia in range(n_qm_i, int(pos.shape[0])):
        dmin = float(np.min(np.linalg.norm(qm_pos - pos[ia], axis=1)))
        if dmin <= cut:
            keep.append(int(ia))

    sub_pos = pos[keep]
    sub_syms = [str(symbols[i]) for i in keep]
    atoms = positions_symbols_to_ase(sub_pos, sub_syms)
    centers = list(range(n_qm_i))
    desc = soap.create(atoms, centers=centers)
    out = np.asarray(desc, dtype=np.float32)
    d_soap = int(soap.get_number_of_features())
    if out.shape != (n_qm_i, d_soap):
        raise RuntimeError(f"QM near-MM SOAP shape {out.shape} != ({n_qm_i}, {d_soap})")
    return out


def compute_soap_all_atom_centers(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    soap=None,
    hyper: SoapE0Hyper | None = None,
) -> np.ndarray:
    """
    SOAP at **every** atom in the frame; environment = full Coo (no QM/MM split).

    Returns (n_atoms, d_soap) float32.
    """
    if soap is None:
        soap, hyper = build_soap_calculator(hyper)
    elif hyper is None:
        hyper = SoapE0Hyper()

    pos = np.asarray(positions, dtype=np.float64)
    n_atoms = int(pos.shape[0])
    if n_atoms == 0:
        d_soap = int(soap.get_number_of_features())
        return np.zeros((0, d_soap), dtype=np.float32)

    atoms = positions_symbols_to_ase(pos, symbols)
    centers = list(range(n_atoms))
    desc = soap.create(atoms, centers=centers)
    out = np.asarray(desc, dtype=np.float32)
    d_soap = int(soap.get_number_of_features())
    if out.shape != (n_atoms, d_soap):
        raise RuntimeError(f"all-atom SOAP shape {out.shape} != ({n_atoms}, {d_soap})")
    return out


def min_dist_qm_per_mm(
    positions: np.ndarray,
    symbols: Sequence[str],
    n_qm: int,
    scf_mm_coords: np.ndarray | None = None,
    scf_mm_symbols: Sequence[str] | None = None,
    *,
    active_only: bool = False,
) -> np.ndarray:
    """Min distance (Å) from each MM site to any QM atom (inspect / audit only)."""
    pos = np.asarray(positions, dtype=np.float64)
    n_qm_i = int(n_qm)
    center_pos, _ = _mm_center_positions(
        pos,
        symbols,
        n_qm_i,
        scf_mm_coords,
        scf_mm_symbols,
        active_only=bool(active_only),
    )
    qm = pos[:n_qm_i]
    out = np.empty((center_pos.shape[0],), dtype=np.float32)
    for j in range(int(center_pos.shape[0])):
        d = np.linalg.norm(qm - center_pos[j], axis=1)
        out[j] = float(np.min(d))
    return out


def recompute_feat_mm_from_geometry(
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    n_qm: int,
    scf_mm_coords: np.ndarray,
    scf_mm_symbols: Sequence[str],
    soap=None,
    hyper: SoapE0Hyper | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild feat_mm + dist_qm + mm_element from Coo (audit / upgrade helper)."""
    feat, mm_el = compute_soap_mm_qm_env_centers(
        positions,
        symbols,
        soap=soap,
        hyper=hyper,
        n_qm=int(n_qm),
        scf_mm_coords=scf_mm_coords,
        scf_mm_symbols=scf_mm_symbols,
    )
    dist = min_dist_qm_per_mm(
        positions,
        symbols,
        int(n_qm),
        scf_mm_coords=scf_mm_coords,
        scf_mm_symbols=scf_mm_symbols,
    )
    return feat, dist, mm_el
