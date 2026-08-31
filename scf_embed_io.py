"""Coo I/O for PySCF QM/MM embedding SCF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embr_io import _parse_coo_core, coo_path, load_e0_txt

try:
    from embr_io import load_e_embed_txt
except ImportError:
    def load_e_embed_txt(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        return load_e0_txt(path), None


DEFAULT_MM_CHARGES: dict[str, float] = {
    "O": -0.834,
    "H": 0.417,
    "Na": 1.0,
    "K": 1.0,
    "Cl": -1.0,
}


@dataclass(frozen=True)
class ScfFrame:
    qm_symbols: tuple[str, ...]
    qm_coords_ang: np.ndarray
    mm_symbols: tuple[str, ...]
    mm_coords_ang: np.ndarray
    mm_charges: np.ndarray


def _norm_elem(tok: str) -> str:
    t = tok.strip()
    if not t:
        raise ValueError("empty element token")
    u = t.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    return t[0].upper() + (t[1:].lower() if len(t) > 1 else "")


def _default_qm_fallback(
    n_qm: int,
    qm_symbols_fallback: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if qm_symbols_fallback is not None:
        return qm_symbols_fallback
    return ()


def load_scf_frame(
    path: Path,
    *,
    n_qm: int = 10,
    mm_charge_map: dict[str, float] | None = None,
    qm_symbols_fallback: tuple[str, ...] | None = None,
) -> ScfFrame:
    from embr_io import load_geometry_lines

    lines = load_geometry_lines(path)
    fb = _default_qm_fallback(int(n_qm), qm_symbols_fallback)
    qm_syms, qm_pos, mm_syms, mm_pos, mm_chg_raw = _parse_coo_core(
        lines, n_qm=int(n_qm), qm_symbols_fallback=fb
    )

    cmap = dict(DEFAULT_MM_CHARGES)
    if mm_charge_map:
        cmap.update(mm_charge_map)

    charges: list[float] = []
    for sym, chg_opt in zip(mm_syms, mm_chg_raw):
        if chg_opt is not None:
            charges.append(float(chg_opt))
            continue
        key = _norm_elem(sym)
        if key not in cmap:
            raise ValueError(f"{path}: no MM charge for element {sym!r}; known: {sorted(cmap)}")
        charges.append(float(cmap[key]))

    return ScfFrame(
        qm_symbols=tuple(qm_syms),
        qm_coords_ang=np.asarray(qm_pos, dtype=np.float64),
        mm_symbols=tuple(mm_syms),
        mm_coords_ang=np.asarray(mm_pos, dtype=np.float64),
        mm_charges=np.asarray(charges, dtype=np.float64),
    )


def load_scf_frame_ala_ions(path: Path, *, n_qm: int) -> ScfFrame:
    return load_scf_frame(path, n_qm=int(n_qm))


def filter_mm_by_distance(
    frame: ScfFrame,
    *,
    r_cut_ang: float | None = None,
) -> ScfFrame:
    if r_cut_ang is None or float(r_cut_ang) <= 0.0:
        return frame
    rc = float(r_cut_ang)
    qm = frame.qm_coords_ang
    mm = frame.mm_coords_ang
    keep: list[int] = []
    for j in range(int(mm.shape[0])):
        d = np.linalg.norm(qm - mm[j], axis=1)
        if float(np.min(d)) <= rc:
            keep.append(j)
    if not keep:
        raise ValueError(f"no MM atom within r_cut={rc} Å of QM region")
    idx = np.asarray(keep, dtype=np.int64)
    return ScfFrame(
        qm_symbols=frame.qm_symbols,
        qm_coords_ang=frame.qm_coords_ang,
        mm_symbols=tuple(frame.mm_symbols[i] for i in idx),
        mm_coords_ang=frame.mm_coords_ang[idx],
        mm_charges=frame.mm_charges[idx],
    )


def write_scf_coo(path: Path, frame: ScfFrame) -> None:
    """Write ``Coo{i}.xyz`` extended XYZ (MM rows include charge)."""
    from embr_io import write_xyz_frame

    write_xyz_frame(
        path,
        qm_symbols=frame.qm_symbols,
        qm_coords_ang=frame.qm_coords_ang,
        mm_symbols=frame.mm_symbols,
        mm_coords_ang=frame.mm_coords_ang,
        mm_charges=frame.mm_charges,
    )


def _mulliken_atom_charges(mf, dm: np.ndarray) -> np.ndarray:
    mol = mf.mol
    dm = np.asarray(dm, dtype=np.float64)
    s = mf.get_ovlp()
    try:
        from pyscf.lo import mulliken as mulliken_mod

        _pop, chg = mulliken_mod.mulliken_pop(mol, dm, s)
        return np.asarray(chg, dtype=np.float64).reshape(-1)
    except Exception:
        pass
    try:
        chg = mf.mulliken_pop(verbose=-1)
        if isinstance(chg, tuple):
            chg = chg[-1]
        return np.asarray(chg, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"Mulliken population failed: {exc}") from exc


def mulliken_mm_charges_for_fragment(
    symbols: tuple[str, ...],
    coords_ang: np.ndarray,
    cfg,
) -> np.ndarray:
    from scf_embed_pyscf import run_gas_mf

    syms = tuple(str(s) for s in symbols)
    coords = np.asarray(coords_ang, dtype=np.float64).reshape(-1, 3)
    if len(syms) != int(coords.shape[0]):
        raise ValueError("symbols/coords length mismatch")
    frame_b = ScfFrame(
        qm_symbols=syms,
        qm_coords_ang=coords,
        mm_symbols=(),
        mm_coords_ang=np.zeros((0, 3), dtype=np.float64),
        mm_charges=np.zeros(0, dtype=np.float64),
    )
    mf, dm = run_gas_mf(frame_b, cfg)
    return np.asarray(_mulliken_atom_charges(mf, dm), dtype=np.float64).reshape(-1)


__all__ = [
    "DEFAULT_MM_CHARGES",
    "ScfFrame",
    "coo_path",
    "filter_mm_by_distance",
    "load_e0_txt",
    "load_e_embed_txt",
    "load_scf_frame",
    "load_scf_frame_ala_ions",
    "mulliken_mm_charges_for_fragment",
    "write_scf_coo",
]
