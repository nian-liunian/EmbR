"""
PySCF driver for the Emb0 / EmbR QM/MM electronic-structure calculations.

Published EmbR workflow
-----------------------
1. **Emb0**: QM Hamiltonian + MM point-charge electrostatics.
2. **EmbR**: Emb0 + the learned short-range repulsive one-electron operator.
3. **Gas**: the same isolated QM geometry without the MM environment.

The manuscript benchmarks use HF/6-31G* and HF/6-31+G*. Legacy B3LYP/D3BJ, Gaussian-envelope,
and angular-cone options are retained for compatibility/diagnostics but are not the paper path.
Counterpoise/full_QM helpers are also defined here for the reference calculations.

Requires PySCF (see ``requirements-scf.txt``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scf_embed_io import ScfFrame

try:
    from pyscf import dft, gto, lib, qmmm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scf_embed_pyscf needs PySCF. Install with:\n  pip install pyscf\nSee requirements-scf.txt"
    ) from e


@dataclass(frozen=True)
class ConeRepAng:
    """Legacy angular window for site-centered repulsion (degrees from H→QM axis)."""

    theta1_deg: float = 180.0
    theta2_deg: float = 180.0

    def __post_init__(self) -> None:
        if float(self.theta2_deg) < float(self.theta1_deg):
            object.__setattr__(self, "theta2_deg", float(self.theta1_deg))

    def is_isotropic(self) -> bool:
        return float(self.theta1_deg) >= 180.0 - 1e-9

    @property
    def theta1_rad(self) -> float:
        return float(np.deg2rad(float(self.theta1_deg)))

    @property
    def theta2_rad(self) -> float:
        return float(np.deg2rad(float(self.theta2_deg)))


def cone_angular_weight_from_cos(
    cos_theta: np.ndarray,
    *,
    theta1_rad: float,
    theta2_rad: float,
) -> np.ndarray:
    """
    θ≤θ1 → 1; θ≥θ2 → 0; smooth cosine taper in between.
    θ1=θ2=π (180°) → all 1 (isotropic repulsion).
    """
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    if float(theta1_rad) >= np.pi - 1e-9:
        return np.ones(cos_theta.shape, dtype=np.float64)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    w = np.ones(theta.shape, dtype=np.float64)
    t2 = float(theta2_rad)
    t1 = float(theta1_rad)
    w[theta >= t2] = 0.0
    mid = (theta > t1) & (theta < t2)
    if t2 > t1 + 1e-12:
        x = (theta[mid] - t1) / (t2 - t1)
        w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))
    elif t2 <= t1 + 1e-12:
        w[theta > t1] = 0.0
    return w


def cone_axes_h_to_nearest_qm(frame: ScfFrame) -> np.ndarray:
    """Unit axes (n_mm, 3): MM H nucleus → nearest QM; zeros on O / non-OH."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    axes = np.zeros((mm.shape[0], 3), dtype=np.float64)
    for i, sym in enumerate(frame.mm_symbols):
        if sym.strip()[0].upper() != "H":
            continue
        dvec = qm - mm[i]
        dists = np.linalg.norm(dvec, axis=1)
        u = dvec[int(np.argmin(dists))]
        n = float(np.linalg.norm(u))
        if n > 1e-10:
            axes[i] = u / n
    return axes


@dataclass(frozen=True)
class ScfEmbedConfig:
    xc: str = "B3LYP"
    basis: str = "6-31g*"
    conv_tol: float = 1e-9
    max_cycle: int = 200
    verbose: int = 0
    unit: str = "Angstrom"
    use_d3bj: bool = False  # Gaussian em=GD3BJ
    num_threads: int = 0  # 0 = PySCF / OMP default
    cart: bool = False  # True = Cartesian d (Gaussian 6-31G* default)
    qm_charge: int = 0  # net formal charge of QM block (Asp⁻ → −1; Ala zwitterion → 0)


def resolve_qm_charge(*sources: object, default: int = 0) -> int:
    """Read ``qm_charge`` / ``qm-charge`` from dict-like sources (first hit wins)."""
    for src in sources:
        if src is None:
            continue
        if isinstance(src, dict):
            if "qm_charge" in src and src["qm_charge"] is not None:
                return int(src["qm_charge"])
            if "qm-charge" in src and src["qm-charge"] is not None:
                return int(src["qm-charge"])
        else:
            q = getattr(src, "qm_charge", None)
            if q is not None:
                return int(q)
    return int(default)


SCF_PRESETS: dict[str, dict[str, object]] = {
    "hf": {
        "method": "hf",
        "basis": "6-31g*",
        "d3bj": False,
        "label": "HF / 6-31G* (HF density)",
    },
    "b3lyp-plus": {
        "method": "b3lyp",
        "basis": "6-31+G*",
        "d3bj": True,
        "label": "B3LYP / 6-31+G* + D3BJ",
    },
    "hf-plus": {
        "method": "hf",
        "basis": "6-31+G*",
        "d3bj": False,
        "label": "HF / 6-31+G* (HF density)",
    },
}


def resolve_scf_method(method: str) -> str:
    """Map CLI method name to PySCF xc string (HF or B3LYP)."""
    m = str(method).strip().lower().replace("-", "").replace("_", "")
    if m in ("hf", "rhf", "hartreefock"):
        return "HF"
    if m in ("b3lyp",):
        return "B3LYP"
    raise ValueError(f"unsupported --method {method!r} (use hf or b3lyp)")


def resolve_scf_basis(basis: str) -> str:
    """Normalize basis-set CLI string for PySCF."""
    b = str(basis).strip()
    if not b:
        raise ValueError("empty --basis")
    aliases = {
        "631g*": "6-31g*",
        "631gs": "6-31g*",
        "631+g*": "6-31+G*",
        "631+gs": "6-31+G*",
        "631++g*": "6-31++G*",
        "631++gs": "6-31++G*",
    }
    key = b.lower().replace(" ", "")
    return aliases.get(key, b)


def is_hf_xc(xc: str) -> bool:
    return resolve_scf_method(xc) == "HF"


def basis_has_diffuse(basis: str) -> bool:
    return "+" in resolve_scf_basis(basis)


def scf_variant_dir_name(residue: str, method: str, basis: str) -> str | None:
    """
    Named output folder for the three comparison runs.

    HF/6-31g* → ala_hf; B3LYP/6-31+G* → ala_+; HF/6-31+G* → ala_hf_+.
    Default B3LYP/6-31g* → None (legacy batch_{residue}_{fs}_{fe}).
    """
    residue = str(residue).lower()
    hf = is_hf_xc(resolve_scf_method(method))
    plus = basis_has_diffuse(basis)
    if hf and not plus:
        return f"{residue}_hf"
    if (not hf) and plus:
        return f"{residue}_+"
    if hf and plus:
        return f"{residue}_hf_+"
    return None


def default_ref_e0_filename(residue: str, method: str, basis: str) -> str:
    """
    Output E0 label filename for PySCF ref batch (``EAla_+.txt`` style).

    HF/6-31g* → EAla_hf.txt; B3LYP/6-31+G* → EAla_+.txt; B3LYP/6-31g* → EAla_ref.txt.
    """
    r = str(residue).strip().lower().capitalize()
    hf = is_hf_xc(resolve_scf_method(method))
    plus = basis_has_diffuse(basis)
    if hf and plus:
        return f"E{r}_hf_+.txt"
    if hf:
        return f"E{r}_hf.txt"
    if plus:
        return f"E{r}_+.txt"
    return f"E{r}_ref.txt"


def default_batch_out_dir(
    residue: str,
    frame_start: int,
    frame_end: int,
    *,
    method: str = "b3lyp",
    basis: str = "6-31g*",
) -> Path:
    tag = scf_variant_dir_name(residue, method, basis)
    if tag is not None:
        return Path(f"theta/{tag}")
    return Path(f"theta/batch_{residue}_{frame_start}_{frame_end}")


def resolve_scf_preset(name: str) -> dict[str, object]:
    key = str(name).strip().lower()
    if key not in SCF_PRESETS:
        raise ValueError(f"unknown --scf-preset {name!r}; choose from {sorted(SCF_PRESETS)}")
    return dict(SCF_PRESETS[key])


def scf_embed_config_from_cli(
    *,
    method: str = "b3lyp",
    basis: str = "6-31g*",
    use_d3bj: bool = False,
    num_threads: int = 0,
    verbose: int = 0,
    conv_tol: float | None = None,
    max_cycle: int | None = None,
    cart: bool = False,
    qm_charge: int = 0,
) -> ScfEmbedConfig:
    xc = resolve_scf_method(method)
    basis_n = resolve_scf_basis(basis)
    d3 = bool(use_d3bj)
    if is_hf_xc(xc) and d3:
        print("  [SCF] HF 不使用 D3BJ，已关闭 --d3bj", flush=True)
        d3 = False
    kw: dict[str, object] = {
        "xc": xc,
        "basis": basis_n,
        "use_d3bj": d3,
        "num_threads": int(num_threads),
        "verbose": int(verbose),
        "cart": bool(cart),
        "qm_charge": int(qm_charge),
    }
    if conv_tol is not None:
        kw["conv_tol"] = float(conv_tol)
    if max_cycle is not None:
        kw["max_cycle"] = int(max_cycle)
    return ScfEmbedConfig(**kw)


def make_mean_field(mol: gto.Mole, cfg: ScfEmbedConfig):
    """RHF/RKS closed-shell; UHF/UKS when ``mol.spin > 0`` (e.g. neutral Na atom)."""
    from pyscf import scf

    spin = int(getattr(mol, "spin", 0) or 0)
    if is_hf_xc(cfg.xc):
        mf = scf.UHF(mol) if spin else scf.RHF(mol)
    else:
        mf = dft.UKS(mol) if spin else dft.RKS(mol)
        mf.xc = cfg.xc
    mf.conv_tol = float(cfg.conv_tol)
    mf.max_cycle = int(cfg.max_cycle)
    mf.verbose = int(cfg.verbose)
    return mf


@dataclass(frozen=True)
class GaussRepParams:
    """V_rep(r) = Σ_J A_J exp(-alpha |r-R_J|²) on MM nuclei (one-electron potential)."""

    alpha_bohr2: float = 0.6
    amp_o_hartree: float = 0.0
    amp_h_hartree: float = 0.0

    def amplitudes_for_mm(self, mm_symbols: tuple[str, ...]) -> np.ndarray:
        out = np.zeros(len(mm_symbols), dtype=np.float64)
        for j, s in enumerate(mm_symbols):
            el = s.strip()[0].upper()
            if el == "O":
                out[j] = float(self.amp_o_hartree)
            elif el == "H":
                out[j] = float(self.amp_h_hartree)
            else:
                raise ValueError(f"GaussRepParams: unsupported MM element {s!r}")
        return out


@dataclass
class ScfEmbedResult:
    e_total_hartree: float
    e_int_hartree: float
    e_gas_hartree: float
    e_emb_scf_hartree: float
    e_gas_scf_hartree: float
    e_disp_emb_hartree: float
    e_disp_gas_hartree: float
    use_d3bj: bool
    converged: bool
    rho_mm: np.ndarray  # ρ at each MM nucleus (e/Bohr³)
    n_mm: float  # electrons in buffer shell (see ``mm_buffer_n_electrons``)
    mf: Any


def set_pyscf_threads(n: int) -> None:
    """Match Gaussian %nprocshared (OpenMP threads for BLAS / grid)."""
    if int(n) <= 0:
        return
    import os

    nt = int(n)
    os.environ["OMP_NUM_THREADS"] = str(nt)
    os.environ["MKL_NUM_THREADS"] = str(nt)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nt)
    # Never assign ``lib.num_threads = nt``: in PySCF 2.x it is a function, not a field.
    setter = getattr(lib, "set_num_threads", None)
    if callable(setter):
        setter(nt)


def _d3bj_dispersion_energy(mf: dft.rks.RKS) -> float:
    """
    Grimme D3(BJ) on QM atoms (Gaussian em=GD3BJ).

    Supports ``pip install dftd3`` (simple-dftd3 >= 1.x) and optional pyscf-dftd3.
    """
    mol = mf.mol
    xc = str(mf.xc).split(",")[0].strip().upper() or "B3LYP"
    errors: list[str] = []

    # 1) simple-dftd3 PySCF helper (dftd3 >= 1.0)
    try:
        import dftd3.pyscf as d3pyscf

        d3 = d3pyscf.DFTD3Dispersion(mol, xc=xc, version="d3bj")
        energy, _grad = d3.kernel()
        return float(np.asarray(energy).reshape(-1)[0])
    except Exception as e:  # noqa: BLE001
        errors.append(f"dftd3.pyscf: {e}")

    # 2) simple-dftd3 low-level interface
    try:
        from dftd3.interface import DispersionModel, RationalDampingParam

        numbers = np.asarray([gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)], dtype=np.int64)
        positions = np.asarray(mol.atom_coords(), dtype=np.float64)  # Bohr (PySCF)
        model = DispersionModel(numbers, positions)
        param = RationalDampingParam(method=xc.lower())
        res = model.get_dispersion(param, grad=False)
        return float(res.get("energy", 0.0))
    except Exception as e:  # noqa: BLE001
        errors.append(f"dftd3.interface: {e}")

    # 3) legacy dftd3 (<1.0) top-level DFTD3
    try:
        import dftd3 as d3lib

        if hasattr(d3lib, "DFTD3"):
            calc = d3lib.DFTD3(
                numbers=np.asarray(mol.atom_charges(), dtype=np.int64),
                positions=np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64),
                charge=int(mol.charge),
                uhf=int(max(0, mol.spin // 2)),
            )
            for method in ("d3bj", "D3BJ"):
                try:
                    res = calc.get_dispersion(xc, method)
                    if isinstance(res, dict):
                        return float(res.get("energy", res.get("dispersion", 0.0)))
                    return float(res)
                except Exception:
                    continue
    except Exception as e:  # noqa: BLE001
        errors.append(f"dftd3 legacy: {e}")

    # 4) optional pyscf-dftd3 extension
    try:
        from pyscf.dftd3 import DFTD3Dispersion

        disp = DFTD3Dispersion(mf, xc=str(mf.xc), d3="d3bj")
        out = disp.kernel()
        if isinstance(out, tuple):
            for x in reversed(out):
                if isinstance(x, (float, int, np.floating)):
                    return float(x)
                arr = np.asarray(x)
                if arr.size == 1:
                    return float(arr.reshape(-1)[0])
            return float(out[-1])
        return float(np.asarray(out).reshape(-1)[0])
    except Exception as e:  # noqa: BLE001
        errors.append(f"pyscf.dftd3: {e}")

    raise ImportError(
        "D3BJ (--d3bj) failed with all backends.\n"
        "  pip install -U dftd3   # simple-dftd3 >= 1.0\n"
        "  conda install -c conda-forge dftd3\n"
        "Details:\n  " + "\n  ".join(errors)
    )


def _total_energy(mf: dft.rks.RKS, *, use_d3bj: bool) -> tuple[float, float, float]:
    """Return (e_total, e_scf, e_disp)."""
    e_scf = float(mf.e_tot)
    if not use_d3bj:
        return e_scf, e_scf, 0.0
    e_disp = _d3bj_dispersion_energy(mf)
    return e_scf + e_disp, e_scf, e_disp


def _nelectron_from_atom_str(atom_str: str, *, charge: int = 0) -> int:
    """Count electrons from ``mol.atom`` lines; ghost-* sites contribute 0."""
    ne = 0
    for line in str(atom_str).strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        if sym.lower().startswith("ghost"):
            continue
        ne += int(gto.charge(sym))
    q = int(charge)
    nelec = int(ne - q)
    if nelec < 0:
        raise ValueError(f"molecule charge {q} exceeds neutral electron count {ne}")
    return nelec


def _apply_mol_cfg(mol: gto.Mole, cfg: ScfEmbedConfig, *, charge: int | None = None) -> None:
    mol.unit = cfg.unit
    mol.basis = cfg.basis
    mol.verbose = int(cfg.verbose)
    if bool(cfg.cart):
        mol.cart = True
    # Explicit override (CP/cluster MM, etc.); else QM net charge from cfg.
    q = int(cfg.qm_charge) if charge is None else int(charge)
    mol.charge = q
    _ensure_mol_spin_consistent(mol)


def _ensure_mol_spin_consistent(mol: gto.Mole) -> None:
    """
    Closed-shell only: ``mol.spin = 0`` → RHF/RKS (not UHF/UKS).

    Requires an **even** electron count after ``mol.charge`` is set (QM ``qm_charge``
    and/or MM ion formal charges: Na/K → +1, Cl → −1; waters → 0 for cluster).
    """
    q = int(getattr(mol, "charge", 0) or 0)
    nelec = _nelectron_from_atom_str(mol.atom, charge=q)
    mol.spin = 0
    if nelec % 2 != 0:
        raise ValueError(
            f"closed-shell SCF needs even electron count, got nelec={nelec} with mol.charge={q:+d}. "
            "Set manifest scf.qm_charge (Asp⁻ → -1; ala/gly zwitterion → 0). "
            "Also check Coo ion formal charges (Na/K +1, Cl −1) and cluster MM net "
            "(neutral O,H,H; ions should usually balance). "
            "A lone Cl− or Na+ in Coo cannot use closed-shell explicit cluster SCF."
        )


def _build_mol(frame: ScfFrame, cfg: ScfEmbedConfig) -> gto.Mole:
    atom_lines = [f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.qm_symbols, frame.qm_coords_ang)]
    mol = gto.Mole()
    mol.atom = "\n".join(atom_lines)
    _apply_mol_cfg(mol, cfg)
    mol.build()
    return mol


def _mm_elem_key(sym: str) -> str:
    """Element key for MM CP / ghost labels (O, H, Na, K, …)."""
    s = str(sym).strip()
    if not s:
        raise ValueError("empty MM symbol")
    u = s.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    if u == "K" or (u.startswith("K") and len(u) <= 2):
        return "K"
    return s[0].upper()


_CP_GHOST_MM_ELEMENTS = frozenset({"O", "H", "C", "N", "Na", "K", "Cl"})
_CP_ION_MM_ELEMENTS = frozenset({"Na", "K", "Cl"})


def cp_ghost_mm_elements() -> frozenset[str]:
    """MM elements that receive ghost basis in CP supermol (A+ghostB)."""
    return _CP_GHOST_MM_ELEMENTS


def missing_cp_ion_mm_support() -> tuple[str, ...]:
    """Empty if Na/K/Cl ghost CP is supported; else missing element keys."""
    return tuple(sorted(_CP_ION_MM_ELEMENTS - _CP_GHOST_MM_ELEMENTS))


def cp_mm_formal_charge(frame: ScfFrame) -> int:
    """
    Net **formal** charge of the explicit MM fragment in CP / cluster SCF.

    Ion sites: ``frame.mm_charges`` from ``load_scf_frame`` (Na/K → +1, Cl → −1
    unless Coo line has an explicit 5th column).  Neutral explicit molecules
    (H₂O, CH₃OH, …) → **0** — Coo column-5 Mulliken values are Emb0 point charges
    only and must not be rounded into CP ``mol.charge``.
    """
    n = len(frame.mm_symbols)
    ch = np.asarray(frame.mm_charges, dtype=np.float64).reshape(-1)
    if ch.size != n:
        raise ValueError(f"mm_charges length {ch.size} != n_mm {n}")
    has_ion = any(_mm_elem_key(s) in _CP_ION_MM_ELEMENTS for s in frame.mm_symbols)
    if not has_ion:
        return 0

    total = 0
    i = 0
    while i < n:
        el = _mm_elem_key(frame.mm_symbols[i])
        if el in ("Na", "K", "Cl"):
            q = int(round(float(ch[i])))
            if q == 0:
                raise ValueError(
                    f"MM {el} site {i}: CP SCF needs ionic formal charge in mm_charges "
                    f"(got {ch[i]:g}); set e.g. +1 for Na+, -1 for Cl- in Coo / load_scf_frame"
                )
            total += q
            i += 1
            continue
        if el == "O" and i + 2 < n:
            trip = [_mm_elem_key(frame.mm_symbols[i + j]) for j in range(3)]
            if trip == ["O", "H", "H"]:
                i += 3
                continue
        i += 1
    return int(total)


def cp_cluster_formal_charge(frame: ScfFrame, *, qm_charge: int = 0) -> int:
    """Total charge for real QM + real MM cluster: ``qm_charge + MM_formal``."""
    return int(qm_charge) + int(cp_mm_formal_charge(frame))


def _mm_ghost_atom_lines(frame: ScfFrame) -> list[str]:
    """Ghost O/H/C/N/Na/K/Cl at MM sites (Gaussian Counterpoise-style supermol)."""
    lines: list[str] = []
    for s, (x, y, z) in zip(frame.mm_symbols, frame.mm_coords_ang):
        el = _mm_elem_key(s)
        if el not in _CP_GHOST_MM_ELEMENTS:
            raise ValueError(
                f"CP supermol: unsupported MM element {s!r} "
                f"(expected one of {sorted(_CP_GHOST_MM_ELEMENTS)})"
            )
        # ``ghost-O`` / ``ghost-H`` / ``ghost-Na``: PySCF applies the basis without nuclear charge.
        # ``mol.ghost`` + global basis does NOT add AOs in many PySCF versions.
        lines.append(f"ghost-{el} {x:.12f} {y:.12f} {z:.12f}")
    return lines


def build_cp_supermol_mol(frame: ScfFrame, cfg: ScfEmbedConfig) -> gto.Mole:
    """
    QM + ghost basis on MM O/H/C/N/Na/K/Cl (no MM point charges).

    Matches the usual Gaussian CP supermol: real QM atoms plus ghost shell sites at
    the same positions with the same basis (e.g. 6-31+G*).
    """
    qm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.qm_symbols, frame.qm_coords_ang)
    ]
    ghost_lines = _mm_ghost_atom_lines(frame)
    mol = gto.Mole()
    mol.atom = "\n".join(qm_lines + ghost_lines)
    _apply_mol_cfg(mol, cfg)
    mol.build()

    mol_qm = _build_mol(frame, cfg)
    if int(mol.nao) <= int(mol_qm.nao) + 5:
        raise RuntimeError(
            f"CP supermol nao={mol.nao} ≈ QM-only ({mol_qm.nao}); ghost basis not active. "
            "Check ghost-O/ghost-H/ghost-Na/ghost-Cl atom labels."
        )
    return mol


def run_gas_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None) -> dft.rks.RKS:
    """Gas-phase QM SCF: same geometry as Coo QM block, no MM charges or ghost."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mol = _build_mol(frame, cfg)
    mf = _run_rks(mol, cfg)
    if not mf.converged:
        raise RuntimeError("gas-phase SCF did not converge")
    return mf


def run_emb0_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None) -> dft.rks.RKS:
    """Emb0 SCF: QM + MM point charges, no Gaussian repulsion."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
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


def run_cp_supermol_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None) -> dft.rks.RKS:
    """
    CP fragment A+B (Gaussian CP supermol job): real QM + ghost MM basis.

    This is **one** SCF (ρ_AB total). Full Boys–Bernardi **energy** CP also needs
    monomer+ghost jobs; see ``run_cp_fragment_b_mf``.
    """
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mol = build_cp_supermol_mol(frame, cfg)
    mf = _run_rks(mol, cfg)
    if not mf.converged:
        raise RuntimeError("CP supermol SCF did not converge")
    return mf


def _qm_ghost_atom_lines(frame: ScfFrame) -> list[str]:
    lines: list[str] = []
    for s, (x, y, z) in zip(frame.qm_symbols, frame.qm_coords_ang):
        el = s.strip()[0].upper()
        lines.append(f"ghost-{el} {x:.12f} {y:.12f} {z:.12f}")
    return lines


def build_cp_fragment_b_mol(frame: ScfFrame, cfg: ScfEmbedConfig) -> gto.Mole:
    """
    CP fragment B with ghost A: **real** MM waters + ghost QM basis on ala.

    Matches the monomer side of Boys–Bernardi (partner ghost basis on A).
    """
    mm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.mm_symbols, frame.mm_coords_ang)
    ]
    ghost_qm = _qm_ghost_atom_lines(frame)
    mol = gto.Mole()
    mol.atom = "\n".join(mm_lines + ghost_qm)
    _apply_mol_cfg(mol, cfg, charge=cp_mm_formal_charge(frame))
    mol.build()
    return mol


def run_cp_fragment_b_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None) -> dft.rks.RKS:
    """SCF on real shell waters + ghost QM (CP monomer B with ghost A)."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mol = build_cp_fragment_b_mol(frame, cfg)
    mf = _run_rks(mol, cfg)
    if not mf.converged:
        raise RuntimeError("CP fragment-B (waters+ghost QM) SCF did not converge")
    return mf


def build_mm_mol(frame: ScfFrame, cfg: ScfEmbedConfig) -> gto.Mole:
    """Real MM shell (O/H/Na/K …; CP monomer B, no ghost)."""
    mm_lines = [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(frame.mm_symbols, frame.mm_coords_ang)
    ]
    mol = gto.Mole()
    mol.atom = "\n".join(mm_lines)
    _apply_mol_cfg(mol, cfg, charge=cp_mm_formal_charge(frame))
    mol.build()
    return mol


def _mm_water_triplets(frame: ScfFrame) -> list[tuple[int, int, int]]:
    """O,H,H index triplets in ``frame.mm_*`` (Coo layout)."""
    out: list[tuple[int, int, int]] = []
    i = 0
    n = len(frame.mm_symbols)
    while i + 2 < n:
        els = [frame.mm_symbols[j].strip()[0].upper() for j in (i, i + 1, i + 2)]
        if els == ["O", "H", "H"]:
            out.append((i, i + 1, i + 2))
            i += 3
        else:
            i += 1
    return out


def _mm_subframe(frame: ScfFrame, idx: tuple[int, ...]) -> ScfFrame:
    ii = [int(j) for j in idx]
    return ScfFrame(
        qm_symbols=frame.qm_symbols,
        qm_coords_ang=frame.qm_coords_ang,
        mm_symbols=tuple(frame.mm_symbols[j] for j in ii),
        mm_coords_ang=np.asarray(frame.mm_coords_ang[ii], dtype=np.float64),
        mm_charges=np.asarray(frame.mm_charges[ii], dtype=np.float64),
    )


def _mm_ion_site_indices(frame: ScfFrame) -> list[int]:
    """Indices of lone Na/K/Cl ion sites (not part of O,H,H waters)."""
    return [i for i, s in enumerate(frame.mm_symbols) if _mm_elem_key(s) in ("Na", "K", "Cl")]


def mm_monomer_energy_per_water_sum_hartree(frame: ScfFrame, cfg: ScfEmbedConfig) -> float:
    """
    Σ_i E(monomer_i) — each water or ion as its own SCF.

    Waters: O,H,H triplets.  Ions: single Na/K sites (uses ``cp_mm_formal_charge`` per site).
    If the MM block has only ions (no waters), this equals ``run_mm_mf`` on the full fragment.
    """
    trips = _mm_water_triplets(frame)
    ions = _mm_ion_site_indices(frame)
    if not trips and not ions:
        mf = run_mm_mf(frame, cfg)
        return float(_mf_total_energy_hartree(mf, cfg))
    total = 0.0
    for trip in trips:
        sub = _mm_subframe(frame, trip)
        mf = run_mm_mf(sub, cfg)
        total += _mf_total_energy_hartree(mf, cfg)
    for ia in ions:
        sub = _mm_subframe(frame, (ia,))
        mf = run_mm_mf(sub, cfg)
        total += _mf_total_energy_hartree(mf, cfg)
    return float(total)


def run_mm_mf(frame: ScfFrame, cfg: ScfEmbedConfig | None = None) -> dft.rks.RKS:
    """SCF on real shell waters only (CP monomer B)."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mol = build_mm_mol(frame, cfg)
    mf = _run_rks(mol, cfg)
    if not mf.converged:
        raise RuntimeError("CP monomer-B (MM only) SCF did not converge")
    return mf


@dataclass(frozen=True)
class CpBsseEnergyResult:
    """
    Boys–Bernardi counterpoise helper energies (legacy field names preserved in NPZ).

    E_int^CP4 = E(A+ghostB) + E(ghostA+B) − E(A) − E(B)   [stored as e_int_cp_hartree]
    E_int^raw = E(real supermol) − E(A) − E(B)             [e_int_raw_hartree]
    E_full_QM = E_int^raw − E_int^CP4                      [used as bsse_kcal in ref npz]

    Training target: ``(E_int^raw − E_int^CP4) − E_int^Emb0`` = ΔE = E_full_QM − E_Emb0.
    """

    e_ab_hartree: float
    e_b_ga_hartree: float
    e_a_hartree: float
    e_b_hartree: float
    e_cluster_hartree: float
    e_int_cp_hartree: float
    e_int_raw_hartree: float
    e_b_per_water_sum_hartree: float
    e_int_cp_pw_hartree: float
    mf_ab: Any
    mf_b_ga: Any
    mf_a: Any
    mf_b: Any
    mf_cluster: Any


def _mf_total_energy_hartree(mf: dft.rks.RKS, cfg: ScfEmbedConfig) -> float:
    e_tot, _, _ = _total_energy(mf, use_d3bj=bool(cfg.use_d3bj))
    return float(e_tot)


def run_cp_bsse_energy(
    frame: ScfFrame,
    cfg: ScfEmbedConfig | None = None,
    *,
    mf_ab: dft.rks.RKS | None = None,
    mf_b_ga: dft.rks.RKS | None = None,
    mf_a: dft.rks.RKS | None = None,
    mf_b: dft.rks.RKS | None = None,
    mf_cluster: dft.rks.RKS | None = None,
) -> CpBsseEnergyResult:
    """
    Gaussian Counterpoise=2 **corrected** ala–shell interaction (Boys–Bernardi).

    Steps 1–4: A+ghostB, ghostA+B, monomer A, monomer B (whole shell in one SCF).
    Also runs real supermol (cluster) to report raw complexation and verify the
    equivalent 3-term formula ``E(cluster) − E(A+ghostB) − E(ghostA+B)``.

    Training E0 label: ``(E_int^raw − E_int^CP) − E_int^Emb0``  (= BSSE correction − E_bg).
    """
    from scf_embed_cluster import run_cluster_supermol_mf

    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))
    mf_ab = mf_ab or run_cp_supermol_mf(frame, cfg)
    mf_b_ga = mf_b_ga or run_cp_fragment_b_mf(frame, cfg)
    mf_a = mf_a or run_gas_mf(frame, cfg)
    mf_b = mf_b or run_mm_mf(frame, cfg)
    mf_cl = mf_cluster or run_cluster_supermol_mf(frame, cfg)
    e_ab = _mf_total_energy_hartree(mf_ab, cfg)
    e_b_ga = _mf_total_energy_hartree(mf_b_ga, cfg)
    e_a = _mf_total_energy_hartree(mf_a, cfg)
    e_b = _mf_total_energy_hartree(mf_b, cfg)
    e_cl = _mf_total_energy_hartree(mf_cl, cfg)
    e_int = float(e_ab + e_b_ga - e_a - e_b)
    e_int_raw = float(e_cl - e_a - e_b)
    e_int_g3 = float(e_cl - e_ab - e_b_ga)
    mm_q = int(cp_mm_formal_charge(frame))
    if mm_q == 0 and abs(e_int - e_int_g3) > 1e-4:
        print(
            f"  WARNING: CP 4-term ({hartree_to_kcal(e_int):+.4f}) vs 3-term "
            f"({hartree_to_kcal(e_int_g3):+.4f}) kcal/mol differ — check ghost/cluster build",
            flush=True,
        )
    elif mm_q != 0 and abs(e_int - e_int_g3) > 1e-3:
        print(
            f"  NOTE: charged MM (q={mm_q:+d}): CP 4-term ({hartree_to_kcal(e_int):+.4f}) vs "
            f"3-term ({hartree_to_kcal(e_int_g3):+.4f}) kcal/mol (expected to differ for ions)",
            flush=True,
        )
    e_b_pw = mm_monomer_energy_per_water_sum_hartree(frame, cfg)
    e_int_pw = float(e_ab + e_b_ga - e_a - e_b_pw)
    return CpBsseEnergyResult(
        e_ab_hartree=e_ab,
        e_b_ga_hartree=e_b_ga,
        e_a_hartree=e_a,
        e_b_hartree=e_b,
        e_cluster_hartree=e_cl,
        e_int_cp_hartree=e_int,
        e_int_raw_hartree=e_int_raw,
        e_b_per_water_sum_hartree=e_b_pw,
        e_int_cp_pw_hartree=e_int_pw,
        mf_ab=mf_ab,
        mf_b_ga=mf_b_ga,
        mf_a=mf_a,
        mf_b=mf_b,
        mf_cluster=mf_cl,
    )


def _envelope_potential_ao(
    mf: dft.rks.RKS,
    centers_bohr: np.ndarray,
    amps: np.ndarray,
    *,
    envelope_cfg=None,
    alpha_bohr2: float = 0.6,
    width_per_center: np.ndarray | None = None,
    C_per_center: np.ndarray | None = None,
    mm_symbols: tuple[str, ...] | list[str] | None = None,
    cone_axes_unit: np.ndarray | None = None,
    cone: ConeRepAng | None = None,
    h_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    V_rep(r) = Σ_J A_J · C_J · envelope_J(|r-R_J|) on SCF grid.

    ``amps`` are A_J [Hartree]; C_J is separate (not folded into A_J).
    """
    from embr_envelope import ENVELOPE_EXP, MmhEnvelopeConfig

    mol = mf.mol
    if envelope_cfg is None:
        envelope_cfg = MmhEnvelopeConfig.uniform(float(alpha_bohr2))
    if centers_bohr.size == 0 or float(np.max(np.abs(amps))) == 0.0:
        return np.zeros((mol.nao, mol.nao), dtype=np.float64)

    use_cone = cone is not None and not cone.is_isotropic()
    grids = mf_lebedev_grid(mf)
    coords = grids.coords
    weights = grids.weights
    v_r = np.zeros(coords.shape[0], dtype=np.float64)
    centers_bohr = np.asarray(centers_bohr, dtype=np.float64).reshape(-1, 3)
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    width_pc = None
    if width_per_center is not None:
        width_pc = np.asarray(width_per_center, dtype=np.float64).reshape(-1)
        if width_pc.size != centers_bohr.shape[0]:
            raise ValueError("width_per_center length mismatch")
    C_pc = None
    if C_per_center is not None:
        C_pc = np.asarray(C_per_center, dtype=np.float64).reshape(-1)
        if C_pc.size != centers_bohr.shape[0]:
            raise ValueError("C_per_center length mismatch")

    for j, (R, a) in enumerate(zip(centers_bohr, amps)):
        if abs(float(a)) < 1e-16:
            continue
        sym = None
        if mm_symbols is not None and j < len(mm_symbols):
            sym = str(mm_symbols[j])
        Cj = float(C_pc[j]) if C_pc is not None else (
            envelope_cfg.C_for_symbol(sym) if sym is not None else math.exp(0.0)
        )
        dr = coords - R
        r2 = np.sum(dr * dr, axis=1)
        r = np.sqrt(np.maximum(r2, 0.0))
        if sym is not None and envelope_cfg is not None and envelope_cfg.has_exp_sum(sym):
            kern = envelope_cfg.envelope_on_grid(sym, r_bohr=r, r2_bohr=r2)
        else:
            if width_pc is not None:
                wj = float(width_pc[j])
            elif sym is not None:
                wj = float(envelope_cfg.width_for_symbol(sym))
            else:
                wj = float(alpha_bohr2)
            if envelope_cfg.kind == ENVELOPE_EXP:
                r_ang = r * float(lib.param.BOHR)
                kern = Cj * np.exp(-wj * r_ang)
            else:
                kern = Cj * np.exp(-wj * r2)
        if use_cone and h_mask is not None and bool(h_mask[j]) and cone_axes_unit is not None:
            axis = np.asarray(cone_axes_unit[j], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(axis)) > 0.5:
                dn = np.linalg.norm(dr, axis=1)
                dn = np.maximum(dn, 1e-30)
                cos_th = np.sum(dr * axis.reshape(1, 3), axis=1) / dn
                kern = kern * cone_angular_weight_from_cos(
                    cos_th,
                    theta1_rad=cone.theta1_rad,
                    theta2_rad=cone.theta2_rad,
                )
        v_r += float(a) * kern

    ao = mf_numint(mf).eval_ao(mol, coords, deriv=0)
    v_ao = np.einsum("k,k,ki,kj->ij", weights, v_r, ao, ao)
    v_ao = (v_ao + v_ao.T) * 0.5
    return v_ao


def _gaussian_potential_ao(
    mf: dft.rks.RKS,
    centers_bohr: np.ndarray,
    amps: np.ndarray,
    alpha: float,
    *,
    alpha_per_center: np.ndarray | None = None,
    C_per_center: np.ndarray | None = None,
    mm_symbols: tuple[str, ...] | list[str] | None = None,
    cone_axes_unit: np.ndarray | None = None,
    cone: ConeRepAng | None = None,
    h_mask: np.ndarray | None = None,
    envelope_cfg=None,
) -> np.ndarray:
    """Gaussian V = Σ A·C·exp(-α r²). Legacy gauss+lnC=0 uses the original per-site α loop."""
    if envelope_cfg is not None and not envelope_cfg.is_legacy_gaussian():
        return _envelope_potential_ao(
            mf,
            centers_bohr,
            amps,
            envelope_cfg=envelope_cfg,
            alpha_bohr2=float(alpha),
            width_per_center=alpha_per_center,
            C_per_center=C_per_center,
            mm_symbols=mm_symbols,
            cone_axes_unit=cone_axes_unit,
            cone=cone,
            h_mask=h_mask,
        )

    mol = mf.mol
    if centers_bohr.size == 0 or float(np.max(np.abs(amps))) == 0.0:
        return np.zeros((mol.nao, mol.nao), dtype=np.float64)

    use_cone = cone is not None and not cone.is_isotropic()
    grids = mf_lebedev_grid(mf)
    coords = grids.coords
    weights = grids.weights
    v_r = np.zeros(coords.shape[0], dtype=np.float64)
    centers_bohr = np.asarray(centers_bohr, dtype=np.float64).reshape(-1, 3)
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    apc = None
    if alpha_per_center is not None:
        apc = np.asarray(alpha_per_center, dtype=np.float64).reshape(-1)
        if apc.size != centers_bohr.shape[0]:
            raise ValueError("alpha_per_center length mismatch")
    Cpc = None
    if C_per_center is not None:
        Cpc = np.asarray(C_per_center, dtype=np.float64).reshape(-1)
        if Cpc.size != centers_bohr.shape[0]:
            raise ValueError("C_per_center length mismatch")

    for j, (R, a) in enumerate(zip(centers_bohr, amps)):
        if abs(float(a)) < 1e-16:
            continue
        aj = float(apc[j]) if apc is not None else float(alpha)
        Cj = float(Cpc[j]) if Cpc is not None else 1.0
        dr = coords - R
        r2 = np.sum(dr * dr, axis=1)
        kern = Cj * np.exp(-aj * r2)
        if use_cone and h_mask is not None and bool(h_mask[j]) and cone_axes_unit is not None:
            axis = np.asarray(cone_axes_unit[j], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(axis)) > 0.5:
                dn = np.linalg.norm(dr, axis=1)
                dn = np.maximum(dn, 1e-30)
                cos_th = np.sum(dr * axis.reshape(1, 3), axis=1) / dn
                kern = kern * cone_angular_weight_from_cos(
                    cos_th,
                    theta1_rad=cone.theta1_rad,
                    theta2_rad=cone.theta2_rad,
                )
        v_r += float(a) * kern

    ao = mf_numint(mf).eval_ao(mol, coords, deriv=0)
    v_ao = np.einsum("k,k,ki,kj->ij", weights, v_r, ao, ao)
    v_ao = (v_ao + v_ao.T) * 0.5
    return v_ao


def _mm_h_mask_from_symbols(mm_symbols: tuple[str, ...] | list[str], n_centers: int) -> np.ndarray:
    mask = np.zeros(int(n_centers), dtype=bool)
    for i, sym in enumerate(mm_symbols):
        if i >= int(n_centers):
            break
        if str(sym).strip()[0].upper() == "H":
            mask[i] = True
    return mask


def _attach_gaussian_repulsion_amps(
    mf: dft.rks.RKS,
    mm_coords_ang: np.ndarray,
    amp_mm_hartree: np.ndarray,
    *,
    alpha_bohr2: float,
    alpha_per_center: np.ndarray | None = None,
    C_per_center: np.ndarray | None = None,
    envelope_cfg=None,
    cone: ConeRepAng | None = None,
    cone_axes_ang: np.ndarray | None = None,
    mm_symbols: tuple[str, ...] | list[str] | None = None,
) -> dft.rks.RKS:
    """Static one-electron repulsion V = Σ A_J C_J envelope_J (full MM list)."""
    centers_bohr = np.asarray(mm_coords_ang, dtype=np.float64) / lib.param.BOHR
    amps = np.asarray(amp_mm_hartree, dtype=np.float64).reshape(-1)
    if amps.size != centers_bohr.shape[0]:
        raise ValueError(f"amp_mm length {amps.size} != n_mm {centers_bohr.shape[0]}")
    cone_axes_unit = None
    h_mask = None
    if cone is not None and not cone.is_isotropic():
        if cone_axes_ang is None:
            raise ValueError("cone_axes_ang required for non-isotropic cone repulsion")
        cone_axes_unit = np.asarray(cone_axes_ang, dtype=np.float64)
        if mm_symbols is None:
            raise ValueError("mm_symbols required for cone repulsion on H sites")
        h_mask = _mm_h_mask_from_symbols(mm_symbols, centers_bohr.shape[0])

    def _v_rep() -> np.ndarray:
        return _gaussian_potential_ao(
            mf,
            centers_bohr,
            amps,
            float(alpha_bohr2),
            alpha_per_center=alpha_per_center,
            C_per_center=C_per_center,
            mm_symbols=mm_symbols,
            envelope_cfg=envelope_cfg,
            cone_axes_unit=cone_axes_unit,
            cone=cone,
            h_mask=h_mask,
        )

    old_get_hcore = mf.get_hcore

    def get_hcore(mol=None):
        h = old_get_hcore(mol)
        return h + _v_rep()

    mf.get_hcore = get_hcore
    return mf


def envelope_repulsion_ao(
    mf: dft.rks.RKS,
    mm_coords_ang: np.ndarray,
    amp_mm_hartree: np.ndarray,
    *,
    envelope_cfg,
    mm_symbols: tuple[str, ...] | list[str],
    cone: ConeRepAng | None = None,
    cone_axes_ang: np.ndarray | None = None,
) -> np.ndarray:
    """One-electron V_rep AO matrix: Σ A_J C_J envelope_J."""
    centers_bohr = np.asarray(mm_coords_ang, dtype=np.float64) / lib.param.BOHR
    amps = np.asarray(amp_mm_hartree, dtype=np.float64).reshape(-1)
    if amps.size != centers_bohr.shape[0]:
        raise ValueError(f"amp_mm length {amps.size} != n_mm {centers_bohr.shape[0]}")
    width_pc = envelope_cfg.width_per_frame_symbols(mm_symbols)
    C_pc = envelope_cfg.C_per_frame_symbols(mm_symbols)
    cone_axes_unit = None
    h_mask = None
    if cone is not None and not cone.is_isotropic():
        if cone_axes_ang is None:
            raise ValueError("cone_axes_ang required for non-isotropic cone repulsion")
        cone_axes_unit = np.asarray(cone_axes_ang, dtype=np.float64)
        h_mask = _mm_h_mask_from_symbols(mm_symbols, centers_bohr.shape[0])
    return _envelope_potential_ao(
        mf,
        centers_bohr,
        amps,
        envelope_cfg=envelope_cfg,
        width_per_center=width_pc,
        C_per_center=C_pc,
        mm_symbols=mm_symbols,
        cone_axes_unit=cone_axes_unit,
        cone=cone,
        h_mask=h_mask,
    )


def _attach_gaussian_repulsion(
    mf: dft.rks.RKS,
    mm_coords_ang: np.ndarray,
    mm_symbols: tuple[str, ...],
    rep: GaussRepParams,
) -> dft.rks.RKS:
    """Static one-electron repulsion: add to hcore only (do not patch get_veff)."""
    centers_bohr = np.asarray(mm_coords_ang, dtype=np.float64) / lib.param.BOHR
    amps = rep.amplitudes_for_mm(mm_symbols)

    def _v_gauss() -> np.ndarray:
        return _gaussian_potential_ao(mf, centers_bohr, amps, float(rep.alpha_bohr2))

    old_get_hcore = mf.get_hcore

    def get_hcore(mol=None):
        h = old_get_hcore(mol)
        return h + _v_gauss()

    mf.get_hcore = get_hcore
    return mf


def _run_rks(
    mol: gto.Mole,
    cfg: ScfEmbedConfig,
    *,
    mm_coords_ang: np.ndarray | None = None,
    mm_charges: np.ndarray | None = None,
    rep: GaussRepParams | None = None,
    mm_symbols: tuple[str, ...] = (),
    amp_mm_hartree: np.ndarray | None = None,
    alpha_bohr2: float | None = None,
) -> Any:
    mf = make_mean_field(mol, cfg)

    if mm_coords_ang is not None and mm_charges is not None and len(mm_charges) > 0:
        mf = qmmm.mm_charge(mf, mm_coords_ang, mm_charges, unit=cfg.unit)

    if amp_mm_hartree is not None:
        if mm_coords_ang is None:
            raise ValueError("amp_mm_hartree requires mm_coords_ang")
        al = float(rep.alpha_bohr2 if rep is not None else alpha_bohr2 if alpha_bohr2 is not None else 0.6)
        if float(np.max(np.abs(amp_mm_hartree))) <= 0.0:
            pass
        else:
            mf = _attach_gaussian_repulsion_amps(
                mf, mm_coords_ang, amp_mm_hartree, alpha_bohr2=al
            )
    elif rep is not None and (abs(rep.amp_o_hartree) + abs(rep.amp_h_hartree)) > 0.0:
        mf = _attach_gaussian_repulsion(mf, mm_coords_ang, mm_symbols, rep)

    mf.kernel()
    return mf


def gaussian_repulsion_ao(
    mf: dft.rks.RKS,
    mm_coords_ang: np.ndarray,
    amp_mm_hartree: np.ndarray,
    *,
    alpha_bohr2: float,
    envelope_cfg=None,
    mm_symbols: tuple[str, ...] | list[str] | None = None,
    cone: ConeRepAng | None = None,
    cone_axes_ang: np.ndarray | None = None,
) -> np.ndarray:
    """One-electron repulsion AO matrix (legacy Gaussian or envelope_cfg)."""
    if envelope_cfg is not None and mm_symbols is not None:
        if not envelope_cfg.is_legacy_gaussian():
            return envelope_repulsion_ao(
                mf,
                mm_coords_ang,
                amp_mm_hartree,
                envelope_cfg=envelope_cfg,
                mm_symbols=mm_symbols,
                cone=cone,
                cone_axes_ang=cone_axes_ang,
            )
    centers_bohr = np.asarray(mm_coords_ang, dtype=np.float64) / lib.param.BOHR
    amps = np.asarray(amp_mm_hartree, dtype=np.float64).reshape(-1)
    if amps.size != centers_bohr.shape[0]:
        raise ValueError(f"amp_mm length {amps.size} != n_mm {centers_bohr.shape[0]}")
    cone_axes_unit = None
    h_mask = None
    if cone is not None and not cone.is_isotropic():
        if cone_axes_ang is None:
            raise ValueError("cone_axes_ang required for non-isotropic cone repulsion")
        cone_axes_unit = np.asarray(cone_axes_ang, dtype=np.float64)
        if mm_symbols is None:
            raise ValueError("mm_symbols required for cone repulsion on H sites")
        h_mask = _mm_h_mask_from_symbols(mm_symbols, centers_bohr.shape[0])
    return _gaussian_potential_ao(
        mf,
        centers_bohr,
        amps,
        float(alpha_bohr2),
        mm_symbols=mm_symbols,
        envelope_cfg=envelope_cfg,
        cone_axes_unit=cone_axes_unit,
        cone=cone,
        h_mask=h_mask,
    )


def amp_mm_for_o_h_sites(frame: ScfFrame, amp_mm: np.ndarray) -> np.ndarray:
    """Expand per-kernel-site ``amp_mm`` (frame.mm_symbols kernel order) to full MM vector."""
    from embr_envelope import is_kernel_mm_symbol

    a_in = np.asarray(amp_mm, dtype=np.float64).reshape(-1)
    out = np.zeros(len(frame.mm_symbols), dtype=np.float64)
    j = 0
    for i, sym in enumerate(frame.mm_symbols):
        if not is_kernel_mm_symbol(sym):
            continue
        if j >= a_in.size:
            raise ValueError("amp_mm shorter than number of MM kernel sites")
        out[i] = float(a_in[j])
        j += 1
    if j != a_in.size:
        raise ValueError(f"amp_mm length {a_in.size} != n MM kernel sites {j}")
    return out


def normalize_amp_mm_full_frame(frame: ScfFrame, amp_mm: np.ndarray) -> np.ndarray:
    """
    Return amps aligned with ``len(frame.mm_symbols)``.

    Peratom labels may already store full-frame amps; only expand
    when the vector is compact (one entry per kernel site).
    """
    a = np.asarray(amp_mm, dtype=np.float64).reshape(-1)
    if a.size == len(frame.mm_symbols):
        return a
    return amp_mm_for_o_h_sites(frame, a)


def _mol_atom_coords_ang(mol: gto.Mole) -> np.ndarray:
    """Cartesian coordinates (Å) for all atoms in ``mol``."""
    try:
        return np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64).reshape(-1, 3)
    except TypeError:
        pass
    c = np.asarray(mol.atom_coords(), dtype=np.float64).reshape(-1, 3)
    u = str(getattr(mol, "unit", "Angstrom")).lower()
    if u.startswith("ang") or u == "a":
        return c
    return c * lib.param.BOHR


def _aoslice_ao_range(aoslices: Any, ia: int) -> tuple[int, int]:
    """
    Extract (start_ao, stop_ao) from PySCF ``aoslice_by_atom()`` row.

    Each row is (start-shell, stop-shell, start-AO, stop-AO).  Older code
    mistakenly used shell columns 0:2, under-counting QM AOs (~50 vs ~98).
    """
    row = aoslices[ia]
    if len(row) >= 4:
        return int(row[2]), int(row[3])
    return int(row[0]), int(row[1])


def qm_ao_mask_for_cp_supermol(
    mf: dft.rks.RKS,
    frame: ScfFrame,
    *,
    tol_ang: float = 0.08,
) -> np.ndarray:
    """
    Boolean mask over AO indices on **real QM atoms** in a CP supermol.

    Match atoms by coordinates (Å).  Falls back to the first ``n_qm`` atoms
    when PySCF preserves Coo input order (usual CP supermol build).
    """
    mol = mf.mol
    qm_coords = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    n_qm = int(qm_coords.shape[0])
    aoslices = mol.aoslice_by_atom()
    mask = np.zeros(int(mol.nao), dtype=bool)
    coords_ang = _mol_atom_coords_ang(mol)

    qm_atom_idx: list[int] = []
    used: set[int] = set()
    for q in qm_coords:
        dists = np.linalg.norm(coords_ang - q.reshape(1, 3), axis=1)
        for ia in np.argsort(dists):
            ii = int(ia)
            if ii in used:
                continue
            if float(dists[ii]) > float(tol_ang):
                break
            used.add(ii)
            qm_atom_idx.append(ii)
            break

    if len(qm_atom_idx) != n_qm:
        # CP supermol is built QM-first in ``build_cp_supermol_mol``.
        qm_atom_idx = list(range(min(n_qm, int(mol.natm))))

    for ia in qm_atom_idx:
        p0, p1 = _aoslice_ao_range(aoslices, ia)
        mask[p0:p1] = True
    if not np.any(mask):
        raise RuntimeError("QM AO mask empty in CP supermol")
    return mask


def project_dm_to_qm_aos(dm: np.ndarray, qm_ao_mask: np.ndarray) -> np.ndarray:
    """Keep only the QM×QM block of ``dm`` (ghost AO rows/cols zeroed)."""
    dm = np.asarray(dm, dtype=np.float64)
    mask = np.asarray(qm_ao_mask, dtype=bool).reshape(-1)
    if dm.shape[0] != mask.size:
        raise ValueError(f"dm shape {dm.shape} != nao {mask.size}")
    idx = np.flatnonzero(mask)
    out = np.zeros_like(dm)
    out[np.ix_(idx, idx)] = dm[np.ix_(idx, idx)]
    return out


def cp_supermol_qm_dm(mf: dft.rks.RKS, frame: ScfFrame) -> np.ndarray:
    """Density matrix with only QM×QM AO block (ghost rows/cols zeroed)."""
    mask = qm_ao_mask_for_cp_supermol(mf, frame)
    return project_dm_to_qm_aos(mf.make_rdm1(), mask)


def cp_supermol_qm_ghost_dm(mf: dft.rks.RKS, frame: ScfFrame) -> np.ndarray:
    """Density matrix with QM+ghost AO block (pure ghost×ghost zeroed)."""
    mol = mf.mol
    qm_mask = qm_ao_mask_for_cp_supermol(mf, frame)
    aoslices = mol.aoslice_by_atom()
    mm_coords = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    coords_ang = _mol_atom_coords_ang(mol)
    ghost_mask = np.zeros(int(mol.nao), dtype=bool)
    qm_atoms = set()
    for ia in range(int(mol.natm)):
        p0, p1 = _aoslice_ao_range(aoslices, ia)
        if np.any(qm_mask[p0:p1]):
            qm_atoms.add(ia)
    ghost_mask[:] = False
    for ia in range(int(mol.natm)):
        if ia in qm_atoms:
            continue
        R = coords_ang[ia]
        if float(np.min(np.linalg.norm(mm_coords - R.reshape(1, 3), axis=1))) > 0.08:
            continue
        p0, p1 = _aoslice_ao_range(aoslices, ia)
        ghost_mask[p0:p1] = True
    both = qm_mask | ghost_mask
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    idx = np.flatnonzero(both)
    out = np.zeros_like(dm)
    out[np.ix_(idx, idx)] = dm[np.ix_(idx, idx)]
    return out


def mf_numint(mf: Any):
    """AO evaluator for ρ; RKS has ``_numint``, RHF/QMMMRHF often does not."""
    ni = getattr(mf, "_numint", None) or getattr(mf, "numint", None)
    if ni is not None:
        return ni
    from pyscf.dft import numint

    return numint.NumInt()


def mf_lebedev_grid(mf: Any, *, level: int = 3):
    """Lebedev grid on ``mf.mol``; RHF objects may lack ``grids`` until built."""
    grids = getattr(mf, "grids", None)
    if grids is None:
        from pyscf.dft import gen_grid

        grids = gen_grid.Grids(mf.mol)
        grids.level = int(level)
        mf.grids = grids
    coords = getattr(grids, "coords", None)
    if coords is None or int(np.asarray(coords).size) == 0:
        grids.build()
    return grids


def eval_density_from_dm(mf: Any, dm: np.ndarray, points_ang: np.ndarray) -> np.ndarray:
    """ρ from density matrix ``dm`` at Cartesian points (Å)."""
    mol = mf.mol
    points_bohr = np.asarray(points_ang, dtype=np.float64) / lib.param.BOHR
    if points_bohr.ndim == 1:
        points_bohr = points_bohr.reshape(1, -1)
    dm = np.asarray(dm, dtype=np.float64)
    ao = mf_numint(mf).eval_ao(mol, points_bohr, deriv=0)
    ao = np.atleast_2d(np.asarray(ao, dtype=np.float64))
    return np.einsum("ij,ki,kj->k", dm, ao, ao)


def eval_density_at_points(mf: Any, points_ang: np.ndarray) -> np.ndarray:
    """Electron density ρ(r) at Cartesian points (Å). Returns shape (n_points,)."""
    return eval_density_from_dm(mf, mf.make_rdm1(), points_ang)


def mm_centers(frame: ScfFrame, element: str) -> np.ndarray:
    """MM O or H coordinates (Å) from Coo shell."""
    el = element.strip()[0].upper()
    pts = [
        frame.mm_coords_ang[j]
        for j, s in enumerate(frame.mm_symbols)
        if s.strip()[0].upper() == el
    ]
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _sphere_unit_directions(n_sphere: int) -> np.ndarray:
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
    """Spherically averaged ρ(r) around each center; mean over all centers."""
    centers = np.asarray(centers_ang, dtype=np.float64).reshape(-1, 3)
    r_grid = np.arange(0.0, float(r_max) + 0.5 * float(dr), float(dr))
    if centers.size == 0:
        return r_grid, np.zeros(r_grid.size, dtype=np.float64)

    dirs = _sphere_unit_directions(n_sphere)
    prof = np.zeros(r_grid.size, dtype=np.float64)
    n_cent = int(centers.shape[0])
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
        if float(rad) < 1e-12:
            prof[ir] = float(np.mean(rho))
        else:
            prof[ir] = float(np.mean(rho.reshape(n_cent, n_sphere)))
    return r_grid, prof


def radial_rho_profiles_mm_oh(
    mf,
    dm: np.ndarray,
    frame: ScfFrame,
    *,
    r_max: float = 3.0,
    dr: float = 0.1,
    n_sphere: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ρ(r) at all MM O and H sites. Returns (r_ang, rho_o, rho_h)."""
    o_cent = mm_centers(frame, "O")
    h_cent = mm_centers(frame, "H")
    r_grid, rho_o = radial_rho_profile(
        mf, dm, o_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    _, rho_h = radial_rho_profile(
        mf, dm, h_cent, r_max=float(r_max), dr=float(dr), n_sphere=int(n_sphere)
    )
    return (
        r_grid,
        np.asarray(rho_o, dtype=np.float64),
        np.asarray(rho_h, dtype=np.float64),
    )


def mm_buffer_n_electrons(
    mf: dft.rks.RKS,
    frame: ScfFrame,
    *,
    inner_ang: float = 1.2,
    outer_ang: float = 5.0,
    n_radial: int = 8,
    n_sphere: int = 26,
) -> float:
    """
    Rough integral of ρ in MM buffer: outside inner_ang from any QM atom, within outer_ang.
    Uses Lebedev spheres on each MM site (diagnostic, not high-accuracy integration).
    """
    qm = frame.qm_coords_ang
    mm = frame.mm_coords_ang
    radii = np.linspace(float(inner_ang), float(outer_ang), int(n_radial))
    # Fibonacci sphere directions
    golden = np.pi * (3.0 - np.sqrt(5.0))
    dirs: list[np.ndarray] = []
    for k in range(int(n_sphere)):
        y = 1.0 - (2.0 * k + 1.0) / float(n_sphere)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        phi = golden * k
        dirs.append(np.array([np.cos(phi) * r, y, np.sin(phi) * r], dtype=np.float64))
    dirs_arr = np.asarray(dirs)

    pts: list[np.ndarray] = []
    for R in mm:
        d_qm = np.linalg.norm(qm - R, axis=1)
        if float(np.min(d_qm)) > float(outer_ang):
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
    rho = eval_density_at_points(mf, points)
    # crude: average ρ * shell volume proxy
    n_pts = max(1, points.shape[0])
    shell_vol = 4.0 / 3.0 * np.pi * (float(outer_ang) ** 3 - float(inner_ang) ** 3)
    return float(np.sum(rho) / n_pts * shell_vol * 0.01)


@dataclass(frozen=True)
class GasEnergyCache:
    """Reuse gas-phase energy when scanning only embedding (Gauss θ) parameters."""

    e_gas_hartree: float
    e_gas_scf_hartree: float
    e_disp_gas_hartree: float


def run_embedding_scf_peratom(
    frame: ScfFrame,
    cfg: ScfEmbedConfig | None = None,
    *,
    amp_mm: np.ndarray,
    alpha_bohr2: float = 0.6,
    gas_cache: GasEnergyCache | None = None,
) -> ScfEmbedResult:
    """Full embedding SCF with per-O/H amplitudes on each MM site."""
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))

    amp_full = amp_mm_for_o_h_sites(frame, amp_mm)
    mol = _build_mol(frame, cfg)

    if gas_cache is None:
        mf_gas = _run_rks(mol, cfg)
        if not mf_gas.converged:
            raise RuntimeError("gas-phase SCF did not converge")
        e_gas, e_gas_scf, d_gas = _total_energy(mf_gas, use_d3bj=bool(cfg.use_d3bj))
    else:
        e_gas = float(gas_cache.e_gas_hartree)
        e_gas_scf = float(gas_cache.e_gas_scf_hartree)
        d_gas = float(gas_cache.e_disp_gas_hartree)

    mf_emb = _run_rks(
        mol,
        cfg,
        mm_coords_ang=frame.mm_coords_ang,
        mm_charges=frame.mm_charges,
        amp_mm_hartree=amp_full,
        alpha_bohr2=float(alpha_bohr2),
    )
    if not mf_emb.converged:
        raise RuntimeError("embedding SCF (per-atom V_rep) did not converge")

    e_emb, e_emb_scf, d_emb = _total_energy(mf_emb, use_d3bj=bool(cfg.use_d3bj))
    rho_mm = eval_density_at_points(mf_emb, frame.mm_coords_ang)
    n_buf = mm_buffer_n_electrons(mf_emb, frame)

    return ScfEmbedResult(
        e_total_hartree=e_emb,
        e_int_hartree=e_emb - e_gas,
        e_gas_hartree=e_gas,
        e_emb_scf_hartree=e_emb_scf,
        e_gas_scf_hartree=e_gas_scf,
        e_disp_emb_hartree=d_emb,
        e_disp_gas_hartree=d_gas,
        use_d3bj=bool(cfg.use_d3bj),
        converged=True,
        rho_mm=rho_mm,
        n_mm=n_buf,
        mf=mf_emb,
    )


def run_embedding_scf(
    frame: ScfFrame,
    cfg: ScfEmbedConfig | None = None,
    *,
    rep: GaussRepParams | None = None,
    gas_cache: GasEnergyCache | None = None,
) -> ScfEmbedResult:
    cfg = cfg or ScfEmbedConfig()
    if int(cfg.num_threads) > 0:
        set_pyscf_threads(int(cfg.num_threads))

    rep = rep or GaussRepParams(alpha_bohr2=0.6, amp_o_hartree=0.0, amp_h_hartree=0.0)
    mol = _build_mol(frame, cfg)

    if gas_cache is None:
        mf_gas = _run_rks(mol, cfg)
        if not mf_gas.converged:
            raise RuntimeError("gas-phase SCF did not converge")
        e_gas, e_gas_scf, d_gas = _total_energy(mf_gas, use_d3bj=bool(cfg.use_d3bj))
    else:
        e_gas = float(gas_cache.e_gas_hartree)
        e_gas_scf = float(gas_cache.e_gas_scf_hartree)
        d_gas = float(gas_cache.e_disp_gas_hartree)

    mf_emb = _run_rks(
        mol,
        cfg,
        mm_coords_ang=frame.mm_coords_ang,
        mm_charges=frame.mm_charges,
        rep=rep,
        mm_symbols=frame.mm_symbols,
    )
    if not mf_emb.converged:
        raise RuntimeError("embedding SCF did not converge")

    e_emb, e_emb_scf, d_emb = _total_energy(mf_emb, use_d3bj=bool(cfg.use_d3bj))
    rho_mm = eval_density_at_points(mf_emb, frame.mm_coords_ang)
    n_buf = mm_buffer_n_electrons(mf_emb, frame)

    return ScfEmbedResult(
        e_total_hartree=e_emb,
        e_int_hartree=e_emb - e_gas,
        e_gas_hartree=e_gas,
        e_emb_scf_hartree=e_emb_scf,
        e_gas_scf_hartree=e_gas_scf,
        e_disp_emb_hartree=d_emb,
        e_disp_gas_hartree=d_gas,
        use_d3bj=bool(cfg.use_d3bj),
        converged=True,
        rho_mm=rho_mm,
        n_mm=n_buf,
        mf=mf_emb,
    )


def hartree_to_kcal(x_hartree: float) -> float:
    return float(x_hartree) * 627.5094740631


def kcal_to_hartree(x_kcal: float) -> float:
    return float(x_kcal) / 627.5094740631
