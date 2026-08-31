"""
Dataset labels from manifest JSON (prefix + QM_atoms). No amino-acid registry.
"""
from __future__ import annotations

from typing import Any, Mapping


def resolve_dataset_label(ds: Mapping[str, Any]) -> str:
    """Stable id: prefix → legacy residue → qm{n}."""
    pref = str(ds.get("prefix") or "").strip()
    if pref:
        return pref.lower()
    res = str(ds.get("residue") or "").strip()
    if res:
        return res.lower()
    n_qm = resolve_dataset_n_qm(ds)
    return f"qm{n_qm}"


def resolve_dataset_n_qm(ds: Mapping[str, Any], *, residue: str | None = None) -> int:
    """QM region size from manifest (QM_atoms / n_qm). Required for unknown systems."""
    if ds.get("n_qm") is not None:
        return int(ds["n_qm"])
    if ds.get("QM_atoms") is not None:
        return int(ds["QM_atoms"])
    if ds.get("qm_atoms") is not None:
        return int(ds["qm_atoms"])
    key = str(residue if residue is not None else ds.get("residue", "")).strip().lower()
    if not key:
        raise KeyError("dataset needs QM_atoms (or n_qm); residue alone is not enough")
    if key in _LEGACY_N_QM:
        return int(_LEGACY_N_QM[key])
    raise KeyError(f"unknown dataset label {key!r}; set QM_atoms in manifest")


_LEGACY_N_QM = {"gly": 10, "ala": 13, "asp": 15, "lys": 25, "thr": 16}

# Legacy name→n_qm map for older manifests. Prefer meta.residue_to_id.
RESIDUES: dict[str, int] = dict(_LEGACY_N_QM)


def qm_fit_from_symbols(symbols: list[str]) -> tuple[int, ...]:
    """H → 0, C/N/O → 1 (legacy QM-E0 path; ksoft does not use this)."""
    out: list[int] = []
    for s in symbols:
        el = str(s).strip()
        if not el:
            raise ValueError("empty symbol in QM list")
        out.append(0 if el[0].upper() == "H" else 1)
    return tuple(out)


class _ResidueSpec:
    """Minimal shim for EmbR scripts that still call get_residue(name)."""

    def __init__(self, name: str, n_qm: int) -> None:
        self.name = str(name)
        self.n_qm = int(n_qm)
        self.qm_fitting_group = tuple(1 for _ in range(int(n_qm)))


def get_residue(name: str) -> _ResidueSpec:
    key = str(name).strip().lower()
    if key not in _LEGACY_N_QM:
        raise KeyError(f"unknown residue label {name!r}; pass --n-qm or set QM_atoms in manifest")
    return _ResidueSpec(key, _LEGACY_N_QM[key])
