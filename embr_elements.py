"""
MM site element integer codes (shared across cache, model, envelope).
"""
from __future__ import annotations

# Codes in mix_mmh npz ``mm_element`` rows (train / infer)
MMH_ELEM_H = 0
MMH_ELEM_NA = 1
MMH_ELEM_K = 2
MMH_ELEM_CL = 3
MMH_ELEM_O = 4
MMH_ELEM_C = 5
MMH_ELEM_N = 6

MMH_ELEM_NAMES: dict[int, str] = {
    MMH_ELEM_H: "H",
    MMH_ELEM_NA: "Na",
    MMH_ELEM_K: "K",
    MMH_ELEM_CL: "Cl",
    MMH_ELEM_O: "O",
    MMH_ELEM_C: "C",
    MMH_ELEM_N: "N",
}

# Codes in HF ref npz kernel arrays (O/H first, then ions, C/N)
MM_KERNEL_EL_O = 0
MM_KERNEL_EL_H = 1
MM_KERNEL_EL_NA = 2
MM_KERNEL_EL_K = 3
MM_KERNEL_EL_CL = 4
MM_KERNEL_EL_C = 5
MM_KERNEL_EL_N = 6

MM_KERNEL_EL_NAMES: dict[int, str] = {
    MM_KERNEL_EL_O: "O",
    MM_KERNEL_EL_H: "H",
    MM_KERNEL_EL_NA: "Na",
    MM_KERNEL_EL_K: "K",
    MM_KERNEL_EL_CL: "Cl",
    MM_KERNEL_EL_C: "C",
    MM_KERNEL_EL_N: "N",
}


def mmh_element_from_symbol(sym: str) -> int:
    el = str(sym).strip()
    if not el:
        raise ValueError("empty MM symbol")
    c = el[0].upper()
    if c == "H":
        return MMH_ELEM_H
    if el.upper().startswith("NA"):
        return MMH_ELEM_NA
    if c == "K":
        return MMH_ELEM_K
    if el.upper().startswith("CL"):
        return MMH_ELEM_CL
    if c == "O":
        return MMH_ELEM_O
    if c == "C":
        return MMH_ELEM_C
    if c == "N":
        return MMH_ELEM_N
    raise ValueError(f"unsupported MMH element {sym!r}")
