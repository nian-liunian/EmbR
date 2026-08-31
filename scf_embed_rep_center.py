"""
高斯排斥势 **中心位置**（与 MM 点电荷位置分离）。

点电荷始终落在真实 MM 核上；仅 ``V_rep = A exp(-α|r-R|²)`` 的 R 可改：

  on_nucleus  — R 在 MM O/H 核（默认，与旧行为一致）
  oh_bond     — Coo 里 MM 按 O,H,H 分组，键上点用 **原始核坐标** 算（不就地改坐标）

改中心位置后，请用 ``precompute_pert_ohbond`` 重算 k_j 并 ``fit_pert_peratom``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyscf import dft, qmmm

from scf_embed_io import ScfFrame
from scf_embed_pyscf import (
    ConeRepAng,
    ScfEmbedConfig,
    _attach_gaussian_repulsion_amps,
    _build_mol,
    amp_mm_for_o_h_sites,
    cone_axes_h_to_nearest_qm,
    normalize_amp_mm_full_frame,
)

RepCenterPlacement = Literal["on_nucleus", "oh_bond"]
REP_CENTER_PLACEMENTS: tuple[str, ...] = ("on_nucleus", "oh_bond")

_DEFAULT_OH_BOND_MAX_ANG = 1.35


@dataclass(frozen=True)
class RepCenterSpec:
    """高斯排斥峰中心（与 MM 点电荷位置可分离）。"""

    placement: RepCenterPlacement = "on_nucleus"
    oh_frac: float = 0.5
    bond_max_ang: float = _DEFAULT_OH_BOND_MAX_ANG

    def validate(self) -> None:
        if self.placement not in REP_CENTER_PLACEMENTS:
            raise ValueError(f"placement must be one of {REP_CENTER_PLACEMENTS}, got {self.placement!r}")

    @property
    def is_on_nucleus(self) -> bool:
        return self.placement == "on_nucleus"

    def label_zh(self) -> str:
        return placement_label_zh(self.placement, self.oh_frac)

    def meta_fields(self) -> dict[str, float | str]:
        return {
            "rep_center": str(self.placement),
            "rep_oh_frac": float(self.oh_frac),
            "rep_oh_bond_max_ang": float(self.bond_max_ang),
        }

    def build_coords_ang(self, frame: ScfFrame) -> np.ndarray:
        self.validate()
        return build_repulsion_center_coords_ang(
            frame,
            self.placement,
            oh_frac=float(self.oh_frac),
            bond_max_ang=float(self.bond_max_ang),
        )


def _element(sym: str) -> str:
    return sym.strip()[0].upper()


def _nearest_bonded_o_for_h(
    h_idx: int,
    mm_coords: np.ndarray,
    mm_symbols: tuple[str, ...],
    *,
    bond_max_ang: float,
) -> int | None:
    rh = mm_coords[int(h_idx)]
    best_j: int | None = None
    best_d = float("inf")
    for j, sym in enumerate(mm_symbols):
        if _element(sym) != "O":
            continue
        d = float(np.linalg.norm(mm_coords[j] - rh))
        if d <= float(bond_max_ang) and d < best_d:
            best_d = d
            best_j = int(j)
    return best_j


def _nearest_bonded_h_for_o(
    o_idx: int,
    mm_coords: np.ndarray,
    mm_symbols: tuple[str, ...],
    *,
    bond_max_ang: float,
) -> int | None:
    ro = mm_coords[int(o_idx)]
    best_j: int | None = None
    best_d = float("inf")
    for j, sym in enumerate(mm_symbols):
        if _element(sym) != "H":
            continue
        d = float(np.linalg.norm(mm_coords[j] - ro))
        if d <= float(bond_max_ang) and d < best_d:
            best_d = d
            best_j = int(j)
    return best_j


def _bond_point(ro: np.ndarray, rh: np.ndarray, frac: float) -> np.ndarray:
    """沿 O→H：frac=0 为 O 核，0.5 键中点，1 为 H 核。"""
    return np.asarray(ro, dtype=np.float64) + float(frac) * (np.asarray(rh, dtype=np.float64) - np.asarray(ro))


def _is_ohh_triplet(s0: str, s1: str, s2: str) -> bool:
    return _element(s0) == "O" and _element(s1) == "H" and _element(s2) == "H"


def _rep_centers_oh_bond_topology(
    coords_orig: np.ndarray,
    mm_symbols: tuple[str, ...],
    *,
    frac: float,
    bond_max_ang: float,
) -> tuple[np.ndarray, int]:
    """Coo 约定 O,H,H 逐水排列；全程用 ``coords_orig``，避免循环中改坐标。"""
    n_mm = int(len(mm_symbols))
    out = np.asarray(coords_orig, dtype=np.float64).reshape(-1, 3).copy()
    n_fallback = 0

    w = 0
    while w < n_mm:
        if w + 2 >= n_mm:
            for i in range(w, n_mm):
                el = _element(mm_symbols[i])
                if el not in ("O", "H"):
                    continue
                if el == "H":
                    oj = _nearest_bonded_o_for_h(
                        i, coords_orig, mm_symbols, bond_max_ang=bond_max_ang
                    )
                    if oj is None:
                        n_fallback += 1
                        continue
                    out[i] = _bond_point(coords_orig[oj], coords_orig[i], frac)
                else:
                    hj = _nearest_bonded_h_for_o(
                        i, coords_orig, mm_symbols, bond_max_ang=bond_max_ang
                    )
                    if hj is None:
                        n_fallback += 1
                        continue
                    out[i] = _bond_point(coords_orig[i], coords_orig[hj], frac)
            break

        s0, s1, s2 = mm_symbols[w], mm_symbols[w + 1], mm_symbols[w + 2]
        if not _is_ohh_triplet(s0, s1, s2):
            for i in (w, w + 1, w + 2):
                el = _element(mm_symbols[i])
                if el == "H":
                    oj = _nearest_bonded_o_for_h(
                        i, coords_orig, mm_symbols, bond_max_ang=bond_max_ang
                    )
                    if oj is None:
                        n_fallback += 1
                        continue
                    out[i] = _bond_point(coords_orig[oj], coords_orig[i], frac)
                elif el == "O":
                    hj = _nearest_bonded_h_for_o(
                        i, coords_orig, mm_symbols, bond_max_ang=bond_max_ang
                    )
                    if hj is None:
                        n_fallback += 1
                        continue
                    out[i] = _bond_point(coords_orig[i], coords_orig[hj], frac)
            w += 3
            continue

        ro = coords_orig[w]
        rh1 = coords_orig[w + 1]
        rh2 = coords_orig[w + 2]
        hnear = w + 1 if float(np.linalg.norm(rh1 - ro)) <= float(np.linalg.norm(rh2 - ro)) else w + 2
        out[w] = _bond_point(ro, coords_orig[hnear], frac)
        out[w + 1] = _bond_point(ro, rh1, frac)
        out[w + 2] = _bond_point(ro, rh2, frac)
        w += 3

    return out, n_fallback


def build_repulsion_center_coords_ang(
    frame: ScfFrame,
    placement: RepCenterPlacement = "on_nucleus",
    *,
    oh_frac: float = 0.5,
    bond_max_ang: float = _DEFAULT_OH_BOND_MAX_ANG,
) -> np.ndarray:
    """
    与 ``frame.mm_coords_ang`` 同序的排斥势中心 [Å]。

    ``oh_bond``：Coo MM 块按 **O,H,H** 分组（与 ``load_coo_frame`` 一致），
    每个 O/H 位 ``R = R_O + oh_frac * (R_H - R_O)``（O 位取较近的那条 O–H）；
    全程用原始核坐标，不在循环里就地改 ``coords``。
    """
    if placement not in REP_CENTER_PLACEMENTS:
        raise ValueError(f"placement must be one of {REP_CENTER_PLACEMENTS}, got {placement!r}")
    coords_orig = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    if placement == "on_nucleus":
        return coords_orig.copy()

    frac = float(np.clip(float(oh_frac), 0.0, 1.0))
    coords, n_fallback = _rep_centers_oh_bond_topology(
        coords_orig,
        frame.mm_symbols,
        frac=frac,
        bond_max_ang=float(bond_max_ang),
    )
    if n_fallback > 0:
        print(
            f"  警告: rep_center=oh_bond 有 {n_fallback} 个 MM 位未能按 O,H,H 拓扑配对"
            f"（或距最近 O/H > {float(bond_max_ang):g} Å），该位仍用核坐标"
        )
    return coords


def placement_label_zh(placement: RepCenterPlacement, oh_frac: float) -> str:
    if placement == "on_nucleus":
        return "排斥峰在 MM 核"
    return f"排斥峰在 O–H 键上（O→H 分数 {float(oh_frac):g}，1=H 核）"


def run_embed_theta_mf_rep(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    amp_mm: np.ndarray,
    *,
    alpha_bohr2: float,
    alpha_per_center: np.ndarray | None = None,
    envelope_cfg=None,
    rep_centers_ang: np.ndarray,
    cone: ConeRepAng | None = None,
) -> dft.rks.RKS:
    """嵌入 SCF：点电荷在真实 MM 核；高斯排斥在 ``rep_centers_ang``。"""
    amp_full = normalize_amp_mm_full_frame(frame, amp_mm)
    rep_centers = np.asarray(rep_centers_ang, dtype=np.float64).reshape(-1, 3)
    if rep_centers.shape[0] != len(frame.mm_symbols):
        raise ValueError("rep_centers_ang length must match n_mm")

    mol = _build_mol(frame, cfg)
    mf = dft.rks.RKS(mol)
    mf.xc = cfg.xc
    mf.conv_tol = float(cfg.conv_tol)
    mf.max_cycle = int(cfg.max_cycle)
    mf.verbose = int(cfg.verbose)

    mf = qmmm.mm_charge(mf, frame.mm_coords_ang, frame.mm_charges, unit=cfg.unit)
    if float(np.max(np.abs(amp_full))) > 0.0:
        cone_kw: dict = {}
        if cone is not None and not cone.is_isotropic():
            cone_kw = {
                "cone": cone,
                "cone_axes_ang": cone_axes_h_to_nearest_qm(frame),
                "mm_symbols": frame.mm_symbols,
            }
        C_pc = None
        alpha_pc = None
        eff_env = None
        if envelope_cfg is not None and not envelope_cfg.is_legacy_gaussian():
            eff_env = envelope_cfg
            alpha_pc = envelope_cfg.width_per_frame_symbols(frame.mm_symbols)
            C_pc = envelope_cfg.C_per_frame_symbols(frame.mm_symbols)
        elif envelope_cfg is not None:
            alpha_pc = envelope_cfg.width_per_frame_symbols(frame.mm_symbols)
        elif alpha_per_center is not None:
            alpha_pc = np.asarray(alpha_per_center, dtype=np.float64).reshape(-1)
            if alpha_pc.size != rep_centers.shape[0]:
                raise ValueError(
                    f"alpha_per_center length {alpha_pc.size} != n_rep_centers {rep_centers.shape[0]}"
                )
        mf = _attach_gaussian_repulsion_amps(
            mf,
            rep_centers,
            amp_full,
            alpha_bohr2=float(alpha_bohr2),
            alpha_per_center=alpha_pc,
            C_per_center=C_pc,
            envelope_cfg=eff_env,
            mm_symbols=frame.mm_symbols,
            **cone_kw,
        )
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("EmbR SCF did not converge (try --embed-scf-conv-tol)")
    return mf
