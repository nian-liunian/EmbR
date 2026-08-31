"""Plane grid through MM H, bonded O, and nearest QM atom (export / plot helpers)."""

from __future__ import annotations

import numpy as np

from scf_embed_io import ScfFrame


def _element(sym: str) -> str:
    return sym.strip()[0].upper()


def _is_ohh_triplet(s0: str, s1: str, s2: str) -> bool:
    return _element(s0) == "O" and _element(s1) == "H" and _element(s2) == "H"


def bonded_o_index_for_mm_h(h_idx: int, frame: ScfFrame) -> int:
    """MM list index of O bonded to ``h_idx`` (Coo O,H,H topology, then distance fallback)."""
    mm_symbols = frame.mm_symbols
    n_mm = len(mm_symbols)
    hi = int(h_idx)
    if hi < 0 or hi >= n_mm:
        raise ValueError(f"mm-h-index {hi} out of range [0, {n_mm})")
    if _element(mm_symbols[hi]) != "H":
        raise ValueError(f"mm-h-index {hi} is {mm_symbols[hi]!r}, not H")

    w = 0
    while w + 2 < n_mm:
        if hi in (w, w + 1, w + 2) and _is_ohh_triplet(mm_symbols[w], mm_symbols[w + 1], mm_symbols[w + 2]):
            return int(w)
        w += 3

    mm = np.asarray(frame.mm_coords_ang, dtype=np.float64)
    rh = mm[hi]
    best_j: int | None = None
    best_d = float("inf")
    for j, sym in enumerate(mm_symbols):
        if _element(sym) != "O":
            continue
        d = float(np.linalg.norm(mm[j] - rh))
        if d <= 1.35 and d < best_d:
            best_d = d
            best_j = int(j)
    if best_j is None:
        raise ValueError(f"no bonded O found for MM H index {hi}")
    return best_j


def nearest_qm_index_to_mm_h(h_idx: int, frame: ScfFrame) -> int:
    """QM atom index closest to MM H ``h_idx``."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    rh = np.asarray(frame.mm_coords_ang[int(h_idx)], dtype=np.float64).reshape(3)
    return int(np.argmin(np.linalg.norm(qm - rh, axis=1)))


def mm_h_qm_distance_ang(h_idx: int, frame: ScfFrame, qm_idx: int | None = None) -> float:
    """3D distance [Å] from MM H to QM atom (default: nearest QM)."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    rh = np.asarray(frame.mm_coords_ang[int(h_idx)], dtype=np.float64).reshape(3)
    if qm_idx is None:
        qm_idx = nearest_qm_index_to_mm_h(h_idx, frame)
    return float(np.linalg.norm(qm[int(qm_idx)] - rh))


def list_mm_h_distances(frame: ScfFrame) -> list[tuple[int, float, int]]:
    """(mm_index, dist_to_nearest_QM [Å], nearest_qm_index) for each MM H."""
    out: list[tuple[int, float, int]] = []
    for i, sym in enumerate(frame.mm_symbols):
        if _element(sym) != "H":
            continue
        iqm = nearest_qm_index_to_mm_h(i, frame)
        d = mm_h_qm_distance_ang(i, frame, iqm)
        out.append((int(i), float(d), int(iqm)))
    return sorted(out, key=lambda x: x[1])


def list_mm_distances_to_nearest_qm(frame: ScfFrame) -> list[tuple[int, float, int]]:
    """(mm_index, dist_to_nearest_QM [Å], nearest_qm_index) for every MM atom."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm = np.asarray(frame.mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    out: list[tuple[int, float, int]] = []
    for j in range(mm.shape[0]):
        dvec = qm - mm[j]
        dists = np.linalg.norm(dvec, axis=1)
        iqm = int(np.argmin(dists))
        out.append((int(j), float(dists[iqm]), iqm))
    return sorted(out, key=lambda x: x[1])


def nearest_mm_h_index_to_qm(frame: ScfFrame) -> int:
    """MM list index of the H closest to any QM atom."""
    rows = list_mm_h_distances(frame)
    if not rows:
        raise ValueError("no MM H in frame")
    return int(rows[0][0])


def list_mm_o_distances(frame: ScfFrame) -> list[tuple[int, float, int]]:
    """(mm_index, dist_to_nearest_QM [Å], nearest_qm_index) for each MM O."""
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    out: list[tuple[int, float, int]] = []
    for i, sym in enumerate(frame.mm_symbols):
        if _element(sym) != "O":
            continue
        ro = np.asarray(frame.mm_coords_ang[int(i)], dtype=np.float64).reshape(3)
        dists = np.linalg.norm(qm - ro, axis=1)
        iqm = int(np.argmin(dists))
        out.append((int(i), float(dists[iqm]), iqm))
    return sorted(out, key=lambda x: x[1])


def nearest_mm_o_index_to_qm(frame: ScfFrame) -> int:
    """MM list index of the O closest to any QM atom."""
    rows = list_mm_o_distances(frame)
    if not rows:
        raise ValueError("no MM O in frame")
    return int(rows[0][0])


def plane_basis_from_atoms(
    p_h: np.ndarray,
    p_o: np.ndarray,
    p_qm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Origin at MM H; u along H→O; v in-plane ⊥ u.

    Returns (origin, e1, e2).
    """
    p_h = np.asarray(p_h, dtype=np.float64).reshape(3)
    p_o = np.asarray(p_o, dtype=np.float64).reshape(3)
    p_qm = np.asarray(p_qm, dtype=np.float64).reshape(3)
    origin = p_h.copy()

    e1 = p_o - p_h
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-8:
        raise ValueError("MM H and O are coincident; cannot define plane")
    e1 = e1 / n1

    normal = np.cross(p_o - p_h, p_qm - p_h)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-8:
        normal = np.cross(e1, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        nn = float(np.linalg.norm(normal))
        if nn < 1e-8:
            normal = np.cross(e1, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            nn = float(np.linalg.norm(normal))
    if nn < 1e-8:
        raise ValueError("H, O, QM are collinear; cannot define plane")
    normal = normal / nn

    e2 = np.cross(normal, e1)
    e2 = e2 / float(np.linalg.norm(e2))
    return origin, e1, e2


def project_to_plane_uv(
    xyz: np.ndarray,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, float]:
    d = np.asarray(xyz, dtype=np.float64).reshape(3) - np.asarray(origin, dtype=np.float64).reshape(3)
    return float(np.dot(d, e1)), float(np.dot(d, e2))


def build_plane_grid(
    p_h: np.ndarray,
    p_o: np.ndarray,
    p_qm: np.ndarray,
    *,
    margin_ang: float = 1.0,
    step_ang: float = 0.08,
    min_half_ang: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (u_axis, v_axis, UU, VV, points_3d_flat, origin, e1, e2).

    Grid auto-sized to include H (u=v=0), bonded O, and nearest QM, plus ``margin_ang``.
    ``rho[i,j]`` samples ``(u_axis[j], v_axis[i])``.
    """
    origin, e1, e2 = plane_basis_from_atoms(p_h, p_o, p_qm)
    uh, vh = project_to_plane_uv(p_h, origin, e1, e2)
    uo, vo = project_to_plane_uv(p_o, origin, e1, e2)
    uq, vq = project_to_plane_uv(p_qm, origin, e1, e2)

    margin = float(margin_ang)
    min_half = float(min_half_ang)
    u_lo = min(uh, uo, uq) - margin
    u_hi = max(uh, uo, uq) + margin
    v_lo = min(vh, vo, vq) - margin
    v_hi = max(vh, vo, vq) + margin

    if u_hi - u_lo < 2 * min_half:
        mid_u = 0.5 * (u_lo + u_hi)
        u_lo, u_hi = mid_u - min_half, mid_u + min_half
    if v_hi - v_lo < 2 * min_half:
        mid_v = 0.5 * (v_lo + v_hi)
        v_lo, v_hi = mid_v - min_half, mid_v + min_half

    step = float(step_ang)
    if step <= 0.0:
        raise ValueError("plane step must be positive")

    u_axis = np.arange(u_lo, u_hi + 0.5 * step, step, dtype=np.float64)
    v_axis = np.arange(v_lo, v_hi + 0.5 * step, step, dtype=np.float64)
    uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
    points = origin + uu[..., None] * e1 + vv[..., None] * e2
    return u_axis, v_axis, uu, vv, points.reshape(-1, 3), origin, e1, e2


def interp_plane_field(
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    field: np.ndarray,
    u: float,
    v: float,
) -> float:
    """Bilinear sample of ``field[i,j]`` at ``(u_axis[j], v_axis[i])``."""
    u_axis = np.asarray(u_axis, dtype=np.float64)
    v_axis = np.asarray(v_axis, dtype=np.float64)
    z = np.asarray(field, dtype=np.float64)
    ui = int(np.searchsorted(u_axis, float(u)) - 1)
    vi = int(np.searchsorted(v_axis, float(v)) - 1)
    ui = int(np.clip(ui, 0, u_axis.size - 2))
    vi = int(np.clip(vi, 0, v_axis.size - 2))
    u0, u1 = float(u_axis[ui]), float(u_axis[ui + 1])
    v0, v1 = float(v_axis[vi]), float(v_axis[vi + 1])
    fu = 0.0 if abs(u1 - u0) < 1e-14 else (float(u) - u0) / (u1 - u0)
    fv = 0.0 if abs(v1 - v0) < 1e-14 else (float(v) - v0) / (v1 - v0)
    z00, z10 = float(z[vi, ui]), float(z[vi, ui + 1])
    z01, z11 = float(z[vi + 1, ui]), float(z[vi + 1, ui + 1])
    return float((1.0 - fu) * (1.0 - fv) * z00 + fu * (1.0 - fv) * z10 + (1.0 - fu) * fv * z01 + fu * fv * z11)


def sample_h_to_qm_axis(
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    atom_uv: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    dr: float = 0.2,
    r_max: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Sample 2D plane fields on segment MM H → QM_near (in-plane).

    ``r=0`` at H; ``r`` increases toward QM. Matches line table axis direction
    for this single H–QM pair (not the multi-pair average from line_rho_pert_cluster).
    """
    uq, vq = float(atom_uv[2, 0]), float(atom_uv[2, 1])
    d_axis = float(np.hypot(uq, vq))
    if d_axis < 1e-10:
        raise ValueError("H and QM coincide in plane")
    tu, tv = uq / d_axis, vq / d_axis
    r_end = min(float(r_max), d_axis)
    r_grid = np.arange(0.0, r_end + 0.5 * float(dr), float(dr), dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, fld in fields.items():
        vals = [
            interp_plane_field(u_axis, v_axis, fld, float(r * tu), float(r * tv)) for r in r_grid
        ]
        out[str(name)] = np.asarray(vals, dtype=np.float64)
    return r_grid, out
