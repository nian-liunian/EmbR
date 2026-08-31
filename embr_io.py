"""
Coo / E0 file I/O for EmbR (QM/MM geometry + reference energies).

Geometry (``Coo{i}.xyz``)
-------------------------
Extended XYZ: line 1 = natoms, line 2 = comment, then ``element x y z`` per atom.
MM lines may include a 5th column (charge) when input is already charged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COO_NAME_FMT = "Coo{}.xyz"

# Legacy 10-atom glycine QM count default in SoapE0Hyper.
M_GLY = 10
SOAP_SPECIES = ("C", "H", "O", "N", "Na", "Cl", "K")
SOAP_SPECIES_QM = ("C", "H", "O", "N")
ION_MM_KEYS = frozenset({"Na", "K", "Cl"})


def _norm_qm_symbol(tok: str) -> str:
    t = tok.strip()
    if not t:
        raise ValueError("empty element token")
    u = t.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    t = t[0].upper() + (t[1:].lower() if len(t) > 1 else "")
    if t not in SOAP_SPECIES_QM:
        raise ValueError(f"unsupported QM element {tok!r}")
    return t


def _norm_mm_symbol(tok: str) -> str:
    t = tok.strip()
    if not t:
        raise ValueError("empty element token")
    u = t.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    if u == "K" or u.startswith("K+"):
        return "K"
    t = t[0].upper() + (t[1:].lower() if len(t) > 1 else "")
    if t not in SOAP_SPECIES:
        raise ValueError(f"unsupported MM element {tok!r} (SOAP species are {SOAP_SPECIES})")
    return t


def _norm_symbol(tok: str) -> str:
    """Backward-compatible: QM species only."""
    return _norm_qm_symbol(tok)


def _mm_block_parser_needed(rest: list[str]) -> bool:
    if not rest:
        return False
    if _mm_lines_all_explicit_charges(rest):
        return False
    g = rest[0].strip().split()
    if len(g) < 1:
        return False
    try:
        el = _norm_mm_symbol(g[0])
    except ValueError:
        return True
    return el in ION_MM_KEYS or len(rest) % 3 != 0


def _mm_lines_all_explicit_charges(rest: list[str]) -> bool:
    """True when every MM line has elem x y z charge (charged Coo format)."""
    if not rest:
        return False
    for ln in rest:
        if len(ln.strip().split()) < 5:
            return False
    return True


def _parse_mm_charged_lines(
    rest: list[str],
) -> tuple[list[str], list[list[float]], list[float | None]]:
    mm_syms: list[str] = []
    mm_pos: list[list[float]] = []
    mm_chg: list[float | None] = []
    for i, ln in enumerate(rest):
        g = ln.strip().split()
        if len(g) < 5:
            raise ValueError(f"MM line {i + 1}: need elem x y z charge")
        mm_syms.append(_norm_mm_symbol(g[0]))
        mm_pos.append([float(g[1]), float(g[2]), float(g[3])])
        mm_chg.append(float(g[4]))
    return mm_syms, mm_pos, mm_chg


def _parse_coo_core(
    lines: list[str],
    *,
    n_qm: int,
    qm_symbols_fallback: tuple[str, ...] = (),
) -> tuple[list[str], list[list[float]], list[str], list[list[float]], list[float | None]]:
    """
    Returns qm_syms, qm_pos, mm_syms, mm_pos, mm_chg_optional (None → use default map).
    """
    if len(lines) < n_qm + 1:
        raise ValueError(f"need at least {n_qm} QM lines + 1 MM line")

    qm_syms: list[str] = []
    qm_pos: list[list[float]] = []
    for i in range(n_qm):
        g = lines[i].strip().split()
        if len(g) < 4:
            raise ValueError(f"QM line {i + 1}: need elem x y z")
        try:
            sym = _norm_qm_symbol(g[0])
        except ValueError:
            if i >= len(qm_symbols_fallback):
                raise ValueError(
                    f"QM line {i + 1}: invalid element {g[0]!r}; "
                    f"Coo must have symbols on each QM line (first {n_qm} lines), "
                    "or pass qm_symbols_fallback="
                ) from None
            sym = qm_symbols_fallback[i]
        qm_syms.append(sym)
        qm_pos.append([float(g[1]), float(g[2]), float(g[3])])

    rest = lines[n_qm:]
    mm_syms: list[str] = []
    mm_pos: list[list[float]] = []
    mm_chg: list[float | None] = []

    if _mm_lines_all_explicit_charges(rest):
        mm_syms, mm_pos, mm_chg = _parse_mm_charged_lines(rest)
        return qm_syms, qm_pos, mm_syms, mm_pos, mm_chg

    if not _mm_block_parser_needed(rest):
        if len(rest) < 3:
            raise ValueError("need at least one water (3 lines) after QM")
        if len(rest) % 3 != 0:
            raise ValueError("after QM, line count must be multiple of 3 (O,H,H) or use ion lines")
        for w in range(len(rest) // 3):
            for j, line_idx in enumerate((3 * w, 3 * w + 1, 3 * w + 2)):
                g = rest[line_idx].strip().split()
                if len(g) < 4:
                    raise ValueError(f"water {w} line {j}: need elem x y z")
                mm_syms.append(_norm_mm_symbol(g[0]))
                mm_pos.append([float(g[1]), float(g[2]), float(g[3])])
                mm_chg.append(float(g[4]) if len(g) >= 5 else None)
        return qm_syms, qm_pos, mm_syms, mm_pos, mm_chg

    i = 0
    n_lines = len(rest)
    while i < n_lines:
        g = rest[i].strip().split()
        if len(g) < 4:
            raise ValueError(f"MM line {i + 1}: need elem x y z")
        el = _norm_mm_symbol(g[0])
        if el in ION_MM_KEYS:
            mm_syms.append(el)
            mm_pos.append([float(g[1]), float(g[2]), float(g[3])])
            mm_chg.append(float(g[4]) if len(g) >= 5 else None)
            i += 1
            continue
        if i + 2 >= n_lines:
            raise ValueError(f"incomplete O,H,H triplet starting MM line {i + 1}")
        trip_els: list[str] = []
        trip_pos: list[list[float]] = []
        trip_chg: list[float | None] = []
        for j in range(3):
            gj = rest[i + j].strip().split()
            if len(gj) < 4:
                raise ValueError(f"water line {i + j + 1}: need elem x y z")
            trip_els.append(_norm_mm_symbol(gj[0]))
            trip_pos.append([float(gj[1]), float(gj[2]), float(gj[3])])
            trip_chg.append(float(gj[4]) if len(gj) >= 5 else None)
        if trip_els != ["O", "H", "H"]:
            raise ValueError(f"expected O,H,H at MM lines {i + 1}..{i + 3}, got {trip_els}")
        mm_syms.extend(trip_els)
        mm_pos.extend(trip_pos)
        mm_chg.extend(trip_chg)
        i += 3

    if not mm_syms:
        raise ValueError("no MM atoms after QM block")
    return qm_syms, qm_pos, mm_syms, mm_pos, mm_chg


def parse_xyz_frame(path: Path) -> tuple[list[str], str]:
    """Extended XYZ → Coo-style atom lines + comment."""
    path = Path(path)
    raw = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if len(raw) < 3:
        raise ValueError(f"{path}: XYZ needs natoms + comment + coordinates")
    natoms = int(raw[0].strip().split()[0])
    comment = raw[1].strip()
    body = raw[2:]
    if len(body) < natoms:
        raise ValueError(f"{path}: expected {natoms} atom lines, got {len(body)}")
    return [ln.strip() for ln in body[:natoms]], comment


def load_geometry_lines(path: Path) -> list[str]:
    """Load atom lines from extended XYZ (``Coo{i}.xyz``)."""
    path = Path(path)
    if path.suffix.lower() != ".xyz":
        raise ValueError(
            f"{path}: geometry must be extended XYZ (.xyz), not {path.suffix!r}. "
            f"Use Coo{{i}}.xyz under out_dir/Coo/."
        )
    lines, _ = parse_xyz_frame(path)
    return lines


def write_xyz_frame(
    path: Path,
    *,
    qm_symbols: list[str] | tuple[str, ...],
    qm_coords_ang: np.ndarray,
    mm_symbols: list[str] | tuple[str, ...],
    mm_coords_ang: np.ndarray,
    mm_charges: np.ndarray,
    comment: str = "EmbR QM/MM",
) -> None:
    """Write extended XYZ with explicit MM charges (5th column on MM rows)."""
    path = Path(path)
    qm_coords_ang = np.asarray(qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm_coords_ang = np.asarray(mm_coords_ang, dtype=np.float64).reshape(-1, 3)
    mm_charges = np.asarray(mm_charges, dtype=np.float64).reshape(-1)
    n = int(qm_coords_ang.shape[0] + mm_coords_ang.shape[0])
    rows: list[str] = [str(n), str(comment).strip() or "EmbR QM/MM"]
    for sym, xyz in zip(qm_symbols, qm_coords_ang):
        x, y, z = (float(v) for v in xyz)
        rows.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    for sym, xyz, chg in zip(mm_symbols, mm_coords_ang, mm_charges):
        x, y, z = (float(v) for v in xyz)
        rows.append(f"{sym} {x:.6f} {y:.6f} {z:.6f} {float(chg):.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def resolve_geometry_path(coo_dir: Path, name_fmt: str, idx: int) -> Path:
    coo_dir = Path(coo_dir)
    fmt = str(name_fmt).strip()
    if fmt.endswith(".txt"):
        fmt = fmt[:-4] + ".xyz"
    if not fmt.endswith(".xyz"):
        fmt = f"{fmt}.xyz" if "{}" in fmt else "Coo{}.xyz"
    if "{}" not in fmt:
        fmt = COO_NAME_FMT
    return coo_dir / fmt.format(int(idx))


def load_coo_frame(
    path: Path,
    n_qm: int = 10,
    *,
    qm_symbols_fallback: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    lines = load_geometry_lines(path)
    qm_syms, qm_pos, mm_syms, mm_pos, _ = _parse_coo_core(
        lines, n_qm=int(n_qm), qm_symbols_fallback=qm_symbols_fallback
    )
    symbols = qm_syms + mm_syms
    positions = qm_pos + mm_pos
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise RuntimeError(f"{path}: bad positions shape {pos.shape}")
    return pos, symbols


def load_coo_frame_atoms(
    path: Path,
    n_qm: int = 10,
    *,
    qm_symbols_fallback: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    return load_coo_frame(path, n_qm=n_qm, qm_symbols_fallback=qm_symbols_fallback)


def coo_path(coo_dir: Path, name_fmt: str, idx: int) -> Path:
    return resolve_geometry_path(coo_dir, name_fmt, idx)


def load_e0_txt(path: Path) -> np.ndarray:
    """First numeric column per line: E0 = E_high - E_int_bg (kcal/mol)."""
    e0, _ = load_e_embed_txt(path)
    return e0


def load_e_embed_txt(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read per-frame embedding energy labels (kcal/mol).

    Column 1 : E0 = E_high - E_int_bg  (required)
    Column 2 : E_int_bg from background-charge SCF (optional)
    """
    e0_vals: list[float] = []
    ebg_vals: list[float] = []
    has_col2 = True
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 1:
            continue
        e0_vals.append(float(parts[0]))
        if len(parts) >= 2:
            ebg_vals.append(float(parts[1]))
        else:
            has_col2 = False
            ebg_vals.append(float("nan"))

    if not e0_vals:
        raise ValueError(f"{path}: no numeric E0 lines")

    e0 = np.asarray(e0_vals, dtype=np.float64)
    if not has_col2 or len(ebg_vals) != len(e0_vals):
        return e0, None
    ebg = np.asarray(ebg_vals, dtype=np.float64)
    if not np.all(np.isfinite(ebg)):
        raise ValueError(f"{path}: column-2 E_int_bg has non-finite values")
    return e0, ebg


def write_e_embed_txt(
    path: Path,
    e0: np.ndarray,
    e_int_bg: np.ndarray | None = None,
    *,
    header: str | None = None,
) -> None:
    e0 = np.asarray(e0, dtype=np.float64).reshape(-1)
    lines: list[str] = []
    if header:
        lines.append(f"# {header}")
    if e_int_bg is None:
        for x in e0:
            lines.append(f"{float(x):.8f}")
    else:
        ebg = np.asarray(e_int_bg, dtype=np.float64).reshape(-1)
        if ebg.shape != e0.shape:
            raise ValueError(f"e0 {e0.shape} != e_int_bg {ebg.shape}")
        for x, b in zip(e0, ebg):
            lines.append(f"{float(x):.8f}  {float(b):.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_all_coo_frames(
    *,
    coo_dir: Path,
    name_fmt: str,
    i0: int,
    n_frames: int,
    n_qm: int,
    qm_symbols_fallback: tuple[str, ...] = (),
) -> list[tuple[np.ndarray, list[str]]]:
    frames: list[tuple[np.ndarray, list[str]]] = []
    for k in range(n_frames):
        p = coo_path(coo_dir, name_fmt, i0 + k)
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")
        frames.append(
            load_coo_frame_atoms(p, n_qm=n_qm, qm_symbols_fallback=qm_symbols_fallback)
        )
    return frames
