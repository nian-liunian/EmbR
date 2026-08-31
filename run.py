#!/usr/bin/env python3
"""
EmbR pipeline orchestrator: ``python run.py --manifest your.json``

Manifest blocks: ``repulsion_parameter``, ``calculate_parameter``, ``datasets[]``, ``train``.

Per dataset use ``prefix`` + ``QM_atoms`` (+ ``qm_charge``, ``e0_file``, ``out_dir``, ``n_frames``).
Legacy ``residue`` label is optional bookkeeping only.

Pipeline (mode=train): batch_hf → ref npz → precompute (--ref-dir) → train ksoft → batch_scf (EmbR).

Train on an existing mixed npz: set ``mode: train_with_npz`` and ``paths.npz``, or pass ``--npz PATH``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys

# Keep public runs from leaving local __pycache__ artifacts.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Production envelope defaults used to generate the released EmbR k_i caches.
# H uses the effective 1.5× width; values are kept at the precision used in production.
_DEFAULT_ALPHA_EXP: dict[str, float] = {
    "H": 5.6685,
    "C": 6.775,
    "N": 6.809,
    "O": 7.006,
    "Na": 7.898,
    "Cl": 2.794,
}
_DEFAULT_LNC_EXP: dict[str, float] = {
    "H": 1.9806021698973124,
    "C": 2.476,
    "N": 2.655,
    "O": 2.855,
    "Na": 3.307,
    "Cl": 0.372,
}

_MODES = ("ab_initio", "train", "scf", "train_with_npz")


def _die(msg: str) -> None:
    raise SystemExit(f"[run.py] {msg}")


def _normalize_mode(raw: Any) -> str:
    key = str(raw if raw is not None else "train").strip().lower().replace("-", "_")
    aliases = {
        "ab_initio": "ab_initio",
        "abinitio": "ab_initio",
        "train": "train",
        "scf": "scf",
        "train_with_npz": "train_with_npz",
    }
    if key not in aliases:
        _die(
            f'mode must be one of {list(_MODES)} (aliases: abinitio→ab_initio), got {raw!r}'
        )
    return aliases[key]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"invalid JSON ({path}): {exc}")
    if not isinstance(raw, dict):
        _die("manifest root must be a JSON object")
    return raw


def _block(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    b = cfg.get(key)
    if b is None:
        return {}
    if not isinstance(b, dict):
        _die(f"{key!r} must be an object {{...}}")
    return b


def _parse_repulsion(cfg: dict[str, Any]) -> dict[str, Any]:
    rep = _block(cfg, "repulsion_parameter")
    # Also allow flat legacy keys at root for convenience
    kind = str(rep.get("envelope_kind", cfg.get("envelope_kind", "exp"))).strip().lower()
    if kind not in ("exp", "gauss"):
        _die(f"envelope_kind must be 'exp' or 'gauss', got {kind!r}")
    if kind == "gauss":
        _die(
            "envelope_kind='gauss' is not wired in this test release of run.py "
            "in this release. Use envelope_kind='exp'."
        )
    # Public manifest name is zeta; alpha/fix_alpha_by_element remain accepted for legacy manifests.
    alpha = rep.get("zeta", rep.get("alpha", rep.get("fix_alpha_by_element")))
    lnc = rep.get("lnC", rep.get("lnC_by_element"))
    if alpha is None:
        alpha = dict(_DEFAULT_ALPHA_EXP)
        print("[run.py] zeta omitted → using exp defaults", flush=True)
    elif not isinstance(alpha, dict):
        _die("zeta must be an object mapping element → float")
    else:
        alpha = {**_DEFAULT_ALPHA_EXP, **{str(k): float(v) for k, v in alpha.items()}}
        print("[run.py] zeta override merged with exp defaults", flush=True)
    if lnc is None:
        lnc = dict(_DEFAULT_LNC_EXP)
        print("[run.py] lnC omitted → using exp defaults", flush=True)
    elif not isinstance(lnc, dict):
        _die("lnC must be an object mapping element → float")
    else:
        lnc = {**_DEFAULT_LNC_EXP, **{str(k): float(v) for k, v in lnc.items()}}
        print("[run.py] lnC override merged with exp defaults", flush=True)
    return {
        "envelope_kind": kind,
        "fix_alpha_by_element": {str(k): float(v) for k, v in alpha.items()},
        "lnC_by_element": {str(k): float(v) for k, v in lnc.items()},
        # H reference width for legacy fix_alpha fields
        "fix_alpha": float(alpha.get("H", _DEFAULT_ALPHA_EXP["H"])),
    }


def _parse_calculate(cfg: dict[str, Any]) -> dict[str, Any]:
    from scf_embed_pyscf import resolve_qm_charge

    calc = _block(cfg, "calculate_parameter")
    # Merge: calculate_parameter overrides; fall back to root (legacy flat)
    def _get(key: str, default: Any) -> Any:
        if key in calc:
            return calc[key]
        return cfg.get(key, default)

    scf = calc.get("scf", cfg.get("scf")) or {}
    if not isinstance(scf, dict):
        _die("scf must be an object")
    # tip3p: waters → TIP3P, ions → formal (±1); mulliken: whole MM/frag2 → Mulliken
    raw_cm = str(_get("charge_mode", "tip3p")).strip().lower()
    if raw_cm in ("milliken",):  # common typo
        raw_cm = "mulliken"
    if raw_cm not in ("tip3p", "mulliken"):
        _die(f'charge_mode must be "tip3p" or "mulliken", got {raw_cm!r}')

    # scf_A_fix: rescale A after EmbR until |ε|<scf_error (scf{N}_fixk.npz)
    # 1st fix slope=E0; later 2-point secant. No fixed fix count.
    raw_fix = _get("scf_A_fix", _get("A_fix", False))
    if isinstance(raw_fix, str):
        scf_a_fix = raw_fix.strip().lower() in ("1", "true", "yes", "on")
    else:
        scf_a_fix = bool(raw_fix)
    raw_err = _get("scf_error", _get("scf_A_fix_eps_kcal", 0.5))
    try:
        scf_error = float(raw_err)
    except (TypeError, ValueError):
        _die(f"scf_error must be a float, got {raw_err!r}")
    if scf_a_fix and scf_error <= 0.0:
        _die(f"scf_error must be > 0, got {scf_error}")
    if scf_a_fix:
        print(
            f"[run.py] scf_A_fix=true  scf_error={scf_error:g} kcal  "
            f"(initial energy-matching scale, then secant; stop when |ε|<scf_error)",
            flush=True,
        )

    # Limit val SCF to first N holdout frames (with scf_frames=all_val).
    raw_nv = _get("scf_n_val_frames", _get("n_val_frames", None))
    scf_n_val_frames: int | None = None
    if raw_nv is not None and str(raw_nv).strip() != "":
        try:
            scf_n_val_frames = int(raw_nv)
        except (TypeError, ValueError):
            _die(f"scf_n_val_frames must be an int, got {raw_nv!r}")
        if scf_n_val_frames <= 0:
            _die(f"scf_n_val_frames must be > 0, got {scf_n_val_frames}")

    raw_amax = _get("scf_A_fix_max", _get("a_fix_max", None))
    scf_a_fix_max: int | None = None
    if raw_amax is not None and str(raw_amax).strip() != "":
        try:
            scf_a_fix_max = int(raw_amax)
        except (TypeError, ValueError):
            _die(f"scf_A_fix_max must be an int, got {raw_amax!r}")
        if scf_a_fix_max < 1:
            _die(f"scf_A_fix_max must be >= 1, got {scf_a_fix_max}")

    # scf_skip_existing: only EmbR/batch_scf. Emb0+CP/DFT ref always --skip-existing.
    raw_scf_skip = _get("scf_skip_existing", False)
    if isinstance(raw_scf_skip, str):
        scf_skip_existing = raw_scf_skip.strip().lower() in ("1", "true", "yes", "on")
    else:
        scf_skip_existing = bool(raw_scf_skip)


    return {
        "r_cut_mm": _get("r_cut_mm", None),
        "r_cut": float(_get("r_cut", 5.0)),
        "n_max": int(_get("n_max", 8)),
        "l_max": int(_get("l_max", 6)),
        "sigma": float(_get("sigma", 0.5)),
        "workers": int(_get("workers", 2)),
        "charge_mode": raw_cm,
        "scf_A_fix": scf_a_fix,
        "scf_error": scf_error,
        "scf_A_fix_max": scf_a_fix_max,
        "scf_n_val_frames": scf_n_val_frames,
        "scf_skip_existing": scf_skip_existing,
        "scf": {
            "method": str(scf.get("method", "hf")),
            "basis": str(scf.get("basis", "6-31g*")),
            "d3bj": bool(scf.get("d3bj", False)),
            "threads": int(scf.get("threads", 4)),
            "verbose_scf": int(scf.get("verbose_scf", 0)),
            # Fallback only; prefer datasets[].qm_charge per system
            "qm_charge": resolve_qm_charge(scf, calc, cfg, default=0),
        },
    }




def _parse_train(cfg: dict[str, Any]) -> dict[str, Any]:
    """ksoft training hyperparameters from manifest ``train`` block."""
    tr = _block(cfg, "train")
    if "lr" not in tr:
        _die('train.lr is required (no default). Example: "lr": 6e-5')

    if tr.get("pure_E0") or tr.get("pure_e0"):
        _die("embr does not support pure_E0 in this release")
    pure_e0 = False

    # Paper path: always ksoft. Optional train.model is ignored if present (must be ksoft).
    if "model" in tr or "k_partition" in tr or "k-partition" in tr:
        raw_model = str(tr.get("model", "ksoft")).strip().lower()
        if raw_model not in ("ksoft", ""):
            _die(
                f'embr only supports ksoft (omit train.model). Got model={raw_model!r}'
            )
        if tr.get("k_partition", tr.get("k-partition")) is not None:
            _die("embr: omit train.k_partition (ksoft is always soft partition)")
    model = "ksoft"
    k_partition = None
    weight_corr_default = 15.0
    weight_corr = float(
        tr.get("weight_corr", tr.get("weight-corr", weight_corr_default))
    )
    corr_target = float(tr.get("corr_target", tr.get("corr-target", 0.9)))
    e_sites_raw = tr.get("e_sites", tr.get("e-sites", "all"))
    e_sites = str(e_sites_raw or "all").strip().lower()
    if e_sites in ("no_ocl", "cations"):
        e_sites = "positive"
    if e_sites not in ("all", "positive", "h_only", "no_o"):
        _die(
            f'train.e_sites must be "all" or "positive" '
            f"(O/Cl e=0), got {e_sites_raw!r}"
        )
    monitor = tr.get("monitor")
    if monitor is not None:
        monitor = str(monitor).strip().lower()
        if monitor not in ("e0", "loss"):
            _die(f'train.monitor must be "e0" or "loss", got {monitor!r}')
    init_ckpt_raw = tr.get("init_ckpt", tr.get("init-ckpt"))
    init_ckpt: Path | None = None
    if init_ckpt_raw is not None and str(init_ckpt_raw).strip():
        init_ckpt = Path(str(init_ckpt_raw)).expanduser()
        if not init_ckpt.is_absolute():
            init_ckpt = (_ROOT / init_ckpt).resolve()
        else:
            init_ckpt = init_ckpt.resolve()
    return {
        "lr": float(tr["lr"]),
        "epochs": int(tr.get("epochs", 2000)),
        "weight_decay": float(tr.get("weight_decay", tr.get("weight-decay", 1e-4))),
        "batch": int(tr.get("batch", 32)),
        "patience": int(tr.get("patience", 150)),
        "device": str(tr.get("device", "cpu")),
        "pure_e0": pure_e0,
        "seed": int(tr.get("seed", 0)),
        "val_frac": float(tr.get("val_frac", 0.15)),
        "model": model,
        "k_partition": k_partition,
        "weight_corr": weight_corr,
        "corr_target": corr_target,
        "e_sites": e_sites,
        "monitor": monitor,
        "weight_e0": float(tr.get("weight_e0", tr.get("weight-e0", 1.0))),
        "init_ckpt": init_ckpt,
    }


def _train_stub() -> dict[str, Any]:
    """Minimal train dict when mode needs no ML training (scf)."""
    return {
        "lr": 0.0,
        "epochs": 2000,
        "weight_decay": 1e-4,
        "batch": 32,
        "patience": 150,
        "device": "cpu",
        "pure_e0": False,
        "seed": 0,
        "val_frac": 0.15,
        "model": "ksoft",
        "k_partition": None,
        "weight_corr": 15.0,
        "corr_target": 0.9,
        "e_sites": "all",
        "monitor": None,
        "weight_e0": 1.0,
    }


def _coo_data_lines(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".xyz":
        from embr_io import load_geometry_lines

        return load_geometry_lines(path)
    return [
        ln
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _count_coo_atom_lines(path: Path) -> int:
    return len(_coo_data_lines(path))


def _norm_elem_token(tok: str) -> str:
    t = str(tok).strip()
    if not t:
        raise ValueError("empty element")
    u = t.upper()
    if u.startswith("NA"):
        return "Na"
    if u.startswith("CL"):
        return "Cl"
    return t[0].upper() + (t[1:].lower() if len(t) > 1 else "")


def _parse_bare_coo(path: Path, *, n_qm: int):
    """elem x y z → QM/MM geometry (no O,H,H check; for Mulliken assign only)."""
    import numpy as np

    from embr_io import load_geometry_lines

    lines = load_geometry_lines(path)
    if len(lines) < int(n_qm) + 1:
        _die(f"{path}: need ≥{n_qm} QM lines + ≥1 MM line")
    qm_syms: list[str] = []
    qm_pos: list[list[float]] = []
    for i in range(int(n_qm)):
        g = lines[i].split()
        if len(g) < 4:
            _die(f"{path}: QM line {i + 1}: need elem x y z")
        qm_syms.append(_norm_elem_token(g[0]))
        qm_pos.append([float(g[1]), float(g[2]), float(g[3])])
    mm_syms: list[str] = []
    mm_pos: list[list[float]] = []
    for j, ln in enumerate(lines[int(n_qm) :]):
        g = ln.split()
        if len(g) < 4:
            _die(f"{path}: MM line {j + 1}: need elem x y z")
        mm_syms.append(_norm_elem_token(g[0]))
        mm_pos.append([float(g[1]), float(g[2]), float(g[3])])
    return (
        tuple(qm_syms),
        np.asarray(qm_pos, dtype=np.float64),
        tuple(mm_syms),
        np.asarray(mm_pos, dtype=np.float64),
    )


def _parse_frame_indices(raw: Any, *, i0_default: int = 1) -> tuple[list[int], int | None]:
    """
    Parse datasets[].n_frames.

    Returns (file_indices, i0_or_None).
    · int N → contiguous i0..i0+N-1 (i0 from caller / default)
    · list → absolute Coo indices; i0 is None (ignored)
    """
    if raw is None:
        _die('dataset requires "n_frames" (int or list of Coo indices)')
    if isinstance(raw, list):
        if not raw:
            _die("n_frames list must be non-empty")
        idxs: list[int] = []
        seen: set[int] = set()
        for x in raw:
            try:
                fi = int(x)
            except (TypeError, ValueError):
                _die(f"n_frames list entries must be ints (Coo file index), got {x!r}")
            if fi in seen:
                _die(f"n_frames list has duplicate Coo index {fi}")
            seen.add(fi)
            idxs.append(fi)
        return idxs, None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        _die(f'n_frames must be an int or a list of Coo indices, got {raw!r}')
    if n < 1:
        _die(f"n_frames int must be >= 1, got {n}")
    i0 = int(i0_default)
    return list(range(i0, i0 + n)), i0


def _coo_mm_has_full_charges(lines: list[str], *, n_qm: int) -> bool:
    if len(lines) < int(n_qm) + 1:
        return False
    for ln in lines[int(n_qm) :]:
        if len(ln.strip().split()) < 5:
            return False
    return True


def _prepare_coo_dir(
    *,
    coo_src: Path,
    out_dir: Path,
    qm_atoms: int,
    file_indices: list[int],
    charge_mode: str,
    scf_cfg: dict[str, Any],
) -> Path:
    """
    Write charged Coo under out_dir/Coo_charge/ as ``Coo{i}.xyz`` (or copy if already charged).
    """
    import shutil

    from embr_io import COO_NAME_FMT, load_geometry_lines, resolve_geometry_path
    from scf_embed_io import ScfFrame, load_scf_frame, mulliken_mm_charges_for_fragment, write_scf_coo
    from scf_embed_pyscf import scf_embed_config_from_cli

    dst = out_dir / "Coo_charge"
    dst.mkdir(parents=True, exist_ok=True)
    name_fmt = COO_NAME_FMT
    cfg = scf_embed_config_from_cli(
        method=str(scf_cfg["method"]),
        basis=str(scf_cfg["basis"]),
        use_d3bj=bool(scf_cfg.get("d3bj", False)),
        num_threads=int(scf_cfg.get("threads", 4)),
        verbose=int(scf_cfg.get("verbose_scf", 0)),
        qm_charge=int(scf_cfg.get("qm_charge", 0)),
    )
    print(
        f"[run.py] charge_mode={charge_mode} → Coo_charge/ under {out_dir}",
        flush=True,
    )
    for fi in file_indices:
        src = resolve_geometry_path(coo_src, name_fmt, fi)
        if not src.is_file():
            _die(f"missing geometry for Coo{fi}: {src}")
        out_coo = dst / f"Coo{fi}.xyz"
        lines = load_geometry_lines(src)
        if _coo_mm_has_full_charges(lines, n_qm=int(qm_atoms)):
            shutil.copy2(src, out_coo)
            continue
        if charge_mode == "tip3p":
            try:
                frame = load_scf_frame(src, n_qm=int(qm_atoms))
            except ValueError as exc:
                _die(
                    f"{src}: tip3p needs water/ion Coo (or already-charged MM). "
                    f"For organic frag2 use charge_mode=mulliken. Detail: {exc}"
                )
        else:
            qm_syms, qm_pos, mm_syms, mm_pos = _parse_bare_coo(src, n_qm=qm_atoms)
            chg = mulliken_mm_charges_for_fragment(mm_syms, mm_pos, cfg)
            frame = ScfFrame(
                qm_symbols=qm_syms,
                qm_coords_ang=qm_pos,
                mm_symbols=mm_syms,
                mm_coords_ang=mm_pos,
                mm_charges=chg,
            )
        write_scf_coo(out_coo, frame)
    n_coo = len(file_indices)
    if n_coo == 1:
        print(f"  [Coo_charge] Coo{file_indices[0]}.xyz ready", flush=True)
    elif n_coo == 2:
        print(
            f"  [Coo_charge] Coo{file_indices[0]}.xyz, Coo{file_indices[1]}.xyz ready",
            flush=True,
        )
    elif n_coo > 2:
        print(
            f"  [Coo_charge] Coo{file_indices[0]}.xyz … Coo{file_indices[-1]}.xyz "
            f"({n_coo} frames) ready",
            flush=True,
        )
    return dst


def _parse_bare_coo_charged(path: Path, *, n_qm: int):
    """elem x y z [charge] → geometry with explicit MM charges when present."""
    import numpy as np

    from embr_io import load_geometry_lines, _parse_coo_core

    lines = load_geometry_lines(path)
    qm_syms, qm_pos, mm_syms, mm_pos, mm_chg_raw = _parse_coo_core(
        lines, n_qm=int(n_qm), qm_symbols_fallback=()
    )
    mm_chg = []
    for sym, chg_opt in zip(mm_syms, mm_chg_raw):
        if chg_opt is None:
            _die(f"{path}: expected MM charge on every line for charged copy")
        mm_chg.append(float(chg_opt))
    return (
        tuple(qm_syms),
        np.asarray(qm_pos, dtype=np.float64),
        tuple(mm_syms),
        np.asarray(mm_pos, dtype=np.float64),
        np.asarray(mm_chg, dtype=np.float64),
    )



def _e0_data_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for lineno, ln in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "," in ln or "\t" in ln:
            _die(
                f"{path}: line {lineno}: use space-separated columns only "
                f"(found comma/tab). Got: {ln!r}"
            )
        parts = s.split()
        if not parts:
            continue
        try:
            float(parts[0])
        except ValueError:
            _die(f"{path}: line {lineno}: first column must be a number, got {parts[0]!r}")
        rows.append(s)
    return rows


def _validate_e0_file(path: Path, *, file_indices: list[int], compact: bool) -> None:
    """
    compact=True  (n_frames list, or contiguous with e0_i0=i0):
        need >= len(file_indices) data lines; line 1 = first frame in list / Coo{i0}.
    compact=False (legacy absolute e0_i0=1):
        need data line fi for each Coo fi.
    """
    if not path.is_file():
        _die(f"e0_file not found: {path}")
    rows = _e0_data_rows(path)
    if compact:
        need = len(file_indices)
        if len(rows) < need:
            _die(
                f"{path}: compact E0 needs >= {need} data line(s) "
                f"(one per n_frames entry in list order), found {len(rows)}"
            )
        return
    missing = [fi for fi in file_indices if fi < 1 or fi > len(rows)]
    if missing:
        _die(
            f"{path}: need data line for each Coo index (e0_i0=1: Coo i → line i); "
            f"file has {len(rows)} lines; missing/out-of-range: {missing}"
        )


def _parse_scf_frames(
    raw: Any,
    *,
    mode: str,
    dataset_frames: list[int],
) -> tuple[str, list[int] | None]:
    """
    Returns (kind, file_indices_or_None).

    kind: \"all_frames\" | \"all_val\" | \"list\"
    For list: absolute Coo indices; must be a subset of dataset_frames (n_frames).
    """
    ds_set = set(dataset_frames)
    if raw is None:
        raw = "all_frames"
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("all_frames", "all-frames", "all"):
            return "all_frames", None
        if key in ("all_val", "all-val", "val"):
            return "all_val", None
        _die(
            f'scf_frames string must be "all_frames" or "all_val", got {raw!r}. '
            f"Or pass a list of Coo indices that is a subset of n_frames."
        )
    if isinstance(raw, list):
        if not raw:
            _die("scf_frames list must be non-empty")
        idxs: list[int] = []
        for x in raw:
            try:
                fi = int(x)
            except (TypeError, ValueError):
                _die(f"scf_frames list entries must be ints (Coo file index), got {x!r}")
            idxs.append(fi)
        bad = [fi for fi in idxs if fi not in ds_set]
        if bad:
            _die(
                f"scf_frames list must be a subset of n_frames={dataset_frames}; "
                f"not in dataset: {bad}"
            )
        return "list", idxs
    _die('scf_frames must be "all_frames", "all_val", or a list of Coo indices')


def _cache_ks_from_file_indices(soap_npz: Path, file_indices: list[int]) -> list[int]:
    """Map absolute Coo file_index → soap-cache row (cache_k) via meta.frames."""
    import numpy as np

    if not soap_npz.is_file():
        _die(f"scf_frames list needs soap cache first: missing {soap_npz}")
    with np.load(soap_npz, allow_pickle=True) as z:
        if "meta_json" not in z:
            _die(f"{soap_npz.name}: missing meta_json (cannot map Coo index → cache_k)")
        meta = json.loads(str(z["meta_json"]))
    frames = meta.get("frames") if isinstance(meta, dict) else None
    if not isinstance(frames, list):
        _die(f"{soap_npz.name}: meta.frames missing (re-run precompute)")
    fi_to_k: dict[int, int] = {}
    for k, fm in enumerate(frames):
        if not isinstance(fm, dict):
            continue
        fi = int(fm.get("file_index", k))
        fi_to_k[fi] = int(fm.get("global_index", k))
    missing = [fi for fi in file_indices if fi not in fi_to_k]
    if missing:
        _die(
            f"scf_frames / Coo indices not in soap cache {soap_npz.name}: {missing}. "
            f"They must appear in this dataset's n_frames."
        )
    return [fi_to_k[fi] for fi in file_indices]


def _write_e0_from_ref_npz(
    e0_path: Path,
    *,
    ref_dir: Path,
    file_indices: list[int],
) -> None:
    """
    Overwrite e0_path with a **compact** E0: one data line per file_indices entry
    (same order). Pairs with compat e0_i0 = fi - p so precompute reads line p.
    """
    import numpy as np

    rows: list[str] = [
        "# E0 from batch_hf ref npz (ab_initio); col1=E0 kcal/mol  col2=E_int_Emb0 kcal/mol",
        f"# compact: line k = Coo file_indices[k-1]; file_indices={file_indices}",
    ]
    for fi in file_indices:
        ref = ref_dir / f"ref_{fi}.npz"
        if not ref.is_file():
            _die(f"E0 write from ref: missing {ref}")
        with np.load(ref, allow_pickle=True) as z:
            if "meta_json" in z:
                meta = json.loads(str(z["meta_json"]))
                e0 = float(meta["e0_kcal"])
                ebg = float(meta["e_int_emb_kcal"])
            else:
                from scf_embed_pyscf import hartree_to_kcal

                e0 = float(hartree_to_kcal(float(z["e0_kcal"])))
                ebg = float(hartree_to_kcal(float(z["e_int_emb_hartree"])))
        rows.append(f"{e0:.8f}  {ebg:.8f}")

    e0_path.parent.mkdir(parents=True, exist_ok=True)
    e0_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        f"[run.py] wrote compact dE from ref → {e0_path}  "
        f"({len(file_indices)} lines, order={file_indices})",
        flush=True,
    )


def _pick_residue_label(qm_atoms: int) -> str:
    """Internal dataset label when manifest has no prefix/residue (compat meta)."""
    return f"qm{int(qm_atoms)}"


def _dataset_qm_charge(ds: dict[str, Any], scf_cfg: dict[str, Any]) -> int:
    """Per-dataset qm_charge; fall back to calculate_parameter.scf.qm_charge."""
    from scf_embed_pyscf import resolve_qm_charge

    return int(resolve_qm_charge(ds, scf_cfg, default=0))


def _write_compat_manifest(
    *,
    path: Path,
    repulsion: dict[str, Any],
    calculate: dict[str, Any],
    prefix: str,
    dataset_label: str,
    qm_atoms: int,
    coo_dir: Path,
    file_indices: list[int],
    contiguous_i0: int | None,
    e0_file: Path,
) -> Path:
    """Flat manifest understood by batch_hf / precompute / batch_scf (unchanged readers)."""
    e0_resolved = str(e0_file.resolve())
    coo_resolved = str(Path(coo_dir).resolve())
    if contiguous_i0 is not None:
        # Compact E0: line k ↔ Coo{i0+k-1}; e0_line = file_index - e0_i0 with e0_i0=i0
        datasets = [
            {
                "prefix": str(prefix),
                "residue": str(dataset_label),
                "n_qm": int(qm_atoms),
                "coo_dir": coo_resolved,
                "coo_name_fmt": "Coo{}.xyz",
                "i0": int(contiguous_i0),
                "n_frames": int(len(file_indices)),
                "e0_i0": int(contiguous_i0),
                "e0_file": e0_resolved,
            }
        ]
    else:
        # Compact E0 in list order: line p ↔ file_indices[p]
        # e0_line = file_index - e0_i0 = p  ⇒  e0_i0 = fi - p
        datasets = [
            {
                "prefix": str(prefix),
                "residue": str(dataset_label),
                "n_qm": int(qm_atoms),
                "coo_dir": coo_resolved,
                "coo_name_fmt": "Coo{}.xyz",
                "i0": int(fi),
                "n_frames": 1,
                "e0_i0": int(fi - p),
                "e0_file": e0_resolved,
            }
            for p, fi in enumerate(file_indices)
        ]
    payload: dict[str, Any] = {
        "_comment": "auto-generated by run.py — do not edit by hand",
        **repulsion,
        "r_cut_mm": calculate["r_cut_mm"],
        "r_cut": calculate["r_cut"],
        "n_max": calculate["n_max"],
        "l_max": calculate["l_max"],
        "sigma": calculate["sigma"],
        "workers": calculate["workers"],
        "charge_mode": calculate.get("charge_mode", "tip3p"),
        "scf": calculate["scf"],
        "e0_source": "txt",
        "datasets": datasets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[run.py] wrote compat manifest → {path}", flush=True)
    return path


def _run(cmd: list[str], *, step: str, hint: str | None = None) -> None:
    print(f"\n=== [run.py] {step} ===", flush=True)
    if hint:
        print(hint, flush=True)
    proc = subprocess.run(cmd, cwd=str(_ROOT))
    if proc.returncode != 0:
        _die(f"step failed ({step}), exit={proc.returncode}")


def _run_tee(
    cmd: list[str],
    *,
    step: str,
    log_path: Path,
    hint: str | None = None,
) -> None:
    """Run a subprocess while streaming its combined stdout/stderr to console and log."""
    print(f"\n=== [run.py] {step} ===", flush=True)
    if hint:
        print(hint, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(f"\n=== {step} ===\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        returncode = proc.wait()
    if returncode != 0:
        _die(f"step failed ({step}), exit={returncode}; see {log_path}")


def _scf_preset_from_calc(scf: dict[str, Any]) -> str | None:
    method = str(scf.get("method", "hf")).lower().replace("-", "")
    basis = str(scf.get("basis", "6-31g*")).lower().replace(" ", "")
    d3bj = bool(scf.get("d3bj", False))
    if method in ("hf", "rhf") and "6-31g*" in basis and "+" not in basis:
        return "hf"
    if method in ("hf", "rhf") and "6-31+g*" in basis:
        return "hf-plus"
    if method == "b3lyp" and "6-31+g*" in basis and d3bj:
        return "b3lyp-plus"
    return None


def _resolve_out_dir(ds: dict[str, Any]) -> Path:
    out_dir = Path(str(ds["out_dir"])).expanduser()
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()
    return out_dir


def _paths_train_with_npz(cfg: dict[str, Any], *, cli: bool = False) -> bool:
    if cli:
        return True
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    if bool(paths.get("train_with_npz")):
        return True
    datasets = cfg.get("datasets") if isinstance(cfg.get("datasets"), list) else []
    return any(
        isinstance(d, dict) and str(d.get("mode", "")).strip().lower() == "train_with_npz"
        for d in datasets
    )


def _resolve_npz(
    cfg: dict[str, Any],
    *,
    cli_npz: Path | None = None,
) -> Path | None:
    """Existing mixed soap cache (manifest paths.npz or --npz)."""
    if cli_npz is not None:
        p = cli_npz.expanduser()
        if not p.is_absolute():
            p = (_ROOT / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            _die(f"--npz not found: {p}")
        return p
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    mix_npz_raw = paths.get("mix_npz") or paths.get("npz")
    if mix_npz_raw is None:
        return None
    soap_mix = Path(str(mix_npz_raw)).expanduser()
    if not soap_mix.is_absolute():
        soap_mix = (_ROOT / soap_mix).resolve()
    else:
        soap_mix = soap_mix.resolve()
    if not soap_mix.is_file():
        _die(f"paths.npz not found: {soap_mix}")
    return soap_mix


def _resolve_mix_dir(cfg: dict[str, Any], datasets: list[dict[str, Any]]) -> tuple[Path, str]:
    """Return (mix_out_dir, mix_prefix) for merged soap + shared ckpt."""
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    calc = cfg.get("calculate_parameter") if isinstance(cfg.get("calculate_parameter"), dict) else {}
    mix_raw = None
    if paths and paths.get("mix_out") is not None:
        mix_raw = paths.get("mix_out")
    elif calc.get("mix_out") is not None:
        mix_raw = calc.get("mix_out")
    if mix_raw is not None:
        mix_out = Path(str(mix_raw)).expanduser()
        if not mix_out.is_absolute():
            mix_out = (_ROOT / mix_out).resolve()
        else:
            mix_out = mix_out.resolve()
    else:
        mix_out = _resolve_out_dir(datasets[0]).parent / "_mix"
    prefix = "mix"
    if paths and paths.get("mix_prefix"):
        prefix = str(paths["mix_prefix"]).strip() or "mix"
    elif calc.get("mix_prefix"):
        prefix = str(calc["mix_prefix"]).strip() or "mix"
    return mix_out, prefix


def _train_mmh_argv(
    train: dict[str, Any],
    *,
    soap_npz: Path | None,
    ckpt_path: Path,
    check: bool = False,
) -> list[str]:
    cmd = [
        str(_ROOT / "train_soap_e0_mix_mmh.py"),
        "--ckpt",
        str(ckpt_path),
        "--epochs",
        str(train["epochs"]),
        "--lr",
        str(train["lr"]),
        "--weight-decay",
        str(train["weight_decay"]),
        "--batch",
        str(train["batch"]),
        "--patience",
        str(train["patience"]),
        "--device",
        str(train["device"]),
        "--seed",
        str(train["seed"]),
        "--val-frac",
        str(train["val_frac"]),
        "--weight-e0",
        str(train["weight_e0"]),
        "--weight-corr",
        str(train["weight_corr"]),
        "--corr-target",
        str(train["corr_target"]),
        "--e-sites",
        str(train.get("e_sites", "all")),
    ]
    if check:
        cmd.append("--check")
    elif soap_npz is not None:
        cmd.extend(["--soap-cache", str(soap_npz)])
    
    if train["monitor"] is not None:
        cmd.extend(["--monitor", str(train["monitor"])])
    return cmd


def _check_train_mmh(*, py: str, train: dict[str, Any]) -> None:
    """Import/validate train stack before precompute or loading soap npz."""
    cmd = [py] + _train_mmh_argv(
        train, soap_npz=None, ckpt_path=Path("soap_e0_mix_mmh.pt"), check=True
    )
    print("[run.py] train preflight (--check only; no real training)", flush=True)
    _run(cmd, step="train preflight")


def _run_train_mmh(
    *,
    py: str,
    soap_npz: Path,
    ckpt_path: Path,
    train: dict[str, Any],
    log_path: Path | None = None,
) -> None:
    cmd = [py] + _train_mmh_argv(
        train, soap_npz=soap_npz, ckpt_path=ckpt_path, check=False
    )
    if log_path is None:
        _run(cmd, step=f"train → {ckpt_path.name}")
    else:
        _run_tee(cmd, step=f"train → {ckpt_path.name}", log_path=log_path)


def _run_one_dataset(
    *,
    ds: dict[str, Any],
    repulsion: dict[str, Any],
    calculate: dict[str, Any],
    train: dict[str, Any],
    py: str,
    ckpt_cli: Path | None = None,
    run_train: bool = True,
    run_scf: bool = True,
    train_log_path: Path | None = None,
    skip_ref: bool = False,
    skip_precompute: bool = False,
    shared_ckpt: Path | None = None,
) -> dict[str, Any]:
    if "out_dir" not in ds:
        _die('dataset requires "out_dir"')
    if "prefix" not in ds:
        _die('dataset requires "prefix" (e.g. "H2O" → H2O.npz / H2O.ckpt)')
    if "QM_atoms" not in ds and "qm_atoms" not in ds:
        _die('dataset requires "QM_atoms"')
    if "e0_file" not in ds:
        _die('dataset requires "e0_file" (filename under out_dir)')

    out_dir = _resolve_out_dir(ds)
    if not out_dir.is_dir():
        _die(f"out_dir does not exist (create it yourself): {out_dir}")

    coo_src = out_dir / "Coo"
    if not coo_src.is_dir():
        _die(f"missing Coo/ under out_dir: {coo_src}")

    prefix = str(ds["prefix"]).strip()
    if not prefix:
        _die("prefix must be non-empty")
    qm_atoms = int(ds.get("QM_atoms", ds.get("qm_atoms")))
    if "i0" in ds and isinstance(ds.get("n_frames"), list):
        print(
            f"[run.py] n_frames is a list → ignoring i0={ds.get('i0')!r}",
            flush=True,
        )
    file_indices, contiguous_i0 = _parse_frame_indices(
        ds.get("n_frames"),
        i0_default=int(ds.get("i0", 1)),
    )
    n_frames = len(file_indices)
    mode = _normalize_mode(ds.get("mode", "train"))
    e0_name = str(ds["e0_file"])
    e0_path = out_dir / e0_name if not Path(e0_name).is_absolute() else Path(e0_name)
    ckpt_override: Path | None = ckpt_cli
    if ckpt_override is None and ds.get("ckpt") is not None:
        raw_ck = Path(str(ds["ckpt"])).expanduser()
        ckpt_override = raw_ck if raw_ck.is_absolute() else (out_dir / raw_ck)

    if contiguous_i0 is None:
        print(
            f"[run.py] mode={mode}  n_frames=list ({n_frames} Coo indices; i0 unused): "
            f"{file_indices}",
            flush=True,
        )
    else:
        print(
            f"[run.py] mode={mode}  n_frames={n_frames} contiguous "
            f"Coo{contiguous_i0}..Coo{contiguous_i0 + n_frames - 1}",
            flush=True,
        )

    scf_kind, scf_file_indices = _parse_scf_frames(
        ds.get("scf_frames", "all_frames"),
        mode=mode,
        dataset_frames=file_indices,
    )
    if scf_kind == "all_val" and n_frames < 2:
        _die(
            'scf_frames="all_val" needs n_frames>=2 (single-frame has empty val split). '
            'Use "all_frames" or a frame list.'
        )
    if scf_kind == "list":
        print(
            f"[run.py] scf_frames=list (subset of n_frames) indices={scf_file_indices}",
            flush=True,
        )
    else:
        print(f"[run.py] scf_frames={scf_kind!r}", flush=True)

    # Validate Coo files + MM_atoms inferred per frame
    from embr_io import COO_NAME_FMT, resolve_geometry_path

    fi0 = file_indices[0]
    coo0 = resolve_geometry_path(coo_src, COO_NAME_FMT, fi0)
    if not coo0.is_file():
        _die(f"missing {coo0}")
    n_lines = _count_coo_atom_lines(coo0)
    if n_lines < qm_atoms:
        _die(f"{coo0}: {n_lines} atom lines < QM_atoms={qm_atoms}")
    n_mm = n_lines - qm_atoms
    for fi in file_indices[1:]:
        coo = resolve_geometry_path(coo_src, COO_NAME_FMT, fi)
        if not coo.is_file():
            _die(f"missing {coo}")
        n_i = _count_coo_atom_lines(coo)
        if n_i != n_lines:
            _die(f"{coo}: {n_i} atom lines != Coo{fi0} ({n_lines})")
    if len(file_indices) == 1:
        print(
            f"[run.py] Coo{fi0}: atoms={n_lines}  QM={qm_atoms}  MM={n_mm}",
            flush=True,
        )
    else:
        print(
            f"[run.py] Coo{file_indices[0]}..Coo{file_indices[-1]} "
            f"({len(file_indices)} frames): atoms={n_lines}  QM={qm_atoms}  MM={n_mm}",
            flush=True,
        )

    # dE labels: train/scf require file; ab_initio creates/overwrites after batch_QM
    if mode in ("train", "scf"):
        _validate_e0_file(e0_path, file_indices=file_indices, compact=True)
    elif mode == "ab_initio":
        if e0_path.is_file():
            n_have = len(_e0_data_rows(e0_path))
            print(
                f"[run.py] ab_initio: {e0_path.name} has {n_have} line(s); "
                f"batch_QM updates dE labels",
                flush=True,
            )
        else:
            print(
                f"[run.py] ab_initio: batch_QM will write {e0_path.name}",
                flush=True,
            )

    dataset_label = str(prefix).strip().lower()
    scf_cfg = dict(calculate["scf"])
    q_qm = _dataset_qm_charge(ds, scf_cfg)
    scf_cfg["qm_charge"] = int(q_qm)
    print(
        f"[run.py] prefix={prefix!r}  QM_atoms={qm_atoms}  qm_charge={q_qm:+d}",
        flush=True,
    )
    charge_mode = str(calculate.get("charge_mode", "tip3p"))
    charged = out_dir / "Coo_charge"
    if skip_ref and skip_precompute and charged.is_dir():
        coo_dir = charged
        print(f"[run.py] reuse charged Coo → {coo_dir}/", flush=True)
    else:
        coo_dir = _prepare_coo_dir(
            coo_src=coo_src,
            out_dir=out_dir,
            qm_atoms=qm_atoms,
            file_indices=file_indices,
            charge_mode=charge_mode,
            scf_cfg=scf_cfg,
        )

    compat_path = out_dir / "_run_compat_manifest.json"
    _write_compat_manifest(
        path=compat_path,
        repulsion=repulsion,
        calculate=calculate,
        prefix=prefix,
        dataset_label=dataset_label,
        qm_atoms=qm_atoms,
        coo_dir=coo_dir,
        file_indices=file_indices,
        contiguous_i0=contiguous_i0,
        e0_file=e0_path,
    )
    ref_dir = out_dir / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    soap_npz = out_dir / f"{prefix}.npz"
    default_ckpt = out_dir / f"{prefix}.ckpt"
    ckpt_path = Path(ckpt_override) if ckpt_override is not None else default_ckpt
    if ckpt_override is not None:
        ckpt_path = ckpt_path.expanduser().resolve()
    if shared_ckpt is not None:
        ckpt_path = Path(shared_ckpt).expanduser().resolve()
    scf_dir = out_dir / "scf"
    frame_start = min(file_indices)
    frame_end = max(file_indices)
    threads = int(scf_cfg["threads"])
    preset = _scf_preset_from_calc(scf_cfg)

    # ab_initio always compute Emb0+CP → E0.
    do_ref = mode in ("ab_initio", "train")
    do_write_e0 = mode == "ab_initio"
    do_train = bool(run_train) and mode in ("ab_initio", "train", "train_with_npz")
    do_scf_after = bool(run_scf) and mode in ("ab_initio", "train", "scf")

    if mode == "scf":
        if not ckpt_path.is_file():
            _die(
                f'mode=scf needs a checkpoint; missing {ckpt_path}. '
                f"Pass --ckpt PATH or datasets[].ckpt, or place {default_ckpt.name} under out_dir."
            )
        print(f"[run.py] scf mode ckpt → {ckpt_path}", flush=True)
    elif shared_ckpt is not None and do_scf_after:
        print(f"[run.py] scf shared ckpt → {ckpt_path}", flush=True)

    # --- 1) Emb0 + Cluster/CP ref (skip-existing; resume-friendly) ---
    if do_ref and not skip_ref:
        cmd = [
            py,
            str(_ROOT / "batch_hf_emb0_cp.py"),
            "--manifest",
            str(compat_path),
            "--prefix",
            str(prefix),
            "--frame-start",
            str(frame_start),
            "--frame-end",
            str(frame_end),
            "--out-dir",
            str(ref_dir),
            "--e0-out",
            str(e0_path),
            "--threads",
            str(threads),
            "--skip-existing",
        ]
        if preset:
            cmd.extend(["--scf-preset", preset])
        else:
            cmd.extend(
                [
                    "--method",
                    str(scf_cfg["method"]),
                    "--basis",
                    str(scf_cfg["basis"]),
                ]
            )
            if scf_cfg.get("d3bj"):
                cmd.append("--d3bj")
            else:
                cmd.append("--no-d3bj")
        cmd.extend(["--verbose-scf", str(int(scf_cfg.get("verbose_scf", 0)))])
        cmd.extend(["--qm-charge", str(int(scf_cfg.get("qm_charge", 0)))])
        if mode == "train":
            print(
                "[run.py] train: Emb0/CP ref via --skip-existing "
                "(reuse out_dir/ref when present)",
                flush=True,
            )
        _run(
            cmd,
            step="batch_QM → ref/",
            hint="Running full_QM reference calculations …",
        )
        missing_ref = [
            ref_dir / f"ref_{fi}.npz"
            for fi in file_indices
            if not (ref_dir / f"ref_{fi}.npz").is_file()
        ]
        if missing_ref:
            names = ", ".join(p.name for p in missing_ref[:5])
            more = f" (+{len(missing_ref) - 5} more)" if len(missing_ref) > 5 else ""
            _die(
                f"batch_hf finished but missing ref npz: {names}{more}. "
                f"Check Coo_charge / charge_mode (tip3p=waters+ions; organic→mulliken)."
            )
        if do_write_e0:
            _write_e0_from_ref_npz(
                e0_path, ref_dir=ref_dir, file_indices=file_indices
            )

    # --- 2) precompute (SOAP + k_j from ref dm) ---
    cmd = [
        py,
        str(_ROOT / "precompute_soap_e0_mix_mmh.py"),
        "--manifest",
        str(compat_path),
        "--out",
        str(soap_npz),
        "--workers",
        str(calculate["workers"]),
        "--mm-env",
        "full",
    ]
    cmd.extend(["--ref-dir", str(ref_dir), "--e0-from-txt"])
    if skip_precompute:
        if not soap_npz.is_file():
            _die(f"skip_precompute but missing soap cache: {soap_npz}")
    else:
        _run(cmd, step=f"precompute → {soap_npz.name}")

    # --- 3) train (ab_initio / train) ---
    if do_train:
        train_ckpt = (
            ckpt_path
            if (ckpt_override is not None or shared_ckpt is not None)
            else default_ckpt
        )
        _run_train_mmh(
            py=py, soap_npz=soap_npz, ckpt_path=train_ckpt, train=train,
            log_path=train_log_path,
        )
        ckpt_path = train_ckpt

    # --- 4) batch_scf → out_dir/scf/ (optional scf_skip_existing; Emb0/DFT always skip) ---
    if do_scf_after:
        scf_dir.mkdir(parents=True, exist_ok=True)
        scf_manifest = compat_path
        scf_skip = bool(calculate.get("scf_skip_existing", False))
        cmd = [
            py,
            str(_ROOT / "batch_scf_mix_mmh_val.py"),
            "--soap-cache",
            str(soap_npz),
            "--manifest",
            str(scf_manifest),
            "--ref-dir",
            str(ref_dir),
            "--ref-embtheta",
            "--out-dir",
            str(scf_dir),
            "--threads",
            str(threads),
            "--val-frac",
            str(train["val_frac"]),
            "--seed",
            str(train["seed"]),
        ]
        if scf_skip:
            cmd.append("--skip-existing")
        print(
            f"[run.py] SCF → {scf_dir}/ "
            + (
                "(scf_skip_existing=true; skip frames with scf*.npz)"
                if scf_skip
                else "(scf_skip_existing=false; re-run EmbR)"
            ),
            flush=True,
        )
        if scf_kind == "all_frames":
            cmd.append("--all-frames")
        elif scf_kind == "all_val":
            print(
                f"[run.py] scf_frames=all_val → val split "
                f"(val_frac={train['val_frac']}, seed={train['seed']})",
                flush=True,
            )
            n_val_take = calculate.get("scf_n_val_frames")
            if n_val_take is not None:
                cmd.extend(["--n-val-frames", str(int(n_val_take))])
                print(
                    f"[run.py] scf_n_val_frames={int(n_val_take)} "
                    f"→ first {int(n_val_take)} val frames only",
                    flush=True,
                )
        else:
            assert scf_file_indices is not None
            cache_ks = _cache_ks_from_file_indices(soap_npz, scf_file_indices)
            cmd.extend(
                [
                    "--only-cache-indices",
                    ",".join(str(k) for k in cache_ks),
                ]
            )
            print(
                f"[run.py] scf_frames list file_index={scf_file_indices} → cache_k={cache_ks}",
                flush=True,
            )
        if not ckpt_path.is_file():
            _die(f"SCF needs checkpoint, missing {ckpt_path}")
        cmd.extend(["--ckpt", str(ckpt_path)])
        e_sites_scf = str(train.get("e_sites", "all") or "all")
        if e_sites_scf != "all":
            cmd.extend(["--repulsion-policy", e_sites_scf])
        print(
            f"[run.py] SCF labels from ML ckpt → {ckpt_path}",
            flush=True,
        )
        if preset:
            cmd.extend(["--scf-preset", preset])
        else:
            cmd.extend(
                [
                    "--method",
                    str(scf_cfg["method"]),
                    "--basis",
                    str(scf_cfg["basis"]),
                ]
            )
            if scf_cfg.get("d3bj"):
                cmd.append("--d3bj")
            else:
                cmd.append("--no-d3bj")
        cmd.extend(["--verbose-scf", str(int(scf_cfg.get("verbose_scf", 0)))])
        cmd.extend(["--qm-charge", str(int(scf_cfg.get("qm_charge", 0)))])
        if bool(calculate.get("scf_A_fix", False)):
            scf_err = float(calculate.get("scf_error", 0.5))
            cmd.extend(["--a-fix", "--a-fix-tol", str(scf_err)])
            amax = calculate.get("scf_A_fix_max")
            if amax is not None:
                cmd.extend(["--a-fix-max", str(int(amax))])
            print(
                f"[run.py] scf_A_fix → batch_scf --a-fix --a-fix-tol {scf_err:g}"
                + (f" --a-fix-max {int(amax)}" if amax is not None else ""),
                flush=True,
            )
        _run(cmd, step=f"batch_scf → {scf_dir}/")

    print(f"\n[run.py] dataset done → {out_dir}", flush=True)
    return {
        "out_dir": out_dir,
        "soap_npz": soap_npz,
        "ckpt_path": ckpt_path,
        "mode": mode,
        "do_scf_after": do_scf_after,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end runner from a unified manifest (ab_initio / train / scf / train_with_npz)"
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="unified JSON (repulsion_parameter / calculate_parameter / datasets / train)",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="checkpoint for mode=scf (or override train output path if set with care)",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="existing mixed soap cache for train_with_npz (overrides manifest paths.npz)",
    )
    args = ap.parse_args()
    manifest_path = Path(args.manifest).resolve()
    cfg = _load_json(manifest_path)

    repulsion = _parse_repulsion(cfg)
    calculate = _parse_calculate(cfg)
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        _die('manifest needs non-empty "datasets": [ {...}, ... ]')

    modes = [
        _normalize_mode(d.get("mode", "train"))
        for d in datasets
        if isinstance(d, dict)
    ]
    need_train = any(m in ("train", "ab_initio", "train_with_npz") for m in modes)
    if need_train:
        train = _parse_train(cfg)
    else:
        train = _train_stub()
        # Optional: pick up val_frac/seed from train{} for scf all_val even without lr
        tr_opt = _block(cfg, "train")
        if tr_opt:
            if "val_frac" in tr_opt:
                train["val_frac"] = float(tr_opt["val_frac"])
            if "seed" in tr_opt:
                train["seed"] = int(tr_opt["seed"])

    ckpt_cli = Path(args.ckpt).expanduser().resolve() if args.ckpt is not None else None

    train_log_path: Path | None = None
    if need_train:
        train_log_path = manifest_path.parent / "loss.out"
        train_log_path.write_text(
            f"# EmbR training log\n# manifest: {manifest_path}\n",
            encoding="utf-8",
        )
        print(f"[run.py] training log → {train_log_path}", flush=True)

    py = sys.executable
    # Fail on import / train-arg errors before any ref / precompute / soap npz I/O.
    if need_train:
        _check_train_mmh(py=py, train=train)

    n_ds = len(datasets)
    mix_train = n_ds > 1 and need_train
    if mix_train:
        if any(m == "train_with_npz" for m in modes) and n_ds > 1 and need_train:
            _die(
                "multi-dataset train_with_npz: use one shared --npz / paths.npz."
            )
        print(
            f"\n[run.py] {n_ds} datasets → MIXED training (one shared ckpt). "
            f"For per-residue ckpts, use one dataset per manifest.",
            flush=True,
        )
        mix_out, mix_prefix = _resolve_mix_dir(cfg, datasets)
        mix_out.mkdir(parents=True, exist_ok=True)
        ckpt_mix = (
            ckpt_cli if ckpt_cli is not None else (mix_out / f"{mix_prefix}.ckpt")
        )
        train_with_npz = _paths_train_with_npz(cfg, cli=args.npz is not None)
        mix_npz = _resolve_npz(cfg, cli_npz=args.npz)
        if train_with_npz:
            if mix_npz is None:
                _die(
                    "train_with_npz requires an existing mixed npz: "
                    "set paths.npz in manifest or pass --npz PATH"
                )
            print(
                f"\n########## mixed train (train_with_npz) ##########\n"
                f"[run.py] skip ref/precompute/merge — train on {mix_npz}\n"
                f"[run.py] ckpt → {ckpt_mix}",
                flush=True,
            )
            soap_mix = mix_npz
        else:
            soap_paths: list[Path] = []
            for i, ds in enumerate(datasets):
                if not isinstance(ds, dict):
                    _die(f"datasets[{i}] must be an object")
                print(
                    f"\n########## dataset[{i}] prepare (no train/scf) ##########",
                    flush=True,
                )
                info = _run_one_dataset(
                    ds=ds,
                    repulsion=repulsion,
                    calculate=calculate,
                    train=train,
                    py=py,
                    ckpt_cli=ckpt_cli,
                    run_train=False,
                    run_scf=False,
                    train_log_path=train_log_path,
                )
                soap_paths.append(Path(info["soap_npz"]))

            soap_mix = mix_out / f"{mix_prefix}.npz"
            if mix_npz is not None:
                soap_mix = mix_npz
                print(
                    f"\n########## mixed train ##########\n"
                    f"[run.py] skip merge — train on existing {soap_mix}",
                    flush=True,
                )
            else:
                print(
                    f"\n########## mixed train ##########\n"
                    f"[run.py] merge {len(soap_paths)} soap caches → {soap_mix}",
                    flush=True,
                )
                from embr_cache import merge_mmh_caches

                merge_mmh_caches(soap_paths, soap_mix)
        _run_train_mmh(
            py=py, soap_npz=soap_mix, ckpt_path=ckpt_mix, train=train,
            log_path=train_log_path,
        )
        if train_with_npz:
            print("[run.py] train_with_npz complete (train-only; no ref/precompute/SCF).", flush=True)
        else:
            for i, ds in enumerate(datasets):
                print(
                    f"\n########## dataset[{i}] SCF (shared ckpt) ##########",
                    flush=True,
                )
                _run_one_dataset(
                    ds=ds,
                    repulsion=repulsion,
                    calculate=calculate,
                    train=train,
                    py=py,
                    ckpt_cli=ckpt_cli,
                    run_train=False,
                    run_scf=True,
                    skip_ref=True,
                    skip_precompute=True,
                    shared_ckpt=ckpt_mix,
                    train_log_path=train_log_path,
                )
    else:
        single_tw = (
            n_ds == 1
            and need_train
            and _normalize_mode(datasets[0].get("mode", "train")) == "train_with_npz"
        )
        if single_tw:
            ds0 = datasets[0]
            npz_path = _resolve_npz(cfg, cli_npz=args.npz)
            if npz_path is None:
                _die("train_with_npz requires --npz or paths.npz in manifest")
            out_dir = _resolve_out_dir(ds0)
            prefix = str(ds0["prefix"])
            ckpt_path = (
                ckpt_cli if ckpt_cli is not None else (out_dir / f"{prefix}.ckpt")
            )
            print(
                f"\n########## train_with_npz (single dataset) ##########\n"
                f"[run.py] train on {npz_path} → {ckpt_path}",
                flush=True,
            )
            _run_train_mmh(
                py=py, soap_npz=npz_path, ckpt_path=ckpt_path, train=train,
                log_path=train_log_path,
            )
            print("[run.py] train_with_npz complete (train-only; no ref/precompute/SCF).", flush=True)
        elif n_ds > 1:
            print(
                f"\n[run.py] {n_ds} datasets, no training → run each independently "
                f"(no mixed ckpt).",
                flush=True,
            )
        if not single_tw:
            for i, ds in enumerate(datasets):
                if not isinstance(ds, dict):
                    _die(f"datasets[{i}] must be an object")
                print(f"\n########## dataset[{i}] ##########", flush=True)
                _run_one_dataset(
                    ds=ds,
                    repulsion=repulsion,
                    calculate=calculate,
                    train=train,
                    py=py,
                    ckpt_cli=ckpt_cli,
                    train_log_path=train_log_path,
                )

    print("\n[run.py] all datasets finished.", flush=True)


if __name__ == "__main__":
    main()
