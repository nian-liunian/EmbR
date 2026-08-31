"""
Persistent per-Coo Emb0 rho-kernel cache for precompute (legacy path without --ref-dir).

Each Coo frame + SCF settings -> one ``.npz`` with ``kernel_mm``, ``mm_element``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scf_embed_io import ScfFrame, filter_mm_by_distance, load_scf_frame


def geom_hash_frame(frame: ScfFrame) -> str:
    blob = np.asarray(
        np.concatenate([frame.qm_coords_ang, frame.mm_coords_ang], axis=0),
        dtype=np.float64,
    ).tobytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def emb0_settings_tag(
    *,
    fix_alpha: float,
    scf_cfg_kw: dict,
    r_cut_mm: float | None,
    envelope_tag: str | None = None,
) -> str:
    payload = {
        "fix_alpha": float(fix_alpha),
        "r_cut_mm": None if r_cut_mm is None else float(r_cut_mm),
        "envelope_tag": str(envelope_tag) if envelope_tag else None,
        "scf": {
            "method": str(scf_cfg_kw.get("method", scf_cfg_kw.get("xc", "b3lyp"))),
            "basis": str(scf_cfg_kw["basis"]),
            "use_d3bj": bool(scf_cfg_kw.get("use_d3bj", False)),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def emb0_cache_filename(
    *,
    geom_hash: str,
    n_qm: int,
    settings_tag: str,
) -> str:
    return f"emb0_{geom_hash}_qm{n_qm}_{settings_tag}.npz"


def emb0_cache_path(
    cache_dir: Path,
    *,
    geom_hash: str,
    n_qm: int,
    settings_tag: str,
) -> Path:
    return cache_dir / emb0_cache_filename(
        geom_hash=geom_hash, n_qm=int(n_qm), settings_tag=settings_tag
    )


def load_emb0_frame_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    if "kernel_mm" not in z or "mm_element" not in z:
        return None
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
    return {
        "kernel_mm": np.asarray(z["kernel_mm"], dtype=np.float64).reshape(-1),
        "mm_element": np.asarray(z["mm_element"], dtype=np.int8).reshape(-1),
        "meta": meta,
    }


def cache_meta_matches(
    cached: dict,
    *,
    coo_path: str,
    n_qm: int,
    geom_hash: str,
    fix_alpha: float,
    scf_cfg_kw: dict,
    r_cut_mm: float | None,
    settings_tag: str,
) -> bool:
    meta = cached.get("meta") or {}
    if str(meta.get("coo_path", "")) != str(coo_path):
        return False
    if int(meta.get("n_qm", -1)) != int(n_qm):
        return False
    if str(meta.get("geom_hash", "")) != str(geom_hash):
        return False
    if str(meta.get("settings_tag", "")) != str(settings_tag):
        return False
    if abs(float(meta.get("fix_alpha", float("nan"))) - float(fix_alpha)) > 1e-12:
        return False
    scf = meta.get("scf_config") or {}
    for key in ("method", "basis", "use_d3bj"):
        if key in scf_cfg_kw and key in scf:
            if key == "use_d3bj":
                if bool(scf[key]) != bool(scf_cfg_kw[key]):
                    return False
            elif str(scf[key]) != str(scf_cfg_kw[key]):
                return False
    rc_meta = meta.get("r_cut_mm")
    if rc_meta is None and r_cut_mm is None:
        pass
    elif rc_meta is None or r_cut_mm is None:
        return False
    elif abs(float(rc_meta) - float(r_cut_mm)) > 1e-9:
        return False
    return True


def save_emb0_frame_cache(
    path: Path,
    *,
    kernel_mm: np.ndarray,
    mm_element: np.ndarray,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kernel_mm=np.asarray(kernel_mm, dtype=np.float64).reshape(-1),
        mm_element=np.asarray(mm_element, dtype=np.int8).reshape(-1),
        meta_json=np.array(json.dumps(meta)),
    )


def load_or_compute_emb0_kernels(
    coo_path: str,
    n_qm: int,
    *,
    cache_dir: Path | None,
    fix_alpha: float,
    scf_cfg_kw: dict,
    r_cut_mm: float | None,
    envelope_tag: str | None = None,
    force: bool,
    compute_fn,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (kernel_mm, mm_element, from_cache). ``compute_fn()`` runs Emb0 SCF."""
    frame = load_scf_frame(Path(coo_path), n_qm=int(n_qm))
    frame = filter_mm_by_distance(frame, r_cut_ang=r_cut_mm)
    gh = geom_hash_frame(frame)
    tag = emb0_settings_tag(
        fix_alpha=float(fix_alpha),
        scf_cfg_kw=scf_cfg_kw,
        r_cut_mm=r_cut_mm,
        envelope_tag=envelope_tag,
    )

    if cache_dir is not None and not force:
        cp = emb0_cache_path(cache_dir, geom_hash=gh, n_qm=int(n_qm), settings_tag=tag)
        hit = load_emb0_frame_cache(cp)
        if hit is not None and cache_meta_matches(
            hit,
            coo_path=str(Path(coo_path).resolve()),
            n_qm=int(n_qm),
            geom_hash=gh,
            fix_alpha=float(fix_alpha),
            scf_cfg_kw=scf_cfg_kw,
            r_cut_mm=r_cut_mm,
            settings_tag=tag,
        ):
            return hit["kernel_mm"], hit["mm_element"], True

    kernel_mm, mm_element = compute_fn()
    if cache_dir is not None:
        cp = emb0_cache_path(cache_dir, geom_hash=gh, n_qm=int(n_qm), settings_tag=tag)
        save_emb0_frame_cache(
            cp,
            kernel_mm=kernel_mm,
            mm_element=mm_element,
            meta={
                "pipeline": "soap_e0_emb0_cache",
                "coo_path": str(Path(coo_path).resolve()),
                "n_qm": int(n_qm),
                "geom_hash": gh,
                "settings_tag": tag,
                "fix_alpha": float(fix_alpha),
                "envelope_tag": envelope_tag,
                "r_cut_mm": r_cut_mm,
                "scf_config": {
                    "method": str(scf_cfg_kw.get("method", "b3lyp")),
                    "basis": str(scf_cfg_kw["basis"]),
                    "use_d3bj": bool(scf_cfg_kw.get("use_d3bj", False)),
                },
            },
        )
    return kernel_mm, mm_element, False
