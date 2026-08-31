"""
Batch SCF + fix output for mix_mmh validation frames (ML e_j → A_j → line_rho).

Uses ``embr_theta.line_rho_pert_cluster``: Emb0 + EmbR SCF → ``scf*.npz``.

Example::

  python batch_scf_mix_mmh_val.py \\
    --ckpt mix.ckpt \\
    --soap-cache mix.npz \\
    --out-dir scf_out \\
    --threads 4

Resume::

  python batch_scf_mix_mmh_val.py ... --skip-existing --skip-export
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _resolve_project_root() -> Path:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--project-root" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if a.startswith("--project-root="):
            return Path(a.split("=", 1)[1]).resolve()
    env = os.environ.get("GLY_DESCRIPTOR_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent


def _bootstrap_import_paths(root: Path) -> Path:
    theta = root / "embr_theta"
    if not theta.is_dir():
        raise SystemExit(
            f"[batch_scf] missing embr_theta/ at {theta}\n"
            "EmbR λ pipeline needs embr_theta/line_rho_pert_cluster.py in this repo."
        )
    for p in (theta, root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


_PROJECT_ROOT = _bootstrap_import_paths(_resolve_project_root())

from embr_theta._deps import require_scf

require_scf()

import numpy as np

from scf_embed_io import filter_mm_by_distance, load_scf_frame
from scf_embed_pyscf import resolve_scf_preset, scf_embed_config_from_cli, resolve_qm_charge
from embr_cache import load_mmh_cache, n_frames
from embr_infer import (
    attach_cache_path,
    build_val_peratom_labels,
    expand_kernel_amp_to_scf_frame,
    expand_kernel_k_to_scf_frame,
    frame_meta_entry,
    save_peratom_npz,
)
from embr_envelope import MmhEnvelopeConfig
from embr_theta.cache import amp_mm_frame_slice, load_pert_peratom_labels


from embr_theta.rho_plane_geom import nearest_mm_h_index_to_qm


def _align_amp_k_to_frame(
    frame,
    amp_raw: np.ndarray,
    ker_raw: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Peratom labels may store kernel-compact or full-frame amp/k; normalize for MM# indexing."""
    amp_raw = np.asarray(amp_raw, dtype=np.float64).reshape(-1)
    n_mm = len(frame.mm_symbols)
    if amp_raw.size == n_mm:
        amp_mm = amp_raw
    else:
        amp_mm = expand_kernel_amp_to_scf_frame(frame, amp_raw)
    k_mm = None
    if ker_raw is not None:
        ker_raw = np.asarray(ker_raw, dtype=np.float64).reshape(-1)
        if ker_raw.size == n_mm:
            k_mm = ker_raw
        else:
            k_mm = expand_kernel_k_to_scf_frame(frame, ker_raw)
    return amp_mm, k_mm


from embr_ref_kernels import (
    compute_mix_err_metrics,
    format_mix_err_one_line,
    load_ref_hf_npz,
    ref_npz_path,
    scf_meta_from_ref_npz,
    spillover_profiles_from_ref_npz,
    spillover_profiles_ref_embtheta,
)
from train_soap_e0_mix_mmh import _split_indices


def _artifact_run_id(task: dict) -> int:
    """Output filenames scf{N}.npz use Coo file_index."""
    fi = task.get("file_index")
    if fi is not None:
        return int(fi)
    return int(task.get("val_run_id", int(task["slice_index"]) + 1))


def _fix_suffix(fix_iter: int) -> str:
    """fix_iter=0 → ''; 1 → '_fix1'."""
    n = int(fix_iter)
    return "" if n <= 0 else f"_fix{n}"


def _scf_npz_path(out_dir: Path, *, file_index: int, fix_iter: int = 0) -> Path:
    return out_dir / f"scf{int(file_index)}{_fix_suffix(fix_iter)}.npz"


def _existing_amp_scale(out_dir: Path, *, file_index: int, fix_iter: int = 0) -> float | None:
    """Return amp_scale stored in scf{N}[_fixk].npz, or None if artifact missing/unreadable."""
    scf_p = _scf_npz_path(out_dir, file_index=file_index, fix_iter=fix_iter)
    if not scf_p.is_file():
        return None
    try:
        with np.load(scf_p, allow_pickle=True) as z:
            if "amp_scale" in z.files:
                return float(z["amp_scale"])
    except OSError:
        return None
    # Older runs without the key: treat as 1.0
    return 1.0


def _scf_a_fix_complete(out_dir: Path, *, file_index: int, fix_iter: int) -> bool:
    """True if scf npz has fields needed for ΔE_ML / ΔE_SCF plotting."""
    scf_p = _scf_npz_path(out_dir, file_index=file_index, fix_iter=fix_iter)
    if not scf_p.is_file():
        return False
    need = ("e0_pred_kcal", "e_int_emb0_kcal", "e_int_embtheta_kcal", "amp_scale")
    try:
        with np.load(scf_p, allow_pickle=True) as z:
            return all(k in z.files for k in need)
    except OSError:
        return False


def _mix_m_from_scf_npz(out_dir: Path, *, file_index: int, fix_iter: int) -> dict[str, float] | None:
    """Rebuild mix_m from finished scf{N}[_fixk].npz (--skip-existing / plot)."""
    scf_p = _scf_npz_path(out_dir, file_index=file_index, fix_iter=fix_iter)
    if not scf_p.is_file():
        return None
    try:
        with np.load(scf_p, allow_pickle=True) as z:
            e0_pred = float(z["e0_pred_kcal"])
            e0_lab = float(z["e0_label_kcal"]) if "e0_label_kcal" in z.files else float("nan")
            e_emb0 = float(z["e_int_emb0_kcal"])
            e_th = float(z["e_int_embtheta_kcal"])
            e_ref = (
                float(z["e_int_qmmm_ref_kcal"])
                if "e_int_qmmm_ref_kcal" in z.files
                else float("nan")
            )
            if "e_int_pred_kcal" in z.files:
                e_pred = float(z["e_int_pred_kcal"])
            else:
                e_pred = (
                    float(e0_pred + e_emb0)
                    if np.isfinite(e0_pred) and np.isfinite(e_emb0)
                    else float("nan")
                )
            lam = float(z["amp_scale"]) if "amp_scale" in z.files else 1.0
            fix_i = int(z["a_fix_iter"]) if "a_fix_iter" in z.files else int(fix_iter)
            if "signed_embtheta_vs_pred_kcal" in z.files:
                signed = float(z["signed_embtheta_vs_pred_kcal"])
            else:
                signed = _signed_embtheta_vs_pred(
                    e_int_embtheta=e_th, e_int_pred=e_pred
                )
            if "abs_embtheta_vs_pred_kcal" in z.files:
                abs_vs_pred = float(z["abs_embtheta_vs_pred_kcal"])
            else:
                abs_vs_pred = abs(signed) if np.isfinite(signed) else float("nan")
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return {
        "e0_pred": e0_pred,
        "e0_label": e0_lab,
        "e_int_emb0": e_emb0,
        "e_int_pred": e_pred,
        "e_int_embtheta": e_th,
        "e_int_qmmm_ref": e_ref,
        "amp_scale": lam,
        "a_fix_iter": float(fix_i),
        "signed_embtheta_err": signed,
        "abs_embtheta_vs_pred": abs_vs_pred,
    }


def _should_skip_existing_frame(
    out_dir: Path,
    *,
    file_index: int,
    amp_scale: float,
) -> bool:
    """Skip only if scf npz exists AND amp_scale matches (config-aware resume)."""
    if not _scf_npz_path(out_dir, file_index=file_index, fix_iter=0).is_file():
        return False
    prev = _existing_amp_scale(out_dir, file_index=file_index, fix_iter=0)
    if prev is None:
        return True
    if abs(float(prev) - float(amp_scale)) > 1e-12:
        return False
    return True


def _a_fix_next_lambda(
    *,
    lam: float,
    eps: float,
    e0_pred_kcal: float,
    hist: list[tuple[float, float]],
) -> tuple[float, str] | None:
    """
    Two-point / Newton update for uniform A-scale λ (A ← λ A_base).

    ε = E_int^EmbR − E_int^pred  with  E_int^pred = E0_pred + Emb0  (kcal).
    Want ε→0 (EmbR → ML interaction target, **not** QM/MM ref).
    λ_new = λ − ε / s.

    First correction: always s = E0_pred (not E0_label).
    Later: secant s = (ε1−ε0)/(λ1−λ0) from the last two points.
    """
    if not np.isfinite(eps) or not np.isfinite(lam):
        return None
    if len(hist) >= 2:
        lam0, eps0 = hist[-2]
        lam1, eps1 = hist[-1]
        denom = float(lam1 - lam0)
        if abs(denom) < 1e-14:
            return None
        s = float(eps1 - eps0) / denom
        tag = f"secant dε/dλ={s:.6g}"
    else:
        s = float(e0_pred_kcal)
        tag = f"first fix s=E0_pred={s:.6g}"
    if not np.isfinite(s) or abs(s) < 1e-12:
        return None
    lam_new = float(lam) - float(eps) / float(s)
    if not np.isfinite(lam_new) or lam_new <= 0.0:
        return None
    return lam_new, tag


def _signed_embtheta_vs_pred(
    *,
    e_int_embtheta: float,
    e_int_pred: float,
) -> float:
    """ε for A-fix: EmbR − (E0_pred+Emb0)."""
    e_th = float(e_int_embtheta)
    e_pr = float(e_int_pred)
    if np.isfinite(e_th) and np.isfinite(e_pr):
        return float(e_th - e_pr)
    return float("nan")


def _filter_run_val_skip_existing(
    cache: dict,
    out_dir: Path,
    run_val: list[int],
    *,
    amp_scale: float = 1.0,
) -> list[int]:
    """Drop cache_k rows whose scf npz matches current amp_scale (same-config resume)."""
    kept: list[int] = []
    n_skip = 0
    n_rerun = 0
    for k in run_val:
        fm = frame_meta_entry(cache, int(k))
        rid = int(fm.get("file_index", k))
        if _should_skip_existing_frame(out_dir, file_index=rid, amp_scale=amp_scale):
            n_skip += 1
            continue
        if _scf_npz_path(out_dir, file_index=rid, fix_iter=0).is_file():
            n_rerun += 1
        kept.append(int(k))
    if n_skip:
        print(
            f"[batch_scf] skip-existing: {n_skip} frames already have "
            f"scf*.npz with amp_scale={float(amp_scale):g}",
            flush=True,
        )
    if n_rerun:
        print(
            f"[batch_scf] skip-existing: {n_rerun} frames have scf*.npz but "
            f"amp_scale≠{float(amp_scale):g} → will re-run",
            flush=True,
        )
    return kept


def _prepare_amp_with_scale(
    amp_mm: np.ndarray,
    amp_scale: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (A_used, A_base) with optional uniform multiplicative perturbation."""
    scale = float(amp_scale)
    amp = np.asarray(amp_mm, dtype=np.float64)
    if abs(scale - 1.0) < 1e-15:
        return amp, None
    if scale <= 0.0:
        raise ValueError(f"amp-scale must be > 0, got {scale}")
    return amp * scale, amp.copy()


def _print_mix_err_summary_table(rows: list[tuple[int, dict[str, float]]]) -> None:
    if not rows:
        return
    print("\n[error summary] kcal/mol", flush=True)
    print(
        f"  {'frame':>8s}  {'dE_ML - dE_ref':>12s}  {'dE_SCF - dE_ML':>14s}  {'dE_SCF - dE_ref':>14s}",
        flush=True,
    )
    signed_e0_list: list[float] = []
    signed_scf_list: list[float] = []
    total_list: list[float] = []
    for run_id, m in rows:
        e0_pred = float(m.get("e0_pred", float("nan")))
        e0_label = float(m.get("e0_label", float("nan")))
        s_e0 = float(m.get("signed_e0_err", float("nan")))
        if not np.isfinite(s_e0) and np.isfinite(e0_pred) and np.isfinite(e0_label):
            s_e0 = float(e0_pred - e0_label)
        s_scf = float(m.get("signed_embtheta_err", float("nan")))
        if not np.isfinite(s_scf):
            s_scf = float(m.get("signed_embtheta_vs_pred", float("nan")))
        tot = float(s_e0 + s_scf) if np.isfinite(s_e0) and np.isfinite(s_scf) else float("nan")
        print(
            f"  {int(run_id):8d}  "
            f"{(s_e0 if np.isfinite(s_e0) else float('nan')):12.4f}  "
            f"{(s_scf if np.isfinite(s_scf) else float('nan')):14.4f}  "
            f"{(tot if np.isfinite(tot) else float('nan')):14.4f}",
            flush=True,
        )
        if np.isfinite(s_e0):
            signed_e0_list.append(s_e0)
        if np.isfinite(s_scf):
            signed_scf_list.append(s_scf)
        if np.isfinite(tot):
            total_list.append(tot)
    if signed_e0_list:
        print(
            f"  mean dE_ML - dE_ref = {float(np.mean(signed_e0_list)):.4f}  "
            f"(RMSE={float(np.sqrt(np.mean(np.square(signed_e0_list)))):.4f})",
            flush=True,
        )
    if signed_scf_list:
        print(
            f"  mean dE_SCF(n) - dE_ML = {float(np.mean(signed_scf_list)):.4f}",
            flush=True,
        )
    if total_list:
        print(f"  mean dE_SCF - dE_ref = {float(np.mean(total_list)):.4f}", flush=True)


def _run_one_from_ref(
    *,
    slice_index: int,
    task: dict,
    ref_dir: Path,
    ref_pattern: str,
    out_dir: Path,
    labels: dict,
    val_run_id: int,
    r_cut_mm: float | None,
    pair_max_ang: float,
    line_dr: float,
    line_max: float,
    amp_scale: float = 1.0,
) -> dict[str, float] | None:
    file_index = int(task.get("file_index", task["global_cache_k"]))
    run_id = int(val_run_id)
    artifact_id = _artifact_run_id(task)
    coo_path_res = Path(task["coo_path"])

    frame = load_scf_frame(coo_path_res, n_qm=int(task["n_qm"]))
    frame = filter_mm_by_distance(frame, r_cut_ang=r_cut_mm)

    amp_raw = amp_mm_frame_slice(labels, slice_index)
    ptr = labels["mm_ptr"]
    s, e = int(ptr[slice_index]), int(ptr[slice_index + 1])
    k_raw = np.asarray(labels["kernel_mm_flat"][s:e], dtype=np.float64) if "kernel_mm_flat" in labels else None
    amp_mm, k_mm = _align_amp_k_to_frame(frame, amp_raw, k_raw)
    amp_mm, _amp_base = _prepare_amp_with_scale(amp_mm, amp_scale)

    plane_hi = nearest_mm_h_index_to_qm(frame)
    e0_kcal = float(labels["e0"][slice_index])
    eerr = float(labels["err_kcal"][slice_index]) if labels.get("err_kcal") is not None else float("nan")
    e0_pred = float(task.get("E0_pred", e0_kcal + eerr))

    ref_path = ref_npz_path(ref_dir, file_index, pattern=ref_pattern)
    print(f"\n[batch_scf] frame {file_index}  (ref preview)", flush=True)
    if not ref_path.is_file():
        raise FileNotFoundError(f"missing ref npz for Coo{file_index}: {ref_path}")

    ref_hf = load_ref_hf_npz(ref_path, load_dm=False)
    mix_m = compute_mix_err_metrics(
        e0_pred_kcal=float(e0_pred),
        e0_label_kcal=float(e0_kcal),
        e_int_emb0_kcal=float(ref_hf["e_int_emb_kcal"]),
        e_int_qmmm_ref_kcal=float(ref_hf["e_int_qmmm_ref_kcal"]),
    )

    t0 = time.perf_counter()
    spillover_profiles_from_ref_npz(
        ref_path,
        frame,
        plane_mm_h_index=int(plane_hi),
        pair_max_ang=float(pair_max_ang),
        line_dr=float(line_dr),
        line_max=float(line_max),
        r_cut_mm=r_cut_mm,
    )
    dt = time.perf_counter() - t0

    print(f"  ok  {dt:.2f}s  (ref preview)", flush=True)
    mix_m = dict(mix_m)
    mix_m["signed_e0_err"] = float(e0_pred) - float(e0_kcal)
    mix_m["e0_pred"] = float(e0_pred)
    mix_m["e0_label"] = float(e0_kcal)
    print(format_mix_err_one_line(mix_m), flush=True)
    return mix_m


def _run_one_ref_embtheta(
    *,
    slice_index: int,
    task: dict,
    ref_dir: Path,
    ref_pattern: str,
    out_dir: Path,
    labels: dict,
    val_run_id: int,
    fix_alpha: float,
    alpha_cfg: MmhAlphaConfig | None = None,
    cone_theta1_deg: float,
    cone_theta2_deg: float,
    r_cut_mm: float | None,
    pair_max_ang: float,
    line_dr: float,
    line_max: float,
    threads: int,
    embed_scf_conv_tol: float,
    amp_scale: float = 1.0,
    fix_iter: int = 0,
) -> dict[str, float] | None:
    file_index = int(task.get("file_index", task["global_cache_k"]))
    run_id = int(val_run_id)
    artifact_id = _artifact_run_id(task)
    coo_path_res = Path(task["coo_path"])

    frame = load_scf_frame(coo_path_res, n_qm=int(task["n_qm"]))
    frame = filter_mm_by_distance(frame, r_cut_ang=r_cut_mm)

    amp_raw = amp_mm_frame_slice(labels, slice_index)
    ptr = labels["mm_ptr"]
    s, e = int(ptr[slice_index]), int(ptr[slice_index + 1])
    k_raw = np.asarray(labels["kernel_mm_flat"][s:e], dtype=np.float64) if "kernel_mm_flat" in labels else None
    amp_mm, k_mm = _align_amp_k_to_frame(frame, amp_raw, k_raw)
    amp_mm, _amp_base = _prepare_amp_with_scale(amp_mm, amp_scale)

    plane_hi = nearest_mm_h_index_to_qm(frame)
    e0_kcal = float(labels["e0"][slice_index])
    eerr = float(labels["err_kcal"][slice_index]) if labels.get("err_kcal") is not None else float("nan")
    e0_pred = float(task.get("E0_pred", e0_kcal + eerr))

    ref_path = ref_npz_path(ref_dir, file_index, pattern=ref_pattern)
    scf_cache = _scf_npz_path(out_dir, file_index=artifact_id, fix_iter=fix_iter)

    print(f"\n[batch_scf] frame {file_index}  fix_iter={int(fix_iter)}", flush=True)
    if not ref_path.is_file():
        raise FileNotFoundError(f"missing ref npz for Coo{file_index}: {ref_path}")

    ref_hf = load_ref_hf_npz(ref_path, load_dm=False)
    t0 = time.perf_counter()
    profiles = spillover_profiles_ref_embtheta(
        ref_path,
        frame,
        amp_mm,
        fix_alpha=float(fix_alpha),
        alpha_cfg=alpha_cfg,
        plane_mm_h_index=int(plane_hi),
        pair_max_ang=float(pair_max_ang),
        line_dr=float(line_dr),
        line_max=float(line_max),
        r_cut_mm=r_cut_mm,
        embed_scf_conv_tol=float(embed_scf_conv_tol),
        cone_theta1_deg=float(cone_theta1_deg),
        cone_theta2_deg=float(cone_theta2_deg),
        num_threads=int(threads),
    )
    dt = time.perf_counter() - t0
    mix_m = compute_mix_err_metrics(
        e0_pred_kcal=float(e0_pred),
        e0_label_kcal=float(profiles.get("e0_kcal", e0_kcal)),
        e_int_emb0_kcal=float(profiles.get("e_int_emb0_kcal", ref_hf["e_int_emb_kcal"])),
        e_int_qmmm_ref_kcal=float(profiles.get("e_int_qmmm_ref_kcal", ref_hf["e_int_qmmm_ref_kcal"])),
        e_int_embtheta_kcal=float(profiles.get("e_int_embtheta_kcal", float("nan"))),
    )
    try:
        scf_kw: dict = {
            "mode": np.array("ref_embtheta"),
            "ref_npz": str(ref_path.resolve()),
            "embed_scf_converged": np.bool_(bool(profiles.get("embed_scf_converged", True))),
            "e0_pred_kcal": np.float64(float(e0_pred)),
            "e0_label_kcal": np.float64(float(profiles.get("e0_kcal", float("nan")))),
            "e_int_embtheta_kcal": np.float64(float(profiles.get("e_int_embtheta_kcal", float("nan")))),
            "e_int_emb0_kcal": np.float64(float(profiles.get("e_int_emb0_kcal", float("nan")))),
            "e_int_pred_kcal": np.float64(float(mix_m.get("e_int_pred", float("nan")))),
            "e_int_qmmm_ref_kcal": np.float64(
                float(profiles.get("e_int_qmmm_ref_kcal", ref_hf["e_int_qmmm_ref_kcal"]))
            ),
            "e_int_cp_kcal": np.float64(float(profiles.get("e_int_cp_kcal", float("nan")))),
            "abs_e0_err_kcal": np.float64(float(mix_m.get("abs_e0_err", float("nan")))),
            "abs_e_int_pred_err_kcal": np.float64(float(mix_m.get("abs_e_int_pred_err", float("nan")))),
            "abs_embtheta_err_kcal": np.float64(float(mix_m.get("abs_embtheta_err", float("nan")))),
            "abs_embtheta_vs_pred_kcal": np.float64(
                float(mix_m.get("abs_embtheta_vs_pred", float("nan")))
            ),
            "signed_embtheta_vs_pred_kcal": np.float64(
                float(mix_m.get("signed_embtheta_vs_pred", float("nan")))
            ),
            "amp_scale": np.float64(float(amp_scale)),
            "a_fix_iter": np.int32(int(fix_iter)),
        }
        if profiles.get("dm_emb") is not None:
            scf_kw["dm_emb"] = np.asarray(profiles["dm_emb"], dtype=np.float64)
        if profiles.get("dm_scf") is not None:
            scf_kw["dm_scf"] = np.asarray(profiles["dm_scf"], dtype=np.float64)
        if profiles.get("dm_cluster_qm") is not None:
            scf_kw["dm_cl_qm"] = np.asarray(profiles["dm_cluster_qm"], dtype=np.float64)
        if profiles.get("e_gas_hartree") is not None:
            scf_kw["e_gas_hartree"] = np.float64(float(profiles["e_gas_hartree"]))
        meta = {
            "xc": str(profiles.get("xc", "")),
            "basis": str(profiles.get("basis", "")),
            "pipeline": "ref_embtheta",
            "n_qm": int(len(frame.qm_symbols)),
            "n_mm": int(len(frame.mm_symbols)),
            "a_fix_iter": int(fix_iter),
        }
        scf_kw["meta_json"] = np.asarray(json.dumps(meta))
        scf_kw["qm_coords_ang"] = np.asarray(frame.qm_coords_ang, dtype=np.float64)
        scf_kw["mm_coords_ang"] = np.asarray(frame.mm_coords_ang, dtype=np.float64)
        scf_kw["mm_charges"] = np.asarray(frame.mm_charges, dtype=np.float64)
        scf_kw["qm_symbols"] = np.asarray(frame.qm_symbols, dtype=object)
        scf_kw["mm_symbols"] = np.asarray(frame.mm_symbols, dtype=object)
        np.savez_compressed(scf_cache, **scf_kw)
    except OSError:
        pass
    print(f"  ok  {dt:.1f}s  scf={scf_cache.name}", flush=True)
    print(format_mix_err_one_line(mix_m), flush=True)
    e_th = float(profiles.get("e_int_embtheta_kcal", float("nan")))
    e_ref = float(profiles.get("e_int_qmmm_ref_kcal", ref_hf["e_int_qmmm_ref_kcal"]))
    e_pred = float(mix_m.get("e_int_pred", float("nan")))
    e_lab = float(profiles.get("e0_kcal", e0_kcal))
    mix_m = dict(mix_m)
    mix_m["e_int_embtheta"] = e_th
    mix_m["e_int_qmmm_ref"] = e_ref
    mix_m["e_int_pred"] = e_pred
    mix_m["signed_embtheta_err"] = _signed_embtheta_vs_pred(
        e_int_embtheta=e_th, e_int_pred=e_pred
    )
    mix_m["signed_e0_err"] = float(e0_pred) - float(e_lab)
    mix_m["amp_scale"] = float(amp_scale)
    mix_m["e0_pred"] = float(e0_pred)
    mix_m["e0_label"] = float(e_lab)
    mix_m["a_fix_iter"] = float(fix_iter)
    return mix_m


def _run_one_line(
    *,
    slice_index: int,
    task: dict,
    peratom_path: Path,
    pert_cache: Path | None,
    out_dir: Path,
    fix_alpha: float,
    cone_theta1_deg: float,
    cone_theta2_deg: float,
    method: str,
    basis: str,
    d3bj: bool,
    threads: int,
    plane_step: float,
    r_cut_mm: float | None,
    force_scf: bool,
    labels: dict,
    val_run_id: int,
    amp_scale: float = 1.0,
) -> None:
    file_index = int(task.get("file_index", task["global_cache_k"]))
    run_id = int(val_run_id)
    artifact_id = _artifact_run_id(task)
    coo_path_res = Path(task["coo_path"])
    residue = str(task["residue"])

    frame = load_scf_frame(coo_path_res, n_qm=int(task["n_qm"]))
    frame = filter_mm_by_distance(frame, r_cut_ang=r_cut_mm)

    amp_raw = amp_mm_frame_slice(labels, slice_index)
    ptr = labels["mm_ptr"]
    s, e = int(ptr[slice_index]), int(ptr[slice_index + 1])
    k_raw = np.asarray(labels["kernel_mm_flat"][s:e], dtype=np.float64) if "kernel_mm_flat" in labels else None
    amp_mm, k_mm = _align_amp_k_to_frame(frame, amp_raw, k_raw)
    if abs(float(amp_scale) - 1.0) > 1e-15:
        print(
            "  WARNING: --amp-scale does not affect line_rho SCF (subprocess reads peratom npz); "
            "use --ref-embtheta for A perturbation tests",
            flush=True,
        )

    plane_hi = nearest_mm_h_index_to_qm(frame)
    e0_kcal = float(labels["e0"][slice_index])
    eerr = float(labels["err_kcal"][slice_index]) if labels.get("err_kcal") is not None else float("nan")

    scf_cache = out_dir / f"scf{artifact_id}.npz"
    plane_npz = out_dir / f"plane{artifact_id}.npz"

    cmd = [
        sys.executable,
        "-m",
        "embr_theta.line_rho_pert_cluster",
        "--coo",
        str(coo_path_res.resolve()),
        "--residue",
        residue,
        "--n-qm",
        str(int(task["n_qm"])),
        "--frame",
        str(int(slice_index)),
        "--peratom",
        str(peratom_path.resolve()),
        "--fix-alpha",
        str(float(fix_alpha)),
        "--cone-theta1-deg",
        str(float(cone_theta1_deg)),
        "--cone-theta2-deg",
        str(float(cone_theta2_deg)),
        "--rep-center",
        "on_nucleus",
        "--method",
        str(method),
        "--basis",
        str(basis),
        "--threads",
        str(int(threads)),
        "--plane-step",
        str(float(plane_step)),
        "--scf-cache",
        str(scf_cache.resolve()),
        "--plane-npz",
        str(plane_npz.resolve()),
        "--plane-mm-h-index",
        str(int(plane_hi)),
    ]
    if pert_cache is not None:
        cmd.extend(["--pert-cache", str(pert_cache.resolve())])
    if d3bj:
        cmd.append("--d3bj")
    if r_cut_mm is not None:
        cmd.extend(["--r-cut-mm", str(float(r_cut_mm))])
    if force_scf:
        cmd.append("--force-scf")

    print(f"\n[batch_scf] frame {file_index}  (line_rho)", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    body = proc.stdout
    if proc.stderr:
        body = body + ("\n" if body and not body.endswith("\n") else "") + proc.stderr

    if proc.returncode != 0:
        tail = body[-2000:] if len(body) > 2000 else body
        raise RuntimeError(
            f"line_rho failed for cache_k={task['global_cache_k']} (exit {proc.returncode}, {dt:.1f}s):\n{tail}"
        )
    print(f"  ok  {dt:.1f}s  scf={scf_cache.name}  plane={plane_npz.name}")






def main() -> None:
    ap = argparse.ArgumentParser(description="mix_mmh val: ML e_j → SCF + fix (line_rho batch)")
    ap.add_argument("--frame-start", type=int, default=1, help="unused (compat)")
    ap.add_argument(
        "--peratom",
        type=Path,
        default=None,
        help="pert_peratom npz (written by export in mix mode)",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="EmbR repo root (directory containing run.py); optional GLY_DESCRIPTOR_ROOT env",
    )
    ap.add_argument("--ckpt", type=Path, required=True, help="ksoft checkpoint")
    ap.add_argument(
        "--repulsion-policy",
        type=str,
        default="all",
        choices=("all", "h_only", "no_o", "positive"),
        help="which sites get A_j: all; h_only; no_o=H+ions (no O); "
        "positive=H/Na/K/C/N (no O/Cl)",
    )
    ap.add_argument("--soap-cache", type=Path, default=None, help="mix_mmh_full.npz (required for --dataset mix)")
    ap.add_argument(
        "--kernel-cache",
        type=Path,
        default=None,
        help="optional mix_ai npz; default: kernels from soap-cache (--ref-dir precompute)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("scf_mmh_val"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0, help="same as train_soap_e0_mix_mmh / inspect")
    ap.add_argument("--fix-alpha", type=float, default=3.0)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="read fix_alpha / fix_alpha_by_element / multi_alpha (else soap-cache meta)",
    )
    ap.add_argument(
        "--multi-alpha",
        action="store_true",
        help="Approach A: α_elem = α_H·(R_H/R_elem)² (ignored if manifest fix_alpha_by_element)",
    )
    ap.add_argument("--alpha-by-element", type=str, default=None)
    ap.add_argument("--alpha-na", type=float, default=None)
    ap.add_argument("--alpha-k", type=float, default=None)
    ap.add_argument("--cone-theta1-deg", type=float, default=180.0)
    ap.add_argument("--cone-theta2-deg", type=float, default=180.0)
    ap.add_argument("--r-cut-mm", type=float, default=None)
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--basis", type=str, default=None)
    ap.add_argument(
        "--scf-preset",
        type=str,
        default=None,
        choices=sorted({"hf", "b3lyp-plus", "hf-plus"}),
    )
    ap.add_argument("--d3bj", action="store_true")
    ap.add_argument("--no-d3bj", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--qm-charge",
        type=int,
        default=None,
        help="QM net formal charge (Asp⁻=-1; default manifest scf.qm_charge or 0)",
    )
    ap.add_argument("--plane-step", type=float, default=0.02)
    ap.add_argument("--verbose-scf", type=int, default=0)
    ap.add_argument(
        "--pert-cache",
        type=Path,
        default=None,
        help="optional theta/pert.npz for ΣAk cross-check (mix_ai peratom kernels usually enough)",
    )
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--force-scf", action="store_true")
    ap.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="batch_hf ref_*.npz dir for ρ_emb/ρ_Cluster^QM (see --ref-embtheta)",
    )
    ap.add_argument(
        "--ref-embtheta",
        action="store_true",
        help="with --ref-dir: run EmbR SCF (ML A_j) + read ref dm for fix; skip Cluster rerun",
    )
    ap.add_argument(
        "--embed-scf-conv-tol",
        type=float,
        default=1e-4,
        help="EmbR SCF conv_tol when --ref-embtheta",
    )
    ap.add_argument(
        "--ref-pattern",
        type=str,
        default="ref_{}.npz",
        help="ref npz filename pattern; {} = Coo file_index (default ref_{}.npz)",
    )
    ap.add_argument("--pair-max", type=float, default=2.0, help="QM–MM axis pair max (Å); ref mode")
    ap.add_argument("--line-max", type=float, default=2.0, help="axis r grid max (Å); ref mode")
    ap.add_argument("--line-dr", type=float, default=0.1, help="axis dr (Å); ref mode")
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="run every cache row 0..n-1 (skip train/val split; use for full-cache batch)",
    )
    ap.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="with --all-frames: run first N cache rows (default: all n in soap-cache)",
    )
    ap.add_argument(
        "--val-frame-index",
        type=str,
        default=None,
        help="val-split local index (0=first val frame); comma-separated, e.g. 0,1,2",
    )
    ap.add_argument(
        "--n-val-frames",
        type=int,
        default=None,
        help="run first N frames **within val split** (not all cache); use --all-frames for full cache",
    )
    ap.add_argument(
        "--only-cache-indices",
        type=str,
        default=None,
        help="global cache_k in npz; with --all-frames any index in [0,n)",
    )
    ap.add_argument(
        "--start-cache-index",
        type=int,
        default=0,
        help="skip cache_k < this (0-based); e.g. 1868 = resume at 1869th frame",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip finished work: without --a-fix, skip frames with matching scf{N}.npz "
        "(same amp_scale); with --a-fix, skip each fix_iter whose scf{N}[_fixk].npz is complete",
    )
    ap.add_argument(
        "--amp-scale",
        type=float,
        default=1.0,
        help="multiply exported A_j by this factor before EmbR SCF "
        "(1.10 = +10%%, 0.90 = -10%%, 1.20 = +20%%); stored in scf{N}.npz",
    )
    ap.add_argument(
        "--a-fix",
        action="store_true",
        help="after base EmbR, rescale A (1st s=E0_pred, then secant) until "
        "|EmbR−(E0_pred+Emb0)|<--a-fix-tol; writes scf{N}_fixk.npz; needs --ref-embtheta",
    )
    ap.add_argument(
        "--a-fix-tol",
        type=float,
        default=0.5,
        help="stop A-fix when |E_int^EmbR − E_int^ref| < this (kcal; default 0.5)",
    )
    ap.add_argument(
        "--a-fix-max",
        type=int,
        default=8,
        help="safety cap on A-fix SCF count (default 8; usually stop earlier by tol)",
    )
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()


    if args.soap_cache is None:
        raise SystemExit("--soap-cache is required for --dataset mix")

    amp_scale = float(args.amp_scale)
    if amp_scale <= 0.0:
        raise SystemExit(f"--amp-scale must be > 0, got {amp_scale}")
    if abs(amp_scale - 1.0) > 1e-15:
        print(
            f"[batch_scf] amp-scale={amp_scale:.6f}  "
            f"(A_used = A_base × {amp_scale:.6f}, ΔA={(amp_scale - 1.0) * 100.0:+.2f}%)",
            flush=True,
        )
    a_fix_on = bool(args.a_fix)
    a_fix_tol = float(args.a_fix_tol)
    a_fix_max = max(1, int(args.a_fix_max)) if a_fix_on else 0
    if a_fix_on and a_fix_tol <= 0.0:
        raise SystemExit(f"--a-fix-tol must be > 0, got {a_fix_tol}")

    use_ref_embtheta = (
        args.ref_dir is not None and bool(args.ref_embtheta) and not bool(args.force_scf)
    )
    use_ref_preview = (
        args.ref_dir is not None and not bool(args.ref_embtheta) and not bool(args.force_scf)
    )
    if a_fix_on and not use_ref_embtheta:
        raise SystemExit("--a-fix requires --ref-embtheta")
    if a_fix_on:
        print(
            f"[batch_scf] A-fix ON  |EmbR−pred|<{a_fix_tol:g} kcal  "
            f"safety_max={a_fix_max}  (1st s=E0_pred, then secant) → scf{{N}}_fixk.npz",
            flush=True,
        )
    if args.ref_dir is not None and args.force_scf:
        print("[batch_scf] --force-scf: ignore --ref-dir, run full line_rho SCF", flush=True)
    if args.ref_embtheta and args.ref_dir is None:
        raise SystemExit("--ref-embtheta requires --ref-dir")

    method = str(args.method or "b3lyp")
    basis = str(args.basis or "6-31g*")
    d3bj = not bool(args.no_d3bj)
    if args.d3bj:
        d3bj = True
    if args.scf_preset:
        preset = resolve_scf_preset(str(args.scf_preset))
        method = str(preset["method"])
        basis = str(preset["basis"])
        d3bj = bool(preset["d3bj"])
    elif not args.no_d3bj and not args.d3bj:
        d3bj = True

    manifest_cfg: dict | None = None
    if args.manifest is not None:
        manifest_cfg = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qm_charge = resolve_qm_charge(
        {"qm_charge": args.qm_charge} if args.qm_charge is not None else None,
        manifest_cfg,
        default=0,
    )
    scf_cfg = scf_embed_config_from_cli(
        method=method,
        basis=basis,
        use_d3bj=d3bj,
        num_threads=int(args.threads),
        verbose=int(args.verbose_scf),
        qm_charge=int(qm_charge),
    )
    if int(qm_charge) != 0:
        pass

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    peratom_path = out_dir / "mix_mmh_val_peratom.npz"

    cache = attach_cache_path(load_mmh_cache(args.soap_cache), args.soap_cache)
    n = n_frames(cache)

    if args.manifest is None:
        cm = cache.get("meta") or {}
        pseudo: dict = {}
        if cm.get("fix_alpha_by_element") is not None:
            pseudo["fix_alpha_by_element"] = dict(cm["fix_alpha_by_element"])
        if cm.get("multi_alpha"):
            pseudo["multi_alpha"] = True
        if cm.get("envelope_kind") is not None:
            pseudo["envelope_kind"] = str(cm["envelope_kind"])
        if cm.get("lnC_by_element") is not None:
            pseudo["lnC_by_element"] = dict(cm["lnC_by_element"])
        if cm.get("exp_sum_by_element") is not None:
            pseudo["exp_sum_by_element"] = dict(cm["exp_sum_by_element"])
        if cm.get("fix_alpha") is not None:
            pseudo["fix_alpha"] = float(cm["fix_alpha"])
        manifest_cfg = pseudo or None

    fix_alpha_cli = float(args.fix_alpha)
    if manifest_cfg is not None and args.fix_alpha == 3.0 and manifest_cfg.get("fix_alpha") is not None:
        fix_alpha_cli = float(manifest_cfg["fix_alpha"])
    multi_alpha = bool(args.multi_alpha) or bool((manifest_cfg or {}).get("multi_alpha", False))
    alpha_cfg = MmhEnvelopeConfig.parse(
        fix_alpha=fix_alpha_cli,
        multi_alpha=multi_alpha,
        alpha_by_element=args.alpha_by_element,
        alpha_na=args.alpha_na,
        alpha_k=args.alpha_k,
        envelope_kind=(manifest_cfg or {}).get("envelope_kind"),
        manifest=manifest_cfg,
    )
    alpha_src = (
        "CLI --alpha-by-element"
        if args.alpha_by_element
        else "manifest fix_alpha_by_element"
        if manifest_cfg and manifest_cfg.get("fix_alpha_by_element")
        else "manifest/cache multi_alpha"
        if multi_alpha
        else "uniform fix_alpha"
    )
    _ = alpha_src

    cache_meta = cache.get("meta") or {}
    if cache_meta.get("envelope_kind") and not alpha_cfg.matches_stored(
        fix_alpha_bohr2=cache_meta.get("fix_alpha"),
        fix_alpha_by_element=cache_meta.get("fix_alpha_by_element"),
        envelope_kind=cache_meta.get("envelope_kind"),
        lnC_by_element=cache_meta.get("lnC_by_element"),
        exp_sum_by_element=cache_meta.get("exp_sum_by_element"),
    ):
        pass

    if args.kernel_cache is None and cache.get("kernel_h_flat") is None:
        raise SystemExit(
            "soap-cache has no kernel_h_flat; pass --kernel-cache mix_ai.npz "
            "or precompute with --ref-dir (batch_hf ref npz)"
        )
    _, val_idx = _split_indices(n, float(args.val_frac), int(args.seed))
    only_cache_early = _parse_int_list(args.only_cache_indices)
    val_local_early = _parse_int_list(args.val_frame_index)
    explicit_frames = (
        bool(args.all_frames)
        or only_cache_early is not None
        or val_local_early is not None
        or args.n_val_frames is not None
    )
    if not val_idx and not explicit_frames:
        raise SystemExit(
            "empty validation split (n_frames too small for val_frac). "
            "Use --all-frames, --only-cache-indices, or --val-frame-index."
        )

    val_local = val_local_early
    only_cache = only_cache_early
    frame_selectors = sum(
        1
        for x in (
            bool(args.all_frames),
            val_local is not None,
            only_cache is not None,
            args.n_val_frames is not None,
        )
        if x
    )
    if frame_selectors > 1:
        raise ValueError(
            "use only one of --all-frames, --val-frame-index, --n-val-frames, --only-cache-indices"
        )

    run_val: list[int]
    if bool(args.all_frames):
        n_take = n if args.n_frames is None else min(int(args.n_frames), n)
        if n_take <= 0:
            raise ValueError("--n-frames must be > 0")
        start_k = max(0, int(args.start_cache_index))
        if start_k >= n_take:
            raise ValueError(f"--start-cache-index {start_k} >= n_take {n_take}")
        run_val = list(range(start_k, n_take))
        print(
            f"[batch_scf] all-frames: cache_k {start_k}..{n_take - 1} (n_cache={n})",
            flush=True,
        )
    elif args.n_val_frames is not None:
        n_take = int(args.n_val_frames)
        if n_take <= 0:
            raise ValueError("--n-val-frames must be > 0")
        val_local = list(range(min(n_take, len(val_idx))))
        bad_local = [i for i in val_local if i < 0 or i >= len(val_idx)]
        if bad_local:
            raise ValueError(
                f"--n-val-frames maps to val index out of range [0, {len(val_idx) - 1}]: {bad_local}"
            )
        run_val = [int(val_idx[i]) for i in val_local]
        print(
            f"[batch_scf] val-frame-index {val_local} → cache_k {run_val} "
            f"(val split n={len(val_idx)}/{n})",
            flush=True,
        )
    elif val_local is not None:
        bad_local = [i for i in val_local if i < 0 or i >= len(val_idx)]
        if bad_local:
            raise ValueError(
                f"--val-frame-index out of range [0, {len(val_idx) - 1}]: {bad_local}"
            )
        run_val = [int(val_idx[i]) for i in val_local]
        print(
            f"[batch_scf] val-frame-index {val_local} → cache_k {run_val} "
            f"(val split n={len(val_idx)}/{n})",
            flush=True,
        )
    elif only_cache is not None:
        only = {int(x) for x in only_cache}
        bad = [x for x in only if x < 0 or x >= n]
        if bad:
            raise ValueError(f"--only-cache-indices out of range [0, {n - 1}]: {sorted(bad)}")
        # Explicit list: any cache rows (not limited to val split)
        run_val = sorted(only)
        print(f"[batch_scf] only-cache-indices → cache_k {run_val}", flush=True)
    else:
        run_val = list(val_idx)
        print(
            f"[batch_scf] default val split → cache_k {run_val} (n={len(val_idx)}/{n})",
            flush=True,
        )

    start_k = max(0, int(args.start_cache_index))
    if start_k > 0 and not bool(args.all_frames):
        before = len(run_val)
        run_val = [int(k) for k in run_val if int(k) >= start_k]
        print(
            f"[batch_scf] start-cache-index={start_k}: "
            f"{before} → {len(run_val)} frames",
            flush=True,
        )
    if bool(args.skip_existing):
        run_val = _filter_run_val_skip_existing(
            cache, out_dir, run_val, amp_scale=amp_scale
        )
    if not run_val:
        print("[batch_scf] nothing to run (empty frame list after filters)", flush=True)
        return

    only: set[int] | None = set(run_val)

    if use_ref_embtheta:
        mode = "ref_embtheta"
    elif use_ref_preview:
        mode = "ref_preview"
    else:
        mode = "line_scf"

    xc_print = str(scf_cfg.xc)
    basis_print = str(scf_cfg.basis)
    d3bj_print = bool(d3bj)
    scf_meta = {"xc": scf_cfg.xc, "basis": scf_cfg.basis, "use_d3bj": bool(scf_cfg.use_d3bj)}
    if use_ref_embtheta or use_ref_preview:
        fm0 = frame_meta_entry(cache, int(run_val[0]))
        fi0 = int(fm0.get("file_index", run_val[0]))
        ref_p = ref_npz_path(Path(args.ref_dir), fi0, pattern=str(args.ref_pattern))
        scf_meta = scf_meta_from_ref_npz(ref_p)
        xc_print = str(scf_meta["xc"])
        basis_print = str(scf_meta["basis"])
        d3bj_print = bool(scf_meta["use_d3bj"])
    print(
        f"[batch_scf] cache={args.soap_cache.name}  frames={n}  "
        f"run={len(run_val)}  out={out_dir}",
        flush=True,
    )

    if not args.skip_export:
        labels_pack = build_val_peratom_labels(
            mmh_cache=cache,
            kernel_cache_path=args.kernel_cache,
            ckpt_path=Path(args.ckpt),
            val_indices=run_val,
            fix_alpha=float(alpha_cfg.legacy_fix_alpha()),
            device=str(args.device),
            h_only=False,
            scf_meta=scf_meta,
            envelope_meta=alpha_cfg.to_full_meta(),
        )
        save_peratom_npz(labels_pack, peratom_path)
        index_path = out_dir / "scf_export_index.json"
        index_path.write_text(
            json.dumps(
                [
                    {
                        "val_run_id": int(fr.get("val_run_id", fr.get("slice_index", 0) + 1)),
                        "slice_index": int(fr.get("slice_index", 0)),
                        "global_cache_k": int(fr.get("global_cache_k", 0)),
                        "coo_file_index": int(fr.get("file_index", 0)),
                        "E0_pred": float(fr.get("E0_pred", float("nan"))),
                    }
                    for fr in (labels_pack["meta"].get("frames") or [])
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {peratom_path}  n_export={len(run_val)}  index={index_path.name}", flush=True)
    elif not peratom_path.is_file():
        raise FileNotFoundError(f"--skip-export but missing {peratom_path}")

    if args.skip_run:
        print("[batch_scf] --skip-run: done after export")
        return

    labels = load_pert_peratom_labels(peratom_path)
    meta = labels["meta"]
    tasks = list(meta.get("frames") or [])
    if len(tasks) != int(labels["e0"].shape[0]):
        raise ValueError(f"meta frames {len(tasks)} != peratom rows {labels['e0'].shape[0]}")

    n_done = 0
    mix_err_rows: list[tuple[int, dict[str, float]]] = []
    for task in tasks:
        if only is not None and int(task["global_cache_k"]) not in only:
            continue
        n_done += 1
        val_run_id = int(task.get("val_run_id", int(task["slice_index"]) + 1))
        artifact_id = _artifact_run_id(task)
        # Without A-fix: skip whole frame if scf{N}.npz matches amp_scale.
        # With A-fix: per-fix_iter resume via scf*.npz (do not skip the frame here).
        if (
            bool(args.skip_existing)
            and not a_fix_on
            and _should_skip_existing_frame(
                out_dir, file_index=artifact_id, amp_scale=amp_scale
            )
        ):
            print(
                f"  skip frame {artifact_id}  scf{artifact_id}.npz exists "
                f"(amp_scale={amp_scale:g})",
                flush=True,
            )
            continue
        mix_m: dict[str, float] | None = None
        if use_ref_embtheta:
            lam = float(amp_scale)
            hist: list[tuple[float, float]] = []

            def _embtheta_or_skip(fix_iter: int, lam_use: float) -> dict[str, float] | None:
                if bool(args.skip_existing) and _scf_a_fix_complete(
                    out_dir, file_index=artifact_id, fix_iter=int(fix_iter)
                ):
                    loaded = _mix_m_from_scf_npz(
                        out_dir, file_index=artifact_id, fix_iter=int(fix_iter)
                    )
                    if loaded is not None:
                        print(
                            f"  skip EmbR fix_iter={int(fix_iter)}  "
                            f"scf{artifact_id}{_fix_suffix(int(fix_iter))}.npz "
                            f"(λ={float(loaded.get('amp_scale', lam_use)):.6g})",
                            flush=True,
                        )
                        return loaded
                return _run_one_ref_embtheta(
                    slice_index=int(task["slice_index"]),
                    task=task,
                    ref_dir=Path(args.ref_dir),
                    ref_pattern=str(args.ref_pattern),
                    out_dir=out_dir,
                    labels=labels,
                    val_run_id=val_run_id,
                    fix_alpha=float(alpha_cfg.legacy_fix_alpha()),
                    alpha_cfg=alpha_cfg,
                    cone_theta1_deg=float(args.cone_theta1_deg),
                    cone_theta2_deg=float(args.cone_theta2_deg),
                    r_cut_mm=args.r_cut_mm,
                    pair_max_ang=float(args.pair_max),
                    line_dr=float(args.line_dr),
                    line_max=float(args.line_max),
                    threads=int(args.threads),
                    embed_scf_conv_tol=float(args.embed_scf_conv_tol),
                    amp_scale=float(lam_use),
                    fix_iter=int(fix_iter),
                )

            mix_m = _embtheta_or_skip(0, lam)
            if mix_m is not None and np.isfinite(float(mix_m.get("amp_scale", float("nan")))):
                lam = float(mix_m["amp_scale"])
            for fix_i in range(1, (a_fix_max if a_fix_on else 0) + 1):
                if mix_m is None:
                    break
                eps = float(mix_m.get("signed_embtheta_err", float("nan")))
                e0_pred_m = float(mix_m.get("e0_pred", float("nan")))
                hist.append((lam, eps))
                if not np.isfinite(eps):
                    print(f"  [A-fix] stop: ε not finite at fix_iter={fix_i - 1}", flush=True)
                    break
                if abs(eps) < float(a_fix_tol):
                    print(
                        f"  [A-fix] |ε|=|EmbR−pred|={abs(eps):.4g} < tol={a_fix_tol:g} → stop",
                        flush=True,
                    )
                    break
                # Prefer already-written fix_i artifact (resume) over recomputing λ.
                if bool(args.skip_existing) and _scf_a_fix_complete(
                    out_dir, file_index=artifact_id, fix_iter=fix_i
                ):
                    mix_m = _embtheta_or_skip(fix_i, lam)
                    if mix_m is not None and np.isfinite(
                        float(mix_m.get("amp_scale", float("nan")))
                    ):
                        lam = float(mix_m["amp_scale"])
                    continue
                nxt = _a_fix_next_lambda(
                    lam=lam,
                    eps=eps,
                    e0_pred_kcal=e0_pred_m,
                    hist=hist,
                )
                if nxt is None:
                    print(f"  [A-fix] stop: cannot update λ at fix_iter={fix_i}", flush=True)
                    break
                lam_new, tag = nxt
                print(
                    f"  [A-fix] → fix{fix_i}: λ {lam:.6g} → {lam_new:.6g}  "
                    f"ε={eps:+.4f}  ({tag})",
                    flush=True,
                )
                lam = float(lam_new)
                mix_m = _embtheta_or_skip(fix_i, lam)
                if mix_m is not None and np.isfinite(float(mix_m.get("amp_scale", float("nan")))):
                    lam = float(mix_m["amp_scale"])
        elif use_ref_preview:
            mix_m = _run_one_from_ref(
                slice_index=int(task["slice_index"]),
                task=task,
                ref_dir=Path(args.ref_dir),
                ref_pattern=str(args.ref_pattern),
                out_dir=out_dir,
                labels=labels,
                val_run_id=val_run_id,
                r_cut_mm=args.r_cut_mm,
                pair_max_ang=float(args.pair_max),
                line_dr=float(args.line_dr),
                line_max=float(args.line_max),
                amp_scale=amp_scale,
            )
        else:
            _run_one_line(
                slice_index=int(task["slice_index"]),
                task=task,
                peratom_path=peratom_path,
                pert_cache=args.pert_cache,
                out_dir=out_dir,
                fix_alpha=float(alpha_cfg.legacy_fix_alpha()),
                cone_theta1_deg=float(args.cone_theta1_deg),
                cone_theta2_deg=float(args.cone_theta2_deg),
                method=method,
                basis=basis,
                d3bj=bool(scf_cfg.use_d3bj),
                threads=int(args.threads),
                plane_step=float(args.plane_step),
                r_cut_mm=args.r_cut_mm,
                force_scf=bool(args.force_scf),
                labels=labels,
                val_run_id=val_run_id,
                amp_scale=amp_scale,
            )
        if mix_m is not None:
            mix_err_rows.append((artifact_id, mix_m))

    _print_mix_err_summary_table(mix_err_rows)

    summary = out_dir / "batch_summary.json"
    summary.write_text(
        json.dumps(
            {
                "n_cache": n,
                "n_val_split": len(val_idx),
                "all_frames": bool(args.all_frames),
                "n_run": int(n_done),
                "val_frame_index": val_local,
                "run_cache_indices": run_val,
                "start_cache_index": int(args.start_cache_index),
                "skip_existing": bool(args.skip_existing),
                "amp_scale": float(amp_scale),
                "peratom": str(peratom_path),
                "fix_alpha": float(alpha_cfg.legacy_fix_alpha()),
                "fix_alpha_by_element": None if alpha_cfg.is_uniform() else alpha_cfg.to_meta(),
                
                "scf_config": scf_meta,
                "ref_dir": None
                if not (use_ref_embtheta or use_ref_preview)
                else str(Path(args.ref_dir).resolve()),
                "mode": mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[batch_scf] finished {n_done} frames → {out_dir}")
    print(f"  summary: {summary.name}")


if __name__ == "__main__":
    main()
