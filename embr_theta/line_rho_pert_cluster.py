"""
QM–MM **axis** line density: ρ_emb vs **Cluster 全 DFT** vs ρ_pert (CP-KS linear response).

Same layout as ``line_rho_pert`` but the reference is one full_QM supermolecule (real QM +
shell waters, total DM — no ghost, no QM-block projection). Also prints dipole
moments @ QM geometry center.

Example::

  python -m embr_theta.line_rho_pert_cluster \\
    --manifest theta/manifest.json \\
    --peratom theta/pert_peratom_a15.npz \\
    --pert-cache theta/pert_a15.npz \\
    --frame 0 --fix-alpha 1.5 --d3bj --threads 4 --embed-scf
"""

from __future__ import annotations

import sys
from pathlib import Path

_THETA_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THETA_DIR.parent
for _p in (_THETA_DIR, _PROJECT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from embr_theta._deps import require_scf

require_scf()

import argparse
import inspect
import json

import numpy as np

from scf_embed_cluster import cluster_qm_dm, run_cluster_supermol_mf
from scf_embed_rep_center import (
    REP_CENTER_PLACEMENTS,
    build_repulsion_center_coords_ang,
    placement_label_zh,
    run_embed_theta_mf_rep,
)
from embr_io import COO_NAME_FMT, coo_path
from scf_embed_io import ScfFrame, filter_mm_by_distance, load_scf_frame
from scf_embed_pyscf import (
    ConeRepAng,
    ScfEmbedConfig,
    amp_mm_for_o_h_sites,
    cone_axes_h_to_nearest_qm,
    eval_density_from_dm,
    hartree_to_kcal,
    run_gas_mf,
    scf_embed_config_from_cli,
)
from scf_embed_perturb import (
    compute_cphf_dm1,
    delta_e_peratom_kcal,
    delta_e_tr_ph,
    delta_e_tr_ph0,
    enumerate_qm_mm_line_pairs,
    gaussian_repulsion_ao,
    line_rho_profile_on_qm_mm_axis,
    run_density_pert_frame,
    run_emb0_mf,
)
from embr_io import load_e_embed_txt
from embr_dataset import get_residue, resolve_dataset_label, resolve_dataset_n_qm
from embr_theta.amp_override import MM_EL_O, apply_amp_overrides
from embr_theta.cache import amp_mm_frame_slice, load_pert_cache, load_pert_peratom_labels, mm_frame_slice
from embr_theta.format_rho import fmt_rho


def _print_mm_amp_dist_table(frame, amp_mm, *, kernel_mm=None, frame_index=None) -> None:
    amp = np.asarray(amp_mm, dtype=np.float64).reshape(-1)
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64)
    tag = f"frame={frame_index}" if frame_index is not None else "frame=?"
    print(f"\n[MM amp vs dist] {tag}")
    print("  MM#  elem   dist(Å)    A_j(Ha)")
    for j, (sym, pos) in enumerate(zip(frame.mm_symbols, frame.mm_coords_ang)):
        d = float(np.min(np.linalg.norm(qm - np.asarray(pos, dtype=np.float64), axis=1)))
        a = float(amp[j]) if j < amp.size else float("nan")
        print(f"  {j:3d}  {sym:<4s}  {d:8.3f}  {a:+.6f}")

from embr_theta.repulsion_policy import enforce_h_only_amplitudes, resolve_h_only

_AU_TO_DEBYE = 2.5417461928


def _cone_rep_kw(frame: ScfFrame, cone: ConeRepAng) -> dict:
    if cone.is_isotropic():
        return {}
    return {
        "cone": cone,
        "cone_axes_ang": cone_axes_h_to_nearest_qm(frame),
        "mm_symbols": frame.mm_symbols,
    }


def _scf_cache_load_supports_cone(load_fn) -> bool:
    return "cone" in inspect.signature(load_fn).parameters


def _cache_expect_for_file(expect: dict, cache_path: Path) -> dict:
    """旧 scf-cache 无 cone 元数据时，不要求 cone 指纹一致。"""
    try:
        with np.load(cache_path, allow_pickle=True) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return dict(expect)
    out = dict(expect)
    if "cone_theta1_deg" not in meta:
        out.pop("cone_theta1_deg", None)
        out.pop("cone_theta2_deg", None)
    return out


def _mean_line_profiles_at_r_grid(
    samples: list[tuple[np.ndarray, np.ndarray]],
    r_grid: np.ndarray,
) -> np.ndarray:
    r_grid = np.asarray(r_grid, dtype=np.float64)
    acc = np.zeros(r_grid.size, dtype=np.float64)
    cnt = np.zeros(r_grid.size, dtype=np.int64)
    for r_loc, rho in samples:
        r_loc = np.asarray(r_loc, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        d_end = float(r_loc[-1])
        for ir, rad in enumerate(r_grid):
            if float(rad) > d_end + 1e-9:
                continue
            acc[ir] += float(np.interp(float(rad), r_loc, rho))
            cnt[ir] += 1
    out = np.zeros_like(acc)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out


def _compute_axis_profiles(
    frame: ScfFrame,
    mf_emb,
    mf_cl,
    mf_gas,
    dm_emb,
    dm_cl_qm,
    dm_cl_tot,
    dm_gas,
    dm_pert,
    *,
    pair_max_ang: float,
    dr: float,
    r_line_max: float,
) -> tuple[np.ndarray, dict, dict, dict, dict, dict, list]:
    """Axis ρ: emb / cluster QM block / cluster total / gas / pert."""
    pairs = enumerate_qm_mm_line_pairs(frame, pair_max_ang=float(pair_max_ang))
    r_grid = np.arange(0.0, float(r_line_max) + 0.5 * float(dr), float(dr))
    emb_samples: dict[str, list] = {"all": [], "mm_O": [], "mm_H": []}
    cl_qm_samples: dict[str, list] = {"all": [], "mm_O": [], "mm_H": []}
    cl_tot_samples: dict[str, list] = {"all": [], "mm_O": [], "mm_H": []}
    gas_samples: dict[str, list] = {"all": [], "mm_O": [], "mm_H": []}
    pert_samples: dict[str, list] = {"all": [], "mm_O": [], "mm_H": []}
    for p in pairs:
        kw = dict(dr=float(dr))
        r_e, rho_e = line_rho_profile_on_qm_mm_axis(mf_emb, dm_emb, p.r_mm_ang, p.r_qm_ang, **kw)
        r_q, rho_q = line_rho_profile_on_qm_mm_axis(mf_cl, dm_cl_qm, p.r_mm_ang, p.r_qm_ang, **kw)
        r_t, rho_t = line_rho_profile_on_qm_mm_axis(mf_cl, dm_cl_tot, p.r_mm_ang, p.r_qm_ang, **kw)
        r_g, rho_g = line_rho_profile_on_qm_mm_axis(mf_gas, dm_gas, p.r_mm_ang, p.r_qm_ang, **kw)
        r_p, rho_p = line_rho_profile_on_qm_mm_axis(mf_emb, dm_pert, p.r_mm_ang, p.r_qm_ang, **kw)
        mel = p.mm_symbol.strip()[0].upper()
        key = "mm_O" if mel == "O" else "mm_H"
        for bucket, pair in (
            (emb_samples, (r_e, rho_e)),
            (cl_qm_samples, (r_q, rho_q)),
            (cl_tot_samples, (r_t, rho_t)),
            (gas_samples, (r_g, rho_g)),
            (pert_samples, (r_p, rho_p)),
        ):
            bucket["all"].append(pair)
            bucket[key].append(pair)

    def _out(samples):
        return {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in samples.items()}

    return (
        r_grid,
        _out(emb_samples),
        _out(cl_qm_samples),
        _out(cl_tot_samples),
        _out(gas_samples),
        _out(pert_samples),
        pairs,
    )


def _axis_groups_for_dm(
    pairs: list,
    mf_emb,
    dm_extra,
    r_grid: np.ndarray,
    *,
    dr: float,
) -> dict[str, np.ndarray]:
    extra_samples: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"all": [], "mm_O": [], "mm_H": []}
    for p in pairs:
        r_p, rho_p = line_rho_profile_on_qm_mm_axis(
            mf_emb, dm_extra, p.r_mm_ang, p.r_qm_ang, dr=float(dr)
        )
        mel = p.mm_symbol.strip()[0].upper()
        extra_samples["all"].append((r_p, rho_p))
        key = "mm_O" if mel == "O" else "mm_H"
        extra_samples[key].append((r_p, rho_p))
    return {k: _mean_line_profiles_at_r_grid(v, r_grid) for k, v in extra_samples.items()}


def _qm_com_bohr(frame) -> np.ndarray:
    from pyscf import lib

    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64).reshape(-1, 3)
    return np.mean(qm, axis=0) / lib.param.BOHR


def _dipole_tot_at_com_debye(mf, dm: np.ndarray, frame) -> np.ndarray:
    mol = mf.mol
    dm = np.asarray(dm, dtype=np.float64)
    com = tuple(float(x) for x in _qm_com_bohr(frame))
    old_verbose = int(getattr(mol, "verbose", 0))
    mol.verbose = 0
    try:
        with mol.with_common_origin(com):
            try:
                mu = mf.dip_moment(unit="AU", dm=dm, verbose=0)
            except TypeError:
                mu = mf.dip_moment(unit="AU", dm=dm)
    finally:
        mol.verbose = old_verbose
    return np.asarray(mu, dtype=np.float64).reshape(3) * _AU_TO_DEBYE


def _print_dipole_row(label: str, mu: np.ndarray, *, mu_ref: np.ndarray | None) -> None:
    mag = float(np.linalg.norm(mu))
    if mu_ref is not None and np.all(np.isfinite(mu_ref)):
        dmu = mu - mu_ref
        dmag = float(np.linalg.norm(dmu))
        print(
            f"    {label:18s}  {mag:10.4f} D  "
            f"({mu[0]:+.3f}, {mu[1]:+.3f}, {mu[2]:+.3f})  "
            f"|Δμ|vs气={dmag:.4f} D  ({dmu[0]:+.3f},{dmu[1]:+.3f},{dmu[2]:+.3f})"
        )
    else:
        print(
            f"    {label:18s}  {mag:10.4f} D  "
            f"({mu[0]:+.3f}, {mu[1]:+.3f}, {mu[2]:+.3f})  (参考)"
        )


def _print_dipole_block(
    frame,
    *,
    mu_gas: np.ndarray,
    mu_cluster: np.ndarray,
    mu_emb: np.ndarray,
    mu_pert: np.ndarray,
    mu_scf: np.ndarray | None,
) -> None:
    print("\n[偶极 μ_tot @ QM 几何中心]")
    print("    说明: 所有态用同一原点；|Δμ|vs气 = 相对孤立 QM 的极化幅度")
    _print_dipole_row("气相 QM", mu_gas, mu_ref=None)
    _print_dipole_row("Cluster 全 DFT", mu_cluster, mu_ref=mu_gas)
    _print_dipole_row("Emb0", mu_emb, mu_ref=mu_gas)
    _print_dipole_row("CP-KS pert", mu_pert, mu_ref=mu_gas)
    if mu_scf is not None:
        _print_dipole_row("EmbR SCF", mu_scf, mu_ref=mu_gas)


def _run_embed_theta_mf(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    amp_mm: np.ndarray,
    *,
    alpha_bohr2: float,
    envelope_cfg=None,
    rep_centers_ang: np.ndarray,
    cone: ConeRepAng,
):
    return run_embed_theta_mf_rep(
        frame,
        cfg,
        amp_mm,
        alpha_bohr2=float(alpha_bohr2),
        envelope_cfg=envelope_cfg,
        rep_centers_ang=rep_centers_ang,
        cone=cone,
    )


def _dm_pert_from_result(
    pert_res,
    mf_emb,
    dm_emb,
    frame,
    amp_mm,
    alpha: float,
    rep_centers_ang: np.ndarray,
    *,
    envelope_cfg=None,
    rep_on_nucleus: bool,
    cone: ConeRepAng,
) -> np.ndarray:
    if rep_on_nucleus:
        dm_pert = getattr(pert_res, "dm1_tot", None)
        if dm_pert is not None:
            return np.asarray(dm_pert, dtype=np.float64)
    amps = amp_mm_for_o_h_sites(frame, amp_mm)
    h1 = gaussian_repulsion_ao(
        mf_emb,
        rep_centers_ang,
        amps,
        alpha_bohr2=float(alpha),
        envelope_cfg=envelope_cfg,
        mm_symbols=frame.mm_symbols,
        **_cone_rep_kw(frame, cone),
    )
    return np.asarray(dm_emb, dtype=np.float64) + np.asarray(compute_cphf_dm1(mf_emb, h1), dtype=np.float64)


def _recompute_vrep_dependent(
    frame: ScfFrame,
    cfg: ScfEmbedConfig,
    cfg_scf: ScfEmbedConfig,
    mf_emb,
    dm_emb: np.ndarray,
    amp_mm: np.ndarray,
    *,
    alpha: float,
    envelope_cfg=None,
    cone: ConeRepAng,
    rep_centers: np.ndarray,
    rep_center: str,
    pert_cache,
    fi: int,
    embed_scf: bool,
) -> tuple:
    """Reuse Cluster reference; refresh Emb0 then CP-KS pert + EmbR (A_H changed)."""
    from embr_theta.cache import mm_frame_slice

    print("  [A_H rescan] Emb0 SCF（跳过 Cluster，新 A_H 需自洽 MO）...", flush=True)
    mf_emb = run_emb0_mf(frame, cfg)
    dm_emb = mf_emb.make_rdm1()

    kernel_mm = None
    if pert_cache is not None:
        kernel_mm, _ = mm_frame_slice(pert_cache, fi)

    print("  [A_H rescan] CP-KS density perturbation ...", flush=True)
    pert_res, mf_emb = run_density_pert_frame(
        frame,
        cfg,
        amp_mm,
        alpha_bohr2=alpha,
        envelope_cfg=envelope_cfg,
        mf=mf_emb,
        kernel_mm=kernel_mm,
        cone=cone,
        rdf_max=0.0,
    )
    dm_pert = _dm_pert_from_result(
        pert_res,
        mf_emb,
        dm_emb,
        frame,
        amp_mm,
        alpha,
        rep_centers,
        envelope_cfg=envelope_cfg,
        rep_on_nucleus=(rep_center == "on_nucleus"),
        cone=cone,
    )
    de_kernel = float(pert_res.de_kernel_kcal)
    if kernel_mm is not None and not np.isfinite(de_kernel):
        de_kernel = float(delta_e_peratom_kcal(kernel_mm, amp_mm))

    de_lin_override = float(pert_res.de1_kcal)
    de_rho1_override = float(getattr(pert_res, "de_rho1_kcal", float("nan")))
    if rep_center != "on_nucleus":
        amps = amp_mm_for_o_h_sites(frame, amp_mm)
        h1 = gaussian_repulsion_ao(
            mf_emb,
            rep_centers,
            amps,
            alpha_bohr2=float(alpha),
            envelope_cfg=envelope_cfg,
            mm_symbols=frame.mm_symbols,
            **_cone_rep_kw(frame, cone),
        )
        de_lin_override = float(hartree_to_kcal(delta_e_tr_ph0(mf_emb, h1)))
        de_rho1_override = float(hartree_to_kcal(delta_e_tr_ph(mf_emb, h1, dm_pert)))

    mf_scf = None
    dm_scf = None
    if embed_scf:
        print(f"  [A_H rescan] EmbR SCF (conv_tol={cfg_scf.conv_tol}) ...", flush=True)
        mf_scf = _run_embed_theta_mf(
            frame,
            cfg_scf,
            amp_mm,
            alpha_bohr2=alpha,
            envelope_cfg=envelope_cfg,
            rep_centers_ang=rep_centers,
            cone=cone,
        )
        dm_scf = mf_scf.make_rdm1()

    return (
        mf_emb,
        dm_emb,
        mf_scf,
        dm_scf,
        dm_pert,
        pert_res,
        de_kernel,
        de_lin_override,
        de_rho1_override,
    )


def _scf_cycles(mf) -> int | None:
    for attr in ("cycles", "cycle"):
        v = getattr(mf, attr, None)
        if v is not None:
            try:
                n = int(v)
                if n >= 0:
                    return n
            except (TypeError, ValueError):
                pass
    hist = getattr(mf, "scf_history", None)
    if hist is not None:
        try:
            return len(hist)
        except TypeError:
            pass
    return None


def _resolve_manifest_task(manifest: Path, frame: int) -> dict:
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    global_k = 0
    for ds in cfg.get("datasets") or []:
        label = resolve_dataset_label(ds)
        n_qm = resolve_dataset_n_qm(ds)
        coo_dir = Path(ds["coo_dir"])
        fmt = str(ds.get("coo_name_fmt", COO_NAME_FMT))
        i0 = int(ds.get("i0", 0))
        n_frames = int(ds["n_frames"])
        e0_path = Path(ds["e0_file"])
        e0_arr, e_int_bg = load_e_embed_txt(e0_path)
        for k in range(n_frames):
            if global_k == int(frame):
                return {
                    "coo_path": coo_path(coo_dir, fmt, i0 + k),
                    "residue": label,
                    "n_qm": int(n_qm),
                    "file_index": i0 + k,
                    "e0_kcal": float(e0_arr[k]),
                    "e_int_bg_kcal": float(e_int_bg[k]) if e_int_bg is not None else float("nan"),
                    "e0_file": e0_path,
                }
            global_k += 1
    raise ValueError(f"frame {frame} not found in {manifest}")


def _print_energy_block(
    *,
    e0: float,
    e_int_bg: float,
    e0_file: Path | None,
    pert_res,
    de_kernel: float,
    mf_emb,
    mf_cl,
    mf_scf,
    embed_scf: bool,
    de_lin_override: float | None = None,
    de_rho1_override: float | None = None,
) -> None:
    e_high = e_int_bg + e0 if np.isfinite(e_int_bg) else float("nan")
    de_lin = float(de_lin_override if de_lin_override is not None else pert_res.de1_kcal)
    de_rho1 = float(
        de_rho1_override if de_rho1_override is not None else getattr(pert_res, "de_rho1_kcal", float("nan"))
    )
    e_emb0_tot = float(mf_emb.e_tot)
    cyc_emb0 = _scf_cycles(mf_emb)
    cyc_cl = _scf_cycles(mf_cl)
    cyc_emb = _scf_cycles(mf_scf) if mf_scf is not None else None

    print("\n[能量对照，单位 kcal/mol]")
    src = f" ({e0_file})" if e0_file is not None else " (peratom 标签)"
    print(f"  E0 (修正标签){src}     = {e0:10.3f}")
    if np.isfinite(e_int_bg):
        print(f"  E_int_bg (E*.txt 第2列) = {e_int_bg:10.3f}")
        print(f"  E_high (= E_int_bg+E0)  = {e_high:10.3f}")
    print("  ※ Tr(P0V)、ΣAk 对照 E0；|err|≲0.1 表示 α/A 与 fit 一致（无 bug 前提）")
    print(f"  Tr(P0V) 一阶 / Emb0 密度        = {de_lin:10.3f}   vs E0  |err|={abs(de_lin - e0):.4f}")
    if np.isfinite(de_rho1):
        print(f"  Tr(P1V) 一阶 / 微扰后密度      = {de_rho1:10.3f}   vs E0  |err|={abs(de_rho1 - e0):.4f}")
    if np.isfinite(de_kernel):
        print(f"  Σ A_j k_j 核积分               = {de_kernel:10.3f}   vs E0  |err|={abs(de_kernel - e0):.4f}")

    cyc_emb0_s = f"  cycles={cyc_emb0}" if cyc_emb0 is not None else ""
    cyc_cl_s = f"  cycles={cyc_cl}" if cyc_cl is not None else ""
    print(f"  Emb0 E_tot                     = {hartree_to_kcal(e_emb0_tot):10.3f}{cyc_emb0_s}")
    print(
        f"  Cluster 全 DFT E_tot           = {hartree_to_kcal(float(mf_cl.e_tot)):10.3f}{cyc_cl_s}  "
        f"nao={int(mf_cl.mol.nao)}"
    )
    if embed_scf and mf_scf is not None:
        e_emb_scf_tot = float(mf_scf.e_tot)
        de_int_kcal = hartree_to_kcal(e_emb_scf_tot - e_emb0_tot)
        cyc_emb_s = f"  cycles={cyc_emb}" if cyc_emb is not None else ""
        print(f"  EmbR E_tot (V in H, 1×SCF)     = {hartree_to_kcal(e_emb_scf_tot):10.3f}{cyc_emb_s}")
        print(
            f"  ΔE_int(EmbR−Emb0)              = {de_int_kcal:10.3f}   vs E0  |err|={abs(de_int_kcal - e0):.4f}"
        )


def _fix_ratio_str(de: float, dp: float, *, eps: float = 1e-18) -> str:
    if not np.isfinite(de) or not np.isfinite(dp) or abs(float(de)) < float(eps):
        return "   n/a"
    return f"{1.0 - float(dp) / float(de):7.4f}"


def _fix_scf_from_rho_triplet(
    rho_emb: float, rho_cl: float, rho_scf: float, *, eps: float = 1e-18
) -> float:
    """fix_s = 1 − Δ_scf/Δ_emb on one axis sample."""
    de = float(rho_emb) - float(rho_cl)
    ds = float(rho_scf) - float(rho_cl)
    if not np.isfinite(de) or not np.isfinite(ds) or abs(de) < float(eps):
        return float("nan")
    return 1.0 - ds / de


def _fix_scf_on_axis_at_r(
    r_grid: np.ndarray,
    rho_emb: np.ndarray,
    rho_cl: np.ndarray,
    rho_scf: np.ndarray,
    r_ang: float,
) -> float:
    r_grid = np.asarray(r_grid, dtype=np.float64)
    ir = int(np.argmin(np.abs(r_grid - float(r_ang))))
    return _fix_scf_from_rho_triplet(
        float(rho_emb[ir]), float(rho_cl[ir]), float(rho_scf[ir])
    )


def _pair_for_mm_h(hi: int, pairs: list):
    hits = [p for p in pairs if int(p.jm) == int(hi)]
    if not hits:
        return None
    return min(hits, key=lambda p: float(p.dist_ang))


def _axis_profiles_on_pair(
    pair,
    mf_emb,
    mf_cl,
    dm_emb: np.ndarray,
    dm_cl_qm: np.ndarray,
    dm_scf: np.ndarray,
    *,
    dr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kw = {"dr": float(dr)}
    r_e, rho_e = line_rho_profile_on_qm_mm_axis(
        mf_emb, dm_emb, pair.r_mm_ang, pair.r_qm_ang, **kw
    )
    r_c, rho_c = line_rho_profile_on_qm_mm_axis(
        mf_cl, dm_cl_qm, pair.r_mm_ang, pair.r_qm_ang, **kw
    )
    r_s, rho_s = line_rho_profile_on_qm_mm_axis(
        mf_emb, dm_scf, pair.r_mm_ang, pair.r_qm_ang, **kw
    )
    return r_e, rho_e, rho_c, rho_s


def _fix_scf_for_mm_h_at_r(
    hi: int,
    pairs: list,
    mf_emb,
    mf_cl,
    dm_emb: np.ndarray,
    dm_cl_qm: np.ndarray,
    dm_scf: np.ndarray | None,
    *,
    r_ang: float = 1.0,
    dr: float = 0.1,
) -> float:
    if dm_scf is None:
        return float("nan")
    pair = _pair_for_mm_h(hi, pairs)
    if pair is None:
        return float("nan")
    r_e, rho_e, rho_c, rho_s = _axis_profiles_on_pair(
        pair, mf_emb, mf_cl, dm_emb, dm_cl_qm, dm_scf, dr=float(dr)
    )
    return _fix_scf_on_axis_at_r(r_e, rho_e, rho_c, rho_s, r_ang)


def _fix_profile_on_axis(
    r_grid: np.ndarray,
    rho_emb: np.ndarray,
    rho_cl: np.ndarray,
    rho_scf: np.ndarray,
) -> np.ndarray:
    r_grid = np.asarray(r_grid, dtype=np.float64)
    out = np.full(r_grid.size, np.nan, dtype=np.float64)
    for ir in range(r_grid.size):
        out[ir] = _fix_scf_from_rho_triplet(
            float(rho_emb[ir]), float(rho_cl[ir]), float(rho_scf[ir])
        )
    return out


def _print_selected_mm_h_axis_fix_table(
    hi: int,
    pairs: list,
    mf_emb,
    mf_cl,
    dm_emb: np.ndarray,
    dm_cl_qm: np.ndarray,
    dm_scf: np.ndarray | None,
    *,
    dr: float,
    r_print_max: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """0.2 Å 步长 fix 轴线表（与上方 MM-H 平均表格式一致）。"""
    if dm_scf is None:
        print(f"\n  [所选 MM-H[{int(hi)}] fix 轴线]  无 EmbR DM（需 embed-scf）")
        return None
    pair = _pair_for_mm_h(hi, pairs)
    if pair is None:
        print(f"\n  [所选 MM-H[{int(hi)}] fix 轴线]  不在 pair_max 配对内")
        return None
    r_e, rho_e, rho_c, rho_s = _axis_profiles_on_pair(
        pair, mf_emb, mf_cl, dm_emb, dm_cl_qm, dm_scf, dr=float(dr)
    )
    iqm = int(pair.iq)
    sq = str(pair.qm_symbol).strip()
    _print_axis_rho_table(
        f"所选 MM-H[{int(hi)}]→QM[{iqm}]={sq}  单对轴线+fix_s（plane/B 表同源）",
        r_e,
        rho_e,
        rho_c,
        rho_s,
        r_print_max=float(r_print_max),
        with_fix=True,
    )
    fix_prof = _fix_profile_on_axis(r_e, rho_e, rho_c, rho_s)
    return r_e, fix_prof


def _print_line_table_pert(
    r: np.ndarray,
    emb: np.ndarray,
    cluster_qm: np.ndarray,
    cluster_tot: np.ndarray,
    gas: np.ndarray,
    pert: np.ndarray,
    *,
    title: str,
    n_pairs: int,
    scf: np.ndarray | None = None,
    r_min_print: float = 0.0,
) -> None:
    d_emb = emb - cluster_qm
    d_pert = pert - cluster_qm
    spill_emb = emb - gas
    spill_cl = cluster_qm - gas
    d_scf = None if scf is None else scf - cluster_qm
    print(f"\n[{title}]  (QM–MM 连线平均, 有效对数={n_pairs})")
    print("  轴: MM 核 r=0 → QM；ρ 均为 **QM 电子**（Emb / Cluster^QM / gas / pert）")
    print("  spill_emb = ρ_emb−ρ_gas  (点电荷造成的 QM 溢出；应 >0 近 MM)")
    print("  Δ_emb = ρ_emb−ρ_Cluster^QM  (Emb 相对真实溶剂化 QM；>0 = 点电荷多溢)")
    print("  ρ_Cluster^tot = QM+水 **总**电子（近 MM 含水核芯/键区，仅作物理背景）")
    print("  fix = 1−Δ_pert/Δ_emb vs Cluster^QM")
    if scf is not None:
        print("  ρ_scf: EmbR 全自洽")
    if float(r_min_print) > 0.0:
        print(f"  ※ 仅打印 r≥{float(r_min_print):g} Å")
    hdr = (
        "  r(Å)   ρ_emb        ρ_cl^QM      ρ_pert       spill_emb    Δ_emb        "
        "fix_p        ρ_cl^tot"
    )
    if scf is not None:
        hdr = (
            "  r(Å)   ρ_emb        ρ_cl^QM      ρ_pert       ρ_scf        spill_emb    "
            "Δ_emb        Δ_pert       fix_p        fix_s        ρ_cl^tot"
        )
    print(hdr)
    for ir, rad in enumerate(r):
        if float(rad) + 1e-9 < float(r_min_print):
            continue
        if ir % 2 == 1 and ir < len(r) - 1:
            continue
        e0 = float(emb[ir])
        c0 = float(cluster_qm[ir])
        p0 = float(pert[ir])
        g0 = float(gas[ir])
        de = float(d_emb[ir])
        dp = float(d_pert[ir])
        se = float(spill_emb[ir])
        fix_p = _fix_ratio_str(de, dp)
        tot = float(cluster_tot[ir])
        if scf is not None:
            s0 = float(scf[ir])
            ds = float(d_scf[ir])  # type: ignore[index]
            fix_s = _fix_ratio_str(de, ds)
            print(
                f"  {rad:4.1f}  {fmt_rho(e0):>12}  {fmt_rho(c0):>12}  {fmt_rho(p0):>12}  {fmt_rho(s0):>12}  "
                f"{fmt_rho(se):>12}  {fmt_rho(de):>12}  {fmt_rho(dp):>12}  {fix_p:>12}  {fix_s:>12}  {fmt_rho(tot):>12}"
            )
        else:
            print(
                f"  {rad:4.1f}  {fmt_rho(e0):>12}  {fmt_rho(c0):>12}  {fmt_rho(p0):>12}  {fmt_rho(se):>12}  "
                f"{fmt_rho(de):>12}  {fix_p:>12}  {fmt_rho(tot):>12}"
            )


def _amp_for_frame(
    labels: dict,
    frame: int,
    *,
    h_only: bool,
    probe_a_o: float | None,
    probe_a_h: float | None,
    scale_a_o: float | None = None,
    scale_a_h: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    amp_mm = amp_mm_frame_slice(labels, frame)
    mm_el = labels.get("mm_element")
    ptr = labels["mm_ptr"]
    el_slice = None if mm_el is None else np.asarray(mm_el[int(ptr[frame]) : int(ptr[frame + 1])], dtype=np.int8)
    if h_only and probe_a_o is None:
        amp_mm = enforce_h_only_amplitudes(amp_mm, el_slice, h_only=True)
    if probe_a_o is not None or probe_a_h is not None:
        amp_mm = apply_amp_overrides(amp_mm, el_slice, set_a_o=probe_a_o, set_a_h=probe_a_h)
    if scale_a_o is not None or scale_a_h is not None:
        amp_mm = apply_amp_overrides(amp_mm, el_slice, scale_a_o=scale_a_o, scale_a_h=scale_a_h)
    return amp_mm, el_slice


def _write_plane_npz_for_plot(
    out_path: Path,
    *,
    frame: ScfFrame,
    frame_index: int,
    mm_h_index: int,
    mf_emb,
    mf_cl,
    dm_emb: np.ndarray,
    dm_cl_qm: np.ndarray,
    dm_scf: np.ndarray,
    coo_path: Path,
    residue: str,
    alpha: float,
    rep_center: str,
    rep_oh_frac: float,
    r_cut_mm: float | None,
    plane_margin: float,
    plane_step: float,
    plane_min_half: float,
    line_dr: float,
    line_r_max: float,
    pairs: list,
    r_line: np.ndarray,
    emb_avg_mm_h: np.ndarray | None,
    cl_avg_mm_h: np.ndarray | None,
    scf_avg_mm_h: np.ndarray | None,
    n_mm_h_pairs: int,
    pair_max_ang: float,
) -> None:
    """
    Write ``plot_rho_plane``-compatible npz using **already converged** mf/dm.

    No extra SCF — only ρ evaluation on a 2D plane grid + axis cross-check print.
    """
    from embr_theta.rho_plane_geom import (
        bonded_o_index_for_mm_h,
        build_plane_grid,
        interp_plane_field,
        nearest_qm_index_to_mm_h,
        project_to_plane_uv,
    )

    hi = int(mm_h_index)
    oi = bonded_o_index_for_mm_h(hi, frame)
    iqm = nearest_qm_index_to_mm_h(hi, frame)
    mm = np.asarray(frame.mm_coords_ang, dtype=np.float64)
    qm = np.asarray(frame.qm_coords_ang, dtype=np.float64)
    p_h, p_o, p_qm = mm[hi], mm[oi], qm[iqm]

    u_axis, v_axis, uu, vv, points_flat, origin, e1, e2 = build_plane_grid(
        p_h,
        p_o,
        p_qm,
        margin_ang=float(plane_margin),
        step_ang=float(plane_step),
        min_half_ang=float(plane_min_half),
    )
    nv, nu = int(v_axis.size), int(u_axis.size)
    n_pts = int(points_flat.shape[0])
    print(
        f"\n[plane-npz] 开始格点 ρ 求值（**不再跑 SCF**；grid {nv}×{nu}={n_pts} 点）",
        flush=True,
    )

    def _rho2d(mf, dm: np.ndarray, label: str) -> np.ndarray:
        print(f"  plane ρ: {label} ...", flush=True)
        rho = eval_density_from_dm(mf, dm, points_flat)
        return np.asarray(rho, dtype=np.float64).reshape(nv, nu)

    dm_scf_a = np.asarray(dm_scf, dtype=np.float64)
    dm_emb_a = np.asarray(dm_emb, dtype=np.float64)
    if dm_scf_a.shape != dm_emb_a.shape:
        raise ValueError(
            f"dm_scf shape {dm_scf_a.shape} != dm_emb {dm_emb_a.shape}; "
            "EmbR DM must live on Emb0 AO basis"
        )
    dm_scf_emb_diff = float(np.max(np.abs(dm_scf_a - dm_emb_a)))
    print(
        f"  [plane-npz] DM 自检 (QM 基组): max|dm_scf−dm_emb|={dm_scf_emb_diff:.3e}  "
        f"(Cluster^QM 在 supermol 基组 nao={mf_cl.mol.nao}，与 Emb0 nao={mf_emb.mol.nao} 不可直接比 DM)"
    )

    rho_emb0 = _rho2d(mf_emb, dm_emb, "Emb0")
    rho_cluster_qm = _rho2d(mf_cl, dm_cl_qm, "Cluster^QM (supermol 基组，此步最慢)")
    rho_emb_scf = _rho2d(mf_emb, dm_scf, "EmbR")

    max_se = float(np.max(np.abs(rho_emb_scf - rho_emb0)))
    max_ce = float(np.max(np.abs(rho_cluster_qm - rho_emb0)))
    print(
        f"  [plane-npz] 平面 ρ 自检: max|ρ_scf−ρ_emb|={max_se:.3e}  "
        f"max|ρ_cl^QM−ρ_emb|={max_ce:.3e} a.u."
    )

    uh, vh = project_to_plane_uv(p_h, origin, e1, e2)
    uo, vo = project_to_plane_uv(p_o, origin, e1, e2)
    uq, vq = project_to_plane_uv(p_qm, origin, e1, e2)
    dist_h_qm_3d = float(np.linalg.norm(p_qm - p_h))
    dist_o_h_3d = float(np.linalg.norm(p_o - p_h))
    dist_h_qm_plane = float(np.hypot(uq - uh, vq - vh))

    # Cross-check: same axis eval as this script's line tables (single H–QM pair).
    r_mm, r_qm = p_h, p_qm
    dr = float(line_dr)
    r_e, rho_e = line_rho_profile_on_qm_mm_axis(mf_emb, dm_emb, r_mm, r_qm, dr=dr)
    r_c, rho_c = line_rho_profile_on_qm_mm_axis(mf_cl, dm_cl_qm, r_mm, r_qm, dr=dr)
    r_s, rho_s = line_rho_profile_on_qm_mm_axis(mf_emb, dm_scf, r_mm, r_qm, dr=dr)
    d_axis = float(np.hypot(uq, vq))
    tu, tv = uq / d_axis, vq / d_axis
    max_plane_err = 0.0
    r_print = min(float(line_r_max), 2.0, d_axis)
    fix0 = _fix_scf_on_axis_at_r(r_e, rho_e, rho_c, rho_s, 0.0)
    fix1 = _fix_scf_on_axis_at_r(r_e, rho_e, rho_c, rho_s, 1.0)
    fix0_s = f"{fix0:.4f}" if np.isfinite(fix0) else "n/a"
    fix1_s = f"{fix1:.4f}" if np.isfinite(fix1) else "n/a"

    print(
        f"\n[plane-npz] 写 {out_path.name}  MM-H[{hi}]→QM[{iqm}]={frame.qm_symbols[iqm]}  "
        f"d(3D)={dist_h_qm_3d:.3f} Å  d_plane(H→QM)={dist_h_qm_plane:.3f} Å  "
        f"d(O–H)={dist_o_h_3d:.3f} Å  step={plane_step:g} Å  "
        f"fix@0Å={fix0_s}  fix@1Å={fix1_s}"
    )
    print(
        "  原子平面坐标 (原点=MM H, u 沿 H→O 键):  "
        f"H=({uh:.3f},{vh:.3f})  O[{oi}]=({uo:.3f},{vo:.3f})  "
        f"QM[{iqm}]=({uq:.3f},{vq:.3f})  ※ O 应在 u≈+0.96"
    )
    print("  ※ A=7个H多对平均；B=仅本 H→QM[7] 单对（与图切片一致）；C=格点vs B")
    if dist_h_qm_plane > 1.6:
        print(
            f"  ※ QM 在平面距 H {dist_h_qm_plane:.2f}Å；画图 --mm-r-max 1.5 会把 QM 裁到视野外"
        )

    pair_hits = [p for p in pairs if int(p.jm) == hi]
    if not pair_hits:
        print(
            f"  ※ 警告: MM-H[{hi}] 不在 pair_max={pair_max_ang:g} Å 的 QM–MM 配对中；"
            "「MM端为 H」平均与 B 不可逐点对比"
        )
    else:
        qm_syms = sorted({p.qm_symbol for p in pair_hits})
        print(f"  ※ MM-H[{hi}] 在配对中出现 {len(pair_hits)} 次 → QM: {', '.join(qm_syms)}")

    if (
        emb_avg_mm_h is not None
        and cl_avg_mm_h is not None
        and scf_avg_mm_h is not None
        and n_mm_h_pairs > 0
    ):
        _print_axis_rho_table(
            f"A  line「MM端为 H」多对平均 n={n_mm_h_pairs}（=上方表格同源）",
            r_line,
            emb_avg_mm_h,
            cl_avg_mm_h,
            scf_avg_mm_h,
            r_print_max=r_print,
        )

    _print_axis_rho_table(
        f"B  仅 MM-H[{hi}]→QM[{iqm}] 单对（平面 H–O–QM 切片用的这一对）",
        r_e,
        rho_e,
        rho_c,
        rho_s,
        r_print_max=r_print,
        with_fix=True,
    )

    fix_axis = _fix_profile_on_axis(r_e, rho_e, rho_c, rho_s)

    print("\n  [C  plane-grid 一致性] B 的 3D 轴线 ρ vs 2D 格点双线性插值")
    for ir, rad in enumerate(r_e):
        if float(rad) > r_print + 1e-9:
            break
        e0, c0, s0 = float(rho_e[ir]), float(rho_c[ir]), float(rho_s[ir])
        u, v = float(float(rad) * tu), float(float(rad) * tv)
        max_plane_err = max(
            max_plane_err,
            abs(e0 - interp_plane_field(u_axis, v_axis, rho_emb0, u, v)),
            abs(c0 - interp_plane_field(u_axis, v_axis, rho_cluster_qm, u, v)),
            abs(s0 - interp_plane_field(u_axis, v_axis, rho_emb_scf, u, v)),
        )
    print(f"  max|Δ| = {max_plane_err:.3e} a.u.", end="")
    if max_plane_err > 1e-4:
        print("  ※ >1e-4：试 --plane-step 0.05")
    else:
        print("  OK")

    meta = {
        "pipeline": "line_rho_pert_cluster_plane",
        "coo_path": str(coo_path.resolve()),
        "frame": int(frame_index),
        "residue": str(residue),
        "mm_h_index": hi,
        "mm_o_index": int(oi),
        "qm_index": int(iqm),
        "qm_symbol": str(frame.qm_symbols[iqm]),
        "mm_h_symbol": str(frame.mm_symbols[hi]),
        "mm_o_symbol": str(frame.mm_symbols[oi]),
        "atom_uv_H": [float(uh), float(vh)],
        "atom_uv_O": [float(uo), float(vo)],
        "atom_uv_QM": [float(uq), float(vq)],
        "fix_scf_r0": float(fix0) if np.isfinite(fix0) else None,
        "fix_scf_r1": float(fix1) if np.isfinite(fix1) else None,
        "fix_axis_r_ang": np.asarray(r_e, dtype=np.float64).tolist(),
        "fix_axis_fix_s": np.asarray(fix_axis, dtype=np.float64).tolist(),
        "fix_alpha": float(alpha),
        "rep_center": str(rep_center),
        "rep_oh_frac": float(rep_oh_frac),
        "plane_step_ang": float(plane_step),
        "plane_margin_ang": float(plane_margin),
        "r_cut_mm": None if r_cut_mm is None else float(r_cut_mm),
        "dist_h_qm_3d_ang": dist_h_qm_3d,
        "dist_o_h_3d_ang": dist_o_h_3d,
        "dist_h_qm_plane_ang": dist_h_qm_plane,
        "line_plane_max_abs_diff": float(max_plane_err),
        "n_mm_h_pairs_avg": int(n_mm_h_pairs),
        "integrity_max_abs_dm_scf_minus_emb": dm_scf_emb_diff,
        "emb_nao": int(mf_emb.mol.nao),
        "cluster_nao": int(mf_cl.mol.nao),
        "integrity_max_abs_rho_scf_minus_emb0": max_se,
        "integrity_max_abs_rho_cl_minus_emb0": max_ce,
        "axis_r0_rho_emb": float(rho_e[0]) if rho_e.size else float("nan"),
        "axis_r0_rho_cl_qm": float(rho_c[0]) if rho_c.size else float("nan"),
        "axis_r0_rho_scf": float(rho_s[0]) if rho_s.size else float("nan"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        u_axis_ang=u_axis,
        v_axis_ang=v_axis,
        uu_ang=uu,
        vv_ang=vv,
        rho_emb0=rho_emb0,
        rho_cluster_qm=rho_cluster_qm,
        rho_emb_scf=rho_emb_scf,
        atom_uv=np.asarray([[uh, vh, 0.0], [uo, vo, 1.0], [uq, vq, 2.0]], dtype=np.float64),
        atom_labels=np.array(["MM_H", "MM_O", "QM_near"], dtype=object),
        dist_h_qm_3d_ang=np.float64(dist_h_qm_3d),
        dist_o_h_3d_ang=np.float64(dist_o_h_3d),
        dist_h_qm_plane_ang=np.float64(dist_h_qm_plane),
        meta_json=np.array(json.dumps(meta)),
    )
    print(f"  wrote plane npz → {out_path.resolve()}")


def _print_mm_h_indices_in_pairs(
    pairs: list,
    *,
    pair_max_ang: float,
    selected_hi: int | None = None,
    mf_emb=None,
    mf_cl=None,
    dm_emb: np.ndarray | None = None,
    dm_cl_qm: np.ndarray | None = None,
    dm_scf: np.ndarray | None = None,
    line_dr: float = 0.1,
) -> None:
    """
    MM-H indices that enter the axis average (same pool as「MM端为 H」).

    One H can link to several QM atoms within ``pair_max_ang``; list is sorted by
    shortest H–QM distance (best candidates for ``--plane-mm-h-index``).
    """
    by_h: dict[int, list[tuple[float, int, str]]] = {}
    for p in pairs:
        if p.mm_symbol.strip()[0].upper() != "H":
            continue
        by_h.setdefault(int(p.jm), []).append(
            (float(p.dist_ang), int(p.iq), str(p.qm_symbol).strip())
        )
    n_links = sum(len(v) for v in by_h.values())
    if not by_h:
        print(f"  [pair_max={pair_max_ang:g} Å 内 MM-H] 无 H–QM 配对")
        return

    rows: list[tuple[float, int, int, str, int, list[tuple[float, int, str]]]] = []
    for hi, hits in by_h.items():
        hits_sorted = sorted(hits, key=lambda x: x[0])
        d_min, iq_min, sq_min = hits_sorted[0]
        rows.append((d_min, hi, iq_min, sq_min, len(hits_sorted), hits_sorted))
    rows.sort(key=lambda x: x[0])

    print(
        f"\n  [pair_max={pair_max_ang:g} Å 内 MM-H] 独立 H={len(rows)}  "
        f"连线条数={n_links}（「MM端为 H」平均 n={n_links}）"
    )
    for d_min, hi, iq_min, sq_min, n_link, hits_sorted in rows:
        qm_bits = ", ".join(f"QM[{iq}]={sym}@{d:.2f}Å" for d, iq, sym in hits_sorted)
        fix_bits = ""
        if (
            dm_scf is not None
            and mf_emb is not None
            and mf_cl is not None
            and dm_emb is not None
            and dm_cl_qm is not None
        ):
            fix1 = _fix_scf_for_mm_h_at_r(
                hi, pairs, mf_emb, mf_cl, dm_emb, dm_cl_qm, dm_scf,
                r_ang=1.0, dr=float(line_dr),
            )
            if np.isfinite(fix1):
                fix_bits = f"  fix@1Å={fix1:.4f}"
            if selected_hi is not None and int(hi) == int(selected_hi):
                fix0 = _fix_scf_for_mm_h_at_r(
                    hi, pairs, mf_emb, mf_cl, dm_emb, dm_cl_qm, dm_scf,
                    r_ang=0.0, dr=float(line_dr),
                )
                f0 = f"{fix0:.4f}" if np.isfinite(fix0) else "n/a"
                f1 = f"{fix1:.4f}" if np.isfinite(fix1) else "n/a"
                fix_bits = f"  (摘要 fix@0={f0} fix@1={f1}；完整见下方 0.2Å 表)"
        mark = "  ← 所选" if selected_hi is not None and int(hi) == int(selected_hi) else ""
        print(
            f"    MM-H[{hi:2d}]  d_min={d_min:.3f} Å → QM[{iq_min}]={sq_min}  ({qm_bits})"
            f"{fix_bits}{mark}"
        )
    if selected_hi is not None and int(selected_hi) not in by_h:
        print(f"  ※ 所选 MM-H[{int(selected_hi)}] 不在 pair_max 配对内，无 fix")
    best_d, best_hi, best_iq, best_sym, _, _ = rows[0]
    print(
        f"  → 近壳 spillover 建议 --plane-mm-h-index {best_hi}  "
        f"(d_min={best_d:.3f} Å → QM[{best_iq}]={best_sym})"
    )


def _print_axis_rho_table(
    title: str,
    r_grid: np.ndarray,
    rho_emb: np.ndarray,
    rho_cl: np.ndarray,
    rho_scf: np.ndarray,
    *,
    r_print_max: float,
    with_fix: bool = False,
) -> None:
    print(f"\n  [{title}]")
    if with_fix:
        print("  fix_s = 1−Δ_scf/Δ_emb = (EmbR−Emb0)/(Cluster−Emb0)")
        print(
            "  r(Å)   ρ_emb        ρ_cl^QM      ρ_scf        Δ_emb        Δ_scf        fix_s"
        )
    else:
        print("  r(Å)   ρ_emb        ρ_cl^QM      ρ_scf        Δ_emb        Δ_scf")
    r_grid = np.asarray(r_grid, dtype=np.float64)
    for ir, rad in enumerate(r_grid):
        if float(rad) > float(r_print_max) + 1e-9:
            break
        if ir % 2 == 1 and ir < r_grid.size - 1:
            continue
        e0 = float(rho_emb[ir])
        c0 = float(rho_cl[ir])
        s0 = float(rho_scf[ir])
        de = e0 - c0
        ds = s0 - c0
        line = (
            f"  {rad:4.1f}  {fmt_rho(e0):>12}  {fmt_rho(c0):>12}  {fmt_rho(s0):>12}  "
            f"{fmt_rho(de):>12}  {fmt_rho(ds):>12}"
        )
        if with_fix:
            line += f"  {_fix_ratio_str(de, ds):>12}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="QM–MM axis ρ: emb vs Cluster 全 DFT vs perturbation + dipole",
        allow_abbrev=False,
    )
    ap.add_argument("--coo", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--peratom", type=Path, required=True, help="theta/pert_peratom.npz")
    ap.add_argument("--pert-cache", type=Path, default=None, help="theta/pert.npz for Σ A_j k_j")
    ap.add_argument("--residue", type=str, default=None)
    ap.add_argument("--n-qm", type=int, default=None)
    ap.add_argument("--r-cut-mm", type=float, default=None)
    ap.add_argument("--pair-max", type=float, default=2.0, help="max QM–MM distance (Å) for axis pairs")
    ap.add_argument("--line-max", type=float, default=2.0, help="r grid 0..line-max (Å) along axis")
    ap.add_argument("--line-dr", type=float, default=0.1)
    ap.add_argument(
        "--line-r-min",
        type=float,
        default=0.0,
        help="表格从 r≥此值 [Å] 起打印；默认 0（含 MM 旁数据）",
    )
    ap.add_argument("--fix-alpha", type=float, default=None, help="Gauss α [Bohr^-2]; default from peratom meta")
    ap.add_argument(
        "--cone-theta1-deg",
        type=float,
        default=180.0,
        help="H→QM cone: full Gaussian inside this angle [deg]; 180 = isotropic (default)",
    )
    ap.add_argument(
        "--cone-theta2-deg",
        type=float,
        default=180.0,
        help="H→QM cone: zero repulsion outside this angle [deg]; smooth taper θ1→θ2",
    )
    ap.add_argument("--probe-a-o", type=float, default=None)
    ap.add_argument("--probe-a-h", type=float, default=None)
    ap.add_argument("--scale-a-o", type=float, default=None, help="multiply fitted A_O by this factor")
    ap.add_argument("--scale-a-h", type=float, default=None, help="multiply fitted A_H by this factor (e.g. 1.1 or 2)")
    ap.add_argument(
        "--rep-center",
        type=str,
        default="on_nucleus",
        choices=REP_CENTER_PLACEMENTS,
        help="高斯排斥峰位置：on_nucleus=MM H/O 核；oh_bond=对应 O–H 键上",
    )
    ap.add_argument(
        "--rep-oh-frac",
        type=float,
        default=0.5,
        help="oh_bond 时沿 O→H 的分数位置：0=O 核，0.5=键中点，1=H 核",
    )
    ap.add_argument("--allow-o-repulsion", action="store_true")
    ap.add_argument("--method", type=str, default="b3lyp", help="SCF: hf or b3lyp")
    ap.add_argument("--basis", type=str, default="6-31g*", help="basis set, e.g. 6-31+G*")
    ap.add_argument("--d3bj", action="store_true")
    ap.add_argument("--no-d3bj", action="store_true", help="disable D3BJ")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--verbose-scf", type=int, default=0)
    ap.add_argument(
        "--no-embed-scf",
        action="store_true",
        help="跳过 EmbR 全自洽嵌入 SCF（默认会跑：点电荷+V_rep 进 hcore）",
    )
    ap.add_argument(
        "--embed-scf",
        action="store_true",
        help="显式开启 EmbR SCF（默认已开启；勿与 --embed-scf-conv-tol 缩写混淆）",
    )
    ap.add_argument("--embed-scf-conv-tol", type=float, default=1e-4)
    ap.add_argument(
        "--plane-npz",
        type=Path,
        default=None,
        help="可选：用**已算好的** mf/dm 在 H–O–QM 平面上求 ρ，写 npz 供 plot_rho_plane（不额外 SCF）",
    )
    ap.add_argument(
        "--plane-mm-h-index",
        type=int,
        default=None,
        help="写 --plane-npz 时指定 MM 列表中的 H 序号（filter 之后）",
    )
    ap.add_argument("--plane-step", type=float, default=0.08, help="平面格点步长 [Å]")
    ap.add_argument("--plane-margin", type=float, default=1.0, help="平面格点边距 [Å]")
    ap.add_argument("--plane-min-half", type=float, default=2.0, help="平面格点最小半宽 [Å]")
    ap.add_argument(
        "--scf-cache",
        type=Path,
        default=None,
        help="SCF 缓存 npz：文件存在且 frame/α/r_cut_mm/幅度等指纹一致则跳过 DFT；否则重算并覆盖",
    )
    ap.add_argument(
        "--force-scf",
        action="store_true",
        help="忽略 --scf-cache，强制重算全部 SCF",
    )
    args = ap.parse_args()
    if args.no_embed_scf and args.embed_scf:
        raise ValueError("cannot use both --embed-scf and --no-embed-scf")
    embed_scf = not bool(args.no_embed_scf)

    if args.coo is None and args.manifest is None:
        raise ValueError("need --coo or --manifest")

    labels = load_pert_peratom_labels(args.peratom)
    meta_lab = labels["meta"]
    from embr_envelope import MmhEnvelopeConfig

    envelope_cfg = MmhEnvelopeConfig.from_meta(meta_lab)
    alpha_cli = args.fix_alpha
    alpha = float(alpha_cli if alpha_cli is not None else meta_lab.get("fix_alpha", 0.6))
    lab_alpha = meta_lab.get("fix_alpha")
    if lab_alpha is not None and alpha_cli is not None and abs(float(lab_alpha) - alpha) > 1e-9:
        print(f"  警告: CLI α={alpha:g} ≠ peratom meta α={float(lab_alpha):g}")
    h_only = resolve_h_only(allow_o_repulsion=bool(args.allow_o_repulsion), meta=meta_lab)
    cone = ConeRepAng(theta1_deg=float(args.cone_theta1_deg), theta2_deg=float(args.cone_theta2_deg))

    e0_file: Path | None = None
    e_int_bg = float("nan")
    if args.coo is not None:
        coo_path_res = Path(args.coo)
        residue = str(args.residue or "gly").lower()
        fi = int(args.frame)
    else:
        fi = int(args.frame)
        task = _resolve_manifest_task(Path(args.manifest), fi)
        coo_path_res = Path(task["coo_path"])
        residue = str(task["residue"])
        e0_file = Path(task["e0_file"])
        e_int_bg = float(task["e_int_bg_kcal"])
        if args.n_qm is None:
            args.n_qm = int(task["n_qm"])

    n_lab = int(labels["e0"].shape[0])
    if fi < 0 or fi >= n_lab:
        raise ValueError(f"--frame {fi} out of range for peratom labels [0, {n_lab})")

    spec = get_residue(residue) if args.n_qm is None else None
    n_qm = int(args.n_qm if args.n_qm is not None else spec.n_qm)
    frame = load_scf_frame(coo_path_res, n_qm=n_qm)
    frame = filter_mm_by_distance(frame, r_cut_ang=args.r_cut_mm)

    d3bj = bool(args.d3bj)
    if args.no_d3bj:
        d3bj = False
    cfg = scf_embed_config_from_cli(
        method=str(args.method),
        basis=str(args.basis),
        use_d3bj=d3bj,
        num_threads=int(args.threads),
        verbose=int(args.verbose_scf),
    )
    d3bj = bool(cfg.use_d3bj)

    amp_mm, el_slice = _amp_for_frame(
        labels,
        fi,
        h_only=h_only,
        probe_a_o=args.probe_a_o,
        probe_a_h=args.probe_a_h,
        scale_a_o=args.scale_a_o,
        scale_a_h=args.scale_a_h,
    )

    pol = "h_only" if h_only else "o_and_h"
    print(f"[line_rho_pert_cluster] coo={coo_path_res}  residue={residue}  frame={fi}  policy={pol}")
    if args.scale_a_h is not None or args.scale_a_o is not None:
        parts = []
        if args.scale_a_h is not None:
            parts.append(f"A_H×{float(args.scale_a_h):g}")
        if args.scale_a_o is not None:
            parts.append(f"A_O×{float(args.scale_a_o):g}")
        print(f"  幅度缩放: {'  '.join(parts)}  (Tr(P0V)/ΣAk 将随 H 缩放；E0 标签不变)")
    k_mm_report = None
    if "kernel_mm_flat" in labels:
        ptr = labels["mm_ptr"]
        s, e = int(ptr[fi]), int(ptr[fi + 1])
        k_mm_report = labels["kernel_mm_flat"][s:e]
    _print_mm_amp_dist_table(frame, amp_mm, kernel_mm=k_mm_report, frame_index=fi)
    print(f"  配对: QM ↔ {args.pair_max}Å 内 MM O/H；轴线 MM(r=0)→QM")
    print(
        f"  pair_max={args.pair_max} Å  line_dr={args.line_dr} Å  "
        f"envelope={envelope_cfg.kind}  width={envelope_cfg.to_meta()}  "
        f"lnC={envelope_cfg.lnC_by_element}  "
        f"cone=({cone.theta1_deg:g}°,{cone.theta2_deg:g}°) isotropic={cone.is_isotropic()}  "
        f"xc={cfg.xc}/{cfg.basis}"
    )
    print(
        "  spillover 对照: ρ_emb / ρ_Cluster^QM / ρ_gas（均为 QM 电子）；"
        "ρ_Cluster^tot 仅列物理总密度（含水，近核可很大）"
    )
    rep_center = str(args.rep_center)
    rep_oh_frac = float(args.rep_oh_frac)
    rep_centers = build_repulsion_center_coords_ang(
        frame, rep_center, oh_frac=rep_oh_frac
    )
    print(f"  排斥峰: {placement_label_zh(rep_center, rep_oh_frac)}  （点电荷仍在 MM 核）")

    pert_cache = None
    pert_cache_meta: dict = {}
    if args.pert_cache is not None:
        pert_cache = load_pert_cache(args.pert_cache)
        pert_cache_meta = pert_cache.get("meta") or {}
        cache_rep = pert_cache_meta.get("rep_center")
        if cache_rep is not None and str(cache_rep) != rep_center:
            print(
                f"  警告: pert-cache rep_center={cache_rep!r} ≠ CLI --rep-center={rep_center!r}；"
                "ΣAk 与 Tr(P0V) 会不一致，请用 precompute_pert_ohbond 重算 k_j"
            )
        elif cache_rep is None and rep_center != "on_nucleus":
            print(
                "  警告: pert-cache 无 rep_center 元数据（旧 pert.npz），"
                "oh_bond 下 ΣAk 按核预计算；请 precompute_pert_ohbond + fit_pert_peratom"
            )
        cache_frac = pert_cache_meta.get("rep_oh_frac")
        if cache_frac is not None and abs(float(cache_frac) - rep_oh_frac) > 1e-9:
            print(
                f"  警告: pert-cache rep_oh_frac={float(cache_frac):g} "
                f"≠ CLI --rep-oh-frac={rep_oh_frac:g}"
            )
        cache_alpha = pert_cache_meta.get("fix_alpha")
        if cache_alpha is not None and abs(float(cache_alpha) - alpha) > 1e-9:
            print(f"  警告: pert.npz α={float(cache_alpha):g} ≠ CLI α={alpha:g}")
        scf_cfg_meta = pert_cache_meta.get("scf_config") or {}
        for key, val in (("xc", cfg.xc), ("basis", cfg.basis)):
            cache_val = scf_cfg_meta.get(key)
            if cache_val is not None and str(cache_val) != str(val):
                print(
                    f"  警告: pert.npz scf_config.{key}={cache_val!r} ≠ CLI {val!r}；"
                    "Emb0 kernel / Cluster 密度不自洽，请用匹配 --method/--basis 重算 precompute"
                )
        for key, val in (
            ("cone_theta1_deg", cone.theta1_deg),
            ("cone_theta2_deg", cone.theta2_deg),
        ):
            cache_val = pert_cache_meta.get(key)
            if cache_val is not None and abs(float(cache_val) - float(val)) > 1e-9:
                print(f"  警告: pert.npz {key}={float(cache_val):g} ≠ CLI {float(val):g}；请重算 precompute")
    elif rep_center != "on_nucleus":
        print(
            "  注意: 未传 --pert-cache；ΣAk 不可用。"
            "oh_bond 请 precompute_pert_ohbond + fit_pert_peratom 后传入匹配的 pert/peratom"
        )

    from embr_theta.scf_line_cache import (
        cache_allows_partial_vrep_reuse,
        cache_expect_without_amp_scale,
        geom_hash,
        load_line_scf_cache,
        load_line_scf_cache_reference,
        save_line_scf_cache,
    )

    n_mm = len(frame.mm_symbols)
    n_qm = len(frame.qm_symbols)
    ghash = geom_hash(frame)
    plane_tag = "ON(post-SCF)" if args.plane_npz is not None else "OFF"
    rcut_s = "all" if args.r_cut_mm is None else f"{float(args.r_cut_mm):g}Å"
    print(
        f"  [SCF 指纹] plane_npz={plane_tag}  frame={fi}  natm={n_qm + n_mm}  "
        f"n_qm={n_qm}  n_mm={n_mm}  r_cut_mm={rcut_s}  α={alpha:g}  "
        f"conv_tol={cfg.conv_tol:g}  threads={cfg.num_threads}  geom={ghash[:12]}"
    )

    cfg_scf = scf_embed_config_from_cli(
        method=str(args.method),
        basis=str(args.basis),
        use_d3bj=d3bj,
        num_threads=int(args.threads),
        verbose=int(args.verbose_scf),
        conv_tol=float(args.embed_scf_conv_tol),
    )
    cache_expect = {
        "frame": int(fi),
        "coo_path": str(coo_path_res.resolve()),
        "residue": str(residue),
        "n_qm": int(n_qm),
        "r_cut_mm": None if args.r_cut_mm is None else float(args.r_cut_mm),
        "scale_a_h": None if args.scale_a_h is None else float(args.scale_a_h),
        "scale_a_o": None if args.scale_a_o is None else float(args.scale_a_o),
        "fix_alpha": float(alpha),
        "cone_theta1_deg": float(cone.theta1_deg),
        "cone_theta2_deg": float(cone.theta2_deg),
        "rep_center": str(rep_center),
        "rep_oh_frac": float(rep_oh_frac),
        "xc": str(cfg.xc),
        "basis": str(cfg.basis),
        "d3bj": bool(d3bj),
        "embed_scf": bool(embed_scf),
        "peratom": str(Path(args.peratom).resolve()),
    }

    cache_expect_base = cache_expect_without_amp_scale(cache_expect)

    scf_ready = False
    de_lin_override = None
    de_rho1_override = None
    if args.scf_cache is not None and Path(args.scf_cache).is_file() and not bool(args.force_scf):
        cache_path = Path(args.scf_cache)
        expect_full = _cache_expect_for_file(cache_expect, cache_path)
        expect_base = _cache_expect_for_file(cache_expect_base, cache_path)
        load_kw = dict(
            path=cache_path,
            cfg=cfg,
            cfg_scf=cfg_scf,
            amp_mm=amp_mm,
            alpha_bohr2=float(alpha),
            rep_centers_ang=rep_centers,
            embed_scf=embed_scf,
            expect_meta=expect_full,
        )
        if _scf_cache_load_supports_cone(load_line_scf_cache):
            load_kw["cone"] = cone
        else:
            print(
                "  警告: theta/scf_line_cache.py 仍是旧版（无 cone 参数）；"
                "请重传该文件。EmbR stub 暂按各向同性重建。"
            )
        try:
            loaded_cache = load_line_scf_cache(**load_kw)
        except ValueError as exc_full:
            try:
                cache_ref = load_line_scf_cache_reference(
                    cache_path,
                    cfg=cfg,
                    expect_meta=expect_base,
                )
            except ValueError:
                print(f"  [scf-cache] 无法加载（{exc_full}）；将全量重算", flush=True)
            else:
                if not cache_allows_partial_vrep_reuse(cache_ref.meta, expect_full):
                    raise exc_full from None
                sc_h = cache_ref.meta.get("scale_a_h")
                sc_o = cache_ref.meta.get("scale_a_o")
                cu_h = cache_expect.get("scale_a_h")
                cu_o = cache_expect.get("scale_a_o")
                print(
                    f"  [scf-cache] 仅换 A_H/peratom：复用 DFT Cluster/Emb0；"
                    f"重算 pert+EmbR  scale ({sc_h}, {sc_o})→({cu_h}, {cu_o})",
                    flush=True,
                )
                frame = cache_ref.frame
                mf_emb = cache_ref.mf_emb
                mf_cl = cache_ref.mf_cl
                mf_gas = cache_ref.mf_gas
                dm_emb = cache_ref.dm_emb
                dm_cl_tot = cache_ref.dm_cl_tot
                dm_cl_qm = cache_ref.dm_cl_qm
                dm_gas = cache_ref.dm_gas
                (
                    mf_emb,
                    dm_emb,
                    mf_scf,
                    dm_scf,
                    dm_pert,
                    pert_res,
                    de_kernel,
                    de_lin_override,
                    de_rho1_override,
                ) = _recompute_vrep_dependent(
                    frame,
                    cfg,
                    cfg_scf,
                    mf_emb,
                    dm_emb,
                    amp_mm,
                    alpha=alpha,
                    envelope_cfg=envelope_cfg,
                    cone=cone,
                    rep_centers=rep_centers,
                    rep_center=rep_center,
                    pert_cache=pert_cache,
                    fi=fi,
                    embed_scf=embed_scf,
                )
                print(
                    f"  Cluster (cached): converged={mf_cl.converged}  "
                    f"E_tot={hartree_to_kcal(float(mf_cl.e_tot)):.3f} kcal/mol  "
                    f"nao={int(mf_cl.mol.nao)}  natm={int(mf_cl.mol.natm)}"
                )
                save_line_scf_cache(
                    cache_path,
                    frame=frame,
                    dm_emb=dm_emb,
                    dm_cl_tot=dm_cl_tot,
                    dm_cl_qm=dm_cl_qm,
                    dm_pert=dm_pert,
                    dm_gas=dm_gas,
                    dm_scf=dm_scf,
                    mf_emb=mf_emb,
                    mf_cl=mf_cl,
                    mf_gas=mf_gas,
                    mf_scf=mf_scf,
                    de_kernel_kcal=de_kernel,
                    de_lin_kcal=float(de_lin_override),
                    de_rho1_kcal=float(de_rho1_override),
                    cache_meta=cache_expect,
                )
                scf_ready = True
        else:
            frame = loaded_cache.frame
            mf_emb = loaded_cache.mf_emb
            mf_cl = loaded_cache.mf_cl
            mf_gas = loaded_cache.mf_gas
            mf_scf = loaded_cache.mf_scf
            dm_emb = loaded_cache.dm_emb
            dm_cl_tot = loaded_cache.dm_cl_tot
            dm_cl_qm = loaded_cache.dm_cl_qm
            dm_pert = loaded_cache.dm_pert
            dm_gas = loaded_cache.dm_gas
            dm_scf = loaded_cache.dm_scf
            pert_res = loaded_cache.pert_res
            de_kernel = float(loaded_cache.de_kernel_kcal)
            de_lin_override = float(loaded_cache.de_lin_kcal)
            de_rho1_override = float(loaded_cache.de_rho1_kcal)
            print(
                f"  Cluster (cached): converged={mf_cl.converged}  "
                f"E_tot={hartree_to_kcal(float(mf_cl.e_tot)):.3f} kcal/mol  "
                f"nao={int(mf_cl.mol.nao)}  natm={int(mf_cl.mol.natm)}"
            )
            scf_ready = True

    if not scf_ready:
        if args.scf_cache is not None and bool(args.force_scf):
            print(f"  [scf-cache] --force-scf：忽略 {args.scf_cache}")
        elif args.scf_cache is not None:
            print(f"  [scf-cache] 无缓存，算完后写入 {args.scf_cache}")

        print("  running Emb0 SCF ...")
        sys.stdout.flush()
        mf_emb = run_emb0_mf(frame, cfg)
        dm_emb = mf_emb.make_rdm1()

        print("  running Cluster 全 DFT supermol SCF ...")
        if int(args.verbose_scf) <= 0:
            print("  提示: 长时间无输出时加 --verbose-scf 4 看迭代；共享 CPU 时试 --threads 1")
        sys.stdout.flush()
        mf_cl = run_cluster_supermol_mf(frame, cfg)
        dm_cl_tot = mf_cl.make_rdm1()
        dm_cl_qm = cluster_qm_dm(mf_cl, frame)
        print(
            f"  Cluster: converged={mf_cl.converged}  "
            f"E_tot={hartree_to_kcal(float(mf_cl.e_tot)):.3f} kcal/mol  "
            f"nao={int(mf_cl.mol.nao)}  natm={int(mf_cl.mol.natm)}  "
            f"(轴线参考=Cluster^QM 块，非总 DM)"
        )

        kernel_mm = None
        if pert_cache is not None:
            kernel_mm, _ = mm_frame_slice(pert_cache, fi)

        print("  running CP-KS density perturbation ...")
        pert_res, mf_emb = run_density_pert_frame(
            frame,
            cfg,
            amp_mm,
            alpha_bohr2=alpha,
            envelope_cfg=envelope_cfg,
            mf=mf_emb,
            kernel_mm=kernel_mm,
            cone=cone,
            rdf_max=0.0,
        )
        dm_pert = _dm_pert_from_result(
            pert_res,
            mf_emb,
            dm_emb,
            frame,
            amp_mm,
            alpha,
            rep_centers,
            envelope_cfg=envelope_cfg,
            rep_on_nucleus=(rep_center == "on_nucleus"),
            cone=cone,
        )
        de_kernel = float(pert_res.de_kernel_kcal)
        if kernel_mm is not None and not np.isfinite(de_kernel):
            de_kernel = float(delta_e_peratom_kcal(kernel_mm, amp_mm))

        de_lin_override = None
        de_rho1_override = None
        if rep_center != "on_nucleus":
            amps = amp_mm_for_o_h_sites(frame, amp_mm)
            h1 = gaussian_repulsion_ao(
                mf_emb,
                rep_centers,
                amps,
                alpha_bohr2=float(alpha),
                envelope_cfg=envelope_cfg,
                mm_symbols=frame.mm_symbols,
                **_cone_rep_kw(frame, cone),
            )
            de_lin_override = float(hartree_to_kcal(delta_e_tr_ph0(mf_emb, h1)))
            de_rho1_override = float(hartree_to_kcal(delta_e_tr_ph(mf_emb, h1, dm_pert)))
        else:
            de_lin_override = float(pert_res.de1_kcal)
            de_rho1_override = float(getattr(pert_res, "de_rho1_kcal", float("nan")))

        print("  running gas-phase QM (dipole ref) ...")
        mf_gas = run_gas_mf(frame, cfg)
        dm_gas = mf_gas.make_rdm1()

        mf_scf = None
        dm_scf = None
        if embed_scf:
            print(f"  running EmbR SCF (点电荷+V_rep 全自洽, conv_tol={cfg_scf.conv_tol}) ...")
            mf_scf = _run_embed_theta_mf(
                frame,
                cfg_scf,
                amp_mm,
                alpha_bohr2=alpha,
                envelope_cfg=envelope_cfg,
                rep_centers_ang=rep_centers,
                cone=cone,
            )
            dm_scf = mf_scf.make_rdm1()

        if args.scf_cache is not None:
            save_line_scf_cache(
                Path(args.scf_cache),
                frame=frame,
                dm_emb=dm_emb,
                dm_cl_tot=dm_cl_tot,
                dm_cl_qm=dm_cl_qm,
                dm_pert=dm_pert,
                dm_gas=dm_gas,
                dm_scf=dm_scf,
                mf_emb=mf_emb,
                mf_cl=mf_cl,
                mf_gas=mf_gas,
                mf_scf=mf_scf,
                de_kernel_kcal=de_kernel,
                de_lin_kcal=float(de_lin_override),
                de_rho1_kcal=float(de_rho1_override),
                cache_meta=cache_expect,
            )

    e0_lab = float(labels["e0"][fi])
    _print_energy_block(
        e0=e0_lab,
        e_int_bg=e_int_bg,
        e0_file=e0_file,
        pert_res=pert_res,
        de_kernel=de_kernel,
        mf_emb=mf_emb,
        mf_cl=mf_cl,
        mf_scf=mf_scf,
        embed_scf=embed_scf,
        de_lin_override=de_lin_override,
        de_rho1_override=de_rho1_override,
    )

    _print_dipole_block(
        frame,
        mu_gas=_dipole_tot_at_com_debye(mf_gas, dm_gas, frame),
        mu_cluster=_dipole_tot_at_com_debye(mf_cl, dm_cl_tot, frame),
        mu_emb=_dipole_tot_at_com_debye(mf_emb, dm_emb, frame),
        mu_pert=_dipole_tot_at_com_debye(mf_emb, dm_pert, frame),
        mu_scf=None if dm_scf is None else _dipole_tot_at_com_debye(mf_scf, dm_scf, frame),
    )

    r, emb_g, cl_qm_g, cl_tot_g, gas_g, pert_g, pairs = _compute_axis_profiles(
        frame,
        mf_emb,
        mf_cl,
        mf_gas,
        dm_emb,
        dm_cl_qm,
        dm_cl_tot,
        dm_gas,
        dm_pert,
        pair_max_ang=float(args.pair_max),
        dr=float(args.line_dr),
        r_line_max=float(args.line_max),
    )

    scf_g: dict[str, np.ndarray] | None = None
    if embed_scf and dm_scf is not None:
        scf_g = _axis_groups_for_dm(pairs, mf_emb, dm_scf, r, dr=float(args.line_dr))

    n_o = sum(1 for p in pairs if p.mm_symbol.strip()[0].upper() == "O")
    n_h = sum(1 for p in pairs if p.mm_symbol.strip()[0].upper() == "H")
    print(f"\n  QM–MM 连线对: 全部={len(pairs)}  (MM-O={n_o}, MM-H={n_h})")
    if not pairs:
        print("  ※ 无配对: 增大 --pair-max")
    else:
        _print_mm_h_indices_in_pairs(
            pairs,
            pair_max_ang=float(args.pair_max),
            selected_hi=args.plane_mm_h_index,
            mf_emb=mf_emb,
            mf_cl=mf_cl,
            dm_emb=dm_emb,
            dm_cl_qm=dm_cl_qm,
            dm_scf=dm_scf,
            line_dr=float(args.line_dr),
        )
        if args.plane_mm_h_index is not None and embed_scf and dm_scf is not None:
            _print_selected_mm_h_axis_fix_table(
                int(args.plane_mm_h_index),
                pairs,
                mf_emb,
                mf_cl,
                dm_emb,
                dm_cl_qm,
                dm_scf,
                dr=float(args.line_dr),
                r_print_max=min(float(args.line_max), 2.0),
            )

    rmin = float(args.line_r_min)
    _print_line_table_pert(
        r, emb_g["all"], cl_qm_g["all"], cl_tot_g["all"], gas_g["all"], pert_g["all"],
        title="全部 QM–MM 连线", n_pairs=len(pairs),
        scf=None if scf_g is None else scf_g["all"], r_min_print=rmin,
    )
    if n_o:
        _print_line_table_pert(
            r, emb_g["mm_O"], cl_qm_g["mm_O"], cl_tot_g["mm_O"], gas_g["mm_O"], pert_g["mm_O"],
            title="MM 端为 O", n_pairs=n_o,
            scf=None if scf_g is None else scf_g["mm_O"], r_min_print=rmin,
        )
        if scf_g is not None:
            _print_axis_rho_table(
                f"【汇报】MM-O 端 QM–MM 轴线 ρ 平均 n={n_o}（0.2Å 步长；轴 MM(r=0)→QM）",
                r,
                emb_g["mm_O"],
                cl_qm_g["mm_O"],
                scf_g["mm_O"],
                r_print_max=min(float(args.line_max), 2.0),
                with_fix=True,
            )
    if n_h:
        _print_line_table_pert(
            r, emb_g["mm_H"], cl_qm_g["mm_H"], cl_tot_g["mm_H"], gas_g["mm_H"], pert_g["mm_H"],
            title="MM 端为 H", n_pairs=n_h,
            scf=None if scf_g is None else scf_g["mm_H"], r_min_print=rmin,
        )

    print("\n[读表] 摘要 r≈0 与 r≈1.0 Å；Δ_emb>0 表示 Emb 比 Cluster^QM 多溢")
    for r_pick, r_label in ((0.0, "r≈0"), (1.0, "r≈1")):
        irx = int(np.argmin(np.abs(r - r_pick)))
        for key, label, n in (("all", "全部", len(pairs)), ("mm_O", "MM-O", n_o), ("mm_H", "MM-H", n_h)):
            if n == 0:
                continue
            de = float(emb_g[key][irx] - cl_qm_g[key][irx])
            dp = float(pert_g[key][irx] - cl_qm_g[key][irx])
            se = float(emb_g[key][irx] - gas_g[key][irx])
            fix_p = _fix_ratio_str(de, dp)
            line = (
                f"\n[摘要 {r_label} Å, {label}]  spill_emb={fmt_rho(se)}  "
                f"Δ_emb={fmt_rho(de)}  Δ_pert={fmt_rho(dp)}  fix_pert={fix_p}"
            )
            if scf_g is not None:
                ds = float(scf_g[key][irx] - cl_qm_g[key][irx])
                fix_s = _fix_ratio_str(de, ds)
                line += f"  Δ_scf={fmt_rho(ds)}  fix_scf={fix_s}"
            print(line)

    if args.plane_npz is not None:
        if args.plane_mm_h_index is None:
            raise ValueError("--plane-npz requires --plane-mm-h-index")
        if not embed_scf or dm_scf is None:
            raise ValueError("--plane-npz requires EmbR SCF (omit --no-embed-scf)")
        _write_plane_npz_for_plot(
            Path(args.plane_npz),
            frame=frame,
            frame_index=fi,
            mm_h_index=int(args.plane_mm_h_index),
            mf_emb=mf_emb,
            mf_cl=mf_cl,
            dm_emb=dm_emb,
            dm_cl_qm=dm_cl_qm,
            dm_scf=dm_scf,
            coo_path=coo_path_res,
            residue=residue,
            alpha=alpha,
            rep_center=rep_center,
            rep_oh_frac=rep_oh_frac,
            r_cut_mm=args.r_cut_mm,
            plane_margin=float(args.plane_margin),
            plane_step=float(args.plane_step),
            plane_min_half=float(args.plane_min_half),
            line_dr=float(args.line_dr),
            line_r_max=float(args.line_max),
            pairs=pairs,
            r_line=r,
            emb_avg_mm_h=emb_g["mm_H"] if n_h else None,
            cl_avg_mm_h=cl_qm_g["mm_H"] if n_h else None,
            scf_avg_mm_h=scf_g["mm_H"] if scf_g is not None and n_h else None,
            n_mm_h_pairs=n_h,
            pair_max_ang=float(args.pair_max),
        )


if __name__ == "__main__":
    main()
