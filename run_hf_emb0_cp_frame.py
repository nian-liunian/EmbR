"""
One frame: Emb0 + counterpoise full_QM reference + optional axis ρ(r).

Definitions (kcal/mol; negative = binding):
  E_int^Emb0  = E(QM + MM point charges) − E(QM monomer)
  E_int^CP4   = E(A+ghostB) + E(ghostA+B) − E(A) − E(B)   [legacy NPZ key e_int_cp]
  E_int^raw   = E(QM+MM cluster) − E(A) − E(B)
  E_full_QM   = E_int^raw − E_int^CP4  (counterpoise-corrected; NPZ key bsse_kcal)
  E0 ≡ ΔE     = E_full_QM − E_int^Emb0  (paper: ΔE = E_full_QM − E_Emb0)

Legacy NPZ key names are unchanged. Axis ρ(r): MM nucleus → nearest QM (r=0 at MM).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scf_embed_cluster import cluster_qm_dm
from scf_embed_io import filter_mm_by_distance, load_scf_frame
from scf_embed_perturb import compute_qm_mm_line_rho_emb_cluster, gauss_rho_kernels_per_mm
from scf_embed_pyscf import (
    ScfEmbedConfig,
    GaussRepParams,
    cp_supermol_qm_dm,
    cp_mm_formal_charge,
    hartree_to_kcal,
    resolve_qm_charge,
    run_cp_bsse_energy,
    run_cp_fragment_b_mf,
    run_cp_supermol_mf,
    run_embedding_scf,
    run_gas_mf,
    run_mm_mf,
    scf_embed_config_from_cli,
)
from embr_io import COO_NAME_FMT, coo_path, load_e0_txt, load_e_embed_txt
from embr_scf_manifest import isotropic_repulsion_cone
from embr_envelope import ION_MM_KEYS, MmhAlphaConfig, _norm_elem_key
from embr_dataset import get_residue, resolve_dataset_n_qm


def _resolve_coo(manifest: Path | None, frame: int, coo: Path | None, residue: str | None) -> tuple[Path, str]:
    if coo is not None:
        return Path(coo).resolve(), str(residue or "gly").lower()
    if manifest is None:
        raise SystemExit("need --coo or --manifest")
    cfg = json.loads(Path(manifest).read_text(encoding="utf-8"))
    global_k = 0
    for ds in cfg.get("datasets") or []:
        res = str(ds["residue"]).lower()
        coo_dir = Path(ds["coo_dir"])
        fmt = str(ds.get("coo_name_fmt", COO_NAME_FMT))
        i0 = int(ds.get("i0", 0))
        n_frames = int(ds["n_frames"])
        for k in range(n_frames):
            if global_k == int(frame):
                return coo_path(coo_dir, fmt, i0 + k), res
            global_k += 1
    raise ValueError(f"frame {frame} not in {manifest}")


def _coo_file_index(coo_path_res: Path) -> int | None:
    m = re.search(r"(\d+)", coo_path_res.stem)
    return int(m.group(1)) if m else None


def _fmt_kcal(x: float) -> str:
    return f"{float(x):+.4f}" if np.isfinite(x) else "   n/a"


def _fmt_rho(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if abs(float(x)) >= 1e-3:
        return f"{float(x):.5f}"
    return f"{float(x):+.3e}"


def _print_axis_table(
    r: np.ndarray,
    rho_emb: np.ndarray,
    rho_ref: np.ndarray,
    *,
    title: str,
    ref_label: str,
    n_pairs: int,
    r_print_max: float,
) -> None:
    drho = rho_emb - rho_ref
    print(f"\n  [{title}]  n_pairs={n_pairs}")
    print(f"  r(Ang)   rho_emb      {ref_label:<12}  Delta(emb-ref)   rel%")
    for ir, rad in enumerate(r):
        if float(rad) > float(r_print_max) + 1e-9:
            break
        if ir % 2 == 1 and ir < r.size - 1:
            continue
        e0 = float(rho_emb[ir])
        d0 = float(rho_ref[ir])
        de = float(drho[ir])
        rel = 100.0 * de / d0 if abs(d0) > 1e-15 else float("nan")
        rel_s = f"{rel:+.1f}%" if np.isfinite(rel) else "  n/a"
        print(
            f"  {rad:5.2f}  {_fmt_rho(e0):>12}  {_fmt_rho(d0):>12}  "
            f"{_fmt_rho(de):>12}  {rel_s:>8}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Emb0 + CP(BSSE) + QM–MM axis rho(r)")
    ap.add_argument("--coo", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--residue", type=str, default=None)
    ap.add_argument("--n-qm", type=int, default=None)
    ap.add_argument("--r-cut-mm", type=float, default=None)
    ap.add_argument("--method", type=str, default="b3lyp")
    ap.add_argument("--basis", type=str, default="6-31g*")
    ap.add_argument("--cart", action="store_true", help="Cartesian d (Gaussian 6-31G* default)")
    ap.add_argument("--d3bj", action="store_true", help="Grimme D3BJ (Gaussian em=GD3BJ)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--qm-charge",
        type=int,
        default=None,
        help="QM net formal charge (Asp⁻=-1; default 0 or from --manifest scf.qm_charge)",
    )
    ap.add_argument("--verbose-scf", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("hf_emb0_cp_frame.npz"))
    ap.add_argument("--e0-file", type=Path, default=None)
    ap.add_argument("--e0-line", type=int, default=None, help="1-based line (default: Coo number)")
    ap.add_argument("--show-totals", action="store_true")
    ap.add_argument("--no-rdf", action="store_true")
    ap.add_argument("--pair-max", type=float, default=2.0, help="max QM–MM distance (Å) for axis pairs")
    ap.add_argument("--line-max", type=float, default=2.0, help="r grid 0..line-max (Å) along axis")
    ap.add_argument("--line-dr", type=float, default=0.1)
    ap.add_argument("--r-print-max", type=float, default=2.0)
    ap.add_argument(
        "--fix-alpha",
        type=float,
        default=3.0,
        help="H reference α [Bohr⁻²]; with --multi-alpha scales Na/K/Cl (approach A)",
    )
    ap.add_argument(
        "--multi-alpha",
        action="store_true",
        help="α_elem = α_H·(R_H/R_elem)² for H/Na/K/Cl (O uses α_H)",
    )
    ap.add_argument(
        "--alpha-by-element",
        type=str,
        default=None,
        help='explicit map JSON, e.g. \'{"H":3,"Na":0.35,"Cl":0.11,"O":3}\'',
    )
    ap.add_argument("--alpha-na", type=float, default=None)
    ap.add_argument("--alpha-k", type=float, default=None)
    ap.add_argument("--no-save-kernels", action="store_true", help="omit kernel_mm from ref npz")
    args = ap.parse_args()
    do_rdf = not bool(args.no_rdf)

    alpha_cfg = MmhAlphaConfig.parse(
        fix_alpha=float(args.fix_alpha),
        multi_alpha=bool(args.multi_alpha),
        alpha_by_element=args.alpha_by_element,
        alpha_na=args.alpha_na,
        alpha_k=args.alpha_k,
    )

    coo_path_res, residue = _resolve_coo(args.manifest, int(args.frame), args.coo, args.residue)
    if args.n_qm is not None:
        n_qm = int(args.n_qm)
    elif args.manifest is not None:
        man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        n_qm = None
        for ds in man.get("datasets") or []:
            if str(ds.get("residue", "")).lower() == str(residue).lower():
                n_qm = resolve_dataset_n_qm(ds, residue=residue)
                break
        if n_qm is None:
            n_qm = int(get_residue(residue).n_qm)
    else:
        n_qm = int(get_residue(residue).n_qm)
    frame = filter_mm_by_distance(load_scf_frame(coo_path_res, n_qm=n_qm), r_cut_ang=args.r_cut_mm)

    from scf_embed_pyscf import resolve_qm_charge

    man_for_q = None
    if args.manifest is not None:
        man_for_q = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qm_charge = int(
        args.qm_charge
        if args.qm_charge is not None
        else resolve_qm_charge(man_for_q, default=0)
    )
    cfg: ScfEmbedConfig = scf_embed_config_from_cli(
        method=str(args.method),
        basis=str(args.basis),
        use_d3bj=bool(args.d3bj),
        num_threads=int(args.threads),
        verbose=int(args.verbose_scf),
        cart=bool(args.cart),
        qm_charge=qm_charge,
    )

    fi = _coo_file_index(coo_path_res)
    n_ion = sum(1 for s in frame.mm_symbols if _norm_elem_key(s) in ION_MM_KEYS)
    n_wat = (len(frame.mm_symbols) - n_ion) // 3
    mm_q = int(cp_mm_formal_charge(frame))
    print(
        f"[hf_ref] coo={coo_path_res}  residue={residue}  n_qm={n_qm}  "
        f"n_mm={len(frame.mm_symbols)}  ions={n_ion}  waters={n_wat}  "
        f"qm_charge={qm_charge:+d}  MM_formal_q={mm_q:+d}  "
        f"cluster_q={qm_charge + mm_q:+d}"
    )
    print(
        f"  xc={cfg.xc}/{cfg.basis}  cart={cfg.cart}  d3bj={cfg.use_d3bj}  "
        f"threads={cfg.num_threads}"
    )

    print("\n  --- SCF: Emb0 + CP (5 jobs: cluster + 4 CP steps) ---")
    print("  [Emb0] QM + MM point charges ...")
    emb_res = run_embedding_scf(frame, cfg, rep=GaussRepParams(amp_o_hartree=0.0, amp_h_hartree=0.0))
    mf_emb = emb_res.mf
    dm_emb = np.asarray(mf_emb.make_rdm1(), dtype=np.float64)
    e_int_emb_kcal = hartree_to_kcal(float(emb_res.e_int_hartree))

    print("  [CP 1/4] A + ghost-B ...")
    mf_ab = run_cp_supermol_mf(frame, cfg)
    dm_cp_tot = np.asarray(mf_ab.make_rdm1(), dtype=np.float64)
    dm_cp_qm = np.asarray(cp_supermol_qm_dm(mf_ab, frame), dtype=np.float64)

    print("  [CP 2/4] ghost-A + B ...")
    mf_b_ga = run_cp_fragment_b_mf(frame, cfg)

    print("  [CP 3/4] monomer A (isolated ala) ...")
    mf_a = run_gas_mf(frame, cfg)
    dm_a = np.asarray(mf_a.make_rdm1(), dtype=np.float64)

    print("  [CP 4/4] monomer B (whole water shell, one SCF) ...")
    mf_b = run_mm_mf(frame, cfg)

    print("  [Cluster] real ala + real waters ...")
    cp = run_cp_bsse_energy(frame, cfg, mf_ab=mf_ab, mf_b_ga=mf_b_ga, mf_a=mf_a, mf_b=mf_b)
    mf_cl = cp.mf_cluster
    dm_cl_tot = np.asarray(mf_cl.make_rdm1(), dtype=np.float64)
    dm_cl_qm = np.asarray(cluster_qm_dm(mf_cl, frame), dtype=np.float64)

    e_int_cp_kcal = hartree_to_kcal(float(cp.e_int_cp_hartree))
    e_int_raw_kcal = hartree_to_kcal(float(cp.e_int_raw_hartree))
    bsse_kcal = hartree_to_kcal(float(cp.e_int_raw_hartree - cp.e_int_cp_hartree))
    e0_kcal = float(bsse_kcal - e_int_emb_kcal)

    if args.show_totals:
        print("\n  --- raw SCF totals (debug only) ---")
        print(f"  E_tot cluster  = {_fmt_kcal(hartree_to_kcal(float(mf_cl.e_tot)))} kcal/mol")
        print(f"  E_tot A+ghostB = {_fmt_kcal(hartree_to_kcal(float(mf_ab.e_tot)))} kcal/mol")
        print(f"  E_tot ghostA+B = {_fmt_kcal(hartree_to_kcal(float(mf_b_ga.e_tot)))} kcal/mol")
        print(f"  E_tot monomerA = {_fmt_kcal(hartree_to_kcal(float(mf_a.e_tot)))} kcal/mol")
        print(f"  E_tot monomerB = {_fmt_kcal(hartree_to_kcal(float(mf_b.e_tot)))} kcal/mol")

    print("\n  ========== Interaction energies ==========")
    print(f"  E_int^Emb0     = E_emb − E(QM)                    = {_fmt_kcal(e_int_emb_kcal)} kcal/mol")
    print(f"  E_int^CP4      = E(A+ghB)+E(ghA+B)−E(A)−E(B)     = {_fmt_kcal(e_int_cp_kcal)} kcal/mol  [legacy key e_int_cp]")
    print(f"  E_int^raw      = E(cluster) − E(A) − E(B)         = {_fmt_kcal(e_int_raw_kcal)} kcal/mol")
    print(f"  E_full_QM      = raw − CP4 (counterpoise-corr.)   = {_fmt_kcal(bsse_kcal)} kcal/mol  [NPZ: bsse_kcal]")
    print(f"  E0 ≡ ΔE        = E_full_QM − E_int^Emb0           = {_fmt_kcal(e0_kcal)} kcal/mol")

    e0_label = float("nan")
    if args.e0_file is not None:
        line1 = int(args.e0_line if args.e0_line is not None else (fi if fi is not None else 1))
        try:
            e0_arr, _ = load_e_embed_txt(args.e0_file)
        except ValueError:
            e0_arr = load_e0_txt(args.e0_file)
        if line1 < 1 or line1 > int(e0_arr.size):
            print(f"  WARNING: e0 line {line1} out of range (file has {e0_arr.size} lines)")
        else:
            e0_label = float(e0_arr[line1 - 1])
            print(f"  E0 label (line {line1})                        = {_fmt_kcal(e0_label)} kcal/mol")
            print(f"  E0 − label                                     = {_fmt_kcal(e0_kcal - e0_label)} kcal/mol")

    rdf_payload: dict = {}
    if do_rdf:
        print(
            f"\n  ========== rho(r) on QM–MM axis (MM r=0 → QM, pair_max={args.pair_max} Å) =========="
        )
        r_grid, emb_g, cl_g, pairs = compute_qm_mm_line_rho_emb_cluster(
            frame,
            mf_emb,
            mf_cl,
            dm_emb,
            dm_cl_qm,
            pair_max_ang=float(args.pair_max),
            dr=float(args.line_dr),
            r_line_max=float(args.line_max),
        )
        n_o = len([p for p in pairs if p.mm_symbol.strip()[0].upper() == "O"])
        n_h = len(pairs) - n_o
        _print_axis_table(
            r_grid,
            emb_g["mm_O"],
            cl_g["mm_O"],
            title="MM O: Emb0 vs Cluster^QM",
            ref_label="rho_cl^QM",
            n_pairs=n_o,
            r_print_max=float(args.r_print_max),
        )
        _print_axis_table(
            r_grid,
            emb_g["mm_H"],
            cl_g["mm_H"],
            title="MM H: Emb0 vs Cluster^QM",
            ref_label="rho_cl^QM",
            n_pairs=n_h,
            r_print_max=float(args.r_print_max),
        )
        rdf_payload = {
            "r_axis_ang": r_grid,
            "rho_emb_o": emb_g["mm_O"],
            "rho_emb_h": emb_g["mm_H"],
            "rho_cluster_qm_o": cl_g["mm_O"],
            "rho_cluster_qm_h": cl_g["mm_H"],
            "drho_emb_cluster_o": emb_g["mm_O"] - cl_g["mm_O"],
            "drho_emb_cluster_h": emb_g["mm_H"] - cl_g["mm_H"],
            "n_axis_pairs_o": np.int64(n_o),
            "n_axis_pairs_h": np.int64(n_h),
            "pair_max_ang": np.float64(float(args.pair_max)),
        }

    meta = {
        "pipeline": "run_hf_emb0_cp_frame",
        "coo_path": str(coo_path_res),
        "file_index": fi,
        "residue": residue,
        "n_qm": n_qm,
        "n_mm": int(len(frame.mm_symbols)),
        "xc": cfg.xc,
        "basis": cfg.basis,
        "cart": bool(cfg.cart),
        "use_d3bj": bool(cfg.use_d3bj),
        "qm_charge": int(cfg.qm_charge),
        "e_int_emb_kcal": float(e_int_emb_kcal),
        "e_int_cp_kcal": float(e_int_cp_kcal),
        "e_int_raw_kcal": float(e_int_raw_kcal),
        "bsse_kcal": float(bsse_kcal),
        "e0_kcal": float(e0_kcal),
        "e0_label_kcal": None if not np.isfinite(e0_label) else float(e0_label),
        "rho_geometry": "qm_mm_axis",
        "formulas": {
            "e_int_emb": "E_emb - E_monomer_QM",
            "e_int_cp": "E(A+ghostB)+E(ghostA+B)-E(A)-E(B)  [legacy name; 4-term ghost basis]",
            "e_int_raw": "E(cluster)-E(A)-E(B)",
            "bsse_kcal": "E_int_raw - E_int_cp  (= counterpoise-corrected E_full_QM)",
            "e0_kcal": "bsse_kcal - E_int_emb  (= ΔE target = E_full_QM - E_Emb0)",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    npz_kw: dict = {
        "meta_json": np.array(json.dumps(meta)),
        "e_int_emb_hartree": np.float64(float(emb_res.e_int_hartree)),
        "e_int_cp_hartree": np.float64(float(cp.e_int_cp_hartree)),
        "e_int_raw_hartree": np.float64(float(cp.e_int_raw_hartree)),
        "e0_kcal": np.float64(float(e0_kcal)),
        "e_emb_hartree": np.float64(float(emb_res.e_total_hartree)),
        "e_gas_hartree": np.float64(float(emb_res.e_gas_hartree)),
        "e_cluster_hartree": np.float64(float(cp.e_cluster_hartree)),
        "e_ab_hartree": np.float64(cp.e_ab_hartree),
        "e_b_ga_hartree": np.float64(cp.e_b_ga_hartree),
        "e_a_hartree": np.float64(cp.e_a_hartree),
        "e_b_hartree": np.float64(cp.e_b_hartree),
        "dm_emb": dm_emb,
        "dm_cp_tot": dm_cp_tot,
        "dm_cp_qm": dm_cp_qm,
        "dm_cluster_qm": dm_cl_qm,
        "dm_cluster_tot": dm_cl_tot,
        "dm_a": dm_a,
    }
    npz_kw.update(rdf_payload)

    if not bool(args.no_save_kernels):
        alpha_tag = (
            f"uniform {alpha_cfg.legacy_fix_alpha():g}"
            if alpha_cfg.is_uniform()
            else alpha_cfg.to_meta()
        )
        print(f"  [k_j] density-overlap kernels (α Bohr^-2: {alpha_tag}) ...")
        k_kw: dict = {"alpha_bohr2": float(alpha_cfg.legacy_fix_alpha())}
        if not alpha_cfg.is_uniform():
            k_kw["alpha_by_element"] = alpha_cfg.to_meta()
        per_k = gauss_rho_kernels_per_mm(
            mf_emb,
            frame,
            cone=isotropic_repulsion_cone(),
            dm=dm_emb,
            **k_kw,
        )
        npz_kw["kernel_mm"] = np.asarray(per_k.kernel_mm, dtype=np.float64)
        npz_kw["mm_element_k"] = np.asarray(per_k.mm_element, dtype=np.int8)
        npz_kw["fix_alpha_bohr2"] = np.float64(float(alpha_cfg.legacy_fix_alpha()))
        npz_kw["rho_o_mean_k"] = np.float64(float(per_k.rho_o_mean))
        meta["fix_alpha_bohr2"] = float(alpha_cfg.legacy_fix_alpha())
        if not alpha_cfg.is_uniform():
            meta["fix_alpha_by_element"] = alpha_cfg.to_meta()
            npz_kw["fix_alpha_by_element_json"] = np.array(json.dumps(alpha_cfg.to_meta()))
        meta["n_mm_k"] = int(per_k.kernel_mm.size)
        npz_kw["meta_json"] = np.array(json.dumps(meta))

    np.savez_compressed(args.out, **npz_kw)
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
