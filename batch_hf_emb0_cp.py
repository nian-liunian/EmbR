"""
Batch ``run_hf_emb0_cp_frame.py`` over a manifest Coo range.

Writes E0 label file (col1 = E0, col2 = E_int^Emb0) and ``ref/ref_*.npz``.

Example::

  python batch_hf_emb0_cp.py \\
    --manifest manifest.json \\
    --prefix Gly+ --frame-start 1 --frame-end 100 \\
    --scf-preset hf --threads 4

Resume: ``--skip-existing`` skips SCF when ``ref_*.npz`` exists; E0 txt appends one line per frame.

Deploy **together** on the server (same git sync / copy batch)::

  batch_hf_emb0_cp.py
  run_hf_emb0_cp_frame.py
  scf_embed_pyscf.py
  scf_embed_cluster.py
  scf_embed_perturb.py   ← gauss_rho_kernels_per_mm(alpha_by_element=…)
  scf_embed_io.py
  embr_io.py
  embr_ref_kernels.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scf_embed_pyscf import (
    default_batch_out_dir,
    hartree_to_kcal,
    missing_cp_ion_mm_support,
    resolve_scf_preset,
    scf_embed_config_from_cli,
)
try:
    from scf_embed_pyscf import basis_has_diffuse, is_hf_xc, resolve_scf_method
except ImportError:
    def resolve_scf_method(method: str) -> str:
        m = str(method).strip().lower().replace("-", "").replace("_", "")
        return "HF" if m in ("hf", "rhf", "hartreefock") else "B3LYP"

    def is_hf_xc(xc: str) -> bool:
        return resolve_scf_method(xc) == "HF"

    def basis_has_diffuse(basis: str) -> bool:
        return "+" in str(basis)

from embr_io import COO_NAME_FMT, coo_path
from embr_scf_manifest import scf_settings_from_manifest
from embr_envelope import MmhAlphaConfig
from embr_dataset import resolve_dataset_label, resolve_dataset_n_qm


def _resolve_manifest_path(manifest: Path, p: Path | str) -> Path:
    """Resolve dataset paths relative to manifest.json location."""
    path = Path(p)
    if path.exists():
        return path.resolve()
    cand = (manifest.parent / path).resolve()
    if cand.exists():
        return cand
    return cand


def _dataset_matches(
    ds: dict,
    label: str,
    ds_index: int,
    *,
    residue: str | None,
    prefix: str | None,
    dataset_index: int | None,
) -> bool:
    if dataset_index is not None and int(dataset_index) != int(ds_index):
        return False
    if prefix is not None:
        p = str(ds.get("prefix") or "").strip().lower()
        return p == str(prefix).strip().lower()
    if residue is not None:
        r = str(residue).strip().lower()
        ds_res = str(ds.get("residue") or "").strip().lower()
        return ds_res == r or str(label).strip().lower() == r
    return True


def build_ref_frame_tasks(
    manifest: Path,
    frame_start: int,
    frame_end: int,
    *,
    residue: str | None = None,
    prefix: str | None = None,
    dataset_index: int | None = None,
) -> tuple[list[dict], Path]:
    """
    Coo tasks for ref batch — does **not** read any existing E0 file.

    Select one ``datasets[]`` block by ``prefix`` (preferred), legacy ``residue``,
    or ``dataset_index``. With a single dataset, no selector is required.

    Returns (tasks, manifest_parent_for_default_e0_out).
    """
    manifest = Path(manifest).resolve()
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    datasets = list(cfg.get("datasets") or [])
    if not datasets:
        raise ValueError(f"{manifest}: no datasets[]")
    if (
        residue is None
        and prefix is None
        and dataset_index is None
        and len(datasets) != 1
    ):
        raise ValueError(
            f"{manifest}: multiple datasets — pass --prefix, --dataset-index, "
            "or legacy --residue"
        )
    tasks: list[dict] = []
    e0_parent: Path | None = None
    global_k = 0
    found_ds = False

    for ds_index, ds in enumerate(datasets):
        label = resolve_dataset_label(ds)
        n_qm = resolve_dataset_n_qm(ds)
        if not _dataset_matches(
            ds,
            label,
            ds_index,
            residue=residue,
            prefix=prefix,
            dataset_index=dataset_index,
        ):
            global_k += int(ds["n_frames"])
            continue
        coo_dir = _resolve_manifest_path(manifest, ds["coo_dir"])
        fmt = str(ds.get("coo_name_fmt", COO_NAME_FMT))
        i0 = int(ds.get("i0", 0))
        n_frames = int(ds["n_frames"])
        if "e0_file" in ds:
            e0_parent = _resolve_manifest_path(manifest, ds["e0_file"]).parent

        for k in range(n_frames):
            file_index = i0 + k
            if frame_start <= file_index <= frame_end:
                p = coo_path(coo_dir, fmt, file_index)
                if not p.is_file():
                    raise FileNotFoundError(p)
                tasks.append(
                    {
                        "slice_index": len(tasks),
                        "global_index": global_k,
                        "residue": label,
                        "local_k": k,
                        "file_index": file_index,
                        "coo_path": str(p.resolve()),
                        "n_qm": n_qm,
                    }
                )
                found_ds = True
            global_k += 1

    if not found_ds:
        sel = (
            f"prefix={prefix!r}"
            if prefix is not None
            else f"residue={residue!r}"
            if residue is not None
            else f"dataset_index={dataset_index}"
        )
        raise ValueError(
            f"no frames for {sel} in [{frame_start}, {frame_end}] "
            f"(check manifest datasets / Coo files)"
        )
    tasks.sort(key=lambda t: int(t["file_index"]))
    for i, t in enumerate(tasks):
        t["slice_index"] = i
    parent = e0_parent if e0_parent is not None and e0_parent.is_dir() else manifest.parent
    return tasks, parent


def resolve_manifest_e0_out(
    manifest: Path,
    *,
    residue: str | None = None,
    prefix: str | None = None,
    dataset_index: int | None = None,
) -> Path | None:
    """Full path from dataset ``e0_file`` if set for the selected block."""
    manifest = Path(manifest).resolve()
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    datasets = list(cfg.get("datasets") or [])
    for ds_index, ds in enumerate(datasets):
        label = resolve_dataset_label(ds)
        if not _dataset_matches(
            ds,
            label,
            ds_index,
            residue=residue,
            prefix=prefix,
            dataset_index=dataset_index,
        ):
            continue
        raw = ds.get("e0_file")
        if not raw:
            continue
        p = _resolve_manifest_path(manifest, raw)
        if not str(p.name).endswith(".txt"):
            p = p.with_suffix(".txt")
        return p
    return None


def _default_ref_e0_filename(residue: str, method: str, basis: str) -> str:
    """EAla_+.txt naming; kept here so batch works without newest scf_embed_pyscf."""
    r = str(residue).strip().lower().capitalize()
    hf = is_hf_xc(resolve_scf_method(method))
    plus = basis_has_diffuse(basis)
    if hf and plus:
        return f"E{r}_hf_+.txt"
    if hf:
        return f"E{r}_hf.txt"
    if plus:
        return f"E{r}_+.txt"
    return f"E{r}_ref.txt"


def _e0_header(
    *,
    residue: str,
    method: str,
    basis: str,
    d3bj: bool,
    frame_start: int,
    frame_end: int,
) -> str:
    return (
        f"PySCF ref batch  residue={residue}  frames={frame_start}..{frame_end}  "
        f"method={method}  basis={basis}  d3bj={d3bj}  "
        f"col1=E0=(E_int_raw-E_int_CP)-E_int_Emb0  col2=E_int_Emb0  kcal/mol"
    )


def _count_e0_data_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n


def _ensure_e0_header(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        path.write_text(f"# {header}\n", encoding="utf-8")


def _append_e0_line(path: Path, e0_kcal: float, e_int_emb_kcal: float) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{float(e0_kcal):.8f}  {float(e_int_emb_kcal):.8f}\n")


def _fmt_elapsed(sec: float) -> str:
    sec = float(sec)
    if sec < 60.0:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    return f"{sec:.1f}s ({m}m{int(sec % 60):02d}s)"


def _frame_script_help(script: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _emb0_sync_hint() -> str:
    return (
        "Keep Emb0/CP scripts in sync within this repo:\n"
        "  batch_hf_emb0_cp.py  run_hf_emb0_cp_frame.py\n"
        "  scf_embed_pyscf.py  scf_embed_cluster.py  scf_embed_perturb.py\n"
        "  scf_embed_io.py  embr_io.py  embr_ref_kernels.py"
    )


def _verify_multi_alpha_kernel_api(*, save_kernels: bool, alpha_cfg: MmhAlphaConfig) -> None:
    if not save_kernels or alpha_cfg.is_uniform():
        return
    try:
        from scf_embed_perturb import gauss_rho_kernels_per_mm
    except ImportError as e:
        raise SystemExit(f"cannot import scf_embed_perturb: {e}\n{_emb0_sync_hint()}") from e
    params = inspect.signature(gauss_rho_kernels_per_mm).parameters
    if "alpha_by_element" not in params:
        raise SystemExit(
            "scf_embed_perturb.py is outdated: gauss_rho_kernels_per_mm() "
            "missing alpha_by_element (multi-α k_j).\n"
            f"{_emb0_sync_hint()}"
        )


def _verify_scf_embed_cp_ions() -> None:
    missing = missing_cp_ion_mm_support()
    if missing:
        raise SystemExit(
            f"scf_embed_pyscf.py on this machine is water-only CP (missing ion ghost: {list(missing)}).\n"
            f"Sync scf_embed_pyscf.py + scf_embed_cluster.py from the same commit as batch_hf_emb0_cp.py."
        )


def _verify_frame_script(
    script: Path,
    *,
    alpha_cfg: MmhAlphaConfig,
    save_kernels: bool,
) -> None:
    if not script.is_file():
        raise SystemExit(f"missing frame script: {script.resolve()}")
    help_text = _frame_script_help(script)
    if proc_failed := ("error:" in help_text.lower() and "usage:" not in help_text):
        raise SystemExit(f"cannot run {script.name} --help:\n{help_text[:2000]}")
    required = ["--r-cut-mm", "--no-save-kernels"]
    if save_kernels and not alpha_cfg.is_uniform():
        required.append("--alpha-by-element")
    missing = [f for f in required if f not in help_text]
    if missing:
        raise SystemExit(
            f"{script.name} on this machine is outdated (missing {missing}).\n"
            f"{_emb0_sync_hint()}\n"
            f"(batch passes multi-α k_j flags to the frame script when save_kernels=True.)"
        )


def _run_one(
    *,
    script: Path,
    task: dict,
    out_npz: Path,
    method: str,
    basis: str,
    d3bj: bool,
    threads: int,
    verbose_scf: int,
    pair_max: float,
    line_max: float,
    line_dr: float,
    no_rdf: bool,
    alpha_cfg: MmhAlphaConfig,
    r_cut_mm: float | None,
    save_kernels: bool = False,
    qm_charge: int = 0,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(script),
        "--coo",
        str(task["coo_path"]),
        "--residue",
        str(task["residue"]),
        "--method",
        str(method),
        "--basis",
        str(basis),
        "--out",
        str(out_npz),
        "--threads",
        str(int(threads)),
        "--verbose-scf",
        str(int(verbose_scf)),
        "--pair-max",
        str(float(pair_max)),
        "--line-max",
        str(float(line_max)),
        "--line-dr",
        str(float(line_dr)),
        "--qm-charge",
        str(int(qm_charge)),
    ]
    if save_kernels:
        cmd.extend(["--fix-alpha", str(float(alpha_cfg.legacy_fix_alpha()))])
        if not alpha_cfg.is_uniform():
            cmd.extend(["--alpha-by-element", json.dumps(alpha_cfg.to_meta())])
    else:
        cmd.append("--no-save-kernels")
    if r_cut_mm is not None:
        cmd.extend(["--r-cut-mm", str(float(r_cut_mm))])
    if int(task.get("n_qm", 0)) > 0:
        cmd.extend(["--n-qm", str(int(task["n_qm"]))])
    if d3bj:
        cmd.append("--d3bj")
    if no_rdf:
        cmd.append("--no-rdf")
    return subprocess.run(cmd, capture_output=True, text=True)


def _read_ref_npz(path: Path) -> tuple[float, float]:
    e0, ebg, _ = _read_ref_npz_energies(path)
    return e0, ebg


def _read_ref_npz_energies(path: Path) -> tuple[float, float, dict]:
    with np.load(path, allow_pickle=True) as z:
        if "meta_json" in z:
            meta = json.loads(str(z["meta_json"]))
            e0 = float(meta["e0_kcal"])
            ebg = float(meta["e_int_emb_kcal"])
            extra = {
                "e_int_cp_kcal": float(meta.get("e_int_cp_kcal", float("nan"))),
                "e_int_raw_kcal": float(meta.get("e_int_raw_kcal", float("nan"))),
                "bsse_kcal": float(meta.get("bsse_kcal", float("nan"))),
                "xc": str(meta.get("xc", "")),
                "basis": str(meta.get("basis", "")),
                "use_d3bj": bool(meta.get("use_d3bj", False)),
            }
            return e0, ebg, extra
        e0 = float(hartree_to_kcal(float(z["e0_kcal"])))
        ebg = float(hartree_to_kcal(float(z["e_int_emb_hartree"])))
        return e0, ebg, {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Emb0+CP ref → new E0 txt (e.g. EAla_+.txt)")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="select datasets[] by prefix (preferred; matches run.py manifest)",
    )
    ap.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="select datasets[] by 0-based index",
    )
    ap.add_argument(
        "--residue",
        type=str,
        default=None,
        help="legacy: select datasets[] by residue label (optional if one dataset)",
    )
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--frame-end", type=int, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="npz directory (default ref_{tag} from SCF variant)",
    )
    ap.add_argument(
        "--e0-out",
        type=Path,
        default=None,
        help="new E0 file (default: beside manifest e0_file, e.g. EAla_+.txt)",
    )
    ap.add_argument(
        "--method",
        "--xc",
        type=str,
        default=None,
        dest="method",
        help="SCF functional (alias --xc); default from manifest scf.method",
    )
    ap.add_argument("--basis", type=str, default=None, help="override manifest scf.basis")
    ap.add_argument("--d3bj", action="store_true", help="force D3BJ on")
    ap.add_argument("--no-d3bj", action="store_true", help="force D3BJ off")
    ap.add_argument(
        "--scf-preset",
        type=str,
        default=None,
        choices=("hf", "b3lyp-plus", "hf-plus"),
        help="override manifest scf (hf → HF/6-31g*; b3lyp-plus → 6-31+G* + D3BJ)",
    )
    ap.add_argument("--threads", type=int, default=None, help="override manifest scf.threads")
    ap.add_argument(
        "--qm-charge",
        type=int,
        default=None,
        help="QM net formal charge (Asp⁻=-1; default: manifest scf.qm_charge or 0)",
    )
    ap.add_argument("--verbose-scf", type=int, default=None)
    ap.add_argument("--pair-max", type=float, default=2.0)
    ap.add_argument("--line-max", type=float, default=2.0)
    ap.add_argument("--line-dr", type=float, default=0.1)
    ap.add_argument(
        "--fix-alpha",
        type=float,
        default=None,
        help="H reference α [Bohr⁻²]; default manifest fix_alpha or 3.0",
    )
    ap.add_argument(
        "--multi-alpha",
        action="store_true",
        help="α_elem = α_H·(R_H/R_elem)² (default if manifest multi_alpha or fix_alpha_by_element)",
    )
    ap.add_argument("--alpha-by-element", type=str, default=None)
    ap.add_argument("--alpha-na", type=float, default=None)
    ap.add_argument("--alpha-k", type=float, default=None)
    ap.add_argument(
        "--r-cut-mm",
        type=float,
        default=None,
        help="optional MM distance filter (Å); default manifest r_cut_mm (null=off)",
    )
    ap.add_argument("--no-rdf", action="store_true", help="energy only (no axis rho in npz)")
    ap.add_argument(
        "--save-kernels",
        action="store_true",
        help="legacy: store k_j in ref npz (default off; precompute recomputes k from dm_emb)",
    )
    ap.add_argument(
        "--no-save-kernels",
        action="store_true",
        help="deprecated alias; k_j off is now the default",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.manifest is None:
        raise SystemExit("--manifest is required")
    manifest_cfg = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    multi_alpha = bool(args.multi_alpha) or bool(manifest_cfg.get("multi_alpha", False))
    fix_alpha_cli = (
        float(args.fix_alpha)
        if args.fix_alpha is not None
        else float(manifest_cfg.get("fix_alpha", 3.0))
    )
    alpha_cfg = MmhAlphaConfig.parse(
        fix_alpha=fix_alpha_cli,
        multi_alpha=multi_alpha,
        alpha_by_element=args.alpha_by_element,
        alpha_na=args.alpha_na,
        alpha_k=args.alpha_k,
        manifest=manifest_cfg,
    )

    scf_manifest = scf_settings_from_manifest(manifest_cfg)
    method = str(args.method or scf_manifest["method"])
    basis = str(args.basis or scf_manifest["basis"])
    d3bj = bool(scf_manifest["use_d3bj"])
    if args.scf_preset is not None:
        preset = resolve_scf_preset(str(args.scf_preset))
        method = str(preset["method"])
        basis = str(preset["basis"])
        d3bj = bool(preset["d3bj"])
    if args.d3bj:
        d3bj = True
    if args.no_d3bj:
        d3bj = False

    qm_charge = int(
        args.qm_charge
        if args.qm_charge is not None
        else scf_manifest.get("qm_charge", 0)
    )
    cfg = scf_embed_config_from_cli(
        method=method,
        basis=basis,
        use_d3bj=d3bj,
        num_threads=int(args.threads if args.threads is not None else scf_manifest["num_threads"]),
        verbose=int(args.verbose_scf if args.verbose_scf is not None else scf_manifest["verbose"]),
        qm_charge=qm_charge,
    )

    r_cut_mm = manifest_cfg.get("r_cut_mm") if args.r_cut_mm is None else args.r_cut_mm
    if r_cut_mm is not None:
        r_cut_mm = float(r_cut_mm)
        if r_cut_mm <= 0.0:
            r_cut_mm = None

    fs, fe = int(args.frame_start), int(args.frame_end)
    residue_tag = (
        str(args.prefix).strip().lower()
        if args.prefix
        else str(args.residue).strip().lower()
        if args.residue
        else "dataset"
    )
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        tag = default_batch_out_dir(residue_tag, fs, fe, method=method, basis=cfg.basis).name
        out_dir = Path(f"ref_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, e0_parent = build_ref_frame_tasks(
        Path(args.manifest),
        fs,
        fe,
        residue=args.residue,
        prefix=args.prefix,
        dataset_index=args.dataset_index,
    )
    if tasks:
        residue_tag = str(tasks[0]["residue"])
    if args.e0_out is not None:
        e0_out = Path(args.e0_out)
    else:
        e0_from_manifest = resolve_manifest_e0_out(
            Path(args.manifest),
            residue=args.residue,
            prefix=args.prefix,
            dataset_index=args.dataset_index,
        )
        if e0_from_manifest is not None:
            e0_out = e0_from_manifest
        else:
            e0_out = e0_parent / _default_ref_e0_filename(residue_tag, method, cfg.basis)

    script = _ROOT / "run_hf_emb0_cp_frame.py"
    save_kernels = bool(args.save_kernels) and not bool(args.no_save_kernels)
    _verify_scf_embed_cp_ions()
    if save_kernels:
        _verify_multi_alpha_kernel_api(save_kernels=True, alpha_cfg=alpha_cfg)
    _verify_frame_script(script, alpha_cfg=alpha_cfg, save_kernels=save_kernels)

    print(
        f"[batch_QM] dataset={residue_tag}  frames={fs}..{fe}  n={len(tasks)}  "
        f"xc={cfg.xc}/{cfg.basis}  d3bj={cfg.use_d3bj}  qm_charge={int(cfg.qm_charge):+d}\n"
        f"  npz → {out_dir.resolve()}\n"
        f"  dE → {e0_out.resolve()}"
    )

    e0_header = _e0_header(
        residue=residue_tag,
        method=method,
        basis=cfg.basis,
        d3bj=bool(cfg.use_d3bj),
        frame_start=fs,
        frame_end=fe,
    )
    if not args.dry_run:
        _ensure_e0_header(e0_out, e0_header)
    n_e0_done = _count_e0_data_lines(e0_out) if not args.dry_run else 0
    if n_e0_done:
        print(f"  dE file already has {n_e0_done} line(s)")

    ok, skip, fail, appended = 0, 0, 0, 0
    scf_times: list[float] = []
    t0 = time.time()

    for task in tasks:
        fi = int(task["file_index"])
        lk = int(task["local_k"])
        out_npz = out_dir / f"ref_{fi}.npz"
        label = f"Coo{fi}"
        has_npz = out_npz.is_file()
        e0_line_done = lk < n_e0_done

        if e0_line_done:
            if args.skip_existing and has_npz:
                print(f"  [skip] {label}  npz+e0 line ok")
                skip += 1
                continue
            print(f"  [skip-e0] {label}  line {lk + 1} already in {e0_out.name}")

        if not e0_line_done and args.skip_existing and has_npz:
            e0_kcal, ebg_kcal = _read_ref_npz(out_npz)
            _append_e0_line(e0_out, e0_kcal, ebg_kcal)
            n_e0_done += 1
            appended += 1
            skip += 1
            print(
                f"  [append] {label}  from npz  dE={e0_kcal:+.4f}  "
                f"E(Emb0)={ebg_kcal:+.4f} kcal/mol"
            )
            continue

        if args.dry_run:
            print(f"  [dry-run] {label}  -> {out_npz}")
            continue

        print(f"  [run] {label}  ...", flush=True)
        t_frame = time.time()
        proc = _run_one(
            script=script,
            task=task,
            out_npz=out_npz,
            method=method,
            basis=cfg.basis,
            d3bj=bool(cfg.use_d3bj),
            threads=int(cfg.num_threads),
            verbose_scf=int(cfg.verbose),
            pair_max=float(args.pair_max),
            line_max=float(args.line_max),
            line_dr=float(args.line_dr),
            no_rdf=bool(args.no_rdf),
            alpha_cfg=alpha_cfg,
            r_cut_mm=r_cut_mm,
            save_kernels=save_kernels,
            qm_charge=int(cfg.qm_charge),
        )
        dt = time.time() - t_frame
        if proc.returncode != 0:
            fail += 1
            tail = (proc.stderr or proc.stdout or "")[-4000:]
            print(f"  [FAIL] {label}  elapsed={_fmt_elapsed(dt)}  exit={proc.returncode}\n{tail}")
            continue

        ok += 1
        scf_times.append(dt)
        e0_kcal, ebg_kcal, en_extra = _read_ref_npz_energies(out_npz)
        if not e0_line_done:
            _append_e0_line(e0_out, e0_kcal, ebg_kcal)
            n_e0_done += 1
            appended += 1
        e_full_qm = float("nan")
        if en_extra.get("bsse_kcal") is not None and np.isfinite(float(en_extra["bsse_kcal"])):
            e_full_qm = float(en_extra["bsse_kcal"])
        elif np.isfinite(e0_kcal) and np.isfinite(ebg_kcal):
            e_full_qm = float(e0_kcal) + float(ebg_kcal)
        e_qm_s = ""
        if np.isfinite(e_full_qm):
            e_qm_s = f"  E(full_QM)={e_full_qm:+.4f}"
        print(
            f"  [ok] {label}  elapsed={_fmt_elapsed(dt)}  "
            f"dE={e0_kcal:+.4f}  E(Emb0)={ebg_kcal:+.4f} kcal/mol{e_qm_s}"
        )

    elapsed = time.time() - t0
    avg_s = f"{float(np.mean(scf_times)):.1f}s" if scf_times else "n/a"
    print(
        f"\n[batch_QM] done  ok={ok}  skip={skip}  fail={fail}  "
        f"appended={appended}  delta_E_lines={_count_e0_data_lines(e0_out) if not args.dry_run else n_e0_done}  "
        f"elapsed={_fmt_elapsed(elapsed)}  avg_per_scf={avg_s}"
    )


if __name__ == "__main__":
    main()
